"""Guard the milestone's experiment controls and test-window lock."""

import json

import chess
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from catboost import CatBoostRanker
from sklearn.ensemble import RandomForestClassifier

from chess_clone.analysis.schemas import EngineLine
from chess_clone.experiments import ranker_width as experiment
from chess_clone.experiments.width_data import file_digest
from chess_clone.features.evaluation import approximate_winning_chance
from chess_clone.modeling.boosted import candidate_pool
from chess_clone.modeling.candidates import FULL_FEATURE_FIELDS, build_candidate_dataset, chronological_game_split
from test_width_experiment import broad_inputs


class FakeEngine:
    engine_identity = 'fixture'
    calls = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def analyze(self, fen, settings):
        self.calls.append((fen, settings))
        moves = sorted(m.uci() for m in chess.Board(fen).legal_moves)
        # Same legal set, intentionally not prefixes between widths.
        offset = settings.multipv
        moves = moves[offset:] + moves[:offset]
        return [EngineLine(rank=i, score_cp=50-i*10, mate_in=None, best_move_uci=move,
                           pv_uci=move, depth=5, seldepth=7, nodes_searched=20001, time_seconds=.01)
                for i, move in enumerate(moves[:settings.multipv], 1)]


def test_actual_search_configuration_and_short_legal_set():
    engine = FakeEngine()
    for k in experiment.WIDTHS:
        lines = experiment.search(engine, {'fen': chess.STARTING_FEN}, k)
        settings = engine.calls[-1][1]
        assert settings.multipv == len(lines) == k
        assert settings.nodes == 20000 and settings.threads == 1 and settings.hash_mb == 16
    fen = '8/8/8/8/8/2k5/r7/K7 w - - 0 1'
    count = chess.Board(fen).legal_moves.count()
    assert count < 3
    assert len(experiment.search(engine, {'fen': fen}, 10)) == count


def test_serving_features_equal_existing_builder_and_ignore_actual_label():
    games, features, analysis = broad_inputs()
    split = chronological_game_split(games)
    for k in experiment.WIDTHS:
        feature = features[0]
        lines = [EngineLine(rank=r['pv_rank'], score_cp=r['score_cp'], mate_in=None,
                            best_move_uci=r['best_move_uci'], pv_uci='', depth=1,
                            seldepth=1, nodes_searched=1, time_seconds=.01)
                 for r in analysis if r['game_id'] == feature['game_id'] and r['pv_rank'] <= k]
        feature = {**feature, 'approximate_winning_chance_before': approximate_winning_chance(lines[0].score_cp, None)}
        expected = build_candidate_dataset([feature], [r for r in analysis if r['game_id'] == feature['game_id']],
                                           split, candidate_k=k, include_outside=True).candidate_rows
        rows = experiment.construct_rows(feature, lines, split, 180.)
        changed = experiment.construct_rows({**feature, 'actual_move_uci': 'a2a4',
                                             'player_clock_seconds_after_move': 1., 'actual_move_rank': 999}, lines, split, 180.)
        for before, actual, other in zip(expected, rows, changed, strict=True):
            assert {f: before[f] for f in FULL_FEATURE_FIELDS} == {f: actual[f] for f in FULL_FEATURE_FIELDS}
            assert {f: actual[f] for f in FULL_FEATURE_FIELDS} == {f: other[f] for f in FULL_FEATURE_FIELDS}
            assert actual['chosen'] is False and actual['actual_move_uci'] == ''


def prediction_fixture(correct):
    return [{'decision_id': f'g{i}:1', 'game_id': f'g{i}', 'actual_move_uci': 'e2e4',
             'covered': True, 'exact_correct': i < correct, 'top_3_correct': True} for i in range(100)]


def test_validation_selection_requires_material_supported_cost_effective_gain():
    val = {k: {'latency_ms': 20.} for k in experiment.WIDTHS}
    equal = {k: prediction_fixture(50) for k in experiment.WIDTHS}
    assert experiment.choose_width(val, equal)['selected_k'] == 3
    small = {3: prediction_fixture(50), 5: prediction_fixture(51), 10: prediction_fixture(51)}
    assert experiment.choose_width(val, small)['selected_k'] == 3
    strong = {3: prediction_fixture(50), 5: prediction_fixture(50), 10: prediction_fixture(70)}
    assert experiment.choose_width(val, strong)['selected_k'] == 10
    expensive = {**val, 10: {'latency_ms': 221.}}
    assert experiment.choose_width(expensive, strong)['selected_k'] == 3


def test_test_gate_precedes_any_source_read(tmp_path):
    with pytest.raises(ValueError, match='test locked'):
        experiment.evaluate(tmp_path, 'test')
    assert not (tmp_path / 'test_opened.json').exists()


def test_coverage_transitions_allow_nonnested_actual_searches():
    from chess_clone.experiments.ranker_width_report import transitions
    left = prediction_fixture(50)
    right = prediction_fixture(60)
    left[55]['covered'] = False
    right[3]['covered'] = False
    right[3]['exact_correct'] = False
    result = transitions(left, right)
    assert result['newly_covered'] == result['newly_covered_exact_correct'] == 1
    assert result['lost_coverage'] == result['lost_exact_correct'] == 1
    assert result['net_exact_correct'] == 9
    with pytest.raises(ValueError, match='decision mismatch'):
        transitions(left, right[:-1])


def tiny_ranker(train, validation, fields):
    model = CatBoostRanker(iterations=3, depth=2, loss_function='QuerySoftMax',
                          thread_count=1, verbose=False, allow_writing_files=False)
    model.fit(candidate_pool(train, fields, grouped=True))
    return model


def test_staged_run_all_decisions_preservation_and_frozen_selection(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, 'PLAYERS', ('Target',))
    monkeypatch.setattr(experiment, 'StockfishAnalyzer', FakeEngine)
    monkeypatch.setattr(experiment, 'train_grouped_ranker', tiny_ranker)
    from chess_clone.modeling import ranker
    monkeypatch.setattr(ranker, 'make_tree_classifier', lambda: RandomForestClassifier(n_estimators=2, random_state=42))
    games, features, analysis = broad_inputs()
    # Ensure enough covered training groups at all widths and outside evaluation moves.
    for f in features:
        f['actual_move_uci'] = 'e2e3'
    features[-1]['actual_move_uci'] = 'a2a3'
    source = tmp_path / 'source' / 'target'
    source.mkdir(parents=True)
    for name, rows in [('games', games), ('features', features), ('analysis', analysis)]:
        pq.write_table(pa.Table.from_pylist(rows), source / f'{name}.parquet')
    split = chronological_game_split(games)
    (source / 'split.json').write_text(json.dumps(split.to_dict()))
    sha = {p.name: file_digest(p) for p in source.iterdir()}
    output = tmp_path / 'output'
    output.mkdir()
    protocol = {'source': str(source.parent), 'engine_identity': 'fixture',
                'selection_policy': experiment.POLICY,
                'inputs': {'Target': {'sha256': sha, 'split': split.to_dict()}}}
    (output / 'protocol.json').write_text(json.dumps(protocol))
    # A width-independent candidate rotation still uses the requested true MultiPV.
    def fake_analysis(self, fen, settings):
        moves = ['e2e3', 'e2e4', 'd2d4', 'd2d3', 'g1f3', 'b1c3', 'c2c4', 'c2c3', 'b2b3', 'a2a4']
        return [EngineLine(i, 30-i, None, move, move, 1, 1, 20000, .01)
                for i, move in enumerate(moves[:settings.multipv], 1)]
    monkeypatch.setattr(FakeEngine, 'analyze', fake_analysis)
    experiment.prepare(output)
    experiment.fit(output)
    with pytest.raises(ValueError, match='test locked'):
        experiment.evaluate(output, 'test')
    experiment.evaluate(output, 'validation')
    selection = experiment.freeze_selection(output)
    assert selection['fit_split'] == 'validation'
    experiment.evaluate(output, 'test')
    assert file_digest(output / 'selection.json') == json.loads((output / 'test_opened.json').read_text())['selection_sha256']
    for k in experiment.WIDTHS:
        dest = output / 'target' / f'k{k}'
        result = json.loads((dest / 'test_evaluation.json').read_text())
        for family, model in result['models'].items():
            assert model['metrics']['decisions'] == 2
            assert model['metrics']['candidate_coverage'] == .5
            assert model['metrics']['exact_move_accuracy'] <= .5
            assert model['metrics']['top_3_accuracy'] <= .5
            assert model['timing']['engine_calls'] == 2 and model['timing']['cache_hits'] == 0
            cost = model['timing']
            assert cost[f'{family}_end_to_end_ms']['mean'] == pytest.approx(
                cost['search_ms']['mean'] + cost['feature_ms']['mean'] + cost[f'{family}_score_ms']['mean'])
    with pytest.raises(ValueError, match='already frozen'):
        experiment.freeze_selection(output)
    frozen = output / 'target/k3/catboost_features.json'
    frozen.write_text('[]')
    with pytest.raises(ValueError, match='frozen validation/model file changed'):
        experiment.verify_selection(output)
