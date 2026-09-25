import hashlib
import json
from pathlib import Path

import chess
import joblib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from sklearn.ensemble import RandomForestClassifier

from chess_clone.benchmark.metrics import coverage, smallest_k, thresholds
from chess_clone.experiments.width_training import select_history_k, train_width_comparison
from chess_clone.experiments.width_metrics import all_decision_metrics, paired_comparison
from chess_clone.modeling.candidates import build_candidate_dataset, chronological_game_split, select_candidate_rows
from chess_clone.modeling.ranker import SparseOneHotPreprocessor, fit_global_rank_frequencies, predict_candidate_probabilities
from chess_clone.modeling.training import predict_one_decision
from test_ranking import _inputs, MOVES


def broad_inputs():
    games, features, analysis = _inputs(count=10, outside_last=True)
    extra = sorted(m.uci() for m in chess.Board().legal_moves if m.uci() not in MOVES)
    for feature in features:
        for rank, move in enumerate(extra, 6):
            analysis.append({'game_id': feature['game_id'], 'ply': 1, 'player_username': 'Target',
                             'canonical_position': 'canonical start', 'pv_rank': rank,
                             'score_cp': 50 - rank * 10, 'mate_in': None, 'best_move_uci': move})
    return games, features, analysis


def test_every_k_and_k70_can_choose_previously_unreported_width():
    values = coverage([2] * 7 + [20] * 3)
    assert all(f'top_{k}' in values for k in range(1, 21))
    assert values['top_1'] == 0 and values['top_2'] == .7
    assert thresholds(values)['70']['k'] == 2
    assert smallest_k(coverage([None]), .7)['status'] == 'greater_than_20_unresolved'
    assert thresholds(coverage([]))['70']['status'] == 'no_data'


def test_history_selection_never_reads_validation_or_test_labels():
    train = [{'game_id': 'train', 'actual_move_rank': 3}] * 7 + [{'game_id': 'train', 'actual_move_rank': 12}] * 3
    future = [{'game_id': 'test', 'actual_move_rank': 1}] * 100
    first = select_history_k(train + future, frozenset({'train'}))
    second = select_history_k(train + [{'game_id': 'test', 'actual_move_rank': None}] * 100, frozenset({'train'}))
    assert first == second
    assert first['selected_k'] == 3 and first['coverage']['decisions'] == 10
    assert first['game_ids'] == ['train'] and first['fit_split'] == 'train'


@pytest.mark.parametrize('k', [1, 3, 5, 10, 20])
def test_configurable_candidates_keep_every_evaluation_decision(k):
    games, features, analysis = broad_inputs()
    dataset = build_candidate_dataset(features, analysis, chronological_game_split(games), candidate_k=k, include_outside=True)
    assert len(dataset.decisions) == 10
    assert len(dataset.candidate_rows) == 10 * k
    assert max(r['engine_rank'] for r in dataset.candidate_rows) == k
    assert dataset.candidate_k == k
    if k == 5:
        assert dataset.outside_candidate_set_decisions == 1
        outside_rows = [r for r in dataset.candidate_rows if r['game_id'] == 'game-09']
        assert len(outside_rows) == 5 and not any(r['chosen'] for r in outside_rows)
        old_default = build_candidate_dataset(features, analysis, chronological_game_split(games))
        assert len(old_default.candidate_rows) == 45


def test_incomplete_candidate_width_is_not_silently_treated_as_outside():
    games, features, analysis = _inputs()
    with pytest.raises(ValueError, match='Incomplete'):
        build_candidate_dataset(features, analysis, chronological_game_split(games), candidate_k=10)
    for k in (0, 21, True):
        with pytest.raises(ValueError):
            build_candidate_dataset(features, analysis, chronological_game_split(games), candidate_k=k)


def test_all_test_decisions_are_accuracy_denominator():
    rows = [{'decision_id': d, 'candidate_move_uci': m, 'engine_rank': i} for d in ('a', 'b') for i, m in enumerate(MOVES, 1)]
    decisions = [{'decision_id': 'a', 'game_id': 'g1', 'actual_move_uci': 'e2e4', 'usable': True},
                 {'decision_id': 'b', 'game_id': 'g2', 'actual_move_uci': 'a2a3', 'usable': False}]
    scores = [.2] * 10
    metrics, predictions = all_decision_metrics(rows, scores, decisions)
    assert metrics['decisions'] == 2 and metrics['candidate_coverage'] == .5
    assert metrics['exact_move_accuracy'] == metrics['top_3_accuracy'] == .5
    assert metrics['conditional_exact_accuracy'] == 1
    assert predictions[1]['exact_correct'] is False and predictions[1]['top_3_correct'] is False
    with pytest.raises(ValueError, match='every evaluation'):
        all_decision_metrics(rows[:5], scores[:5], decisions)


def test_variable_rank_frequencies_and_single_class_k1():
    rows = [{'engine_rank': 10, 'chosen': True}, {'engine_rank': 1, 'chosen': False}]
    assert fit_global_rank_frequencies(rows, candidate_k=10)[10] == 1
    model = RandomForestClassifier(n_estimators=2, random_state=1).fit(np.array([[1], [1]]), np.array([1, 1]))
    rows = [{'decision_id': 'a', 'engine_rank': 1}]
    preprocessing = SparseOneHotPreprocessor(('engine_rank',)).fit(rows)
    assert predict_candidate_probabilities(model, preprocessing, rows) == [1.0]


def test_inference_uses_saved_width_and_old_artifacts_default_to_five(tmp_path):
    games, features, analysis = broad_inputs()
    rows = build_candidate_dataset(features, analysis, chronological_game_split(games), candidate_k=10, include_outside=True).candidate_rows[:10]
    preprocessing = SparseOneHotPreprocessor(('engine_rank',)).fit(rows)
    model = RandomForestClassifier(n_estimators=2, random_state=1).fit(preprocessing.transform(rows), [int(r['chosen']) for r in rows])
    joblib.dump(model, tmp_path / 'engine_and_context_model.joblib')
    preprocessing.save(tmp_path / 'engine_and_context_preprocessing.json')
    assert len(predict_one_decision(tmp_path, rows)) == 5
    (tmp_path / 'manifest.json').write_text(json.dumps({'candidate_k': 10}))
    assert len(predict_one_decision(tmp_path, rows)) == 10
    with pytest.raises(ValueError, match='frozen'):
        predict_one_decision(tmp_path, rows, candidate_k=5)
    with pytest.raises(ValueError, match='Incomplete'):
        predict_one_decision(tmp_path, rows[:5])
    short = [dict(rows[0], legal_move_count=1)]
    assert len(select_candidate_rows(short, 10)) == 1


def test_paired_comparison_uses_identical_games_and_is_reproducible():
    base = [{'decision_id': f'g{i}:1', 'game_id': f'g{i}', 'actual_move_uci': 'e2e4', 'covered': True,
             'exact_correct': False, 'top_3_correct': False} for i in range(5)]
    wider = [{**p, 'exact_correct': True, 'top_3_correct': True} for p in base]
    result = paired_comparison(base, wider)
    assert result == paired_comparison(base, list(reversed(wider)))
    assert result['exact_correct']['delta'] == 1
    assert result['exact_correct']['paired_game_bootstrap_95_interval'] == [1, 1]
    with pytest.raises(ValueError, match='identical'):
        paired_comparison(base, wider[:-1])


def test_width_experiment_rejects_incomplete_preparation(tmp_path):
    (tmp_path / 'manifest.json').write_text(json.dumps({'status': 'running'}))
    with pytest.raises(ValueError, match='complete'):
        train_width_comparison(tmp_path, tmp_path / 'out')


def test_default_k5_matches_pre_milestone_golden_rows():
    # Digest generated from the parent commit's candidate builder, with the
    # existing deterministic fixture. Guards context values, row order, labels,
    # outside accounting and historical-position context, not just row count.
    games, features, analysis = _inputs(count=10, outside_last=True)
    dataset = build_candidate_dataset(features, analysis, chronological_game_split(games))
    serialized = json.dumps({'rows': dataset.candidate_rows, 'decisions': dataset.decisions}, sort_keys=True, default=str)
    assert hashlib.sha256(serialized.encode()).hexdigest() == 'c81b47738e39c07798c6286a0c74a33bab4e48ae2fd1744c485e7851e219c937'


def test_end_to_end_controlled_widths_and_saved_inference(tmp_path, monkeypatch):
    from chess_clone.experiments.width_data import file_digest
    from chess_clone.modeling import ranker
    monkeypatch.setattr(ranker, 'make_tree_classifier', lambda: RandomForestClassifier(n_estimators=2, random_state=42, n_jobs=1))
    games, features, analysis = broad_inputs()
    for feature in features:
        feature['actual_move_rank'] = next(r['pv_rank'] for r in analysis if r['game_id'] == feature['game_id'] and r['best_move_uci'] == feature['actual_move_uci'])
    prepared = tmp_path / 'prepared'
    source = prepared / 'target'
    source.mkdir(parents=True)
    for name, rows in [('features', features), ('games', games), ('analysis', analysis)]:
        pq.write_table(pa.Table.from_pylist(rows), source / f'{name}.parquet')
    split = chronological_game_split(games)
    (source / 'split.json').write_text(json.dumps(split.to_dict()))
    sha = {name: file_digest(source / name) for name in ('features.parquet', 'analysis.parquet', 'games.parquet', 'split.json')}
    manifest = {'status': 'complete', 'engine_identity': 'test', 'settings': {'multipv': 20}, 'runtime': {},
                'players': [{'username': 'Target', 'games': 10, 'decisions': 10, 'sha256': sha}]}
    (prepared / 'manifest.json').write_text(json.dumps(manifest))
    out = tmp_path / 'models'
    report = train_width_comparison(prepared, out)
    player = report['players'][0]
    assert player['history_k_selection']['selected_k'] == 3
    assert player['split_game_counts'] == {'train': 7, 'validation': 1, 'test': 2}
    for experiment in player['experiments'].values():
        assert experiment['metrics']['test']['decisions'] == 2
        assert experiment['inference_artifact_bytes'] > experiment['model_bytes'] > 0
        assert experiment['metrics']['test']['exact_move_accuracy'] <= experiment['metrics']['test']['candidate_coverage']
    assert player['experiments']['fixed_k5']['metrics']['test']['candidate_coverage'] == .5
    assert player['experiments']['fixed_k10']['metrics']['test']['candidate_coverage'] == 1
    assert player['experiments']['history_k70']['metrics']['test']['candidate_coverage'] == 0
    assert player['experiments']['history_k70']['metrics']['test']['conditional_exact_accuracy'] is None
    rows = build_candidate_dataset(features, analysis, split, candidate_k=10, include_outside=True).candidate_rows[:10]
    assert len(predict_one_decision(out / 'target/history_k70', rows)) == 3
    with pytest.raises(FileExistsError):
        train_width_comparison(prepared, out)
    (source / 'split.json').write_text('{}')
    with pytest.raises(ValueError, match='changed'):
        train_width_comparison(prepared, tmp_path / 'tampered')
