from dataclasses import asdict, replace
import json
from pathlib import Path

import chess
import httpx
import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

from chess_clone.analysis import EngineLine, EngineSettings, FileAnalysisCache, build_analysis_cache_key
from chess_clone.benchmark.config import CohortPlayer, load_cohort
from chess_clone.benchmark.engine import CoverageEngine, benchmark_cache_key
from chess_clone.benchmark.metrics import aggregate_bands, correlations, coverage, sample_games, smallest_k, thresholds
from chess_clone.benchmark.profiles import build_strength_profile, build_style_profile
from chess_clone.benchmark.runner import behavior_rows, ingest_player, run_benchmark
from chess_clone.cli import app
from chess_clone.providers.lichess import LichessProvider

SETTINGS = EngineSettings(nodes=20000, multipv=20)


class FakeFish:
    engine_identity = 'Stockfish 18|sha256:fake-test-only'

    def __init__(self):
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def analyze(self, fen, settings):
        self.calls.append((fen, settings))
        moves = sorted(chess.Board(fen).legal_moves, key=lambda m: m.uci())[:settings.multipv]
        return [EngineLine(i, 100 - i * 10, None, m.uci(), m.uci(), 3, 3, settings.nodes, .01)
                for i, m in enumerate(moves, 1)]


def config(tmp_path, **changes):
    player = {'username': 'TargetPlayer', 'intended_rating_band': '1800',
              'games_requested': 300, 'perf_type': 'blitz', **changes}
    path = tmp_path / 'cohort.json'
    path.write_text(json.dumps({'schema_version': 1, 'players': [player]}))
    return path


def pgn(tmp_path):
    source = Path('tests/fixtures/sample_games.pgn').read_text().replace('Take Take Take Arena', 'rated blitz game')
    path = tmp_path / 'source.pgn'
    path.write_text(source)
    return path


def test_cohort_parsing_and_relative_source(tmp_path):
    path = config(tmp_path, raw_pgn='source.pgn')
    player, = load_cohort(path)
    assert player.raw_pgn == tmp_path / 'source.pgn'
    assert player.intended_rating_band == '1800'
    assert not hasattr(player, 'observed_rating')


@pytest.mark.parametrize('changes', [
    {'username': '../bad'}, {'intended_rating_band': '1500'}, {'games_requested': 0},
    {'games_requested': True}, {'perf_type': 'rapid'}, {'extra': 1}, {'notes': 42},
])
def test_invalid_cohort(tmp_path, changes):
    with pytest.raises(ValueError):
        load_cohort(config(tmp_path, **changes))


def test_empty_and_duplicate_cohorts(tmp_path):
    path = config(tmp_path)
    value = json.loads(path.read_text())
    value['players'].append({**value['players'][0], 'username': 'targetplayer'})
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='unique'):
        load_cohort(path)
    value['players'] = []
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='Supply explicit'):
        load_cohort(path)


def test_coverage_thresholds_and_unresolved():
    values = coverage([1, 3, 5, 10, 15, 20, None])
    assert [values[f'top_{k}'] for k in (1, 3, 5, 10, 15, 20)] == [i / 7 for i in range(1, 7)]
    assert values['outside_top_20'] == 1 / 7
    assert smallest_k(values, .9) == {'k': None, 'status': 'greater_than_20_unresolved'}
    values = coverage([1] * 9 + [20])
    assert smallest_k(values, .9)['k'] == 1
    assert smallest_k(values, .95)['k'] == 20
    assert smallest_k(coverage([]), .9)['status'] == 'no_data'
    with pytest.raises(ValueError):
        coverage([0])


def test_sampling_is_whole_game_deterministic_and_reports_omissions():
    rows = [{'game_id': f'g{i}', 'ply': p} for i in range(30) for p in range(1, 5)]
    selected, summary = sample_games(rows, 19, 42, 'Player')
    assert (selected, summary) == sample_games(list(reversed(rows)), 19, 42, 'PLAYER')
    assert len(selected) == 16
    assert summary['omitted_decisions'] == 104
    assert len(summary['selected_game_ids']) == 4
    assert len(summary['omitted_game_ids']) == 26
    assert selected != sample_games(rows, 19, 43, 'Player')[0]
    assert len(sample_games(rows, None, 42, 'Player')[0]) == len(rows)
    with pytest.raises(ValueError, match='at least 4'):
        sample_games(rows, 2, 42, 'Player')
    with pytest.raises(ValueError, match='duplicate'):
        sample_games(rows + rows[:1], 19, 42, 'Player')


def member(name, rating, ranks, band='1800'):
    values = coverage(ranks)
    return {'username': name, 'intended_rating_band': band, 'coverage': values,
            'metadata': {'observed_ratings': [rating], 'median_player_rating': rating, 'games_obtained': 1},
            'sampling': {'selected_games': 1}, 'adaptive_k': thresholds(values)}


def test_band_aggregation_equal_player_weighting_and_spread():
    players = [member('a', 1700, [1]), member('b', 1900, [10] * 9)]
    band = aggregate_bands(players)['1800']
    assert band['coverage_mean']['top_5'] == .5  # not decision weighted .1
    assert band['median_observed_rating'] == 1800
    assert band['decisions'] == 10 and band['players'] == 2
    assert band['coverage_range']['top_5'] == [0, 1]
    assert band['coverage_stdev']['top_5'] == pytest.approx(2 ** -.5)
    assert band['adaptive_k']['90']['k'] == 10
    assert aggregate_bands([member('a', 1200, [None])])['1800']['adaptive_k']['90']['status'] == 'greater_than_20_unresolved'


def test_correlations_ties_missing_constant_and_censored():
    players = [member('a', 1200, [None]), member('b', 1400, [10]), member('c', 1600, [1])]
    stats = correlations(players)['statistics']
    assert stats['top_5']['spearman'] == pytest.approx(.8660254038)
    assert stats['top_5']['pearson'] == pytest.approx(.8660254038)
    assert stats['k90']['n'] == 2 and stats['k90']['excluded_players'] == 1
    assert stats['k90']['pearson'] is None
    assert correlations([member(str(i), 1200 + i, [1]) for i in range(3)])['statistics']['top_5']['undefined_reason'] == 'constant_values'
    assert correlations(players[:1])['statistics']['top_5']['undefined_reason'] == 'fewer_than_3_players'


def test_benchmark_cache_settings_and_halfmove_separate_from_production():
    identity = FakeFish.engine_identity
    base = benchmark_cache_key(chess.STARTING_FEN, SETTINGS, identity)
    assert base != build_analysis_cache_key(chess.STARTING_FEN, SETTINGS, identity)
    for changed in (replace(SETTINGS, multipv=5), replace(SETTINGS, nodes=500), replace(SETTINGS, threads=2)):
        assert base != benchmark_cache_key(chess.STARTING_FEN, changed, identity)
    fen = ' '.join(chess.STARTING_FEN.split()[:4]) + ' 90 42'
    assert base != benchmark_cache_key(fen, SETTINGS, identity)
    assert base == benchmark_cache_key(chess.STARTING_FEN.rsplit(' ', 1)[0] + ' 42', SETTINGS, identity)


def test_engine_caps_width_to_legal_moves_and_replays_cache(tmp_path):
    analyzer = FakeFish()
    engine = CoverageEngine(analyzer, FileAnalysisCache(tmp_path), SETTINGS)
    fen = '7k/8/8/8/8/8/8/K7 w - - 0 1'
    row = {'fen': fen, 'actual_move_uci': 'a1a2', 'game_id': 'g', 'ply': 1}
    result = engine.analyze_decision(row)
    assert result['effective_multipv'] == 3
    assert result['actual_move_rank'] == 1
    assert analyzer.calls[0][1].multipv == 3
    assert len(analyzer.calls) == 1  # terminal insufficient material needs no quality search
    assert engine.analyze_decision(row)['cache_hit'] is True
    assert len(analyzer.calls) == 1
    assert engine.snapshot()['cache_hits'] == 1
    assert engine.snapshot()['cache_misses'] == 1


def test_engine_broad_search_only_once_and_uniform_quality(tmp_path):
    analyzer = FakeFish()
    engine = CoverageEngine(analyzer, FileAnalysisCache(tmp_path), SETTINGS)
    row = {'fen': chess.STARTING_FEN, 'actual_move_uci': 'e2e4', 'game_id': 'g', 'ply': 1}
    engine.analyze_decision(row)
    engine.analyze_decision({**row, 'game_id': 'g2'})
    assert [s.multipv for _, s in analyzer.calls] == [20, 1]
    assert engine.stats.cache_hits == 2
    assert engine.snapshot()['unique_positions'] == 1


def test_engine_rejects_wrong_budget_version_and_partial_lines(tmp_path):
    analyzer = FakeFish()
    with pytest.raises(ValueError, match='requires MultiPV'):
        CoverageEngine(analyzer, FileAnalysisCache(tmp_path), replace(SETTINGS, nodes=500))
    analyzer.engine_identity = 'Stockfish 17|sha256:fake'
    with pytest.raises(ValueError, match='Stockfish 18'):
        CoverageEngine(analyzer, FileAnalysisCache(tmp_path), SETTINGS)
    analyzer.engine_identity = FakeFish.engine_identity
    analyzer.analyze = lambda *args: []
    engine = CoverageEngine(analyzer, FileAnalysisCache(tmp_path), SETTINGS)
    with pytest.raises(ValueError, match='incomplete'):
        engine.analyze_decision({'fen': chess.STARTING_FEN, 'actual_move_uci': 'e2e4'})


def test_ingestion_homogeneity_and_documented_api_parameters(tmp_path):
    raw = pgn(tmp_path).read_bytes()
    requests = []
    def handler(request):
        requests.append(request)
        return httpx.Response(200, content=raw)
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        games, rows, summary = ingest_player(CohortPlayer('TargetPlayer', '1800'), tmp_path / 'out', LichessProvider(client=client))
    assert requests[0].url.params['rated'] == 'true'
    assert requests[0].url.params['perfType'] == 'blitz'
    assert len(games) == 1 and len(rows) == 3
    assert games[0]['rated'] is True and games[0]['variant'] == 'Standard'
    assert summary['excluded_nonhomogeneous_games'] == 1
    assert summary['ingestion']['raw_path'].read_bytes() == raw


def test_profiles_distinct_and_training_scope_excludes_heldout(tmp_path):
    player = CohortPlayer('TargetPlayer', '1800', raw_pgn=pgn(tmp_path))
    games, positions, _ = ingest_player(player, tmp_path / 'out')
    rows = [{**r, 'actual_move_rank': 1, 'centipawn_loss': 10} for r in behavior_rows(positions)]
    heldout = {**games[0], 'game_id': 'heldout', 'white_rating': 9999}
    contaminated = rows + [{**rows[0], 'game_id': 'heldout', 'centipawn_loss': 99999}]
    kwargs = {'game_ids': {games[0]['game_id']}, 'purpose': 'training_only'}
    strength = build_strength_profile('TargetPlayer', contaminated, games + [heldout], **kwargs)
    style = build_style_profile('TargetPlayer', contaminated, games + [heldout], **kwargs)
    assert strength.median_rating == 1800 and strength.mean_centipawn_loss == 10
    assert strength.scope.decisions == 3 and strength.scope.purpose == 'training_only'
    assert 'opening_distribution' not in asdict(strength)
    assert 'median_rating' not in asdict(style)
    assert style.average_move_time_seconds == 4
    assert style.move_time_observations == 2
    assert style.opening_distribution['C20']['count'] == 1
    assert style.time_pressure_observations == 3


def test_mates_and_missing_values_not_sentinel_averages(tmp_path):
    player = CohortPlayer('TargetPlayer', '1800', raw_pgn=pgn(tmp_path))
    games, positions, _ = ingest_player(player, tmp_path / 'out')
    rows = [{**r, 'actual_move_rank': None, 'centipawn_loss': None} for r in behavior_rows(positions)]
    rows[0]['centipawn_loss'] = 1201
    profile = build_strength_profile('TargetPlayer', rows, games, game_ids={games[0]['game_id']})
    assert profile.finite_cp_decisions == 1 and profile.excluded_cp_decisions == 2
    assert profile.cp_loss_over_1000_count == 1 and profile.mean_centipawn_loss == 1201


def test_full_runner_warm_cache_outputs_and_no_overwrite(tmp_path):
    source = pgn(tmp_path)
    cohort = config(tmp_path, raw_pgn=str(source))
    analyzer = FakeFish()
    out = tmp_path / 'run'
    report = run_benchmark(cohort, output_dir=out, cache_dir=tmp_path / 'cache', analyzer=analyzer)
    player, = report['players']
    assert player['metadata']['games_obtained'] == 1
    assert player['metadata']['median_player_rating'] == 1800
    assert player['coverage']['decisions'] == 3
    assert len(pq.read_table(out / 'targetplayer/decisions.parquet')) == 3
    assert (out / 'targetplayer/strength_profile.json').is_file()
    assert (out / 'targetplayer/style_profile.json').is_file()
    assert json.loads((out / 'manifest.json').read_text())['status'] == 'complete'
    with pytest.raises(FileExistsError):
        run_benchmark(cohort, output_dir=out, analyzer=analyzer)
    replay = run_benchmark(cohort, output_dir=tmp_path / 'replay', cache_dir=tmp_path / 'cache', analyzer=FakeFish())
    assert replay['runtime']['engine_calls'] == 0
    assert replay['runtime']['cache_hits'] == 6
    assert replay['players'][0]['coverage'] == player['coverage']


def test_cli_failure_and_unchanged_magnus_top_five(tmp_path):
    result = CliRunner().invoke(app, ['benchmark-candidate-coverage', '--cohort', str(tmp_path / 'missing.json')])
    assert result.exit_code == 1 and 'Benchmark failed' in result.output
    # Running broad analysis must leave the original five-candidate dataset
    # and outside-top-five accounting unchanged.
    from test_ranking import _inputs
    from chess_clone.modeling.candidates import build_candidate_dataset, chronological_game_split
    games, features, analysis = _inputs(count=10, outside_last=True)
    split = chronological_game_split(games)
    before = build_candidate_dataset(features, analysis, split)
    engine = CoverageEngine(FakeFish(), FileAnalysisCache(tmp_path / 'cache'), SETTINGS)
    engine.analyze_decision(features[0])
    after = build_candidate_dataset(features, analysis, split)
    assert asdict(before) == asdict(after)
    assert len(after.candidate_rows) == 45
    assert after.inside_top_5_decisions == 9
    assert after.outside_top_5_decisions == 1


def test_checkmate_diagnostics_are_excluded_from_cp(tmp_path):
    board = chess.Board()
    for san in ['e4', 'e5', 'Bc4', 'Nc6', 'Qh5', 'Nf6']:
        board.push_san(san)
    analyzer = FakeFish()
    engine = CoverageEngine(analyzer, FileAnalysisCache(tmp_path), SETTINGS)
    result = engine.analyze_decision({'fen': board.fen(), 'actual_move_uci': 'h5f7', 'game_id': 'mate', 'ply': 7})
    assert result['actual_mate_in'] == 0
    assert result['centipawn_loss'] is None
    assert len(analyzer.calls) == 1


def test_outside_top_twenty_still_gets_quality_analysis(tmp_path):
    board = chess.Board()
    board.push_san('e4')
    board.push_san('e5')
    moves = sorted(board.legal_moves, key=lambda m: m.uci())
    assert len(moves) > 20
    analyzer = FakeFish()
    engine = CoverageEngine(analyzer, FileAnalysisCache(tmp_path), SETTINGS)
    result = engine.analyze_decision({'fen': board.fen(), 'actual_move_uci': moves[-1].uci(), 'game_id': 'g', 'ply': 3})
    assert result['actual_move_rank'] is None
    assert result['centipawn_loss'] is not None
    assert len(analyzer.calls) == 2
    assert coverage([result['actual_move_rank']])['outside_top_20'] == 1


def test_no_eligible_games_report_no_data_and_failure_manifest(tmp_path):
    source = pgn(tmp_path)
    source.write_text(source.read_text().replace('[Event "rated blitz game"]', '[Event "casual blitz game"]\n[Rated "false"]'))
    cohort = config(tmp_path, raw_pgn=str(source))
    report = run_benchmark(cohort, output_dir=tmp_path / 'empty', cache_dir=tmp_path / 'cache', analyzer=FakeFish())
    assert report['players'][0]['coverage']['decisions'] == 0
    assert report['players'][0]['adaptive_k']['90']['status'] == 'no_data'
    assert report['runtime']['engine_calls'] == 0
    source = pgn(tmp_path)
    with pytest.raises(ValueError, match='at least 3'):
        run_benchmark(cohort, output_dir=tmp_path / 'failed', cache_dir=tmp_path / 'cache', analyzer=FakeFish(), max_decisions_per_player=1)
    assert json.loads((tmp_path / 'failed/manifest.json').read_text())['status'] == 'failed'
