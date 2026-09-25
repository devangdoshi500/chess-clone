"""Sequential, isolated cross-rating candidate-coverage experiment."""

from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from typing import Callable

import chess
import pyarrow as pa
import pyarrow.parquet as pq

from chess_clone.analysis.cache import FileAnalysisCache
from chess_clone.analysis.schemas import EngineSettings
from chess_clone.analysis.stockfish import StockfishAnalyzer
from chess_clone.benchmark.config import CohortPlayer, RATING_BANDS, load_cohort
from chess_clone.benchmark.engine import CoverageEngine, benchmark_position
from chess_clone.benchmark.metrics import aggregate_bands, correlations, coverage, sample_games, thresholds
from chess_clone.benchmark.profiles import build_strength_profile, build_style_profile, stratified_quality
from chess_clone.features.board import extract_board_state_features, extract_move_behavior_features, player_color_from_name
from chess_clone.features.time import TimePressureThresholds, derive_time_features
from chess_clone.ingestion.pipeline import ingest_games
from chess_clone.providers.lichess import LichessProvider


class _PGNReplayProvider:
    """Feed preserved bytes through the existing ingestion pipeline without HTTP."""
    def __init__(self, path: Path):
        self.path = path

    def download_games(self, username, **kwargs):
        return self.path.read_bytes()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str, allow_nan=False) + '\n')


def _metadata(games: list[dict], positions: list[dict], username: str) -> dict:
    colors = ['white' if g['white_username'].casefold() == username.casefold() else 'black' for g in games]
    ratings = [g[f'{color}_rating'] for g, color in zip(games, colors)]
    ratings = [r for r in ratings if r is not None]
    dates = sorted(g['played_at'] for g in games if g['played_at'] is not None)
    return {
        'games_obtained': len(games), 'decision_count': len(positions),
        'date_range': [dates[0].isoformat(), dates[-1].isoformat()] if dates else None,
        'games_with_known_date': len(dates), 'games_with_known_rating': len(ratings),
        'median_player_rating': median(ratings) if ratings else None,
        'mean_player_rating': mean(ratings) if ratings else None,
        'rating_range': [min(ratings), max(ratings)] if ratings else None,
        'observed_ratings': ratings,
        'time_control_distribution': dict(Counter(g['time_control'] or 'unknown' for g in games)),
        'game_color_split': dict(Counter(colors)),
        'decision_color_split': dict(Counter(r['player_color'] for r in positions)),
    }


def ingest_player(player: CohortPlayer, output: Path, provider=None) -> tuple[list[dict], list[dict], dict]:
    summary = ingest_games(
        _PGNReplayProvider(player.raw_pgn) if player.raw_pgn else (provider or LichessProvider()),
        player.username, max_games=player.games_requested, perf_type='blitz',
        raw_dir=output / 'raw', processed_dir=output / 'normalized',
    )
    games = pq.read_table(summary.games_path).to_pylist()
    positions = pq.read_table(summary.positions_path).to_pylist()
    eligible = [g for g in games if g['rated'] is True and g['variant'].casefold() == 'standard'
                and g['speed'] == 'blitz' and g['provider'] == 'lichess'
                and player.username.casefold() in (g['white_username'].casefold(), g['black_username'].casefold())]
    ids = [g['game_id'] for g in eligible]
    if len(ids) != len(set(ids)):
        raise ValueError(f'duplicate source games for {player.username}; deduplicate source PGN explicitly')
    # Local exports may exceed the request; choose most recent games explicitly.
    eligible.sort(key=lambda g: (g['played_at'].isoformat() if g['played_at'] else '', g['game_id']), reverse=True)
    obtained = eligible[:player.games_requested]
    selected_ids = {g['game_id'] for g in obtained}
    selected_positions = [r for r in positions if r['game_id'] in selected_ids]
    for row in selected_positions:
        if row['player_username'].casefold() != player.username.casefold() or row['speed'] != 'blitz':
            raise ValueError('inconsistent normalized position metadata')
    return obtained, selected_positions, {
        'ingestion': asdict(summary), 'source': 'local_pgn' if player.raw_pgn else 'lichess_api',
        'raw_sha256': hashlib.sha256(summary.raw_path.read_bytes()).hexdigest(),
        'normalized_games': len(games), 'eligible_games_before_request_limit': len(eligible),
        'excluded_nonhomogeneous_games': len(games) - len(eligible),
        'games_omitted_by_request_limit': len(eligible) - len(obtained),
        'obtained_game_ids': [g['game_id'] for g in obtained],
        'game_selection': 'most recent eligible games, game_id tie break',
    }


def behavior_rows(rows: list[dict]) -> list[dict]:
    result = []
    previous = {}
    for row in sorted(rows, key=lambda r: (r['game_id'], r['ply'])):
        board = chess.Board(row['fen'])
        clock = row['clock_seconds_after_move']
        time = derive_time_features(
            time_control=row['time_control'], clock_after_move=clock,
            previous_player_clock_after_move=previous.get(row['game_id']),
            thresholds=TimePressureThresholds(),
        )
        previous[row['game_id']] = clock
        queen_capture_available = any(
            board.piece_type_at(m.from_square) == chess.QUEEN and board.is_capture(m)
            and board.piece_type_at(m.to_square) == chess.QUEEN for m in board.legal_moves)
        result.append({**row,
                       **asdict(extract_board_state_features(board, player_color_from_name(row['player_color']))),
                       **asdict(extract_move_behavior_features(board, row['actual_move_uci'])),
                       **asdict(time), 'queen_capture_available': queen_capture_available})
    return result


def run_benchmark(cohort: str | Path, *, output_dir: Path, cache_dir: Path = Path('data/cache/candidate-coverage'),
                  stockfish_path: str = 'stockfish', settings: EngineSettings = EngineSettings(nodes=20000, multipv=20),
                  max_decisions_per_player: int | None = 1000, seed: int = 42,
                  progress: Callable[[str], None] = print, analyzer=None, provider=None) -> dict:
    players = load_cohort(cohort)
    if max_decisions_per_player is not None and max_decisions_per_player < 1:
        raise ValueError('max_decisions_per_player must be positive or None')
    started = perf_counter()
    # Existing directories are never reused, including model/split directories.
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f'refusing to overwrite existing output directory: {output_dir}')
    results = []
    with (analyzer or StockfishAnalyzer(stockfish_path)) as active:
        engine = CoverageEngine(active, FileAnalysisCache(cache_dir), settings)
        output_dir.mkdir(parents=True, exist_ok=False)
        write_json(output_dir / 'cohort.json', {'schema_version': 1, 'players': [asdict(p) for p in players]})
        manifest = {'schema_version': 1, 'status': 'running', 'started_at': datetime.now(UTC).isoformat(),
                    'engine_identity': active.engine_identity, 'settings': settings.cache_payload(),
                    'max_decisions_per_player': max_decisions_per_player, 'seed': seed,
                    'cache_directory': str(Path(cache_dir).resolve())}
        write_json(output_dir / 'manifest.json', manifest)
        try:
            for player in players:
                player_started = perf_counter()
                before = asdict(engine.stats)
                destination = output_dir / player.username.casefold()
                destination.mkdir()
                games, positions, ingestion = ingest_player(player, destination, provider)
                selected, sampling = sample_games(positions, max_decisions_per_player, seed, player.username)
                selected_ids = set(sampling['selected_game_ids'])
                sampled_games = [g for g in games if g['game_id'] in selected_ids]
                progress(f'{player.username}: obtained {len(games)} games / {len(positions)} decisions; selected {len(sampled_games)} complete games / {len(selected)} decisions; omitted {sampling["omitted_decisions"]} decisions')
                write_json(destination / 'sampling.json', sampling)
                write_json(destination / 'ingestion.json', ingestion)
                analyzed = []
                last_progress = perf_counter()
                for row in behavior_rows(selected):
                    analyzed.append({**row, **engine.analyze_decision(row)})
                    if perf_counter() - last_progress >= 20:
                        progress(f'{player.username}: analyzed {len(analyzed)}/{len(selected)} decisions; total calls={engine.stats.engine_calls}, hits={engine.stats.cache_hits}')
                        last_progress = perf_counter()
                values = coverage([r['actual_move_rank'] for r in analyzed])
                strength = build_strength_profile(player.username, analyzed, games, game_ids=selected_ids)
                style = build_style_profile(player.username, analyzed, games, game_ids=selected_ids)
                after = asdict(engine.stats)
                stats = {key: after[key] - before[key] for key in after}
                stats.update(runtime_seconds=perf_counter() - player_started,
                             unique_positions=len({benchmark_position(r['fen']) for r in analyzed}),
                             average_time_per_miss_seconds=stats['engine_seconds'] / stats['cache_misses'] if stats['cache_misses'] else None)
                by_control = stratified_quality(analyzed, 'time_control', ('180+0', '180+2'))
                for key, subgroup in by_control.items():
                    subgroup['games'] = len({r['game_id'] for r in analyzed if (r['time_control'] or 'unknown') == key})
                    subgroup['sufficient_for_descriptive_comparison'] = subgroup['games'] >= 5 and subgroup['decisions'] >= 100
                result = {
                    'username': player.username, 'intended_rating_band': player.intended_rating_band,
                    'games_requested': player.games_requested, 'notes': player.notes,
                    'metadata': _metadata(games, positions, player.username),
                    'sample_metadata': _metadata(sampled_games, selected, player.username),
                    'sampling': sampling, 'coverage': values, 'adaptive_k': thresholds(values),
                    'by_phase': strength.accuracy_by_phase,
                    'by_time_pressure': strength.accuracy_under_time_pressure,
                    'by_color': stratified_quality(analyzed, 'player_color', ('white', 'black')),
                    'by_time_control': by_control, 'runtime': stats,
                }
                write_json(destination / 'summary.json', result)
                write_json(destination / 'strength_profile.json', asdict(strength))
                write_json(destination / 'style_profile.json', asdict(style))
                if analyzed:
                    pq.write_table(pa.Table.from_pylist(analyzed), destination / 'decisions.parquet')
                results.append(result)
            missing_bands = [b for b in RATING_BANDS if b not in {p['intended_rating_band'] for p in results if p['coverage']['decisions']}]
            report = {
                'schema_version': 1, 'players': results, 'rating_bands': aggregate_bands(results),
                'correlations': correlations(results), 'missing_rating_bands': missing_bands,
                'runtime': {**engine.snapshot(), 'runtime_seconds': perf_counter() - started},
                'engine_identity': active.engine_identity, 'settings': settings.cache_payload(),
                'interpretation': {
                    'cross_rating_evidence_complete': not missing_bands,
                    'production_candidate_generator_changed': False,
                    'note': 'Compare individual and band coverage before choosing K. A single elite reference cannot identify a rating relationship. All thresholds are descriptive in-sample estimates, not guarantees for future games.',
                },
            }
            write_json(output_dir / 'report.json', report)
            write_json(output_dir / 'rating_bands.json', report['rating_bands'])
            write_json(output_dir / 'correlations.json', report['correlations'])
            pq.write_table(pa.Table.from_pylist(report['correlations']['scatter']), output_dir / 'scatter.parquet')
            manifest['status'] = 'complete'
            manifest['runtime'] = report['runtime']
            write_json(output_dir / 'manifest.json', manifest)
            progress(f'Benchmark complete: {output_dir / "report.json"}; missing bands: {", ".join(missing_bands) or "none"}')
            return report
        except Exception as exc:
            manifest.update(status='failed', error=str(exc), completed_players=[p['username'] for p in results])
            write_json(output_dir / 'manifest.json', manifest)
            raise
