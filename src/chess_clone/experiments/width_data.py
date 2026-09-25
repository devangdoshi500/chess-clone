"""Prepare complete per-player histories once for nested candidate experiments."""

from dataclasses import asdict
from datetime import UTC, datetime
import hashlib
from pathlib import Path
from time import perf_counter

import pyarrow as pa
import pyarrow.parquet as pq

from chess_clone.analysis import EngineSettings, FileAnalysisCache, StockfishAnalyzer
from chess_clone.benchmark.config import load_cohort
from chess_clone.benchmark.engine import CoverageEngine
from chess_clone.benchmark.metrics import coverage, thresholds
from chess_clone.benchmark.runner import behavior_rows, ingest_player, write_json
from chess_clone.features.evaluation import approximate_winning_chance
from chess_clone.modeling import canonical_position_key
from chess_clone.modeling.candidates import chronological_game_split


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_width_data(cohort: Path, output_dir: Path, *,
                       cache_dir: Path = Path('data/cache/candidate-coverage'),
                       stockfish_path: str = 'stockfish', analyzer=None, progress=print) -> dict:
    """No decision sampling; reuse existing PGN ingestion and MultiPV=20 cache."""
    players = load_cohort(cohort)
    if output_dir.exists():
        raise FileExistsError(f'refusing to overwrite {output_dir}')
    started = perf_counter()
    manifest = {'format_version': 1, 'status': 'running', 'created_at': datetime.now(UTC).isoformat(),
                'cohort': [asdict(p) for p in players], 'players': [],
                'sampling': 'none; every decision in obtained games',
                'analysis_protocol': 'shared MultiPV=20 prefixes; no post-move searches'}
    with (analyzer or StockfishAnalyzer(stockfish_path)) as active:
        settings = EngineSettings(nodes=20000, multipv=20, threads=1)
        engine = CoverageEngine(active, FileAnalysisCache(cache_dir), settings)
        output_dir.mkdir(parents=True)
        manifest.update(engine_identity=active.engine_identity, settings=settings.cache_payload())
        write_json(output_dir / 'manifest.json', manifest)
        try:
            for player in players:
                destination = output_dir / player.username.casefold()
                destination.mkdir()
                before = asdict(engine.stats)
                player_started = perf_counter()
                games, positions, ingestion = ingest_player(player, destination)
                if not positions or len(games) < 4:
                    raise ValueError(f'{player.username}: insufficient complete history')
                if len({(p['game_id'], p['ply']) for p in positions}) != len(positions):
                    raise ValueError('duplicate decisions in input history')
                split = chronological_game_split(games)
                write_json(destination / 'split.json', split.to_dict())
                write_json(destination / 'ingestion.json', ingestion)
                pq.write_table(pa.Table.from_pylist(games), destination / 'games.parquet')
                progress(f'{player.username}: analyzing ALL {len(positions)} decisions from {len(games)} games; split games={len(split.train_game_ids)}/{len(split.validation_game_ids)}/{len(split.test_game_ids)}', flush=True)
                features, analysis = [], []
                last = perf_counter()
                for row in behavior_rows(positions):
                    lines, key, hit = engine.analyze_candidates(row['fen'])
                    rank = next((line.rank for line in lines if line.best_move_uci == row['actual_move_uci']), None)
                    features.append({**row, 'opening_eco': row['eco'],
                                     'rating_difference': row['player_rating'] - row['opponent_rating'] if row['player_rating'] is not None and row['opponent_rating'] is not None else None,
                                     'approximate_winning_chance_before': approximate_winning_chance(lines[0].score_cp, lines[0].mate_in),
                                     'actual_move_rank': rank, 'cache_key': key, 'cache_hit': hit})
                    for line in lines:
                        analysis.append({**line.to_dict(), 'pv_rank': line.rank,
                                         'game_id': row['game_id'], 'ply': row['ply'],
                                         'player_username': player.username,
                                         'canonical_position': canonical_position_key(row['fen']),
                                         'multipv_requested': 20})
                    if perf_counter() - last >= 20:
                        progress(f'{player.username}: {len(features)}/{len(positions)} decisions; calls={engine.stats.engine_calls}, hits={engine.stats.cache_hits}', flush=True)
                        last = perf_counter()
                pq.write_table(pa.Table.from_pylist(features), destination / 'features.parquet')
                pq.write_table(pa.Table.from_pylist(analysis), destination / 'analysis.parquet')
                by_split = {}
                for name in ('train', 'validation', 'test'):
                    values = coverage([r['actual_move_rank'] for r in features if split.name_for(r['game_id']) == name])
                    by_split[name] = {'coverage': values, 'thresholds': thresholds(values)}
                after = asdict(engine.stats)
                summary = {'username': player.username, 'games': len(games), 'games_requested': player.games_requested,
                           'decisions': len(features), 'coverage_by_split': by_split,
                           'runtime': {k: after[k] - before[k] for k in before},
                           'runtime_seconds': perf_counter() - player_started,
                           'sha256': {f: file_digest(destination / f) for f in ('games.parquet', 'features.parquet', 'analysis.parquet', 'split.json')}}
                write_json(destination / 'summary.json', summary)
                manifest['players'].append(summary)
                write_json(output_dir / 'manifest.json', manifest)
            manifest.update(status='complete', runtime={**engine.snapshot(), 'runtime_seconds': perf_counter() - started})
            write_json(output_dir / 'manifest.json', manifest)
            return manifest
        except Exception as exc:
            manifest.update(status='failed', error=str(exc), runtime=engine.snapshot())
            write_json(output_dir / 'manifest.json', manifest)
            raise
