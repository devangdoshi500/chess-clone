"""Offline width/ranker milestone with a durable validation-before-test gate.

No production defaults or models are modified. Run as a module; each phase is
explicit so test rows cannot be loaded until both players' widths are frozen.
"""

import argparse
from collections import defaultdict
from dataclasses import asdict
from datetime import UTC, datetime
import json
from pathlib import Path
import platform
from time import perf_counter

import chess
import joblib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from catboost import CatBoostRanker

from chess_clone.analysis import EngineSettings, StockfishAnalyzer
from chess_clone.benchmark.runner import write_json
from chess_clone.experiments.width_data import file_digest
from chess_clone.experiments.width_metrics import all_decision_metrics, paired_comparison
from chess_clone.features.board import extract_board_state_features
from chess_clone.features.evaluation import approximate_winning_chance
from chess_clone.modeling.boosted import groupwise_softmax, predict_relevance_scores, train_grouped_ranker
from chess_clone.modeling.candidates import (
    FULL_FEATURE_FIELDS, _candidate_row, _line_evaluation, _pre_move_clocks,
    chronological_game_split, validate_no_leakage_fields,
)
from chess_clone.modeling.historical import canonical_position_key
from chess_clone.modeling.ranker import (
    SparseOneHotPreprocessor, predict_candidate_probabilities, train_candidate_model,
)

WIDTHS = (3, 5, 10)
PLAYERS = ('DrNykterstein', 'EneaScarabeo')
POLICY = {
    'fit_split': 'validation', 'minimum_exact_gain': .01,
    'require_positive_paired_game_bootstrap_95_lower_bound': True,
    'maximum_added_ms_per_percentage_point': 10.,
    'order': 'start at K=3; compare K=5 then K=10 to current selection',
    'tie_break': 'smaller K', 'bootstrap_seed': 42, 'resamples': 2000,
}


def now():
    return datetime.now(UTC).isoformat()


def digest_tree(root):
    return {str(p.relative_to(root)): file_digest(p) for p in sorted(root.rglob('*')) if p.is_file()}


def initialize(source: Path, baseline: Path, output: Path):
    if output.exists():
        raise FileExistsError(f'refusing to overwrite {output}')
    manifest = json.loads((source / 'manifest.json').read_text())
    if manifest['status'] != 'complete':
        raise ValueError('source must be complete')
    summaries = {p['username']: p for p in manifest['players']}
    inputs = {}
    for player in PLAYERS:
        summary = summaries[player]
        if summary['games'] != 300:
            raise ValueError('requires full 300-game histories')
        directory = source / player.lower()
        for name, expected in summary['sha256'].items():
            if file_digest(directory / name) != expected:
                raise ValueError(f'input changed: {directory / name}')
        games = pq.read_table(directory / 'games.parquet').to_pylist()
        split = chronological_game_split(games)
        if split.to_dict() != json.loads((directory / 'split.json').read_text()):
            raise ValueError('chronological split mismatch')
        inputs[player] = {'sha256': summary['sha256'], 'decisions': summary['decisions'],
                          'games': 300, 'split': split.to_dict()}
    validate_no_leakage_fields(FULL_FEATURE_FIELDS)
    output.mkdir(parents=True)
    result = {'created_at': now(), 'status': 'initialized',
              'source': str(source.resolve()), 'baseline': str(baseline.resolve()),
              'source_manifest_sha256': file_digest(source / 'manifest.json'),
              'preserved_baseline_sha256': digest_tree(baseline),
              'preserved_production_sha256': digest_tree(Path('artifacts/models')),
              'inputs': inputs, 'widths': list(WIDTHS), 'features': list(FULL_FEATURE_FIELDS),
              'selection_policy': POLICY, 'production_changed': False,
              'engine_identity': manifest['engine_identity'],
              'search': {'nodes': 20000, 'threads': 1, 'hash_mb': 16,
                         'multipv': 'min(K, legal moves)', 'cache': 'disabled',
                         'state': 'new UCI game per position; no cross-position hash reuse'},
              'feature_protocol': 'existing FULL_FEATURE_FIELDS for both RF and grouped CatBoost; no history augmentation',
              'machine': platform.platform(), 'processor': platform.processor()}
    write_json(output / 'protocol.json', result)
    return result


def load_source(output, player, split_name):
    protocol = json.loads((output / 'protocol.json').read_text())
    if split_name == 'test':
        verify_selection(output)
    source = Path(protocol['source']) / player.lower()
    expected = protocol['inputs'][player]
    # Hashing bytes verifies provenance without exposing future labels/metrics.
    if file_digest(source / 'features.parquet') != expected['sha256']['features.parquet']:
        raise ValueError('source features changed')
    games = pq.read_table(source / 'games.parquet').to_pylist()
    split = chronological_game_split(games)
    if split.to_dict() != expected['split']:
        raise ValueError('saved split changed')
    ids = expected['split'][split_name]['game_ids']
    features = pq.read_table(source / 'features.parquet', filters=[('game_id', 'in', ids)]).to_pylist()
    if any(f['player_username'].casefold() != player.casefold() for f in features):
        raise ValueError('cross-player input')
    if {f['game_id'] for f in features} != set(ids):
        raise ValueError('missing whole game')
    features.sort(key=lambda f: (split.game_dates[f['game_id']], f['game_id'], f['ply']))
    return features, split


def construct_rows(feature, lines, split, pre_clock):
    """Construct the unchanged features without consulting the current label."""
    board = chess.Board(feature['fen'])
    context = {**feature, **asdict(extract_board_state_features(board, board.turn)),
               'approximate_winning_chance_before': approximate_winning_chance(lines[0].score_cp, lines[0].mate_in)}
    canonical = canonical_position_key(feature['fen'])
    best = _line_evaluation(lines[0].to_dict())
    return [_candidate_row(context, {**line.to_dict(), 'pv_rank': line.rank},
                           split_name=split.name_for(feature['game_id']),
                           canonical_position=canonical, actual_move='', best_evaluation=best,
                           pre_move_clock=pre_clock, played_at=split.game_dates[feature['game_id']],
                           previous_player_post_position=None) for line in lines]


def label_rows(rows, feature):
    # Only after constructing/scoring features. Labels are never model inputs.
    for row in rows:
        row['actual_move_uci'] = feature['actual_move_uci']
        row['chosen'] = row['candidate_move_uci'] == feature['actual_move_uci']
    return {'decision_id': rows[0]['decision_id'], 'game_id': feature['game_id'],
            'actual_move_uci': feature['actual_move_uci'], 'usable': any(r['chosen'] for r in rows)}


def search(analyzer, feature, k):
    legal = chess.Board(feature['fen']).legal_moves
    expected = min(k, legal.count())
    lines = analyzer.analyze(feature['fen'], EngineSettings(nodes=20000, multipv=expected))
    moves = [line.best_move_uci for line in lines]
    if ([line.rank for line in lines] != list(range(1, expected + 1))
            or len(set(moves)) != expected
            or any(chess.Move.from_uci(m) not in legal for m in moves)):
        raise ValueError('incomplete or illegal engine candidates')
    return lines


def prepare(output: Path):
    protocol = json.loads((output / 'protocol.json').read_text())
    with StockfishAnalyzer() as analyzer:
        if analyzer.engine_identity != protocol['engine_identity']:
            raise ValueError('engine identity changed')
        for player in PLAYERS:
            for name in ('train', 'validation'):
                features, split = load_source(output, player, name)
                clocks = _pre_move_clocks(features)
                destinations = {k: output / player.lower() / f'k{k}' for k in WIDTHS}
                pending = [k for k in WIDTHS if not (destinations[k] / f'{name}_data.json').exists()]
                rows = {k: [] for k in pending}
                decisions = {k: [] for k in pending}
                times = {k: [] for k in pending}
                last = perf_counter()
                for i, feature in enumerate(features):
                    # Rotate order to avoid making width coincide with warmup or thermal drift.
                    for k in pending[i % len(pending):] + pending[:i % len(pending)] if pending else []:
                        started = perf_counter()
                        lines = search(analyzer, feature, k)
                        searched = perf_counter()
                        candidates = construct_rows(feature, lines, split, clocks[(feature['game_id'], feature['ply'])])
                        built = perf_counter()
                        decisions[k].append(label_rows(candidates, feature))
                        rows[k].extend(candidates)
                        times[k].append({'decision_id': candidates[0]['decision_id'],
                                         'search_seconds': searched - started, 'feature_seconds': built - searched,
                                         'nodes': max(line.nodes_searched or 0 for line in lines)})
                    if perf_counter() - last >= 20:
                        print(f'{player} {name}: {i+1}/{len(features)} decisions, K={pending}', flush=True)
                        last = perf_counter()
                for k in pending:
                    dest = destinations[k]
                    dest.mkdir(parents=True, exist_ok=True)
                    pq.write_table(pa.Table.from_pylist(rows[k]), dest / f'{name}_rows.parquet')
                    pq.write_table(pa.Table.from_pylist(decisions[k]), dest / f'{name}_decisions.parquet')
                    pq.write_table(pa.Table.from_pylist(times[k]), dest / f'{name}_preparation_timing.parquet')
                    write_json(dest / f'{name}_data.json', {'decisions': len(decisions[k]), 'rows': len(rows[k]),
                               'covered': sum(d['usable'] for d in decisions[k]), 'finished_at': now(),
                               'sha256': {f'{name}_{suffix}.parquet': file_digest(dest / f'{name}_{suffix}.parquet')
                                          for suffix in ('rows', 'decisions', 'preparation_timing')}})
    write_json(output / 'preparation_complete.json', {'finished_at': now()})


def score(model, preprocessor, rows):
    if preprocessor is None:
        return groupwise_softmax(rows, predict_relevance_scores(model, rows, FULL_FEATURE_FIELDS))
    return predict_candidate_probabilities(model, preprocessor, rows)


def fit(output: Path, *, players=None):
    # An explicitly named ready player can fit while another player's offline
    # preparation continues. Serving measurements always run after all fitting.
    if players is None and not (output / 'preparation_complete.json').exists():
        raise ValueError('preparation incomplete')
    players = PLAYERS if players is None else tuple(players)
    if not players or any(player not in PLAYERS for player in players):
        raise ValueError('unknown experiment player')
    for player in players:
        for k in WIDTHS:
            dest = output / player.lower() / f'k{k}'
            if (dest / 'models.json').exists():
                continue
            datasets = {}
            for name in ('train', 'validation'):
                metadata = json.loads((dest / f'{name}_data.json').read_text())
                for file, sha in metadata['sha256'].items():
                    if file_digest(dest / file) != sha:
                        raise ValueError('prepared rows changed')
                rows = pq.read_table(dest / f'{name}_rows.parquet').to_pylist()
                decisions = pq.read_table(dest / f'{name}_decisions.parquet').to_pylist()
                covered = {d['decision_id'] for d in decisions if d['usable']}
                datasets[name] = [r for r in rows if r['decision_id'] in covered]
            print(f'{player} K={k}: fitting matched RF and grouped CatBoost', flush=True)
            models = {}
            for family in ('rf', 'catboost'):
                started = perf_counter()
                if family == 'rf':
                    model, preprocessor = train_candidate_model(datasets['train'], datasets['validation'], FULL_FEATURE_FIELDS)
                    model_path, extra = dest / 'rf.joblib', dest / 'rf_preprocessing.json'
                    joblib.dump(model, model_path)
                    preprocessor.save(extra)
                else:
                    model = train_grouped_ranker(datasets['train'], datasets['validation'], FULL_FEATURE_FIELDS)
                    model_path, extra = dest / 'catboost.cbm', dest / 'catboost_features.json'
                    model.save_model(model_path)
                    write_json(extra, list(FULL_FEATURE_FIELDS))
                models[family] = {'fit_and_serialization_seconds': perf_counter() - started,
                                  'model_bytes': model_path.stat().st_size,
                                  'inference_artifact_bytes': model_path.stat().st_size + extra.stat().st_size,
                                  'model_sha256': file_digest(model_path), 'extra_sha256': file_digest(extra),
                                  'hyperparameters': model.get_params(),
                                  'tree_count': model.tree_count_ if family == 'catboost' else len(model.estimators_)}
            write_json(dest / 'models.json', models)


def load_models(dest):
    manifest = json.loads((dest / 'models.json').read_text())
    files = {'rf': ('rf.joblib', 'rf_preprocessing.json'), 'catboost': ('catboost.cbm', 'catboost_features.json')}
    for family, (model_file, extra) in files.items():
        if (file_digest(dest / model_file) != manifest[family]['model_sha256']
                or file_digest(dest / extra) != manifest[family]['extra_sha256']):
            raise ValueError('frozen model changed')
    cb = CatBoostRanker()
    cb.load_model(dest / 'catboost.cbm')
    return {'catboost': (cb, None),
            'rf': (joblib.load(dest / 'rf.joblib'), SparseOneHotPreprocessor.load(dest / 'rf_preprocessing.json'))}


def timing_summary(rows, family):
    result = {'decisions': len(rows), 'engine_calls': len(rows), 'cache_hits': 0,
              'scope': 'serial warm in-memory model, actual uncached per-K UCI search + board/candidate features + single-decision preprocessing/scoring/normalization; excludes startup, disk/network and evaluation labels',
              'nodes_mean': float(np.mean([r['nodes'] for r in rows]))}
    for field in ('search_ms', 'feature_ms', f'{family}_score_ms', f'{family}_end_to_end_ms'):
        values = [r[field] for r in rows]
        result[field] = {'mean': float(np.mean(values)), 'p50': float(np.median(values)),
                         'p95': float(np.quantile(values, .95)), 'total_seconds': sum(values) / 1000}
    return result


def evaluate(output: Path, name: str):
    if name not in ('validation', 'test'):
        raise ValueError('held-out splits only')
    if name == 'test':
        verify_selection(output)
        if not (output / 'test_opened.json').exists():
            write_json(output / 'test_opened.json', {'opened_at': now(), 'selection_sha256': file_digest(output / 'selection.json')})
    elif (output / 'selection.json').exists():
        raise ValueError('validation frozen; use a new output for a new experiment')
    protocol = json.loads((output / 'protocol.json').read_text())
    with StockfishAnalyzer() as analyzer:
        if analyzer.engine_identity != protocol['engine_identity']:
            raise ValueError('engine identity changed')
        for player in PLAYERS:
            features, split = load_source(output, player, name)
            clocks = _pre_move_clocks(features)
            destinations = {k: output / player.lower() / f'k{k}' for k in WIDTHS}
            pending = [k for k in WIDTHS if not (destinations[k] / f'{name}_evaluation.json').exists()]
            models = {k: load_models(destinations[k]) for k in pending}
            predictions = {k: {family: [] for family in models[k]} for k in pending}
            all_rows, decisions, timings = ({k: [] for k in pending} for _ in range(3))
            last = perf_counter()
            for i, feature in enumerate(features):
                for k in pending[i % len(pending):] + pending[:i % len(pending)] if pending else []:
                    started = perf_counter()
                    lines = search(analyzer, feature, k)
                    searched = perf_counter()
                    rows = construct_rows(feature, lines, split, clocks[(feature['game_id'], feature['ply'])])
                    built = perf_counter()
                    record = {'decision_id': rows[0]['decision_id'], 'search_ms': (searched-started)*1000,
                              'feature_ms': (built-searched)*1000,
                              'nodes': max(line.nodes_searched or 0 for line in lines)}
                    if i == 0:
                        for model, prep in models[k].values():
                            score(model, prep, rows)  # model warmup excluded
                    for family in (('catboost', 'rf') if i % 2 else ('rf', 'catboost')):
                        t = perf_counter()
                        values = score(*models[k][family], rows)
                        record[f'{family}_score_ms'] = (perf_counter()-t)*1000
                        record[f'{family}_end_to_end_ms'] = record['search_ms'] + record['feature_ms'] + record[f'{family}_score_ms']
                        predictions[k][family].extend(values)
                    decisions[k].append(label_rows(rows, feature))
                    all_rows[k].extend(rows)
                    timings[k].append(record)
                if perf_counter() - last >= 20:
                    print(f'{player} {name} serving: {i+1}/{len(features)} decisions, K={pending}', flush=True)
                    last = perf_counter()
            for k in pending:
                dest = destinations[k]
                result = {'split': name, 'candidate_k': k, 'finished_at': now(), 'models': {}}
                saved = {}
                for family, values in predictions[k].items():
                    metrics, saved[family] = all_decision_metrics(all_rows[k], values, decisions[k])
                    result['models'][family] = {'metrics': metrics, 'timing': timing_summary(timings[k], family)}
                    pq.write_table(pa.Table.from_pylist(saved[family]), dest / f'{name}_{family}_predictions.parquet')
                result['catboost_minus_rf'] = paired_comparison(saved['rf'], saved['catboost'])
                pq.write_table(pa.Table.from_pylist(timings[k]), dest / f'{name}_serving_timing.parquet')
                write_json(dest / f'{name}_evaluation.json', result)
                print(f'{player} K={k} {name}: CB exact={result["models"]["catboost"]["metrics"]["exact_move_accuracy"]:.4f}', flush=True)


def choose_width(validation, predictions, policy=POLICY):
    """Only validation objects accepted. A positive CI is not equivalence proof."""
    selected, comparisons = min(validation), []
    for k in sorted(validation):
        if k == selected:
            continue
        comparison = paired_comparison(predictions[selected], predictions[k])
        gain = comparison['exact_correct']['delta']
        lower = comparison['exact_correct']['paired_game_bootstrap_95_interval'][0]
        extra_ms = validation[k]['latency_ms'] - validation[selected]['latency_ms']
        acceptable_cost = extra_ms <= policy['maximum_added_ms_per_percentage_point'] * gain * 100
        promote = gain > policy['minimum_exact_gain'] and lower > 0 and acceptable_cost
        comparisons.append({'from_k': selected, 'to_k': k, 'paired': comparison,
                            'added_latency_ms': extra_ms, 'cost_acceptable': acceptable_cost, 'promote': promote})
        if promote:
            selected = k
    return {'selected_k': selected, 'comparisons': comparisons}


def freeze_selection(output: Path):
    if (output / 'selection.json').exists() or (output / 'test_opened.json').exists():
        raise ValueError('selection already frozen or test already opened')
    protocol = json.loads((output / 'protocol.json').read_text())
    if protocol['selection_policy'] != POLICY:
        raise ValueError('selection policy changed')
    result = {'frozen_at': now(), 'fit_split': 'validation', 'policy': POLICY,
              'protocol_sha256': file_digest(output / 'protocol.json'), 'players': {}, 'frozen_files': {}}
    for player in PLAYERS:
        validation, predictions = {}, {}
        for k in WIDTHS:
            dest = output / player.lower() / f'k{k}'
            evaluation = json.loads((dest / 'validation_evaluation.json').read_text())['models']['catboost']
            validation[k] = {'exact_move_accuracy': evaluation['metrics']['exact_move_accuracy'],
                             'latency_ms': evaluation['timing']['catboost_end_to_end_ms']['mean']}
            predictions[k] = pq.read_table(dest / 'validation_catboost_predictions.parquet').to_pylist()
            for p in dest.iterdir():
                if p.is_file():
                    result['frozen_files'][str(p.relative_to(output))] = file_digest(p)
        result['players'][player] = {**choose_width(validation, predictions), 'validation': validation}
    write_json(output / 'selection.json', result)
    write_json(output / 'selection.sha256.json', {'sha256': file_digest(output / 'selection.json')})
    return result


def verify_selection(output: Path):
    path = output / 'selection.json'
    if not path.exists():
        raise ValueError('test locked: validation selection must be frozen for both players')
    seal = json.loads((output / 'selection.sha256.json').read_text())
    if file_digest(path) != seal['sha256']:
        raise ValueError('selection changed after freezing')
    selection = json.loads(path.read_text())
    if set(selection['players']) != set(PLAYERS) or selection['fit_split'] != 'validation':
        raise ValueError('incomplete validation selection')
    if file_digest(output / 'protocol.json') != selection['protocol_sha256']:
        raise ValueError('protocol changed')
    for file, digest in selection['frozen_files'].items():
        if file_digest(output / file) != digest:
            raise ValueError(f'frozen validation/model file changed: {file}')
    return selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['init', 'prepare', 'fit', 'validate', 'select', 'test'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=Path('artifacts/benchmarks/width-full-history'))
    parser.add_argument('--baseline', type=Path, default=Path('artifacts/benchmarks/width-model-comparison'))
    args = parser.parse_args()
    if args.phase == 'init':
        initialize(args.source, args.baseline, args.output)
    elif args.phase == 'prepare':
        prepare(args.output)
    elif args.phase == 'fit':
        fit(args.output)
    elif args.phase == 'validate':
        evaluate(args.output, 'validation')
    elif args.phase == 'select':
        print(json.dumps(freeze_selection(args.output)['players'], indent=2))
    else:
        evaluate(args.output, 'test')


if __name__ == '__main__':
    main()
