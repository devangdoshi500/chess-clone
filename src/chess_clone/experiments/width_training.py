"""Controlled, per-player RF width experiments on a shared chronological split."""

import json
from pathlib import Path
from time import perf_counter

import joblib
import pyarrow as pa
import pyarrow.parquet as pq

from chess_clone.benchmark.metrics import coverage, smallest_k
from chess_clone.benchmark.runner import write_json
from chess_clone.experiments.width_data import file_digest
from chess_clone.experiments.width_metrics import all_decision_metrics, paired_comparison
from chess_clone.experiments.width_report import render_width_report
from chess_clone.modeling.candidates import FULL_FEATURE_FIELDS, build_candidate_dataset, chronological_game_split
from chess_clone.modeling.ranker import predict_candidate_probabilities, train_candidate_model


def select_history_k(features: list[dict], train_game_ids: frozenset[str]) -> dict:
    """Never inspect validation/test ranks when choosing the frozen width."""
    rows = [r for r in features if r['game_id'] in train_game_ids]
    values = coverage([r['actual_move_rank'] for r in rows])
    selected = smallest_k(values, .70)
    return {'threshold': .70, 'selected_k': selected['k'], 'status': selected['status'],
            'fit_split': 'train', 'game_ids': sorted(train_game_ids), 'coverage': values}


def train_width_comparison(prepared_dir: Path, output_dir: Path, *, progress=print) -> dict:
    """Fit the same deterministic full-context Random Forest at each width.

    Training excludes outside-candidate decisions as in the existing baseline.
    Validation/test score candidates for every decision, including outside moves.
    Neither features nor model hyperparameters are selected using held-out results.
    """
    prepared_dir, output_dir = Path(prepared_dir), Path(output_dir)
    source_manifest = json.loads((prepared_dir / 'manifest.json').read_text())
    if source_manifest['status'] != 'complete':
        raise ValueError('prepared analysis must be complete')
    if output_dir.exists():
        raise FileExistsError(f'refusing to overwrite {output_dir}')
    output_dir.mkdir(parents=True)
    started = perf_counter()
    report = {'format_version': 1, 'status': 'running', 'players': [],
              'prepared_dir': str(prepared_dir.resolve()),
              'engine_identity': source_manifest['engine_identity'], 'settings': source_manifest['settings'],
              'model_family': 'RandomForestClassifier; existing engine_and_context features and hyperparameters',
              'production_default_k': 5, 'production_changed': False,
              'candidate_protocol': 'nested prefixes of one shared MultiPV=20 search per position',
              'analysis_runtime': source_manifest['runtime']}
    write_json(output_dir / 'report.json', report)
    try:
        for player in source_manifest['players']:
            username = player['username']
            source = prepared_dir / username.casefold()
            for name, expected in player['sha256'].items():
                if file_digest(source / name) != expected:
                    raise ValueError(f'prepared input changed: {source / name}')
            features = pq.read_table(source / 'features.parquet').to_pylist()
            analysis = pq.read_table(source / 'analysis.parquet').to_pylist()
            games = pq.read_table(source / 'games.parquet').to_pylist()
            if any(r['player_username'].casefold() != username.casefold() for r in features + analysis):
                raise ValueError('personalized experiments cannot pool players')
            if len(features) != player['decisions'] or len(games) != player['games']:
                raise ValueError('prepared history count mismatch')
            game_ids = {g['game_id'] for g in games}
            if {r['game_id'] for r in features} - game_ids:
                raise ValueError('features contain unknown games')
            split = chronological_game_split(games)
            if split.to_dict() != json.loads((source / 'split.json').read_text()):
                raise ValueError('split changed after preparation')
            selection = select_history_k(features, split.train_game_ids)
            destination = output_dir / username.casefold()
            destination.mkdir()
            write_json(destination / 'split_metadata.json', split.to_dict())
            write_json(destination / 'history_k_selection.json', selection)
            specs = [('fixed_k5', 5), ('fixed_k10', 10), ('history_k70', selection['selected_k'])]
            player_result = {'username': username, 'games': len(games), 'decisions': len(features),
                             'history_k_selection': selection, 'experiments': {},
                             'split_game_counts': {name: split.to_dict()[name]['game_count'] for name in ('train', 'validation', 'test')}}
            predictions = {}
            for label, k in specs:
                if k is None:
                    player_result['experiments'][label] = {'status': selection['status'], 'candidate_k': None}
                    continue
                progress(f'{username}: fitting {label}, frozen K={k}', flush=True)
                experiment_started = perf_counter()
                dataset = build_candidate_dataset(features, analysis, split, candidate_k=k, include_outside=True)
                all_rows = {name: [r for r in dataset.candidate_rows if r['split'] == name] for name in ('train', 'validation', 'test')}
                decisions = {name: [d for d in dataset.decisions if d['split'] == name] for name in all_rows}
                usable_ids = {d['decision_id'] for d in dataset.decisions if d['usable']}
                usable_rows = {name: [r for r in rows if r['decision_id'] in usable_ids] for name, rows in all_rows.items()}
                fit_started = perf_counter()
                model, preprocessor = train_candidate_model(usable_rows['train'], usable_rows['validation'], FULL_FEATURE_FIELDS)
                fit_seconds = perf_counter() - fit_started
                metrics, prediction_rows, inference_seconds = {}, {}, {}
                for name in ('validation', 'test'):
                    score_started = perf_counter()
                    probabilities = predict_candidate_probabilities(model, preprocessor, all_rows[name])
                    inference_seconds[name] = perf_counter() - score_started
                    metrics[name], prediction_rows[name] = all_decision_metrics(all_rows[name], probabilities, decisions[name])
                artifact = destination / label
                artifact.mkdir()
                model_path = artifact / 'engine_and_context_model.joblib'
                preprocessor_path = artifact / 'engine_and_context_preprocessing.json'
                joblib.dump(model, model_path)
                preprocessor.save(preprocessor_path)
                for name, rows in prediction_rows.items():
                    pq.write_table(pa.Table.from_pylist(rows), artifact / f'{name}_predictions.parquet')
                manifest = {'format_version': 1, 'username': username, 'candidate_k': k,
                            'experiment': label, 'selection': 'training-only K70' if label == 'history_k70' else 'fixed a priori',
                            'analysis_multipv': 20, 'engine_identity': source_manifest['engine_identity'],
                            'engine_settings': source_manifest['settings'], 'features': list(FULL_FEATURE_FIELDS),
                            'hyperparameters': model.get_params(), 'input_sha256': player['sha256'],
                            'preprocessing_fit_split': 'train', 'split_file': '../split_metadata.json',
                            'training_decisions': len(decisions['train']),
                            'training_covered_decisions': sum(d['usable'] for d in decisions['train']),
                            'training_candidate_rows': len(usable_rows['train']),
                            'all_training_candidate_rows': len(all_rows['train']),
                            'fit_seconds': fit_seconds, 'inference_seconds': inference_seconds,
                            'runtime_seconds': perf_counter() - experiment_started,
                            'model_bytes': model_path.stat().st_size,
                            'inference_artifact_bytes': model_path.stat().st_size + preprocessor_path.stat().st_size,
                            'artifact_size_definition': 'model_bytes=joblib; inference_artifact_bytes=model plus preprocessing; directory_bytes includes predictions and metadata'}
                write_json(artifact / 'manifest.json', manifest)
                write_json(artifact / 'evaluation_metrics.json', metrics)
                result = {**manifest, 'status': 'complete', 'metrics': metrics,
                          'artifact_directory_bytes': sum(f.stat().st_size for f in artifact.iterdir() if f.is_file())}
                player_result['experiments'][label] = result
                predictions[label] = prediction_rows['test']
                progress(f'{username}: {label}: all-test exact={metrics["test"]["exact_move_accuracy"]:.4f}, coverage={metrics["test"]["candidate_coverage"]:.4f}, fit={fit_seconds:.2f}s', flush=True)
            player_result['k10_minus_k5_test'] = paired_comparison(predictions['fixed_k5'], predictions['fixed_k10'])
            if 'history_k70' in predictions:
                player_result['history_minus_k5_test'] = paired_comparison(predictions['fixed_k5'], predictions['history_k70'])
            write_json(destination / 'comparison.json', player_result)
            report['players'].append(player_result)
            write_json(output_dir / 'report.json', report)
        report.update(status='complete', runtime_seconds=perf_counter() - started)
        write_json(output_dir / 'report.json', report)
        (output_dir / 'REPORT.md').write_text(render_width_report(report))
        return report
    except Exception as exc:
        report.update(status='failed', error=str(exc), runtime_seconds=perf_counter() - started)
        write_json(output_dir / 'report.json', report)
        raise
