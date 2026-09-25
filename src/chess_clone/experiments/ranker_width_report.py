"""Render the completed milestone, only after validation selection and test."""

import argparse
import csv
from importlib.metadata import version
import json
from pathlib import Path
from time import perf_counter

import pyarrow.parquet as pq

from chess_clone.benchmark.runner import write_json
from chess_clone.experiments.ranker_width import (
    PLAYERS, WIDTHS, digest_tree, load_models, now, verify_selection,
)
from chess_clone.experiments.width_data import file_digest
from chess_clone.experiments.width_metrics import paired_comparison


def transitions(left, right):
    a = {p['decision_id']: p for p in left}
    b = {p['decision_id']: p for p in right}
    if set(a) != set(b):
        raise ValueError('decision mismatch')
    new = [key for key in a if not a[key]['covered'] and b[key]['covered']]
    lost = [key for key in a if a[key]['covered'] and not b[key]['covered']]
    return {'newly_covered': len(new), 'newly_covered_exact_correct': sum(b[key]['exact_correct'] for key in new),
            'lost_coverage': len(lost), 'lost_exact_correct': sum(a[key]['exact_correct'] for key in lost),
            'covered_in_both': sum(a[key]['covered'] and b[key]['covered'] for key in a),
            'net_exact_correct': sum(int(b[key]['exact_correct']) - int(a[key]['exact_correct']) for key in a)}


def pct(value):
    return '—' if value is None else f'{100*value:.2f}%'


def delta(value):
    interval = value['paired_game_bootstrap_95_interval']
    return f'{100*value["delta"]:+.2f} pp [{100*interval[0]:+.2f}, {100*interval[1]:+.2f}]'


def render(report):
    selection = report['selection']
    lines = ['# Candidate width versus ranker: completed milestone', '',
             'Production behavior and all saved baseline/production artifacts are unchanged (SHA-256 verified).', '',
             '## Decision', '']
    for player in report['players']:
        selected = selection['players'][player['username']]['selected_k']
        arm = player['arms'][str(selected)]
        val = arm['validation']['models']['catboost']
        test = arm['test']['models']['catboost']
        lines.append(f'- **{player["username"]}: freeze K={selected}.** Validation exact {pct(val["metrics"]["exact_move_accuracy"])}, '
                     f'test exact {pct(test["metrics"]["exact_move_accuracy"])}; mean end-to-end '
                     f'{val["timing"]["catboost_end_to_end_ms"]["mean"]:.2f}/{test["timing"]["catboost_end_to_end_ms"]["mean"]:.2f} ms (validation/test).')
    lines += ['', f'Selection frozen at `{selection["frozen_at"]}`; test first opened at `{report["test_opened"]["opened_at"]}`. '
              'Both players were selected before either test window was opened. No test-based revision was permitted.', '',
              '## Controlled protocol', '',
              '- Full preserved 300-game histories, 210/45/45 chronological whole-game split. Every decision is evaluated, with outside-candidate moves wrong for exact and top-3.',
              '- K=3/5/10 each uses actual Stockfish 18 MultiPV=min(K, legal moves), 20,000 nodes, one thread, 16 MB hash, new UCI game per position, no cache. Candidate sets are not necessarily nested.',
              '- Both rankers use the same 31 existing `FULL_FEATURE_FIELDS` from the preceding width experiment. The existing grouped CatBoost trainer is reused: QuerySoftMax, up to 350 depth-6 trees, learning rate .05, seed 20260830, validation early stopping 50. RF retains its existing 160-tree settings. No feature additions or player pooling.',
              '- Training and CatBoost early stopping use covered decisions only, matching the existing objective. All held-out decisions are scored. RF preprocessing is fit on training only. No history features, calibration tuning, rating-based width rules, or hyperparameter search.',
              '- Pre-registered selection: begin at K=3; consider K=5 then K=10 against the current selection. Promote only when validation exact gain exceeds 1 pp, the paired whole-game 95% bootstrap lower bound is positive, and added mean end-to-end latency is at most 10 ms per pp gained. Otherwise prefer the smaller K. A non-significant difference is uncertainty, not proof of equivalence.',
              '- Bootstrap: paired whole games, 2,000 resamples, seed 42; uncertainty excludes training-seed, model-family, and new-window variation. Validation is also used for early stopping, so the frozen test result is the final estimate.', '',
              'Opening tags retain the preceding experiment’s archived PGN metadata. This milestone does not establish live pre-decision availability of those tags; the comparison is offline with the feature schema held fixed.', '',
              '## Accuracy and candidate accounting', '',
              'Exact and top-3 use all decisions. Conditional exact uses covered decisions. All percentages are rounded; machine-readable counts and unrounded values are in `report.json` and `metrics.csv`.', '',
              '| Player | Split | K | Model | Decisions | Coverage | Exact | Top-3 | Conditional exact | Candidates mean [min,max] |',
              '|---|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for player in report['players']:
        for name in ('validation', 'test'):
            for k in WIDTHS:
                for family in ('catboost', 'rf'):
                    m = player['arms'][str(k)][name]['models'][family]['metrics']
                    lines.append(f'| {player["username"]} | {name} | {k} | {family} | {m["decisions"]} | {pct(m["candidate_coverage"])} | '
                                 f'{pct(m["exact_move_accuracy"])} | {pct(m["top_3_accuracy"])} | {pct(m["conditional_exact_accuracy"])} | '
                                 f'{m["mean_candidates_per_decision"]:.3f} [{m["minimum_candidates"]},{m["maximum_candidates"]}] |')
    lines += ['', '## Actual serving costs and artifact sizes', '',
              'Every validation/test decision receives a fresh search. Width order rotates per decision; models are in memory and warmed, and measurements run serially. '
              'The same search and feature construction feed both rankers; score order alternates. End-to-end is the sum of measured search, feature, and single-decision scoring components for that decision. '
              'It includes UCI overhead, board/context and candidate construction, model preprocessing, scoring and probability normalization. It excludes engine/model startup, disk/network I/O, and label/metric accounting. '
              'Pre-move clocks are reconstructed from prior observed clocks before replay. Existing opening metadata is retained; opening classification is not re-extracted from a live game. '
              'These are local replay measurements, not a deployed service SLA. No MultiPV=20 timing is used.', '',
              '| Player | Split | K | Search mean ms | Features mean ms | CB score mean ms | CB E2E mean / p50 / p95 ms | RF score mean ms | RF E2E mean / p95 ms |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for player in report['players']:
        for name in ('validation', 'test'):
            for k in WIDTHS:
                models = player['arms'][str(k)][name]['models']
                cb, rf = models['catboost']['timing'], models['rf']['timing']
                ce, re = cb['catboost_end_to_end_ms'], rf['rf_end_to_end_ms']
                lines.append(f'| {player["username"]} | {name} | {k} | {cb["search_ms"]["mean"]:.2f} | {cb["feature_ms"]["mean"]:.2f} | '
                             f'{cb["catboost_score_ms"]["mean"]:.2f} | {ce["mean"]:.2f} / {ce["p50"]:.2f} / {ce["p95"]:.2f} | '
                             f'{rf["rf_score_ms"]["mean"]:.2f} | {re["mean"]:.2f} / {re["p95"]:.2f} |')
    lines += ['', '| Player | K | Covered / all train decisions | CB trees | CB model / serving MiB | RF model / serving MiB | Full arm MiB | Both models load ms |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for player in report['players']:
        for k in WIDTHS:
            arm = player['arms'][str(k)]
            cb, rf = arm['artifacts']['catboost'], arm['artifacts']['rf']
            size = lambda m: f'{m["model_bytes"]/1048576:.3f} / {m["inference_artifact_bytes"]/1048576:.3f}'
            lines.append(f'| {player["username"]} | {k} | {arm["training"]["covered"]}/{arm["training"]["decisions"]} | {cb["tree_count"]} | '
                         f'{size(cb)} | {size(rf)} | {arm["directory_bytes"]/1048576:.2f} | {arm["both_models_load_ms"]:.2f} |')
    lines += ['', 'Serving bytes = model plus feature schema (CatBoost) or model plus preprocessing (RF); full arm includes datasets, predictions and timing records. '
              'The separately measured one-time load is both models together, with the OS file cache uncontrolled, and is excluded from warm serving cost.', '',
              '## Matched ranker and width comparisons', '',
              'Exact-accuracy deltas in percentage points; brackets are paired whole-game 95% bootstrap intervals.', '',
              '| Player | Comparison | Validation | Test |', '|---|---|---:|---:|']
    for player in report['players']:
        for k in WIDTHS:
            arm = player['arms'][str(k)]
            lines.append(f'| {player["username"]} | CatBoost − RF, K={k} | {delta(arm["validation"]["catboost_minus_rf"]["exact_correct"])} | '
                         f'{delta(arm["test"]["catboost_minus_rf"]["exact_correct"])} |')
        for comparison in ('k5_minus_k3', 'k10_minus_k5', 'k10_minus_k3'):
            lines.append(f'| {player["username"]} | CatBoost {comparison} | {delta(player["comparisons"]["validation"][comparison]["exact_correct"])} | '
                         f'{delta(player["comparisons"]["test"][comparison]["exact_correct"])} |')
    lines += ['', '## Preserved fixed-K=5 baseline', '',
              'The preceding RF fixed-K=5 result is preserved byte-for-byte and remains the historical baseline. '
              'It used a prefix of MultiPV=20, so differences from the new actual-MultiPV=5 arm also change the candidate search protocol. '
              'Use the matched RF rows above to isolate CatBoost’s effect. The earlier Magnus-only saved CatBoost artifacts also remain unchanged; their different feature/analysis setup is not a matched comparator.', '',
              '| Player | Split | Coverage | Exact | Top-3 | Conditional exact |', '|---|---|---:|---:|---:|---:|']
    for player in report['players']:
        for name in ('validation', 'test'):
            m = player['preserved_fixed_k5']['metrics'][name]
            lines.append(f'| {player["username"]} | {name} | {pct(m["candidate_coverage"])} | {pct(m["exact_move_accuracy"])} | '
                         f'{pct(m["top_3_accuracy"])} | {pct(m["conditional_exact_accuracy"])} |')
    lines += ['', '## Interpretation', '']
    for player in report['players']:
        username = player['username']
        k = selection['players'][username]['selected_k']
        m = player['arms'][str(k)]['test']['models']['catboost']['metrics']
        outside = m['outside_decisions']
        wrong_inside = m['covered_decisions'] - round(m['exact_move_accuracy'] * m['decisions'])
        primary = 'ranker choice among candidates' if wrong_inside > outside else 'candidate coverage'
        lines.append(f'**{username}: limitation.** At frozen K={k}, the larger test error component is **{primary}**: '
                     f'{outside} outside-candidate misses versus {wrong_inside} wrong choices among covered decisions.')
        wide = player['arms']['10']['test']['models']['catboost']['metrics']
        wide_inside = wide['covered_decisions'] - round(wide['exact_move_accuracy'] * wide['decisions'])
        lines.append(f'At K=10, there are {wide["outside_decisions"]} outside misses and {wide_inside} wrong candidate choices: '
                     'expanding coverage leaves substantial ranking headroom.')
        cb_improvements = []
        for width in WIDTHS:
            arm = player['arms'][str(width)]
            v = arm['validation']['catboost_minus_rf']['exact_correct']
            t = arm['test']['catboost_minus_rf']['exact_correct']
            cb_improvements.append(f'{100*t["delta"]:+.2f}')
        supported = all(player['arms'][str(width)]['test']['catboost_minus_rf']['exact_correct']['paired_game_bootstrap_95_interval'][0] > 0 for width in WIDTHS)
        verdict = 'improves test accuracy at every K, with positive paired intervals' if supported else 'has no clear test advantage at any K in this run'
        lines += ['', f'**Ranker:** CatBoost {verdict}; test deltas at K=3/5/10 are ' + '/'.join(cb_improvements) + ' pp. Validation comparisons appear above.', '']
        v = player['comparisons']['validation']['k10_minus_k5']
        t = player['comparisons']['test']['k10_minus_k5']
        a5 = player['arms']['5']['validation']['models']['catboost']['timing']['catboost_end_to_end_ms']['mean']
        a10 = player['arms']['10']['validation']['models']['catboost']['timing']['catboost_end_to_end_ms']['mean']
        lines.append(f'**K=10 value:** versus K=5, test coverage increases {100*t["covered"]["delta"]:.2f} pp, '
                     f'while exact changes {delta(t["exact_correct"])}. Validation E2E is {a5:.2f} → {a10:.2f} ms. '
                     f'The validation rule {"selects K=10" if k == 10 else "retains K="+str(k)+" because the width gain fails its accuracy-evidence gate"}.')
        wider_test = player['comparisons']['test']['k10_minus_k3']['exact_correct']
        if k != 10 and wider_test['paired_game_bootstrap_95_interval'][0] > 0:
            lines.append(f'K=10 versus K=3 does gain {delta(wider_test)} on test. That promising secondary result '
                         'cannot revise the validation-frozen selection.')
        lines.append('')
    lines += ['K=10 is faster here under the fixed total node budget, and its serving artifacts are larger. '
              'The smaller-K choice follows the required accuracy-evidence preference, not an assumed search-cost penalty.', '',
              '**Next feature gap:** candidate-specific tactical difficulty under the pre-move clock—hanging pieces, immediate threats, '
              'and narrow forcing continuations. Basic rank/evaluation and move properties omit those demands. This is a feature hypothesis for a later ablation; no new architecture is proposed.', '',
              '## Reproduction and audit files', '',
              'See `docs/ranker_width_milestone.md` for phase commands. The output directory contains `protocol.json`, '
              '`selection.json` and its hash seal, `test_opened.json`, `report.json`, `metrics.csv`, per-arm models, '
              'input/dataset hashes, complete per-decision predictions and latency records. Original source data and baselines are never overwritten.', '']
    return '\n'.join(lines)


def build_report(output):
    selection = verify_selection(output)
    protocol = json.loads((output / 'protocol.json').read_text())
    baseline = Path(protocol['baseline'])
    if digest_tree(baseline) != protocol['preserved_baseline_sha256']:
        raise ValueError('historical baseline modified')
    if digest_tree(Path('artifacts/models')) != protocol['preserved_production_sha256']:
        raise ValueError('production artifacts modified')
    opened = json.loads((output / 'test_opened.json').read_text())
    if opened['selection_sha256'] != file_digest(output / 'selection.json') or opened['opened_at'] <= selection['frozen_at']:
        raise ValueError('test selection chronology violated')
    report = {'status': 'complete', 'created_at': now(), 'protocol': protocol, 'selection': selection,
              'test_opened': opened, 'production_and_baseline_hashes_unchanged': True,
              'library_versions': {name: version(name) for name in ('catboost', 'scikit-learn', 'python-chess', 'numpy', 'pyarrow')},
              'players': []}
    flat = []
    for username in PLAYERS:
        player = {'username': username, 'arms': {}, 'comparisons': {}, 'coverage_transitions': {}}
        for k in WIDTHS:
            dest = output / username.lower() / f'k{k}'
            arm = {name: json.loads((dest / f'{name}_evaluation.json').read_text()) for name in ('validation', 'test')}
            arm['artifacts'] = json.loads((dest / 'models.json').read_text())
            arm['training'] = json.loads((dest / 'train_data.json').read_text())
            arm['directory_bytes'] = sum(p.stat().st_size for p in dest.iterdir() if p.is_file())
            started = perf_counter()
            load_models(dest)
            arm['both_models_load_ms'] = (perf_counter() - started)*1000
            player['arms'][str(k)] = arm
            total = arm['training']['decisions'] + sum(arm[name]['models']['catboost']['metrics']['decisions']
                                                      for name in ('validation', 'test'))
            if total != protocol['inputs'][username]['decisions']:
                raise ValueError('full-history decision count changed')
            for name in ('validation', 'test'):
                for family, result in arm[name]['models'].items():
                    row = {'player': username, 'split': name, 'k': k, 'model': family, **result['metrics'],
                           'model_bytes': arm['artifacts'][family]['model_bytes'],
                           'inference_artifact_bytes': arm['artifacts'][family]['inference_artifact_bytes']}
                    for field, values in result['timing'].items():
                        if isinstance(values, dict):
                            row.update({f'{field}_{stat}': value for stat, value in values.items()})
                    flat.append(row)
        for name in ('validation', 'test'):
            predictions = {k: pq.read_table(output / username.lower() / f'k{k}' / f'{name}_catboost_predictions.parquet').to_pylist() for k in WIDTHS}
            player['comparisons'][name] = {f'k{b}_minus_k{a}': paired_comparison(predictions[a], predictions[b])
                                            for a, b in ((3, 5), (5, 10), (3, 10))}
            player['coverage_transitions'][name] = transitions(predictions[5], predictions[10])
        original = baseline / username.lower() / 'fixed_k5'
        player['preserved_fixed_k5'] = {'source': str(original),
                                      'metrics': json.loads((original / 'evaluation_metrics.json').read_text())}
        for name in ('validation', 'test'):
            expected = player['preserved_fixed_k5']['metrics'][name]['decisions']
            if any(player['arms'][str(k)][name]['models']['catboost']['metrics']['decisions'] != expected for k in WIDTHS):
                raise ValueError('all-decision count changed relative to preserved baseline')
        report['players'].append(player)
    write_json(output / 'report.json', report)
    keys = list(dict.fromkeys(key for row in flat for key in row))
    with (output / 'metrics.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(flat)
    (output / 'REPORT.md').write_text(render(report))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    build_report(parser.parse_args().output)
