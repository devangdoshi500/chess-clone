"""Human-readable companion to the controlled experiment's JSON results."""


def render_width_report(report: dict) -> str:
    def pct(value):
        return '—' if value is None else f'{value * 100:.2f}%'

    def table(headers, rows):
        return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                          '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                         ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])

    parts = ['# Controlled candidate-width experiment',
             'Each player is modeled separately using their complete obtained history. All arms share chronological whole-game splits, the same Random Forest hyperparameters and engine/context features, and nested prefixes of one Stockfish 18 MultiPV=20 analysis. Production K=5, existing RF/CatBoost artifacts, and canonical legacy splits are unchanged.',
             'The new fixed-K=5 arm is a controlled baseline on these 300-game histories and the shared broad-search protocol. It is not a replacement for, or direct rerun of, the previously saved model on its older inputs. Broad MultiPV can change ordering compared with a separate narrow search.',
             '## Data and frozen selection',
             table(['Player', 'Games', 'Decisions', 'Train/validation/test games', 'Training-only K70', 'Training coverage at K70'],
                   [[p['username'], p['games'], p['decisions'], '/'.join(str(p['split_game_counts'][s]) for s in ('train', 'validation', 'test')),
                     p['history_k_selection']['selected_k'], pct(p['history_k_selection']['coverage'].get(f"top_{p['history_k_selection']['selected_k']}"))] for p in report['players']]),
             'K70 selects the smallest integer K from 1 through 20 reaching 70% coverage on training games only. The selection file records those game IDs. Validation/test ranks are not consulted. All arms fit on their covered training decisions; increasing K therefore also admits additional training decisions. Preprocessing is fitted on training rows only. No cohort pooling, rating formula, test-time width tuning, or new strength/style input features are introduced.']
    for split in ('validation', 'test'):
        rows = []
        for player in report['players']:
            for label in ('fixed_k5', 'fixed_k10', 'history_k70'):
                e = player['experiments'][label]
                if e['status'] != 'complete':
                    continue
                m = e['metrics'][split]
                rows.append([player['username'], label, e['candidate_k'], m['decisions'], pct(m['candidate_coverage']),
                             pct(m['exact_move_accuracy']), pct(m['top_3_accuracy']), pct(m['conditional_exact_accuracy']),
                             pct(m['conditional_top_3_accuracy']), m['candidate_rows'], f"{m['mean_candidates_per_decision']:.3f}"])
        parts += [f'## {split.title()} results',
                  table(['Player', 'Arm', 'K', 'All decisions', 'Coverage', 'Exact', 'Top 3', 'Conditional exact', 'Conditional top 3', 'Candidate rows', 'Mean candidates'], rows)]
    parts += ['Exact and top-three accuracies use all held-out decisions as the denominator. Actual moves outside the candidate set count as failures. Conditional accuracies use only covered decisions. Every held-out decision has predicted candidates saved in its arm’s predictions Parquet. Actual candidate counts can be below K when fewer legal moves exist.',
              '## Paired held-out differences',
              'Differences are percentage points relative to fixed K=5. Intervals are paired whole-game percentile bootstrap intervals (2,000 resamples, seed 42), preserving within-game clustering. They describe uncertainty in this test window, not robustness to another player, training seed, era, or model family.']
    rows = []
    for p in report['players']:
        for label, key in [('K10 minus K5', 'k10_minus_k5_test'), ('History minus K5', 'history_minus_k5_test')]:
            if key not in p:
                continue
            c = p[key]
            for field in ('covered', 'exact_correct', 'top_3_correct'):
                value = c[field]
                interval = value['paired_game_bootstrap_95_interval']
                rows.append([p['username'], label, field, f"{100 * value['delta']:+.2f}", f"[{100 * interval[0]:+.2f}, {100 * interval[1]:+.2f}]", c['games']])
    parts.append(table(['Player', 'Comparison', 'Metric', 'Delta (pp)', '95% game-bootstrap interval (pp)', 'Test games'], rows))
    parts.append('## Cost and artifacts')
    rows = []
    for p in report['players']:
        for label in ('fixed_k5', 'fixed_k10', 'history_k70'):
            e = p['experiments'][label]
            if e['status'] != 'complete':
                continue
            rows.append([p['username'], label, e['training_covered_decisions'], e['training_candidate_rows'],
                         f"{e['runtime_seconds']:.3f}", f"{e['fit_seconds']:.3f}", f"{e['inference_seconds']['test']:.3f}",
                         e['model_bytes'], e['inference_artifact_bytes'], e['artifact_directory_bytes']])
    parts.append(table(['Player', 'Arm', 'Covered train decisions', 'Train candidate rows', 'Arm seconds', 'Fit seconds', 'Test score seconds', 'Model bytes', 'Model + preprocessing bytes', 'Directory bytes'], rows))
    a = report['analysis_runtime']
    parts += [f"Shared analysis: {a.get('coverage_requests')} decisions, {a.get('unique_positions')} unique positions, {a.get('engine_calls')} engine calls, {a.get('cache_hits')} cache hits, {a.get('runtime_seconds')} wall seconds, {a.get('engine_seconds')} engine seconds, and {a.get('average_time_per_miss_seconds')} engine seconds per miss. There are no post-move quality searches in preparation. Model comparison wall time: {report.get('runtime_seconds')} seconds.",
              'Arm runtime includes dataset construction, fitting, evaluation, and artifact serialization. Test score time includes preprocessing and model scoring over every test candidate, but excludes engine search, candidate feature construction, disk loading, and metric computation. These are single-run timings, not a calibrated latency benchmark. Engine cost is shared; this comparison does not measure standalone MultiPV=5 versus MultiPV=10 search latency.',
              'Model bytes count the joblib estimator. Inference bytes add preprocessing. Directory bytes also include manifests, evaluations, and prediction files. Saved manifests freeze candidate width, engine protocol, feature list, hyperparameters, and input hashes. Old artifacts without width metadata default to K=5; inference rejects conflicting requested widths.',
              '## Decision boundary',
              'Assess whether a coverage gain survives reranking as exact/top-three imitation accuracy, and compare that gain with actual candidate counts and measured costs for each player. These RF results do not establish the behavior of grouped CatBoost or a generic human model. No arm is automatically promoted and no rating-based formula is fitted. A future production decision needs independent validation beyond these two histories.']
    return '\n\n'.join(parts) + '\n'
