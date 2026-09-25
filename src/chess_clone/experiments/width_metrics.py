"""All-decision held-out scoring, including moves outside the candidate set."""

from collections import defaultdict
import math
from statistics import mean

import numpy as np


def all_decision_metrics(rows: list[dict], probabilities: list[float], decisions: list[dict]) -> tuple[dict, list[dict]]:
    if len(rows) != len(probabilities):
        raise ValueError('probability and candidate counts differ')
    grouped = defaultdict(list)
    for row, probability in zip(rows, probabilities, strict=True):
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError('invalid candidate probability')
        grouped[row['decision_id']].append((row, probability))
    if len({d['decision_id'] for d in decisions}) != len(decisions):
        raise ValueError('duplicate evaluation decisions')
    if set(grouped) != {d['decision_id'] for d in decisions}:
        raise ValueError('must score candidates for every evaluation decision exactly once')
    predictions = []
    for decision in decisions:
        candidates = sorted(grouped[decision['decision_id']], key=lambda pair: (-pair[1], pair[0]['engine_rank'], pair[0]['candidate_move_uci']))
        if abs(sum(p for _, p in candidates) - 1) > 1e-6:
            raise ValueError('candidate probabilities must sum to one')
        moves = [row['candidate_move_uci'] for row, _ in candidates]
        if len(moves) != len(set(moves)):
            raise ValueError('duplicate candidate moves')
        actual = decision['actual_move_uci']
        covered = actual in moves
        if covered != bool(decision['usable']):
            raise ValueError('candidate coverage and decision labels disagree')
        predictions.append({
            'decision_id': decision['decision_id'], 'game_id': decision['game_id'],
            'actual_move_uci': actual, 'predicted_move_uci': moves[0],
            'top_3_moves': moves[:3], 'candidate_count': len(moves),
            'covered': covered, 'exact_correct': actual == moves[0],
            'top_3_correct': actual in moves[:3],
        })
    n = len(predictions)
    covered = sum(p['covered'] for p in predictions)
    exact = sum(p['exact_correct'] for p in predictions)
    top3 = sum(p['top_3_correct'] for p in predictions)
    return {
        'decisions': n, 'covered_decisions': covered, 'outside_decisions': n - covered,
        'candidate_coverage': covered / n if n else None,
        'exact_move_accuracy': exact / n if n else None,
        'top_3_accuracy': top3 / n if n else None,
        'conditional_exact_accuracy': exact / covered if covered else None,
        'conditional_top_3_accuracy': top3 / covered if covered else None,
        'candidate_rows': len(rows), 'mean_candidates_per_decision': mean(p['candidate_count'] for p in predictions) if n else None,
        'minimum_candidates': min((p['candidate_count'] for p in predictions), default=None),
        'maximum_candidates': max((p['candidate_count'] for p in predictions), default=None),
        'denominator': 'all held-out decisions; outside-candidate moves are misses',
    }, predictions


def paired_comparison(baseline: list[dict], wider: list[dict], *, seed: int = 42, resamples: int = 2000) -> dict:
    """Paired whole-game bootstrap; preserve clustering within each sampled game."""
    left = {p['decision_id']: p for p in baseline}
    right = {p['decision_id']: p for p in wider}
    if set(left) != set(right) or len(left) != len(baseline) or len(right) != len(wider):
        raise ValueError('comparisons require identical unique test decisions')
    groups = defaultdict(list)
    for key, p in left.items():
        if p['game_id'] != right[key]['game_id'] or p['actual_move_uci'] != right[key]['actual_move_uci']:
            raise ValueError('comparison decision identity mismatch')
        groups[p['game_id']].append(key)
    game_ids = sorted(groups)
    if not game_ids:
        raise ValueError('empty paired comparison')
    counts = np.array([len(groups[g]) for g in game_ids])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(game_ids), size=(resamples, len(game_ids)))
    result = {'games': len(game_ids), 'decisions': len(left), 'bootstrap_seed': seed, 'resamples': resamples}
    for field in ('covered', 'exact_correct', 'top_3_correct'):
        differences = np.array([sum(int(right[k][field]) - int(left[k][field]) for k in groups[g]) for g in game_ids])
        samples = differences[draws].sum(axis=1) / counts[draws].sum(axis=1)
        result[field] = {'delta': float(differences.sum() / counts.sum()),
                         'paired_game_bootstrap_95_interval': [float(v) for v in np.quantile(samples, [.025, .975])]}
    return result
