"""Coverage, complete-game sampling, band summaries and descriptive correlations."""

from collections import defaultdict
import hashlib
from statistics import mean, median, stdev

KS = tuple(range(1, 21))


def coverage(ranks: list[int | None]) -> dict:
    if any(r is not None and (type(r) is not int or r < 1) for r in ranks):
        raise ValueError('ranks must be positive integers or None (outside analyzed width)')
    return {
        'decisions': len(ranks),
        **{f'top_{k}': sum(r is not None and r <= k for r in ranks) / len(ranks)
           if ranks else None for k in KS},
        'outside_top_20': sum(r is None or r > 20 for r in ranks) / len(ranks) if ranks else None,
    }


def smallest_k(values: dict, threshold: float) -> dict:
    if not 0 < threshold <= 1:
        raise ValueError('threshold must be in (0, 1]')
    if values['top_20'] is None:
        return {'k': None, 'status': 'no_data'}
    for k in KS:
        if values[f'top_{k}'] >= threshold:
            return {'k': k, 'status': 'reached'}
    return {'k': None, 'status': 'greater_than_20_unresolved'}


def thresholds(values: dict) -> dict:
    return {str(t): smallest_k(values, t / 100) for t in (70, 90, 95)}


def sample_games(rows: list[dict], max_decisions: int | None, seed: int, username: str) -> tuple[list[dict], dict]:
    """Take a hash-shuffled prefix of complete games, stopping before the cap.

    Never skip a long game to pack in shorter games: that would favor short games.
    If the first game does not fit, require a larger budget rather than truncate it.
    """
    if max_decisions is not None and (type(max_decisions) is not int or max_decisions < 1):
        raise ValueError('max_decisions must be positive or None')
    groups = defaultdict(list)
    seen = set()
    for row in rows:
        key = (row['game_id'], row['ply'])
        if key in seen:
            raise ValueError(f'duplicate decision {key}')
        seen.add(key)
        groups[row['game_id']].append(row)
    order = sorted(groups, key=lambda g: (hashlib.sha256(
        f'{seed}:{username.casefold()}:{g}'.encode()).hexdigest(), g))
    selected = []
    selected_ids = []
    for game_id in order:
        game_rows = sorted(groups[game_id], key=lambda r: r['ply'])
        if max_decisions is not None and len(selected) + len(game_rows) > max_decisions:
            if not selected:
                raise ValueError(f'max-decisions-per-player must be at least {len(game_rows)} to fit first sampled game {game_id}')
            break
        selected.extend(game_rows)
        selected_ids.append(game_id)
    return selected, {
        'method': 'sha256_permuted_whole_game_prefix_v1', 'seed': seed,
        'max_decisions': max_decisions, 'available_games': len(groups),
        'available_decisions': len(rows), 'selected_games': len(selected_ids),
        'selected_decisions': len(selected), 'omitted_decisions': len(rows) - len(selected),
        'selected_game_ids': selected_ids,
        'omitted_game_ids': [g for g in order if g not in set(selected_ids)],
    }


def aggregate_bands(players: list[dict]) -> dict:
    grouped = defaultdict(list)
    for player in players:
        grouped[player['intended_rating_band']].append(player)
    result = {}
    for band, members in sorted(grouped.items()):
        observed = [r for p in members for r in p['metadata']['observed_ratings']]
        values = {f'top_{k}': [p['coverage'][f'top_{k}'] for p in members
                              if p['coverage'][f'top_{k}'] is not None] for k in KS}
        means = {key: mean(v) if v else None for key, v in values.items()}
        result[band] = {
            'players': len(members), 'players_with_coverage': len(values['top_1']),
            'usernames': [p['username'] for p in members],
            'games': sum(p['metadata']['games_obtained'] for p in members),
            'sampled_games': sum(p['sampling']['selected_games'] for p in members),
            'decisions': sum(p['coverage']['decisions'] for p in members),
            'median_observed_rating': median(observed) if observed else None,
            'coverage_mean': means,
            'coverage_stdev': {key: stdev(v) if len(v) > 1 else None for key, v in values.items()},
            'coverage_range': {key: [min(v), max(v)] if v else None for key, v in values.items()},
            'adaptive_k': thresholds(means),
            'weighting': 'equal player coverage; median rating pools game-level observations',
        }
    return result


def _average_ranks(values: list[float]) -> list[float]:
    ordered = sorted(set(values))
    result = {}
    offset = 0
    for value in ordered:
        count = values.count(value)
        result[value] = offset + (count + 1) / 2
        offset += count
    return [result[v] for v in values]


def _pearson(x: list[float], y: list[float]) -> float:
    mx, my = mean(x), mean(y)
    numerator = sum((a - mx) * (b - my) for a, b in zip(x, y))
    denominator = (sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y)) ** .5
    return max(-1., min(1., numerator / denominator))


def correlations(players: list[dict]) -> dict:
    scatter = [{
        'username': p['username'], 'intended_rating_band': p['intended_rating_band'],
        'median_rating': p['metadata']['median_player_rating'],
        'top_5': p['coverage']['top_5'], 'top_10': p['coverage']['top_10'],
        'k90': p['adaptive_k']['90']['k'], 'k90_status': p['adaptive_k']['90']['status'],
    } for p in players]
    statistics = {}
    for field in ('top_5', 'top_10', 'k90'):
        pairs = [(p['median_rating'], p[field]) for p in scatter
                 if p['median_rating'] is not None and p[field] is not None]
        x, y = [p[0] for p in pairs], [p[1] for p in pairs]
        reason = 'fewer_than_3_players' if len(pairs) < 3 else (
            'constant_values' if len(set(x)) < 2 or len(set(y)) < 2 else None)
        statistics[field] = {
            'n': len(pairs), 'excluded_players': len(players) - len(pairs),
            'pearson': None if reason else _pearson(x, y),
            'spearman': None if reason else _pearson(_average_ranks(x), _average_ranks(y)),
            'undefined_reason': reason,
        }
    return {'statistics': statistics, 'scatter': scatter,
            'caution': 'Descriptive player-level associations, no linearity assumption. K90 omits right-censored unresolved players; defined-only correlation can be biased.'}
