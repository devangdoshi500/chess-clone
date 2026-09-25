"""Separate descriptive strength and observed-style profiles.

Builders accept explicit game IDs, so future callers can supply training IDs only.
No builder discovers a latest dataset or reads validation/test data implicitly.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass
from statistics import mean, median

from chess_clone.benchmark.metrics import coverage


@dataclass(frozen=True, slots=True)
class ProfileScope:
    purpose: str
    game_ids: tuple[str, ...]
    decisions: int


@dataclass(frozen=True, slots=True)
class PlayerStrengthProfile:
    username: str
    scope: ProfileScope
    median_rating: float | None
    mean_rating: float | None
    median_centipawn_loss: float | None
    mean_centipawn_loss: float | None
    finite_cp_decisions: int
    excluded_cp_decisions: int
    cp_loss_over_1000_count: int
    move_coverage: dict
    error_severity: dict
    accuracy_by_phase: dict
    accuracy_under_time_pressure: dict


@dataclass(frozen=True, slots=True)
class PlayerStyleProfile:
    username: str
    scope: ProfileScope
    opening_distribution: dict
    capture_rate: float | None
    check_rate: float | None
    queen_trade_tendency: dict
    castling_tendency: dict
    piece_move_distribution: dict
    average_move_time_seconds: float | None
    move_time_observations: int
    time_pressure_rate: float | None
    time_pressure_observations: int
    change_in_move_quality_under_time_pressure: dict
    phase_distribution: dict


def distribution(values: list) -> dict:
    counts = Counter('unknown' if v is None else str(v) for v in values)
    return {key: {'count': count, 'fraction': count / len(values)} for key, count in sorted(counts.items())}


def quality(rows: list[dict]) -> dict:
    losses = [r['centipawn_loss'] for r in rows if r['centipawn_loss'] is not None]
    return {**coverage([r['actual_move_rank'] for r in rows]),
            'finite_cp_decisions': len(losses),
            'mean_centipawn_loss': mean(losses) if losses else None,
            'median_centipawn_loss': median(losses) if losses else None}


def stratified_quality(rows: list[dict], field: str, expected: tuple[str, ...] = ()) -> dict:
    groups = defaultdict(list)
    for row in rows:
        value = row.get(field)
        key = 'unknown' if value is None else str(value).lower()
        groups[key].append(row)
    return {key: quality(groups[key]) for key in sorted(set(groups) | set(expected))}


def _selected(username: str, rows: list[dict], games: list[dict], game_ids: set[str], purpose: str):
    if purpose not in ('descriptive_benchmark', 'training_only'):
        raise ValueError('purpose must be descriptive_benchmark or training_only')
    selected = [r for r in rows if r['game_id'] in game_ids and r['player_username'].casefold() == username.casefold()]
    selected_games = [g for g in games if g['game_id'] in game_ids]
    if {g['game_id'] for g in selected_games} != game_ids:
        raise ValueError('profile game IDs must exist in supplied games')
    for game in selected_games:
        if username.casefold() not in (game['white_username'].casefold(), game['black_username'].casefold()):
            raise ValueError('profile game does not contain player')
    scope = ProfileScope(purpose, tuple(sorted(game_ids)), len(selected))
    return selected, selected_games, scope


def build_strength_profile(username: str, rows: list[dict], games: list[dict], *, game_ids: set[str], purpose: str = 'descriptive_benchmark') -> PlayerStrengthProfile:
    rows, games, scope = _selected(username, rows, games, game_ids, purpose)
    ratings = [g['white_rating'] if g['white_username'].casefold() == username.casefold() else g['black_rating'] for g in games]
    ratings = [r for r in ratings if r is not None]
    losses = [r['centipawn_loss'] for r in rows if r['centipawn_loss'] is not None]
    severity = [('under_50' if c < 50 else '50_to_99' if c < 100 else '100_to_299' if c < 300 else '300_plus') for c in losses]
    return PlayerStrengthProfile(
        username, scope, median(ratings) if ratings else None, mean(ratings) if ratings else None,
        median(losses) if losses else None, mean(losses) if losses else None,
        len(losses), len(rows) - len(losses), sum(c > 1000 for c in losses),
        coverage([r['actual_move_rank'] for r in rows]), distribution(severity),
        stratified_quality(rows, 'game_phase', ('opening', 'middlegame', 'endgame')),
        stratified_quality(rows, 'time_pressure', ('true', 'false', 'unknown')),
    )


def build_style_profile(username: str, rows: list[dict], games: list[dict], *, game_ids: set[str], purpose: str = 'descriptive_benchmark') -> PlayerStyleProfile:
    rows, games, scope = _selected(username, rows, games, game_ids, purpose)
    def rate(field):
        return sum(r[field] for r in rows) / len(rows) if rows else None
    times = [r['seconds_spent_on_move'] for r in rows if r['seconds_spent_on_move'] is not None]
    clocks = [r['time_pressure'] for r in rows if r['time_pressure'] is not None]
    pressure = quality([r for r in rows if r['time_pressure'] is True])
    normal = quality([r for r in rows if r['time_pressure'] is False])
    delta = {}
    for field in ('top_5', 'mean_centipawn_loss'):
        delta[field] = pressure[field] - normal[field] if pressure[field] is not None and normal[field] is not None else None
    opportunities = [r for r in rows if r['queen_capture_available']]
    castled_games = {r['game_id'] for r in rows if r['is_castle']}
    return PlayerStyleProfile(
        username, scope, distribution([g['eco'] for g in games]), rate('is_capture'), rate('is_check'),
        {'queen_captures_by_queen': sum(r['is_queen_trade'] for r in rows),
         'per_decision_rate': rate('is_queen_trade'), 'opportunity_decisions': len(opportunities),
         'opportunity_rate': sum(r['is_queen_trade'] for r in opportunities) / len(opportunities) if opportunities else None},
        {'games_castled': len(castled_games), 'games': len(games),
         'per_game_rate': len(castled_games) / len(games) if games else None},
        distribution([r['piece_moved'] for r in rows]), mean(times) if times else None, len(times),
        mean(clocks) if clocks else None, len(clocks),
        {'pressure': pressure, 'no_pressure': normal, 'pressure_minus_no_pressure': delta},
        distribution([r['game_phase'] for r in rows]),
    )
