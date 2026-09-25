"""Explicit, versioned cohorts; labels are never treated as observed ratings."""

from dataclasses import dataclass
import json
from pathlib import Path
import re

RATING_BANDS = ('1200', '1400', '1600', '1800', '2000', '2200', '2400+', 'elite')


@dataclass(frozen=True, slots=True)
class CohortPlayer:
    username: str
    intended_rating_band: str
    games_requested: int = 300
    perf_type: str = 'blitz'
    notes: str | None = None
    raw_pgn: Path | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.username, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,30}', self.username):
            raise ValueError('username must be a valid explicit Lichess username')
        if self.intended_rating_band not in RATING_BANDS:
            raise ValueError(f'intended_rating_band must be one of {RATING_BANDS}')
        if type(self.games_requested) is not int or self.games_requested < 1:
            raise ValueError('games_requested must be a positive integer')
        if self.perf_type != 'blitz':
            raise ValueError('benchmark cohorts must use perf_type=blitz')
        if self.notes is not None and not isinstance(self.notes, str):
            raise ValueError('notes must be a string')


def load_cohort(path: str | Path) -> list[CohortPlayer]:
    """Read JSON; relative PGN paths are resolved relative to the config file."""
    path = Path(path)
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or set(payload) != {'schema_version', 'players'}:
        raise ValueError('cohort must contain only schema_version and players')
    if type(payload['schema_version']) is not int or payload['schema_version'] != 1:
        raise ValueError('unsupported cohort schema_version')
    if not isinstance(payload['players'], list) or not payload['players']:
        raise ValueError('Supply explicit benchmark usernames in players; no automatic discovery')
    players = []
    for item in payload['players']:
        if not isinstance(item, dict):
            raise ValueError('each player must be an object')
        item = dict(item)
        if item.get('raw_pgn') is not None:
            if not isinstance(item['raw_pgn'], str) or not item['raw_pgn']:
                raise ValueError('raw_pgn must be a nonempty path')
            item['raw_pgn'] = (path.parent / item['raw_pgn']).resolve()
        try:
            players.append(CohortPlayer(**item))
        except TypeError as exc:
            raise ValueError(f'invalid cohort player fields: {exc}') from exc
    names = [p.username.casefold() for p in players]
    if len(names) != len(set(names)):
        raise ValueError('cohort usernames must be unique (case insensitive)')
    return players
