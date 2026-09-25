"""Capture/check choice rates from strictly earlier training games."""

from collections import defaultdict
from copy import deepcopy
import math

KINDS = {"capture": "candidate_is_capture", "check": "candidate_gives_check"}
OPPORTUNITY_FIELDS = tuple(f"profile_{kind}_{field}" for kind in KINDS
                           for field in ("match_probability", "observations", "available"))


def _empty():
    return {kind: {"opportunities": 0, "chosen": 0} for kind in KINDS}


class OpportunityProfile:
    """Smoothed player rates with chronological population fallback.

    All rows of a game receive a profile frozen before that game's timestamp.
    Games sharing a timestamp are encoded before any of their labels are added.
    Forced capture/check positions do not represent discretionary opportunities.
    """

    def __init__(self, smoothing: float = 20.):
        if not math.isfinite(smoothing) or smoothing <= 0:
            raise ValueError("smoothing must be positive and finite")
        self.smoothing = smoothing
        self.players = {}
        self.population = _empty()
        self.fitted = False

    def _encode(self, rows):
        groups = defaultdict(list)
        for row in rows:
            groups[row["decision_id"]].append(row)
        output = []
        for group in groups.values():
            state = self.players.get(str(group[0]["player_username"]).casefold(), _empty())
            opportunity = {kind: len({bool(r[field]) for r in group}) == 2 for kind, field in KINDS.items()}
            for row in group:
                item = dict(row)
                for kind, field in KINDS.items():
                    pool = self.population[kind]
                    prior = (pool["chosen"] + 1) / (pool["opportunities"] + 2)
                    local = state[kind]
                    probability = (local["chosen"] + self.smoothing * prior) / (local["opportunities"] + self.smoothing)
                    item[f"profile_{kind}_match_probability"] = probability if bool(row[field]) else 1 - probability
                    item[f"profile_{kind}_observations"] = local["opportunities"]
                    item[f"profile_{kind}_available"] = opportunity[kind]
                output.append(item)
        return output

    def _update(self, rows):
        groups = defaultdict(list)
        for row in rows:
            groups[row["decision_id"]].append(row)
        for group in groups.values():
            chosen = [r for r in group if r["chosen"]]
            if len(chosen) != 1:
                raise ValueError("Each decision requires one human move")
            player = str(group[0]["player_username"]).casefold()
            state = self.players.setdefault(player, _empty())
            for kind, field in KINDS.items():
                if len({bool(r[field]) for r in group}) != 2:
                    continue
                for counts in (state[kind], self.population[kind]):
                    counts["opportunities"] += 1
                    counts["chosen"] += int(bool(chosen[0][field]))

    def fit_transform_ordered(self, rows):
        if any(r["split"] != "train" for r in rows):
            raise ValueError("Profiles may only be fitted on training games")
        self.players, self.population = {}, _empty()
        by_date = defaultdict(list)
        game_dates = {}
        for row in rows:
            key = row["game_id"]
            if key in game_dates and game_dates[key] != row["played_at"]:
                raise ValueError("Inconsistent game timestamp")
            game_dates[key] = row["played_at"]
            by_date[row["played_at"]].append(row)
        encoded = {}
        for date in sorted(by_date):
            batch = by_date[date]
            for row in self._encode(batch):
                key = (row["decision_id"], row["candidate_move_uci"])
                if key in encoded:
                    raise ValueError("Duplicate candidate")
                encoded[key] = row
            self._update(batch)
        self.fitted = True
        return [encoded[(r["decision_id"], r["candidate_move_uci"])] for r in rows]

    def transform(self, rows):
        if not self.fitted:
            raise RuntimeError("Profile must be fitted before transform")
        encoded = {(r["decision_id"], r["candidate_move_uci"]): r for r in self._encode(rows)}
        return [encoded[(r["decision_id"], r["candidate_move_uci"])] for r in rows]

    def with_player_history(self, rows, *, before):
        """Return a copy with replacement personal histories and a frozen prior.

        Caller supplies complete, disjoint historical games. Only explicitly
        marked history rows strictly before the evaluation cutoff are accepted.
        Other players and the training-population counts remain unchanged.
        """
        if not self.fitted:
            raise RuntimeError("Profile must be fitted before adaptation")
        seen = set()
        decisions = {}
        game_dates = {}
        for row in rows:
            if row["split"] != "history" or row["played_at"] >= before:
                raise ValueError("Adaptation requires history strictly before cutoff")
            key = (row["decision_id"], row["candidate_move_uci"])
            if key in seen:
                raise ValueError("Duplicate history candidate")
            seen.add(key)
            identity = (row["game_id"], str(row["player_username"]).casefold())
            if decisions.setdefault(row["decision_id"], identity) != identity:
                raise ValueError("Inconsistent decision identity")
            if game_dates.setdefault(row["game_id"], row["played_at"]) != row["played_at"]:
                raise ValueError("Inconsistent game timestamp")
        temporary = OpportunityProfile(self.smoothing)
        temporary._update(rows)
        result = self.from_dict(self.to_dict())
        result.players.update(deepcopy(temporary.players))
        return result

    def to_dict(self):
        if not self.fitted:
            raise RuntimeError("Profile must be fitted before serialization")
        return deepcopy({"version": 1, "smoothing": self.smoothing,
                         "players": self.players, "population": self.population})

    @classmethod
    def from_dict(cls, value):
        if value["version"] != 1:
            raise ValueError("Unsupported profile version")
        instance = cls(value["smoothing"])
        instance.players = deepcopy(value["players"])
        instance.population = deepcopy(value["population"])
        instance.fitted = True
        return instance
