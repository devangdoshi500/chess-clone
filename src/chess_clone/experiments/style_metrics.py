"""Opportunity-conditioned style distributions, not a 'style accuracy' percent.

Metrics compare expected behavior under the full move policy with observed
behavior on the same positions. Forced attributes are excluded. Equal-player
and equal-cell aggregation prevents prolific players and common phases from
dominating. These are replay diagnostics, not evidence of long-horizon plans.
"""

from collections import defaultdict
import math
from statistics import mean

from chess_clone.modeling.legal_policy import BEHAVIOR_BOOLEAN_FIELDS

STYLE_FIELDS = BEHAVIOR_BOOLEAN_FIELDS + (
    "candidate_piece_moved", "candidate_destination_wing",
)


def style_metrics(rows, probabilities, *, min_opportunities=20):
    if min_opportunities < 1:
        raise ValueError("min_opportunities must be positive")
    if len(rows) != len(probabilities):
        raise ValueError("Candidate and probability counts differ")
    groups = defaultdict(list)
    for row, probability in zip(rows, probabilities, strict=True):
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Invalid probability")
        groups[row["decision_id"]].append((row, probability))
    if not groups:
        raise ValueError("At least one decision is required")
    cells = defaultdict(list)
    forced = defaultdict(int)
    for decision, group in groups.items():
        if abs(sum(p for _, p in group) - 1) > 1e-6:
            raise ValueError("Probabilities must sum to one")
        actual = [r for r, _ in group if r["chosen"]]
        if len(actual) != 1:
            raise ValueError("Each decision requires one human move")
        actual = actual[0]
        if len({r["candidate_move_uci"] for r, _ in group}) != len(group):
            raise ValueError("Duplicate candidate")
        identity = (str(actual["player_username"]).casefold(), actual["game_phase"])
        if any((str(r["player_username"]).casefold(), r["game_phase"], r["game_id"])
               != (*identity, actual["game_id"]) for r, _ in group):
            raise ValueError("Inconsistent decision metadata")
        top = min(group, key=lambda pair: (-pair[1], pair[0]["candidate_move_uci"]))[0]
        for field in STYLE_FIELDS:
            key = (*identity, field)
            categories = {r[field] for r, _ in group}
            if len(categories) == 1:
                forced[key] += 1
                continue
            distribution = {value: sum(p for r, p in group if r[field] == value)
                            for value in categories}
            cells[key].append((actual[field], top[field], distribution))
    output = []
    for key in sorted(set(cells) | set(forced)):
        values = cells[key]
        categories = set().union(*(set(v[2]) for v in values)) if values else set()
        rates = []
        for category in sorted(categories, key=str):
            rates.append({"category": category,
                          "human": mean(a == category for a, _, _ in values),
                          "policy": mean(d.get(category, 0) for _, _, d in values),
                          "argmax": mean(t == category for _, t, _ in values)})
        output.append({"player": key[0], "phase": key[1], "attribute": key[2],
                       "opportunities": len(values), "forced_excluded": forced[key],
                       "supported": len(values) >= min_opportunities,
                       "policy_tv": sum(abs(r["human"] - r["policy"]) for r in rates) / 2 if values else None,
                       "argmax_tv": sum(abs(r["human"] - r["argmax"]) for r in rates) / 2 if values else None,
                       "attribute_brier": mean(sum((p - (c == a)) ** 2 for c, p in d.items())
                                               for a, _, d in values) if values else None,
                       "attribute_nll": -mean(math.log(max(d[a], 1e-15)) for a, _, d in values) if values else None,
                       "rates": rates})
    players = {}
    for player in sorted({key[0] for key in cells}):
        supported = [c for c in output if c["player"] == player and c["supported"]]
        players[player] = {"supported_cells": len(supported),
                           **{metric: mean(c[metric] for c in supported) if supported else None
                              for metric in ("policy_tv", "argmax_tv", "attribute_brier", "attribute_nll")}}
    return {"decisions": len(groups), "minimum_cell_opportunities": min_opportunities,
            "supported_players": sum(p["supported_cells"] > 0 for p in players.values()),
            "macro_policy_tv": mean(p["policy_tv"] for p in players.values() if p["supported_cells"])
            if any(p["supported_cells"] for p in players.values()) else None,
            "players": players, "cells": output}
