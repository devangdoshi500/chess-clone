"""Development-only personal-style diagnostic on an already inspected cohort."""

import argparse
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path

import chess
from catboost import CatBoostRanker
import numpy as np
import pyarrow.parquet as pq

from chess_clone.experiments.expanded_evaluation import load_sealed
from chess_clone.experiments.move_quality import _write_json
from chess_clone.experiments.population_metrics import legal_policy_metrics
from chess_clone.experiments.profile_ablation import digest
from chess_clone.experiments.style_metrics import style_metrics
from chess_clone.modeling.boosted import groupwise_softmax, predict_relevance_scores
from chess_clone.modeling.legal_policy import build_all_legal_candidate_rows
from chess_clone.modeling.opportunity_profile import OpportunityProfile


def split_history(positions, dates):
    """Earliest half of whole games; timestamp ties go to evaluation together."""
    games = sorted({r["game_id"] for r in positions}, key=lambda g: (dates[g], g))
    if len(games) < 10:
        raise ValueError("At least ten games required")
    cutoff = dates[games[len(games) // 2]]
    history = [r for r in positions if dates[r["game_id"]] < cutoff]
    later = [r for r in positions if dates[r["game_id"]] >= cutoff]
    if not history or not later:
        raise ValueError("No strictly separated history/evaluation games")
    return history, later, cutoff


def history_candidates(positions, dates):
    """Only the legal capture/check attributes needed for frozen v4 adaptation."""
    rows = []
    for position in positions:
        board = chess.Board(position["fen"])
        actual = chess.Move.from_uci(position["actual_move_uci"])
        if actual not in board.legal_moves:
            raise ValueError("Illegal history move")
        for move in board.legal_moves:
            rows.append({"decision_id": f"{position['game_id']}:{position['ply']}",
                         "game_id": position["game_id"], "played_at": dates[position["game_id"]],
                         "player_username": position["player_username"], "split": "history",
                         "candidate_move_uci": move.uci(), "chosen": move == actual,
                         "candidate_is_capture": board.is_capture(move),
                         "candidate_gives_check": board.gives_check(move)})
    return rows


def paired_player_interval(deltas):
    if not deltas:
        return None
    values = np.asarray(deltas, dtype=float)
    rng = np.random.default_rng(42)
    means = rng.choice(values, (2000, len(values)), replace=True).mean(axis=1)
    return {"players": len(values), "mean_delta": float(values.mean()),
            "paired_player_95_interval": np.quantile(means, [.025, .975]).tolist()}


def run(source, output):
    if output.exists():
        raise FileExistsError(output)
    declaration, state = load_sealed(source)
    artifact = Path(declaration["config"]["artifact_dir"])
    frozen = OpportunityProfile.from_dict(json.loads((artifact / "profile.json").read_text()))
    cohorts, seen_games = {}, set()
    for player, record in sorted(state["players"].items()):
        if not record["accepted"] or record["kind"] != "unseen":
            continue
        if player.casefold() in frozen.players:
            raise ValueError("Target appears in training profile")
        positions = pq.read_table(record["positions"]).to_pylist()
        dates = {g: datetime.fromisoformat(d) for g, d in record["selected_dates"].items()}
        games = {r["game_id"] for r in positions}
        if seen_games & games:
            raise ValueError("Shared games across targets")
        seen_games.update(games)
        history, later, cutoff = split_history(positions, dates)
        cohorts[player] = (history, later, dates, cutoff)
    if len(cohorts) < 2:
        raise ValueError("At least two unseen players required")
    output.mkdir(parents=True)
    protocol = {"scope": "development only: already inspected external cohort; no model promotion",
                "split": "earliest half of each player's selected whole games; ties to evaluation",
                "arms": ["safe_population", "fallback", "personal", "wrong_player"],
                "wrong_player": "first cyclic alphabetical donor with history strictly before recipient cutoff; only donor history partition",
                "minimum_cell_opportunities": 20, "bootstrap_replicates": 2000, "seed": 42,
                "primary": "equal-player, equal-supported-phase/attribute-cell policy total variation; lower is better",
                "limitations": "replay only; 11 overlapping attributes, not independent traits; no fresh confirmation or engine quality rerun",
                "source_acquisition_sha256": digest(source / "acquisition.json"),
                "input_sha256": {str(p): digest(p) for p in
                                 [artifact / n for n in ("profile.json", "feature_sets.json", "metrics.json",
                                                        "safe_population.cbm", "opportunity_profile.cbm")]},
                "players": {p: {"cutoff": c.isoformat(),
                                 "history_games": sorted({r["game_id"] for r in h}),
                                 "evaluation_games": sorted({r["game_id"] for r in e})}
                            for p, (h, e, _, c) in cohorts.items()}}
    _write_json(output / "protocol.json", protocol)
    manifest = {"status": "running", "protocol_sha256": digest(output / "protocol.json")}
    _write_json(output / "manifest.json", manifest)
    try:
        schemas = json.loads((artifact / "feature_sets.json").read_text())
        fitted = json.loads((artifact / "metrics.json").read_text())
        models = {name: CatBoostRanker().load_model(artifact / f"{name}.cbm") for name in schemas}
        result = {}
        names = list(cohorts)
        for index, (player, (history, later, dates, cutoff)) in enumerate(cohorts.items()):
            print(f"Style diagnostic {index + 1}/{len(names)}: {player}", flush=True)
            personal = frozen.with_player_history(history_candidates(history, dates), before=cutoff)
            wrong = None
            donor_name, donor_games = None, 0
            for offset in range(1, len(names)):
                donor = names[(index + offset) % len(names)]
                donor_history, _, donor_dates, _ = cohorts[donor]
                eligible = [r for r in donor_history if donor_dates[r["game_id"]] < cutoff]
                if not eligible:
                    continue
                donor_profile = frozen.with_player_history(history_candidates(eligible, donor_dates), before=cutoff)
                wrong = OpportunityProfile.from_dict(frozen.to_dict())
                wrong.players[player.casefold()] = deepcopy(donor_profile.players[donor.casefold()])
                donor_name, donor_games = donor, len({r["game_id"] for r in eligible})
                break
            rows = build_all_legal_candidate_rows(later, dates, {g: "development" for g in dates})
            arms = {"safe_population": None, "fallback": frozen, "personal": personal}
            if wrong is not None:
                arms["wrong_player"] = wrong
            record = {"history_decisions": len(history), "evaluation_decisions": len(later),
                      "donor": donor_name, "donor_games": donor_games, "arms": {}}
            for arm, profile in arms.items():
                name = "safe_population" if profile is None else "opportunity_profile"
                data = rows if profile is None else profile.transform(rows)
                scores = predict_relevance_scores(models[name], data, tuple(schemas[name]))
                probabilities = groupwise_softmax(data, scores, temperature=fitted[name]["temperature"])
                ranking, _ = legal_policy_metrics(data, probabilities)
                record["arms"][arm] = {"ranking": ranking, "style": style_metrics(data, probabilities)}
            result[player] = record
            _write_json(output / "players.json", result)
        comparisons = {}
        for baseline in ("safe_population", "fallback", "wrong_player"):
            pairs = [r["arms"] for r in result.values() if baseline in r["arms"]
                     and r["arms"][baseline]["style"]["macro_policy_tv"] is not None
                     and r["arms"]["personal"]["style"]["macro_policy_tv"] is not None]
            comparisons[f"personal_minus_{baseline}"] = {
                "style_tv_lower_is_better": paired_player_interval([
                    a["personal"]["style"]["macro_policy_tv"] - a[baseline]["style"]["macro_policy_tv"] for a in pairs]),
                "exact_accuracy_higher_is_better": paired_player_interval([
                    a["personal"]["ranking"]["exact_move_accuracy"] - a[baseline]["ranking"]["exact_move_accuracy"] for a in pairs])}
        summary = {"scope": protocol["scope"], "players": len(result),
                   "history_decisions": sum(r["history_decisions"] for r in result.values()),
                   "evaluation_decisions": sum(r["evaluation_decisions"] for r in result.values()),
                   "comparisons": comparisons}
        _write_json(output / "summary.json", summary)
        manifest["status"] = "complete"
        _write_json(output / "manifest.json", manifest)
        print(json.dumps(summary, indent=2), flush=True)
        return summary
    except Exception as exc:
        manifest.update(status="failed", error=str(exc))
        _write_json(output / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.output)
