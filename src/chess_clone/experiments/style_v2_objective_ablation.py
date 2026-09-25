"""Preregistered, development-only factorized objective ablation.

The float64 sparse objective operates on complete histories. It is deliberately
separate from the original minibatch trainer and never opens reserved outcomes.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np
import pyarrow.parquet as pq
from scipy.optimize import minimize
from scipy.special import expit
from scipy.sparse import csr_matrix, hstack, vstack

from chess_clone.experiments.deep_style import macro, style_delta
from chess_clone.experiments.population_metrics import legal_policy_metrics
from chess_clone.experiments.style_metrics import style_metrics
from chess_clone.experiments.style_v2_dataset import digest
from chess_clone.experiments.style_v2_prototype import (
    load_json_sealed, save_json_sealed, split_audit, verify_file,
)
from chess_clone.modeling.legal_policy import BEHAVIOR_BOOLEAN_FIELDS
from chess_clone.modeling.style_residual import (
    BASE_FIELDS, DIMENSION, PHASES, PIECES, VERSION, WINGS,
    feature_matrix, group_structure, shared_residual,
)


@dataclass
class History:
    matrix: csr_matrix
    base: np.ndarray
    starts: np.ndarray
    lengths: np.ndarray
    chosen: np.ndarray
    moments: csr_matrix
    moment_target: np.ndarray
    moment_cells: list[dict]


def log_policy(logits, starts, lengths):
    centered = logits - np.repeat(np.maximum.reduceat(logits, starts), lengths)
    return centered - np.repeat(np.log(np.add.reduceat(np.exp(centered), starts)), lengths)


def opportunity_matrix(matrix, starts, lengths, *, minimum=20):
    """Linear history rates with equal supported phase/attribute weighting.

    M.T @ p - M.T @ y is scaled so its squared norm is the mean
    half-squared categorical distribution distance over supported cells.
    """
    width = len(BASE_FIELDS)
    families = [(name, [BASE_FIELDS.index(name)], True) for name in BEHAVIOR_BOOLEAN_FIELDS]
    families += [("candidate_piece_moved", [BASE_FIELDS.index(f"piece:{p}") for p in PIECES], False),
                 ("candidate_destination_wing", [BASE_FIELDS.index(f"wing:{w}") for w in WINGS], False)]
    pieces = [BASE_FIELDS.index(f"piece:{p}") for p in PIECES]
    columns, cells = [], []
    for phase_index, phase in enumerate(PHASES):
        phase_present = np.asarray(matrix[starts, :][:, [width * (phase_index + 1) + i for i in pieces]].sum(axis=1)).ravel() > .5
        for name, indexes, boolean in families:
            attributes = matrix[:, indexes].toarray()
            available = np.maximum.reduceat(attributes, starts, axis=0)
            if boolean:
                varied = (available[:, 0] > .5) & (np.minimum.reduceat(attributes[:, 0], starts) < .5)
            else:
                varied = available.sum(axis=1) > 1.5
            eligible = phase_present & varied
            count = int(eligible.sum())
            if count < minimum:
                continue
            scale = np.repeat(eligible / count, lengths)
            if not boolean:
                scale = scale / np.sqrt(2.)
            columns.append(matrix[:, indexes].multiply(scale[:, None]).tocsr())
            cells.append({"phase": phase, "attribute": name, "opportunities": count})
    if not cells:
        return csr_matrix((matrix.shape[0], 0)), []
    return hstack(columns, format="csr") / np.sqrt(len(cells)), cells


def make_history(matrix, base, starts, lengths, chosen, *, minimum=20):
    if matrix.shape[1] != DIMENSION or not np.isfinite(matrix.data).all() or not np.isfinite(base).all():
        raise ValueError("Invalid history features/logits")
    if (not len(starts) or starts[0] != 0 or np.any(lengths <= 0)
            or not np.array_equal(starts, np.r_[0, np.cumsum(lengths)[:-1]])
            or lengths.sum() != len(base) or np.any(chosen < 0) or np.any(chosen >= lengths)):
        raise ValueError("Invalid history decision boundaries")
    moments, cells = opportunity_matrix(matrix, starts, lengths, minimum=minimum)
    absolute_chosen = starts + chosen
    target = np.asarray(moments[absolute_chosen].sum(axis=0)).ravel()
    return History(matrix, base, starts, lengths, absolute_chosen, moments, target, cells)


def load_histories(manifest, players, *, minimum):
    histories = []
    for player in players:
        identity = manifest["player_to_index"][player]
        matrices, bases, starts, lengths, chosen = [], [], [], [], []
        offset = 0
        # Frozen builder uses one player per shard and names each shard by player.
        shards = [s for s in manifest["shards"] if Path(s["path"]).name.startswith(player + "-")]
        if not shards:
            raise ValueError(f"No history shards for {player}")
        for shard in shards:
            path = Path(shard["path"])
            verify_file(path, shard["sha256"])
            with np.load(path) as values:
                if not np.all(values["players"] == identity):
                    raise ValueError("History shard mixes identities")
                matrix = csr_matrix((values["data"].astype(np.float64), values["indices"], values["indptr"]),
                                    shape=tuple(values["shape"]))
                matrices.append(matrix)
                bases.append(values["base_logits"].astype(np.float64))
                starts.append(values["starts"].astype(int) + offset)
                lengths.append(values["lengths"].astype(int))
                chosen.append(values["chosen"].astype(int))
                offset += matrix.shape[0]
        history = make_history(vstack(matrices, format="csr"), np.concatenate(bases),
                               np.concatenate(starts), np.concatenate(lengths), np.concatenate(chosen), minimum=minimum)
        histories.append(history)
        print(f"History loaded: {player}, {len(history.starts)} decisions, {len(history.moment_cells)} supported cells", flush=True)
    return histories


def reconstruct(coefficients, dimension=32):
    coefficients = np.asarray(coefficients, dtype=np.float64)
    shared = coefficients.mean(axis=0)
    u, singular, vt = np.linalg.svd(coefficients - shared, full_matrices=False)
    rank = int(np.linalg.matrix_rank(coefficients - shared))
    if rank > dimension:
        raise ValueError("Insufficient embedding dimension for exact reconstruction")
    embeddings = np.zeros((len(coefficients), dimension))
    projection = np.zeros((dimension, coefficients.shape[1]))
    kept = min(dimension, len(singular))
    embeddings[:, :kept] = u[:, :kept] * np.sqrt(singular[:kept])
    projection[:kept] = np.sqrt(singular[:kept, None]) * vt[:kept]
    return shared, embeddings, projection, rank


def unpack(parameters, players, dimension):
    boundary = DIMENSION + players * dimension
    return (parameters[:DIMENSION], parameters[DIMENSION:boundary].reshape(players, dimension),
            parameters[boundary:].reshape(dimension, DIMENSION))


def effective_coefficients(parameters, players, dimension):
    shared, embeddings, projection = unpack(parameters, players, dimension)
    return shared + embeddings @ projection


def coefficient_objective(coefficients, histories, penalty, arm):
    """Loss and analytic derivative with respect to each effective beta."""
    gradient = np.zeros_like(coefficients)
    totals = dict(imitation=0., identity=0., behavior=0., population_kl=0., opportunity=0.)
    count = len(histories)
    for player, history in enumerate(histories):
        x, starts, lengths, chosen = history.matrix, history.starts, history.lengths, history.chosen
        n = len(starts)
        lp = log_policy(history.base + x @ coefficients[player], starts, lengths)
        p = np.exp(lp)
        error = p.copy()
        error[chosen] -= 1
        row_gradient = error / n
        totals["imitation"] -= float(lp[chosen].mean()) / count
        if arm.get("identity_weight", 0):
            donor = (player + 1) % count
            wrong_lp = log_policy(history.base + x @ coefficients[donor], starts, lengths)
            margin = .1 - lp[chosen] + wrong_lp[chosen]
            totals["identity"] += float(np.logaddexp(0., margin).mean()) / count
            weights = np.repeat(expit(margin), lengths) * arm["identity_weight"] / n
            row_gradient += weights * error
            wrong_error = np.exp(wrong_lp)
            wrong_error[chosen] -= 1
            gradient[donor] -= np.asarray(x.T @ (weights * wrong_error)).ravel() / count
        if arm.get("behavior_weight", 0):
            features = x[:, :len(BASE_FIELDS)]
            delta = np.asarray(features.T @ error).ravel() / n
            totals["behavior"] += float(np.mean(delta ** 2)) / count
            v = np.asarray(features @ delta).ravel() * (2 * arm["behavior_weight"] / (n * len(BASE_FIELDS)))
            row_gradient += p * (v - np.repeat(np.add.reduceat(p * v, starts), lengths))
        if arm.get("population_kl_weight", 0):
            base_lp = log_policy(history.base, starts, lengths)
            log_ratio = lp - base_lp
            kl = np.add.reduceat(p * log_ratio, starts)
            totals["population_kl"] += float(kl.mean()) / count
            row_gradient += arm["population_kl_weight"] * p * (log_ratio - np.repeat(kl, lengths)) / n
        if arm.get("opportunity_weight", 0) and history.moments.shape[1]:
            delta = np.asarray(history.moments.T @ p).ravel() - history.moment_target
            totals["opportunity"] += float(delta @ delta) / count
            v = np.asarray(history.moments @ delta).ravel() * (2 * arm["opportunity_weight"])
            row_gradient += p * (v - np.repeat(np.add.reduceat(p * v, starts), lengths))
        gradient[player] += np.asarray(x.T @ row_gradient).ravel() / count
    regularization = penalty * np.square(coefficients).sum() / (2 * count)
    gradient += penalty * coefficients / count
    loss = totals["imitation"] + regularization
    for component in ("identity", "behavior", "population_kl", "opportunity"):
        loss += arm.get(component + "_weight", 0) * totals[component]
    return float(loss), gradient, {**totals, "regularization": float(regularization)}


def factor_objective(parameters, histories, dimension, penalty, arm):
    shared, embeddings, projection = unpack(parameters, len(histories), dimension)
    loss, gradient, _ = coefficient_objective(shared + embeddings @ projection, histories, penalty, arm)
    return loss, np.r_[gradient.sum(axis=0), (gradient @ projection.T).ravel(), (embeddings.T @ gradient).ravel()]


def model(beta):
    return {"version": VERSION, "coefficients": np.asarray(beta).tolist()}


def train_arm(histories, players, v1, arm, config):
    dimension = config["embedding_dimension"]
    if arm["initialization"] == "v1_svd":
        shared, embeddings, projection, rank = reconstruct(v1, dimension)
    else:
        random = np.random.default_rng(config["seed"])
        shared = np.zeros(DIMENSION)
        embeddings = random.normal(0, .02, (len(players), dimension))
        projection = random.normal(0, .02, (dimension, DIMENSION))
        rank = None
    initial = np.r_[shared, embeddings.ravel(), projection.ravel()]
    started = time.perf_counter()
    if arm["train"]:
        result = minimize(factor_objective, initial, args=(histories, dimension, config["penalty"], arm),
                          method="L-BFGS-B", jac=True,
                          options={"maxiter": config["max_iterations"], "ftol": config["ftol"], "gtol": config["gtol"]})
        parameters = result.x
        optimization = {"converged": bool(result.success), "message": str(result.message),
                        "iterations": int(result.nit), "evaluations": int(result.nfev),
                        "objective": float(result.fun), "gradient_max_abs": float(np.max(np.abs(result.jac)))}
    else:
        parameters = initial
        optimization = {"converged": True, "message": "untrained exact reconstruction", "iterations": 0}
    coefficients = effective_coefficients(parameters, len(players), dimension)
    if not np.isfinite(coefficients).all():
        raise ValueError("Non-finite trained coefficients")
    loss, _, components = coefficient_objective(coefficients, histories, config["penalty"], arm)
    return {"arm": arm, "optimization": optimization, "seconds": time.perf_counter() - started,
            "effective_objective": loss, "objective_components": components, "reconstruction_rank": rank,
            "personal_models": {p: model(beta) for p, beta in zip(players, coefficients, strict=True)},
            "shared_model": model(coefficients.mean(axis=0))}


METRICS = ("style_tv", "negative_log_likelihood", "exact_move_accuracy", "top_3_accuracy", "top_5_accuracy",
           "mean_reciprocal_rank", "multiclass_brier_score", "top_1_expected_calibration_error")


def gates(records, config, *, converged=True):
    panel = {arm: {metric: macro(records, arm, metric) for metric in METRICS}
             for arm in ("personal", "shared", "wrong", "v1", "population")}
    personal = panel["personal"]
    checks = {"optimizer_converged": converged,
              "meaningful_v1_gain": personal["style_tv"] <= panel["v1"]["style_tv"] - config["minimum_style_tv_gain"],
              "beats_shared": personal["style_tv"] < panel["shared"]["style_tv"],
              "beats_wrong": personal["style_tv"] < panel["wrong"]["style_tv"],
              "nll_no_worse_than_shared": personal["negative_log_likelihood"] <= panel["shared"]["negative_log_likelihood"],
              "exact_within_two_points_of_shared": personal["exact_move_accuracy"] >= panel["shared"]["exact_move_accuracy"] - .02}
    return {"passed": all(checks.values()), "checks": checks, "macro": panel,
            "style_differences": {f"personal_minus_{arm}": style_delta(records, "personal", arm)
                                  for arm in ("v1", "shared", "wrong", "population")}}


def score_players(data, output, players, trained, v1, quarantine, config):
    results = {arm: {} for arm in trained}
    parity_max = 0.
    for index, player in enumerate(players):
        checkpoint = output / "replay" / f"{player}.json"
        if checkpoint.exists():
            record = load_json_sealed(checkpoint)
        else:
            rows = pq.read_table(data / "candidate-cache/validation" / f"{player}.parquet").to_pylist()
            rows = [r for r in rows if r["game_id"] not in set(quarantine.get(player, []))]
            if any(r["split"] != "validation" or r["player_username"].casefold() != player for r in rows):
                raise ValueError("Wrong validation identity/split")
            x = feature_matrix(rows)
            starts, lengths, _ = group_structure(rows, require_labels=True)
            base = np.asarray([r["base_logit"] for r in rows])
            def probabilities(residual):
                logits = base if residual is None else base + x @ np.asarray(residual["coefficients"])
                return np.exp(log_policy(logits, starts, lengths))
            def evaluate(probability):
                ranking, _ = legal_policy_metrics(rows, probability.tolist())
                return {"ranking": ranking, "style": style_metrics(rows, probability.tolist())}
            v1_probabilities = probabilities(v1[player])
            reference = {"v1": evaluate(v1_probabilities), "population": evaluate(probabilities(None))}
            record = {"player": player, "scored_games": len({r["game_id"] for r in rows}),
                      "decisions": len(starts), "quarantined_games": quarantine.get(player, []), "arms": {}}
            reconstructed = probabilities(trained["svd_parity"]["personal_models"][player])
            record["parity_max_probability_error"] = float(np.max(np.abs(reconstructed - v1_probabilities)))
            if record["parity_max_probability_error"] > config["parity_tolerance"]:
                raise ValueError("SVD probability parity failed")
            for arm, artifact in trained.items():
                donor = players[(index + 1) % len(players)]
                policies = {"personal": artifact["personal_models"][player], "shared": artifact["shared_model"],
                            "wrong": artifact["personal_models"][donor]}
                scores = {name: evaluate(probabilities(residual)) for name, residual in policies.items()}
                record["arms"][arm] = {**reference, **scores, "donor": donor}
            save_json_sealed(checkpoint, record)
        parity_max = max(parity_max, record["parity_max_probability_error"])
        for arm in trained:
            results[arm][player] = record["arms"][arm]
        print(f"Development replay: {index + 1}/{len(players)} players", flush=True)
    return results, parity_max


def run(config_path, data, baseline, output):
    config = json.loads(config_path.read_text())
    bundle_path = data / "bundle.json"
    bundle = json.loads(bundle_path.read_text())
    verify_file(data / "selection.json", bundle["selection_sha256"])
    selection = json.loads((data / "selection.json").read_text())
    players = sorted(bundle["development_players"])
    reserved = bundle["evaluation_players"]
    if set(players) & set(reserved):
        raise ValueError("Development and reserved identities overlap")
    audit = split_audit(selection, players, reserved)
    inputs = {str(p): digest(p) for p in (config_path, Path(__file__), bundle_path, data / "selection.json")}
    manifest_path = Path(bundle["splits"]["history"]["path"])
    verify_file(manifest_path, bundle["splits"]["history"]["sha256"])
    manifest = json.loads(manifest_path.read_text())
    inputs[str(manifest_path)] = digest(manifest_path)
    v1 = {}
    for player in players:
        for split in ("history", "validation"):
            path = data / "candidate-cache" / split / f"{player}.parquet"
            metadata = json.loads(path.with_suffix(".json").read_text())
            verify_file(path, metadata["sha256"])
            inputs[str(path)] = metadata["sha256"]
        v1_path = baseline / "development-replay" / f"{player}-v1.json"
        saved = load_json_sealed(v1_path)
        if (saved["inputs"]["history_sha256"] != inputs[str(data / "candidate-cache/history" / f"{player}.parquet")]
                or saved["inputs"]["penalty"] != config["penalty"]):
            raise ValueError("V1 baseline history or penalty mismatch")
        inputs[str(v1_path)] = digest(v1_path)
        v1[player] = saved["model"]
    declaration = {"schema_version": 1, "config": config, "inputs": inputs,
                   "development_players": players, "reserved_players_unopened": reserved, "split_audit": audit}
    output.mkdir(parents=True, exist_ok=True)
    (output / "trained").mkdir(exist_ok=True)
    (output / "replay").mkdir(exist_ok=True)
    declaration_path = output / "declaration.json"
    if declaration_path.exists():
        if load_json_sealed(declaration_path) != declaration:
            raise ValueError("Frozen ablation declaration/inputs changed")
    else:
        save_json_sealed(declaration_path, declaration)
    if (output / "selection.json").exists():
        selected = load_json_sealed(output / "selection.json")
        for path, expected in selected["hashes"].items():
            verify_file(Path(path), expected)
        load_json_sealed(output / "policy_bundle.json")
        return selected
    trained = {}
    histories = None
    v1_array = np.asarray([v1[p]["coefficients"] for p in players])
    for arm in config["arms"]:
        path = output / "trained" / f"{arm['name']}.json"
        if path.exists():
            trained[arm["name"]] = load_json_sealed(path)
            continue
        if histories is None:
            histories = load_histories(manifest, players, minimum=config["minimum_style_opportunities"])
        print(f"Training fixed arm: {arm['name']}", flush=True)
        artifact = train_arm(histories, players, v1_array, arm, config)
        save_json_sealed(path, artifact)
        trained[arm["name"]] = artifact
        print(f"Finished {arm['name']}: {artifact['optimization']}, {artifact['seconds']:.1f}s", flush=True)
    del histories
    records, parity = score_players(data, output, players, trained, v1,
                                    audit["development_validation_quarantine"], config)
    panels = {name: gates(records[name], config, converged=artifact["optimization"]["converged"])
              for name, artifact in trained.items()}
    eligible = [a["name"] for a in config["arms"] if a["eligible"] and panels[a["name"]]["passed"]]
    selected_arm = min(eligible, key=lambda name: panels[name]["macro"]["personal"]["style_tv"]) if eligible else None
    hashes = {str(declaration_path): digest(declaration_path),
              **{str(path): digest(path) for path in sorted((output / "trained").glob("*.json"))},
              **{str(path): digest(path) for path in sorted((output / "replay").glob("*.json"))}}
    result = {"schema_version": 1, "selection_status": "selected" if selected_arm else "no_candidate",
              "selected_arm": selected_arm, "development_players": players, "results": panels,
              "parity_max_probability_error": parity, "hashes": hashes, "evidence_status": config["evidence_status"],
              "reserved_players_unadapted_and_unscored": reserved,
              "selection_is_model_promotion": False}
    policies = trained[selected_arm] if selected_arm else {"personal_models": v1, "shared_model": shared_residual(list(v1.values()))}
    export = {key: result[key] for key in ("schema_version", "selection_status", "selected_arm", "development_players", "hashes", "evidence_status")}
    export.update(personal_models=policies["personal_models"], shared_model=policies["shared_model"], v1_models=v1,
                  policy_source=selected_arm or "v1_fallback_no_new_candidate")
    save_json_sealed(output / "policy_bundle.json", export)
    save_json_sealed(output / "selection.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/style_v2_objective_ablation_v1.json"))
    parser.add_argument("--data", type=Path, default=Path("artifacts/benchmarks/style-v2-prototype-data"))
    parser.add_argument("--baseline", type=Path, default=Path("artifacts/benchmarks/style-v2-smallest-candidate-v1"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/benchmarks/style-v2-objective-ablation-v1"))
    args = parser.parse_args()
    result = run(args.config, args.data, args.baseline, args.output)
    print(json.dumps({k: v for k, v in result.items() if k not in ("hashes", "results")}, indent=2))
