import json
from pathlib import Path

import numpy as np
import pytest

from chess_clone.experiments.style_v2_objective_ablation import (
    coefficient_objective, effective_coefficients, factor_objective, gates,
    load_histories, log_policy, make_history, reconstruct,
)
from chess_clone.experiments.style_v2_dataset import prepare_candidate_shards
from chess_clone.modeling.style_residual import DIMENSION, feature_matrix, group_structure, objective
from test_player_embedding import candidate_rows


def history(player="alpha", move="e2e4"):
    rows = candidate_rows(player, move, count=5)
    matrix = feature_matrix(rows)
    starts, lengths, labels = group_structure(rows, require_labels=True)
    chosen = np.flatnonzero(labels) - starts
    base = np.asarray([r["base_logit"] for r in rows])
    return make_history(matrix, base, starts, lengths, chosen, minimum=2)


def test_exact_reconstruction_preserves_probabilities_and_mean():
    random = np.random.default_rng(7)
    beta = random.normal(size=(14, DIMENSION))
    shared, embeddings, projection, rank = reconstruct(beta)
    rebuilt = shared + embeddings @ projection
    assert rank == 13
    assert np.max(np.abs(rebuilt - beta)) < 1e-13
    assert np.allclose(rebuilt.mean(axis=0), shared, atol=1e-14)
    h = history()
    assert np.allclose(log_policy(h.base + h.matrix @ beta[0], h.starts, h.lengths),
                       log_policy(h.base + h.matrix @ rebuilt[0], h.starts, h.lengths), atol=1e-12)
    with pytest.raises(ValueError, match="Insufficient"):
        reconstruct(beta, 2)


@pytest.mark.parametrize("arm", [{}, {"identity_weight": .1}, {"behavior_weight": .1},
                                  {"population_kl_weight": .02}, {"opportunity_weight": 10.},
                                  {"identity_weight": .1, "behavior_weight": .1, "population_kl_weight": .02}])
def test_analytic_coefficient_gradient_matches_finite_difference(arm):
    histories = [history(), history("beta", "g1f3")]
    random = np.random.default_rng(17)
    beta = random.normal(0, .15, (2, DIMENSION))
    direction = random.normal(size=beta.shape)
    direction /= np.linalg.norm(direction)
    value, gradient, _ = coefficient_objective(beta, histories, .001, arm)
    epsilon = 1e-5
    plus = coefficient_objective(beta + epsilon * direction, histories, .001, arm)[0]
    minus = coefficient_objective(beta - epsilon * direction, histories, .001, arm)[0]
    assert np.isfinite(value)
    assert (plus - minus) / (2 * epsilon) == pytest.approx(float(np.sum(gradient * direction)), abs=2e-8)


def test_factor_gradient_and_effective_regularization_match_v1():
    histories = [history(), history("beta", "g1f3")]
    random = np.random.default_rng(13)
    beta = random.normal(0, .1, (2, DIMENSION))
    shared, embeddings, projection, _ = reconstruct(beta, 4)
    parameters = np.r_[shared, embeddings.ravel(), projection.ravel()]
    assert np.allclose(effective_coefficients(parameters, 2, 4), beta)
    value, gradient = factor_objective(parameters, histories, 4, .001, {})
    reference = []
    for b, h in zip(beta, histories, strict=True):
        labels = np.zeros(len(h.base))
        labels[h.chosen] = 1
        reference.append(objective(b, h.matrix, h.base, h.starts, h.lengths, labels, .001)[0])
    assert value == pytest.approx(np.mean(reference), abs=1e-13)
    direction = random.normal(size=parameters.shape)
    direction /= np.linalg.norm(direction)
    epsilon = 1e-5
    plus = factor_objective(parameters + epsilon * direction, histories, 4, .001, {})[0]
    minus = factor_objective(parameters - epsilon * direction, histories, 4, .001, {})[0]
    assert (plus - minus) / (2 * epsilon) == pytest.approx(float(gradient @ direction), abs=2e-8)


def test_opportunity_cells_exclude_forced_attributes_and_use_history_support():
    h = history()
    attributes = {c["attribute"] for c in h.moment_cells}
    assert "candidate_piece_moved" in attributes
    assert "candidate_destination_wing" in attributes
    assert "candidate_is_capture" not in attributes
    assert all(c["opportunities"] == 5 and c["phase"] == "opening" for c in h.moment_cells)
    observed = np.zeros(len(h.base))
    observed[h.chosen] = 1
    assert np.max(np.abs(h.moments.T @ observed - h.moment_target)) == 0
    from chess_clone.experiments.style_metrics import style_metrics
    probabilities = np.exp(log_policy(h.base, h.starts, h.lengths))
    reference = style_metrics(candidate_rows("alpha", "e2e4", count=5), probabilities.tolist(), min_opportunities=2)
    expected = np.mean([sum((r["policy"] - r["human"]) ** 2 for r in c["rates"]) / 2
                        for c in reference["cells"] if c["supported"]])
    delta = h.moments.T @ probabilities - h.moment_target
    assert float(delta @ delta) == pytest.approx(expected, abs=1e-14)


def test_history_loading_filters_reserved_identities_and_checks_hashes(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    sources = {}
    for player in ("alpha", "reserved"):
        path = tmp_path / f"{player}.parquet"
        pq.write_table(pa.Table.from_pylist(candidate_rows(player, "e2e4", count=3)), path)
        sources[player] = path
    manifest = prepare_candidate_shards(sources, tmp_path / "shards", decisions_per_shard=2)
    for shard in manifest["shards"]:
        if Path(shard["path"]).name.startswith("reserved-"):
            Path(shard["path"]).unlink()
    histories = load_histories(manifest, ["alpha"], minimum=2)
    assert len(histories) == 1 and len(histories[0].starts) == 3
    target = next(s for s in manifest["shards"] if Path(s["path"]).name.startswith("alpha-"))
    Path(target["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="Changed sealed"):
        load_histories(manifest, ["alpha"], minimum=2)


def test_declared_parity_is_ineligible_and_gain_threshold_is_meaningful():
    config = json.loads(Path("configs/style_v2_objective_ablation_v1.json").read_text())
    assert config["arms"][0]["name"] == "svd_parity" and not config["arms"][0]["eligible"]
    from chess_clone.experiments.style_v2_objective_ablation import METRICS
    def scores(tv):
        return {"style": {"macro_policy_tv": tv}, "ranking": {m: .4 for m in METRICS if m != "style_tv"}}
    records = {"a": {"personal": scores(.03), "v1": scores(.03), "shared": scores(.04),
                     "wrong": scores(.05), "population": scores(.06)}}
    assert not gates(records, config)["passed"]
    records["a"]["personal"] = scores(.029)
    assert gates(records, config)["passed"]
    assert not gates(records, config, converged=False)["passed"]


def test_end_to_end_sealed_fallback_resume_and_tampering(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from chess_clone.experiments.style_v2_dataset import digest
    from chess_clone.experiments.style_v2_objective_ablation import run
    from chess_clone.experiments.style_v2_prototype import save_json_sealed
    from chess_clone.modeling.style_residual import fit_residual

    data, baseline, output = [tmp_path / name for name in ("data", "baseline", "output")]
    (baseline / "development-replay").mkdir(parents=True)
    sources, allocations = {}, {}
    for player, move in (("alpha", "e2e4"), ("beta", "g1f3")):
        rows = candidate_rows(player, move, count=24)
        allocations[player] = {"game_ids": {}, "zero_decision_games": []}
        for split in ("history", "validation"):
            local = [dict(r, split=split, game_id=split + r["game_id"],
                          decision_id=split + r["decision_id"]) for r in rows]
            path = data / "candidate-cache" / split / f"{player}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.Table.from_pylist(local), path)
            path.with_suffix(".json").write_text(json.dumps({"sha256": digest(path)}))
            allocations[player]["game_ids"][split] = sorted({r["game_id"] for r in local})
            if split == "history":
                sources[player] = path
                save_json_sealed(baseline / "development-replay" / f"{player}-v1.json",
                                 {"inputs": {"history_sha256": digest(path), "penalty": .001},
                                  "model": fit_residual(local, penalty=.001)})
        allocations[player]["game_ids"]["evaluation"] = [player + "-evaluation"]
    allocations["reserved"] = {"game_ids": {"history": ["reserved-h"], "validation": ["reserved-v"],
                                              "evaluation": ["reserved-e"]}, "zero_decision_games": []}
    prepare_candidate_shards(sources, data / "shards/history", decisions_per_shard=12)
    selection_path, manifest_path = data / "selection.json", data / "shards/history/manifest.json"
    selection_path.write_text(json.dumps(allocations))
    (data / "bundle.json").write_text(json.dumps({
        "development_players": ["alpha", "beta"], "evaluation_players": ["reserved"],
        "selection_sha256": digest(selection_path),
        "splits": {"history": {"path": str(manifest_path), "sha256": digest(manifest_path)}}}))
    config = json.loads(Path("configs/style_v2_objective_ablation_v1.json").read_text())
    config["arms"] = [config["arms"][0], config["arms"][2]]
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    result = run(config_path, data, baseline, output)
    assert result["selection_status"] == "no_candidate"
    assert result["reserved_players_unadapted_and_unscored"] == ["reserved"]
    export = json.loads((output / "policy_bundle.json").read_text())
    assert export["policy_source"] == "v1_fallback_no_new_candidate"
    assert set(export["personal_models"]) == {"alpha", "beta"}
    assert result == run(config_path, data, baseline, output)
    path = output / "trained/imitation_reconstructed.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="Changed sealed"):
        run(config_path, data, baseline, output)
