import json
import math

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

from chess_clone.cli import app
from chess_clone.experiments.move_quality import QualityScore, compare_scores
from chess_clone.experiments import winning_chance as wc


def row(key="a", *, actual=0, predicted=50, reference=100, method="population", exact=False, game=None):
    def score(cp):
        return QualityScore(cp, "finite" if cp is not None else "winning_mate")
    return wc.convert_row({"method": method, "decision_id": key, "game_id": game or key,
                           "actual_move_uci": "e2e4", "predicted_move_uci": "e2e4" if exact else "d2d4",
                           "exact_correct": exact, "player_username": "A", "game_phase": "opening",
                           "rating_band": "1400", "player_color": "white", "time_control": "180+0",
                           **compare_scores(score(actual), score(predicted), score(reference))})


def test_formula_units_symmetry_extremes_and_decisive_positions():
    assert wc.win_percent(0) == 50
    assert wc.win_percent(100) == pytest.approx(59.10259036219175)
    assert wc.win_percent(-100) + wc.win_percent(100) == pytest.approx(100)
    assert wc.win_percent(1e308) == 100
    assert wc.win_percent(-1e308) == 0
    assert wc.win_percent(None) is None
    for value in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError):
            wc.win_percent(value)
    assert wc.win_percent(1100) - wc.win_percent(1000) < wc.win_percent(100) - wc.win_percent(0)


def test_selection_is_validation_only_deduplicates_human_and_ignores_predictions():
    rows = [row(str(i), actual=0, reference=10, predicted=i * 100) for i in range(4)]
    thresholds = wc.select_thresholds(rows, split="validation")
    assert thresholds["human_similarity_pp"] == 1
    duplicates = [dict(r, method="other", predicted_loss_pp=999, human_gap_pp=999) for r in rows]
    repeated = wc.select_thresholds(rows + duplicates, split="validation")
    assert repeated["finite_human_reference_pairs"] == 4
    assert repeated["human_similarity_pp"] == 1
    with pytest.raises(ValueError, match="only"):
        wc.select_thresholds(rows, split="test")
    with pytest.raises(ValueError, match="Inconsistent"):
        wc.select_thresholds(rows + [row("0", actual=30)], split="validation")
    with pytest.raises(ValueError, match="No finite"):
        wc.select_thresholds([row(actual=None)], split="validation")
    with pytest.raises(ValueError, match="unresolved"):
        wc.select_thresholds([row(actual=-1000, reference=1000)], split="validation")


def test_mate_exclusions_different_moves_and_soundness_are_separate():
    thresholds = {"human_similarity_pp": 2, "reference_soundness_pp": 2}
    rows = [row("a", actual=-300, predicted=-300, reference=100, exact=True),
            row("b", actual=-300, predicted=-300, reference=100),
            row("c", actual=None, predicted=100, reference=None),
            row("d", actual=100, predicted=200, reference=100)]
    s = wc.summary(rows, thresholds)
    assert s["human_gap_pp"]["count"] == 3
    assert s["mate_pairs"] == 1
    assert s["all"]["similar_rate"] == pytest.approx(2 / 3)
    assert s["different_move"]["similar_rate"] == .5
    assert s["all"]["similar_and_sound_rate"] == 0
    assert rows[-1]["predicted_loss_pp"] == 0
    assert s["predicted_exceeds_reference_count"] == 1
    assert wc.summary([], thresholds)["all"]["similar_rate"] is None


def test_paired_comparison_common_masks_and_whole_game_bootstrap():
    thresholds = {"human_similarity_pp": 2, "reference_soundness_pp": 2}
    left = [row("a", predicted=100, game="g"), row("b", predicted=100, game="g"),
            row("c", predicted=None), row("d", predicted=0, exact=True)]
    right = [row("a", predicted=0, game="g"), row("b", predicted=0, game="g"),
             row("c", predicted=0), row("d", predicted=100)]
    result = wc.paired_comparison(left, right, thresholds, resamples=100)
    common = result["both_different_common_finite/human_gap_pp"]
    assert common["decisions"] == 2
    assert common["games"] == 1
    assert common["threshold_rate_delta"]["estimate"] == 1
    assert common["threshold_rate_delta"]["paired_game_95_interval"] == [1, 1]
    assert result["all_common_finite/human_gap_pp"]["decisions"] == 3
    with pytest.raises(ValueError, match="identical"):
        wc.paired_comparison(left, right[:-1], thresholds)


def test_runner_freezes_before_test_loading_and_refuses_overlap(tmp_path, monkeypatch):
    validation = [row("val", actual=0, reference=10)]
    test = [row("test", actual=0, reference=10)]
    manifest = {"engine_identity": "fake", "engine_settings": {"nodes": 20000}}
    monkeypatch.setattr(wc, "replay_validation", lambda *args: None)
    def quality(*args, **kwargs):
        path = args[2]
        path.mkdir()
        pq.write_table(pa.Table.from_pylist(validation), path / "decisions.parquet")
    monkeypatch.setattr(wc, "run_move_quality", quality)
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    pq.write_table(pa.Table.from_pylist(test), test_dir / "decisions.parquet")
    output = tmp_path / "out"
    def read(path, *, expected_split):
        if expected_split == "validation":
            return validation, manifest
        assert (output / "thresholds.json").exists()
        return test, manifest
    monkeypatch.setattr(wc, "read_quality", read)
    result = wc.run_winning_chance(tmp_path, tmp_path, test_dir, output)
    saved = json.loads((output / "manifest.json").read_text())
    assert saved["status"] == "complete"
    assert saved["thresholds_sha256"] == wc._hash(output / "thresholds.json")
    assert result["inspected_test"]["methods"]["population"]["decisions"] == 1
    with pytest.raises(FileExistsError):
        wc.run_winning_chance(tmp_path, tmp_path, test_dir, output)
    output = tmp_path / "overlap"
    test[0]["game_id"] = "val"
    with pytest.raises(ValueError, match="overlap"):
        wc.run_winning_chance(tmp_path, tmp_path, test_dir, output)
    assert json.loads((output / "manifest.json").read_text())["status"] == "failed"


def test_quality_reader_checks_provenance_split_and_duplicates(tmp_path):
    source = tmp_path / "source.json"
    source.write_text("{}")
    manifest = {"status": "complete", "split": "validation", "input_sha256": {str(source): wc._hash(source)}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    pq.write_table(pa.Table.from_pylist([row()]), tmp_path / "decisions.parquet")
    assert len(wc.read_quality(tmp_path, expected_split="validation")[0]) == 1
    with pytest.raises(ValueError, match="split"):
        wc.read_quality(tmp_path, expected_split="test")
    pq.write_table(pa.Table.from_pylist([row(), row()]), tmp_path / "decisions.parquet")
    with pytest.raises(ValueError, match="Duplicate"):
        wc.read_quality(tmp_path, expected_split="validation")
    source.write_text("changed")
    with pytest.raises(ValueError, match="hash changed"):
        wc.read_quality(tmp_path, expected_split="validation")


def test_cli_failure_is_actionable(tmp_path):
    result = CliRunner().invoke(app, ["benchmark-winning-chance", "--cohort", str(tmp_path / "missing"),
                                    "--artifact-dir", str(tmp_path), "--test-quality-dir", str(tmp_path),
                                    "--output-dir", str(tmp_path / "out")])
    assert result.exit_code == 1
    assert "Winning-chance benchmark failed" in result.output


def test_player_clustered_quality_intervals_keep_game_and_player_counts_distinct():
    left = [row("a", predicted=100), row("b", predicted=100)]
    right = [row("a", predicted=0), row("b", predicted=0)]
    thresholds = {"human_similarity_pp": 2, "reference_soundness_pp": 2}
    result = wc.paired_comparison(left, right, thresholds, resamples=100, unit="player_username")
    common = result["both_different_common_finite/human_gap_pp"]
    assert common["games"] == 2
    assert common["resampling_units"] == 1
    assert common["threshold_rate_delta"]["paired_player_95_interval"] == [1, 1]
