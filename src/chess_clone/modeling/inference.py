"""Read-only, label-free inference for the sealed deep-style policy."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Literal

import chess
from catboost import CatBoostRanker

from chess_clone.modeling.boosted import predict_relevance_scores
from chess_clone.modeling.legal_policy import build_all_legal_inference_rows
from chess_clone.modeling.style_residual import DIMENSION, VERSION, predict_residual

PolicyName = Literal["auto", "personal", "shared", "population"]


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _validate_residual_model(model: dict, *, label: str) -> None:
    coefficients = model.get("coefficients")
    if model.get("version") != VERSION or not isinstance(coefficients, list):
        raise ValueError(f"Invalid residual model schema: {label}")
    if len(coefficients) != DIMENSION:
        raise ValueError(f"Invalid residual coefficient count: {label}")


def _expected_hash(frozen: dict[str, str], filename: str) -> str:
    matches = [value for name, value in frozen.items() if Path(name).name == filename]
    if len(matches) != 1:
        raise ValueError(f"Frozen artifact manifest is missing a unique {filename}")
    return matches[0]


@dataclass(frozen=True, slots=True)
class PredictionContext:
    """Game-start context not recoverable from a FEN."""

    player_rating: int | None
    opponent_rating: int | None
    speed: str | None
    time_control: str | None
    player_username: str | None = None


@dataclass(frozen=True, slots=True)
class MovePrediction:
    rank: int
    uci: str
    san: str
    probability: float


@dataclass(frozen=True, slots=True)
class PredictionResult:
    fen: str
    terminal: bool
    terminal_reason: str | None
    policy: str
    player_username: str | None
    personal_model_available: bool
    legal_move_count: int
    probability_sum: float
    latency_ms: float
    moves: tuple[MovePrediction, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "moves": [asdict(move) for move in self.moves],
        }


class DeepStylePredictor:
    """Load the frozen backbone and residuals once for repeated prediction."""

    def __init__(self, source: Path | str) -> None:
        started = perf_counter()
        self.source = Path(source)
        declaration_path = self.source / "declaration.json"
        acquisition_path = self.source / "acquisition.json"
        selection_path = self.source / "residual/selection.json"
        seal_path = self.source / "residual/selection_seal.json"
        declaration = _read_json(declaration_path)
        selection = _read_json(selection_path)
        seal = _read_json(seal_path)
        if _digest(selection_path) != seal.get("sha256"):
            raise ValueError("Deep-style selection seal mismatch")
        if selection.get("status") != "selected":
            raise ValueError("Deep-style run has no selected residual")
        if selection.get("declaration_sha256") != _digest(declaration_path):
            raise ValueError("Deep-style declaration changed after selection")
        if selection.get("acquisition_sha256") != _digest(acquisition_path):
            raise ValueError("Deep-style acquisition changed after selection")

        artifact = Path(declaration["config"]["artifact_dir"])
        if not artifact.is_absolute():
            working_path = Path.cwd() / artifact
            project_path = self.source.resolve().parents[2] / artifact
            artifact = working_path if working_path.exists() else project_path
        artifact_selection = _read_json(artifact / "selection.json")
        frozen = artifact_selection["frozen_sha256"]
        model_path = artifact / "safe_population.cbm"
        for filename in ("safe_population.cbm", "feature_sets.json", "metrics.json"):
            if _digest(artifact / filename) != _expected_hash(frozen, filename):
                raise ValueError(f"Frozen population artifact hash mismatch: {filename}")

        features = _read_json(artifact / "feature_sets.json")
        metrics = _read_json(artifact / "metrics.json")
        self.feature_fields = tuple(features["safe_population"])
        self.temperature = float(metrics["safe_population"]["temperature"])
        if self.temperature <= 0:
            raise ValueError("Invalid frozen population temperature")
        self.base_model = CatBoostRanker().load_model(model_path)

        penalty = str(selection["selected_penalty"])
        self.shared_model = _read_json(self.source / "residual/shared.json")[penalty]
        _validate_residual_model(self.shared_model, label="shared")
        self.personal_models: dict[str, dict] = {}
        self.display_names: dict[str, str] = {}
        for path in sorted((self.source / "residual/models").glob("*.json")):
            payload = _read_json(path)
            model = payload.get("models", {}).get(penalty)
            if model is None:
                continue
            _validate_residual_model(model, label=path.stem)
            key = path.stem.casefold()
            self.personal_models[key] = model
            self.display_names[key] = path.stem
        if not self.personal_models:
            raise ValueError("No selected personal residual models found")
        self.load_latency_ms = (perf_counter() - started) * 1000

    @property
    def known_players(self) -> tuple[str, ...]:
        return tuple(self.display_names[key] for key in sorted(self.display_names))

    def predict(
        self,
        fen: str,
        context: PredictionContext,
        *,
        policy: PolicyName = "auto",
        top_k: int | None = 5,
    ) -> PredictionResult:
        started = perf_counter()
        if policy not in {"auto", "personal", "shared", "population"}:
            raise ValueError(f"Unknown policy: {policy}")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be positive or None")

        try:
            board = chess.Board(fen)
        except ValueError as exc:
            raise ValueError(f"Invalid FEN: {exc}") from exc
        if not board.is_valid():
            raise ValueError(f"Invalid chess position: {fen}")

        player_key = (
            context.player_username.casefold()
            if context.player_username
            else None
        )
        personal_available = bool(player_key and player_key in self.personal_models)
        resolved_policy = policy
        if policy == "auto":
            resolved_policy = "personal" if personal_available else "population"
        if resolved_policy == "personal" and not personal_available:
            raise ValueError(
                f"No personal residual for player: {context.player_username!r}"
            )

        if board.is_game_over(claim_draw=False):
            return PredictionResult(
                fen=board.fen(),
                terminal=True,
                terminal_reason=board.outcome(claim_draw=False).termination.name.lower(),
                policy=resolved_policy,
                player_username=context.player_username,
                personal_model_available=personal_available,
                legal_move_count=0,
                probability_sum=0.0,
                latency_ms=(perf_counter() - started) * 1000,
                moves=(),
            )

        rows = build_all_legal_inference_rows(
            board.fen(),
            player_username=context.player_username,
            player_rating=context.player_rating,
            opponent_rating=context.opponent_rating,
            speed=context.speed,
            time_control=context.time_control,
        )
        scores = predict_relevance_scores(
            self.base_model, rows, self.feature_fields
        )
        for row, score in zip(rows, scores, strict=True):
            row["position_key"] = " ".join(board.fen().split()[:4])
            row["base_logit"] = score / self.temperature

        residual = None
        if resolved_policy == "personal":
            residual = self.personal_models[player_key]  # type: ignore[index]
        elif resolved_policy == "shared":
            residual = self.shared_model
        probabilities = predict_residual(rows, residual)
        ordered = sorted(
            zip(rows, probabilities, strict=True),
            key=lambda item: (-item[1], str(item[0]["candidate_move_uci"])),
        )
        if top_k is not None:
            ordered = ordered[:top_k]
        moves = tuple(
            MovePrediction(
                rank=index,
                uci=str(row["candidate_move_uci"]),
                san=board.san(chess.Move.from_uci(str(row["candidate_move_uci"]))),
                probability=float(probability),
            )
            for index, (row, probability) in enumerate(ordered, start=1)
        )
        return PredictionResult(
            fen=board.fen(),
            terminal=False,
            terminal_reason=None,
            policy=resolved_policy,
            player_username=context.player_username,
            personal_model_available=personal_available,
            legal_move_count=len(rows),
            probability_sum=float(sum(probabilities)),
            latency_ms=(perf_counter() - started) * 1000,
            moves=moves,
        )
