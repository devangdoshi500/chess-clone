"""Small regularized conditional-logit adapter over a frozen chess policy."""

import hashlib
import math

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csr_matrix

from chess_clone.modeling.legal_policy import BEHAVIOR_BOOLEAN_FIELDS

BOOLEAN_FIELDS = BEHAVIOR_BOOLEAN_FIELDS + (
    "candidate_is_development", "candidate_is_hanging_after", "candidate_gives_mate",
)
NUMERIC_FIELDS = {"candidate_material_gain": 9., "candidate_attackers_after": 4.,
                  "candidate_defenders_after": 4., "candidate_center_control_after": 8.,
                  "candidate_opponent_mobility_after": 40.}
PIECES = ("pawn", "knight", "bishop", "rook", "queen", "king")
WINGS = ("queenside", "center", "kingside")
PHASES = ("opening", "middlegame", "endgame")
BASE_FIELDS = BOOLEAN_FIELDS + tuple(NUMERIC_FIELDS) + tuple(f"piece:{p}" for p in PIECES) + tuple(f"wing:{w}" for w in WINGS)
OPENING_BUCKETS = 128
DIMENSION = len(BASE_FIELDS) * (1 + len(PHASES)) + OPENING_BUCKETS
VERSION = "conditional-style-residual-v1"


def feature_matrix(rows):
    """Fixed sparse schema; labels and player identity never enter features."""
    data, indexes, indptr = [], [], [0]
    width = len(BASE_FIELDS)
    for row in rows:
        phase = row["game_phase"]
        if phase not in PHASES:
            raise ValueError(f"Unknown game phase: {phase}")
        values = [float(bool(row[f])) for f in BOOLEAN_FIELDS]
        values += [float(np.clip(float(row[f]) / scale, -1, 1)) for f, scale in NUMERIC_FIELDS.items()]
        if row["candidate_piece_moved"] not in PIECES or row["candidate_destination_wing"] not in WINGS:
            raise ValueError("Unknown piece/wing category")
        values += [float(row["candidate_piece_moved"] == p) for p in PIECES]
        values += [float(row["candidate_destination_wing"] == w) for w in WINGS]
        for offset in (0, width * (1 + PHASES.index(phase))):
            for index, value in enumerate(values):
                if not math.isfinite(value):
                    raise ValueError("Non-finite feature")
                if value:
                    indexes.append(offset + index)
                    data.append(value)
        if int(row["move_number"]) <= 10:
            key = f"{row['position_key']}|{row['candidate_move_uci']}"
            bucket = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") % OPENING_BUCKETS
            indexes.append(width * (1 + len(PHASES)) + bucket)
            data.append(1.)
        indptr.append(len(data))
    return csr_matrix((data, indexes, indptr), shape=(len(rows), DIMENSION), dtype=np.float64)


def group_structure(rows, *, require_labels):
    starts, seen, previous = [], set(), None
    for index, row in enumerate(rows):
        key = row["decision_id"]
        if key != previous:
            if key in seen:
                raise ValueError("Decision rows must be contiguous")
            seen.add(key)
            starts.append(index)
            previous = key
    if not starts:
        raise ValueError("At least one decision required")
    starts = np.asarray(starts, dtype=int)
    lengths = np.diff(np.r_[starts, len(rows)])
    if require_labels:
        labels = np.asarray([float(bool(r["chosen"])) for r in rows])
        if not np.all(np.add.reduceat(labels, starts) == 1):
            raise ValueError("Each decision requires exactly one human move")
    else:
        labels = None
    return starts, lengths, labels


def softmax(logits, starts, lengths):
    centered = logits - np.repeat(np.maximum.reduceat(logits, starts), lengths)
    weights = np.exp(centered)
    return weights / np.repeat(np.add.reduceat(weights, starts), lengths)


def objective(beta, matrix, base, starts, lengths, labels, penalty):
    logits = base + matrix @ beta
    maxima = np.maximum.reduceat(logits, starts)
    centered = logits - np.repeat(maxima, lengths)
    normalizers = np.log(np.add.reduceat(np.exp(centered), starts))
    loss = (normalizers.sum() - np.dot(centered, labels)) / len(starts)
    probabilities = softmax(logits, starts, lengths)
    gradient = np.asarray(matrix.T @ (probabilities - labels)).ravel() / len(starts)
    return float(loss + penalty * np.dot(beta, beta) / 2), gradient + penalty * beta


def fit_residual(rows, *, penalty):
    if not math.isfinite(penalty) or penalty <= 0:
        raise ValueError("Penalty must be positive and finite")
    if any(r["split"] != "history" for r in rows):
        raise ValueError("Residuals may only fit history")
    matrix = feature_matrix(rows)
    starts, lengths, labels = group_structure(rows, require_labels=True)
    base = np.asarray([r["base_logit"] for r in rows], dtype=float)
    if not np.isfinite(base).all():
        raise ValueError("Non-finite base logits")
    result = minimize(objective, np.zeros(DIMENSION), args=(matrix, base, starts, lengths, labels, penalty),
                      method="L-BFGS-B", jac=True, options={"maxiter": 150, "ftol": 1e-9})
    if not result.success:
        raise RuntimeError(f"Residual optimization failed: {result.message}")
    return {"version": VERSION, "penalty": penalty, "coefficients": result.x.tolist(),
            "iterations": int(result.nit), "objective": float(result.fun), "decisions": len(starts)}


def predict_residual(rows, model=None):
    starts, lengths, _ = group_structure(rows, require_labels=False)
    base = np.asarray([r["base_logit"] for r in rows], dtype=float)
    if not np.isfinite(base).all():
        raise ValueError("Non-finite base logits")
    if model is not None:
        beta = np.asarray(model["coefficients"], dtype=float)
        if model["version"] != VERSION or beta.shape != (DIMENSION,) or not np.isfinite(beta).all():
            raise ValueError("Invalid residual model schema")
        base += feature_matrix(rows) @ beta
    return softmax(base, starts, lengths).tolist()


def shared_residual(models):
    """Equal-player mean of history-fitted adapters: non-personalized control."""
    if not models or any(m["version"] != VERSION for m in models):
        raise ValueError("Compatible models required")
    return {"version": VERSION, "coefficients": np.mean([m["coefficients"] for m in models], axis=0).tolist(),
            "description": "equal-player mean of development history adapters"}
