"""Filesystem-backed cache for deterministic engine results."""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile

from chess_clone.analysis.schemas import EngineLine, EngineSettings
from chess_clone.modeling import canonical_position_key

_CACHE_SCHEMA_VERSION = 1


def build_analysis_cache_key(
    fen: str, settings: EngineSettings, engine_identity: str
) -> str:
    """Hash a canonical position together with engine identity and settings."""

    payload = {
        "schema_version": _CACHE_SCHEMA_VERSION,
        "position": canonical_position_key(fen),
        "engine_identity": engine_identity,
        "settings": settings.cache_payload(),
    }
    serialized = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class CachedAnalysis:
    position_key: str
    engine_identity: str
    settings: dict[str, object]
    lines: tuple[EngineLine, ...]


class FileAnalysisCache:
    """One immutable JSON file per cache key, written atomically."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> CachedAnalysis | None:
        path = self._path(key)
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("schema_version") != _CACHE_SCHEMA_VERSION:
                return None
            return CachedAnalysis(
                position_key=str(payload["position_key"]),
                engine_identity=str(payload["engine_identity"]),
                settings=dict(payload["settings"]),
                lines=tuple(EngineLine.from_dict(line) for line in payload["lines"]),
            )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def put(
        self,
        key: str,
        *,
        position_key: str,
        engine_identity: str,
        settings: EngineSettings,
        lines: list[EngineLine],
    ) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": _CACHE_SCHEMA_VERSION,
            "position_key": position_key,
            "engine_identity": engine_identity,
            "settings": settings.cache_payload(),
            "lines": [line.to_dict() for line in lines],
        }
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))

        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{key}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary.write(serialized)
                temporary.flush()
                os.fsync(temporary.fileno())
                temporary_path = Path(temporary.name)
            os.replace(temporary_path, path)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()
