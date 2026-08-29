"""Stateless-per-position Stockfish adapter."""

import hashlib
from pathlib import Path
import shutil
from types import TracebackType

import chess
import chess.engine

from chess_clone.analysis.schemas import EngineLine, EngineSettings


class EngineAnalysisError(RuntimeError):
    """An engine could not analyze a position."""


class StockfishNotFoundError(EngineAnalysisError):
    """The configured Stockfish executable does not exist."""


class StockfishAnalyzer:
    """An isolated Stockfish worker with no cross-position model state."""

    def __init__(self, executable: str | Path = "stockfish") -> None:
        self.executable = self._resolve_executable(executable)
        self._engine: chess.engine.SimpleEngine | None = None
        self._engine_identity: str | None = None

    @staticmethod
    def _resolve_executable(executable: str | Path) -> Path:
        requested = Path(executable)
        resolved = shutil.which(str(executable)) if requested.parent == Path(".") else None
        path = Path(resolved) if resolved else requested.expanduser()
        if not path.is_file():
            raise StockfishNotFoundError(
                f"Stockfish executable not found: {executable}. "
                "Install Stockfish or pass --stockfish-path."
            )
        return path.resolve()

    def __enter__(self) -> "StockfishAnalyzer":
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def start(self) -> None:
        if self._engine is not None:
            return
        try:
            self._engine = chess.engine.SimpleEngine.popen_uci(str(self.executable))
        except (OSError, chess.engine.EngineError) as exc:
            raise EngineAnalysisError(f"Could not start Stockfish: {exc}") from exc
        name = self._engine.id.get("name", "Stockfish")
        digest = hashlib.sha256(self.executable.read_bytes()).hexdigest()[:16]
        self._engine_identity = f"{name}|sha256:{digest}"

    def close(self) -> None:
        if self._engine is None:
            return
        engine, self._engine = self._engine, None
        try:
            engine.quit()
        except (chess.engine.EngineError, TimeoutError):
            engine.close()

    @property
    def engine_identity(self) -> str:
        self.start()
        if self._engine_identity is None:
            raise EngineAnalysisError("Stockfish identity is unavailable")
        return self._engine_identity

    def analyze(self, fen: str, settings: EngineSettings) -> list[EngineLine]:
        """Analyze one position; the result depends only on input and settings."""

        self.start()
        if self._engine is None:
            raise EngineAnalysisError("Stockfish is not running")

        board = chess.Board(fen)
        options: dict[str, str | int | bool] = {
            "Threads": settings.threads,
            "Hash": settings.hash_mb,
            **dict(settings.options),
        }
        try:
            result = self._engine.analyse(
                board,
                chess.engine.Limit(nodes=settings.nodes),
                multipv=settings.multipv,
                game=object(),
                info=chess.engine.INFO_ALL,
                options=options,
            )
        except (chess.engine.EngineError, chess.engine.EngineTerminatedError) as exc:
            raise EngineAnalysisError(f"Stockfish analysis failed: {exc}") from exc

        infos = result if isinstance(result, list) else [result]
        lines: list[EngineLine] = []
        for rank, info in enumerate(infos, start=1):
            score = info.get("score")
            pov_score = score.pov(board.turn) if score is not None else None
            pv = info.get("pv", [])
            lines.append(
                EngineLine(
                    rank=rank,
                    score_cp=pov_score.score() if pov_score is not None else None,
                    mate_in=pov_score.mate() if pov_score is not None else None,
                    best_move_uci=pv[0].uci() if pv else None,
                    pv_uci=" ".join(move.uci() for move in pv),
                    depth=_as_int(info.get("depth")),
                    seldepth=_as_int(info.get("seldepth")),
                    nodes_searched=_as_int(info.get("nodes")),
                    time_seconds=_as_float(info.get("time")),
                )
            )
        return lines


def _as_int(value: object) -> int | None:
    return None if value is None else int(value)


def _as_float(value: object) -> float | None:
    return None if value is None else float(value)
