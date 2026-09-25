"""Value objects shared by engine analysis components."""

from dataclasses import asdict, dataclass

EngineOption = tuple[str, str | int | bool]


@dataclass(frozen=True, slots=True)
class EngineSettings:
    """All evaluation-affecting settings included in an analysis cache key."""

    nodes: int = 500
    multipv: int = 1
    threads: int = 1
    hash_mb: int = 16
    options: tuple[EngineOption, ...] = ()

    def __post_init__(self) -> None:
        if self.nodes < 1:
            raise ValueError("nodes must be at least 1")
        if self.multipv < 1:
            raise ValueError("multipv must be at least 1")
        if self.threads < 1:
            raise ValueError("threads must be at least 1")
        if self.hash_mb < 1:
            raise ValueError("hash_mb must be at least 1")
        names = [name for name, _ in self.options]
        if len(names) != len(set(names)):
            raise ValueError("engine option names must be unique")

    def cache_payload(self) -> dict[str, object]:
        return {
            "nodes": self.nodes,
            "multipv": self.multipv,
            "threads": self.threads,
            "hash_mb": self.hash_mb,
            "options": dict(sorted(self.options)),
        }


@dataclass(frozen=True, slots=True)
class EngineLine:
    """One principal variation returned by the engine."""

    rank: int
    score_cp: int | None
    mate_in: int | None
    best_move_uci: str | None
    pv_uci: str
    depth: int | None
    seldepth: int | None
    nodes_searched: int | None
    time_seconds: float | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "EngineLine":
        return cls(
            rank=int(value["rank"]),
            score_cp=_optional_int(value.get("score_cp")),
            mate_in=_optional_int(value.get("mate_in")),
            best_move_uci=_optional_str(value.get("best_move_uci")),
            pv_uci=str(value["pv_uci"]),
            depth=_optional_int(value.get("depth")),
            seldepth=_optional_int(value.get("seldepth")),
            nodes_searched=_optional_int(value.get("nodes_searched")),
            time_seconds=_optional_float(value.get("time_seconds")),
        )


def _optional_int(value: object) -> int | None:
    return None if value is None else int(value)


def _optional_float(value: object) -> float | None:
    return None if value is None else float(value)


def _optional_str(value: object) -> str | None:
    return None if value is None else str(value)
