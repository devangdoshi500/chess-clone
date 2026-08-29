"""Transform PGN games into normalized game and position records."""

from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import io
import re

import chess
import chess.pgn

from chess_clone.models import GameRecord, PositionRecord

_SPEEDS = ("ultrabullet", "bullet", "blitz", "rapid", "classical", "correspondence")


@dataclass(frozen=True, slots=True)
class ParsedGames:
    games: list[GameRecord]
    positions: list[PositionRecord]
    skipped_games: int = 0


def _optional_text(value: str | None) -> str | None:
    return value if value and value not in {"?", "-"} else None


def _optional_int(value: str | None) -> int | None:
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


def _played_at(headers: chess.pgn.Headers) -> datetime | None:
    date_text = headers.get("UTCDate") or headers.get("Date")
    time_text = headers.get("UTCTime", "00:00:00")
    if not date_text or "?" in date_text or "?" in time_text:
        return None
    try:
        return datetime.strptime(f"{date_text} {time_text}", "%Y.%m.%d %H:%M:%S").replace(
            tzinfo=UTC
        )
    except ValueError:
        return None


def _speed(headers: chess.pgn.Headers) -> str | None:
    explicit = _optional_text(headers.get("Speed"))
    if explicit:
        return explicit.lower()
    event = headers.get("Event", "").casefold()
    for speed in _SPEEDS:
        if re.search(rf"\b{re.escape(speed)}\b", event):
            return speed
    time_control = headers.get("TimeControl", "")
    clock = re.fullmatch(r"(\d+)\+(\d+)", time_control)
    if clock:
        estimated_seconds = int(clock.group(1)) + 40 * int(clock.group(2))
        if estimated_seconds <= 29:
            return "ultrabullet"
        if estimated_seconds <= 179:
            return "bullet"
        if estimated_seconds <= 479:
            return "blitz"
        if estimated_seconds <= 1499:
            return "rapid"
        return "classical"
    return None


def _is_rated(headers: chess.pgn.Headers) -> bool:
    rated = headers.get("Rated")
    if rated is not None:
        return rated.casefold() in {"true", "yes", "1"}
    # Named arenas do not say "rated" in Event, but rating deltas are only
    # present when a game's result affects the players' ratings.
    if "WhiteRatingDiff" in headers or "BlackRatingDiff" in headers:
        return True
    return bool(re.search(r"\brated\b", headers.get("Event", ""), re.IGNORECASE))


def _game_id(game: chess.pgn.Game) -> str:
    explicit = _optional_text(game.headers.get("GameId"))
    if explicit:
        return explicit
    site = _optional_text(game.headers.get("Site"))
    if site and site.startswith(("https://lichess.org/", "http://lichess.org/")):
        candidate = site.rstrip("/").rsplit("/", 1)[-1].split("#", 1)[0]
        if candidate:
            return candidate
    digest_input = "\n".join(f"{key}:{value}" for key, value in game.headers.items())
    digest_input += "\n" + " ".join(move.uci() for move in game.mainline_moves())
    return hashlib.sha256(digest_input.encode("utf-8")).hexdigest()[:16]


def parse_pgn(pgn_bytes: bytes, username: str) -> ParsedGames:
    """Parse standard games and retain positions where ``username`` must move."""

    games: list[GameRecord] = []
    positions: list[PositionRecord] = []
    skipped_games = 0
    stream = io.StringIO(pgn_bytes.decode("utf-8-sig"))
    requested = username.casefold()

    while game := chess.pgn.read_game(stream):
        headers = game.headers
        white = headers.get("White", "")
        black = headers.get("Black", "")
        variant = headers.get("Variant", "Standard")
        if variant.casefold() != "standard":
            skipped_games += 1
            continue
        if requested not in {white.casefold(), black.casefold()}:
            skipped_games += 1
            continue

        game_id = _game_id(game)
        white_rating = _optional_int(headers.get("WhiteElo"))
        black_rating = _optional_int(headers.get("BlackElo"))
        speed = _speed(headers)
        time_control = _optional_text(headers.get("TimeControl"))
        eco = _optional_text(headers.get("ECO"))
        opening_name = _optional_text(headers.get("Opening"))
        opening_variation = _optional_text(headers.get("Variation"))
        moves = list(game.mainline_moves())

        games.append(
            GameRecord(
                game_id=game_id,
                provider="lichess",
                game_url=_optional_text(headers.get("Site")),
                played_at=_played_at(headers),
                white_username=white,
                black_username=black,
                white_rating=white_rating,
                black_rating=black_rating,
                result=headers.get("Result", "*"),
                rated=_is_rated(headers),
                variant=variant,
                speed=speed,
                time_control=time_control,
                eco=eco,
                opening_name=opening_name,
                opening_variation=opening_variation,
                termination=_optional_text(headers.get("Termination")),
                total_plies=len(moves),
            )
        )

        board = game.board()
        node: chess.pgn.GameNode = game
        for ply, move in enumerate(moves, start=1):
            next_node = node.next()
            if next_node is None:  # Defensive guard for malformed game trees.
                break
            mover_is_white = board.turn == chess.WHITE
            mover = white if mover_is_white else black
            color = "white" if mover_is_white else "black"
            if mover.casefold() == requested:
                player_rating = white_rating if mover_is_white else black_rating
                opponent_rating = black_rating if mover_is_white else white_rating
                positions.append(
                    PositionRecord(
                        game_id=game_id,
                        ply=ply,
                        move_number=board.fullmove_number,
                        player_username=mover,
                        player_color=color,
                        fen=board.fen(),
                        actual_move_uci=move.uci(),
                        actual_move_san=board.san(move),
                        player_rating=player_rating,
                        opponent_rating=opponent_rating,
                        white_rating=white_rating,
                        black_rating=black_rating,
                        speed=speed,
                        time_control=time_control,
                        eco=eco,
                        opening_name=opening_name,
                        opening_variation=opening_variation,
                        clock_seconds_after_move=next_node.clock(),
                    )
                )
            board.push(move)
            node = next_node

    return ParsedGames(games=games, positions=positions, skipped_games=skipped_games)
