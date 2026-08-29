"""Command-line interface for chess-clone ingestion."""

from pathlib import Path
from typing import Annotated

import typer

from chess_clone.analysis import (
    EngineAnalysisError,
    EngineSettings,
    FileAnalysisCache,
    StockfishAnalyzer,
    analyze_position_dataset,
    default_analysis_output_path,
)
from chess_clone.ingestion import ingest_games
from chess_clone.modeling import HistoricalMoveModel
from chess_clone.providers import LichessProvider, ProviderError

app = typer.Typer(no_args_is_help=True, help="Personalized chess data ingestion.")


@app.callback()
def main() -> None:
    """Personalized chess data ingestion."""


@app.command()
def ingest(
    username: Annotated[str, typer.Argument(help="Public Lichess username")],
    max_games: Annotated[
        int, typer.Option("--max-games", min=1, help="Maximum games to download")
    ] = 100,
    since: Annotated[
        str | None,
        typer.Option(help="Epoch milliseconds or ISO-8601 start date/time"),
    ] = None,
    until: Annotated[
        str | None,
        typer.Option(help="Epoch milliseconds or ISO-8601 end date/time"),
    ] = None,
    raw_dir: Annotated[
        Path, typer.Option(hidden=True, help="Raw PGN destination")
    ] = Path("data/raw"),
    processed_dir: Annotated[
        Path, typer.Option(hidden=True, help="Processed Parquet destination")
    ] = Path("data/processed"),
) -> None:
    """Download and normalize rated standard games from Lichess."""

    try:
        summary = ingest_games(
            LichessProvider(),
            username,
            max_games=max_games,
            since=since,
            until=until,
            raw_dir=raw_dir,
            processed_dir=processed_dir,
        )
    except (ProviderError, ValueError) as exc:
        typer.echo(f"Ingestion failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo("Ingestion complete")
    typer.echo(f"  User: {summary.username}")
    typer.echo(f"  Games: {summary.games}")
    typer.echo(f"  Player positions: {summary.positions}")
    typer.echo(f"  Skipped games: {summary.skipped_games}")
    typer.echo(f"  Raw PGN: {summary.raw_path}")
    typer.echo(f"  Games Parquet: {summary.games_path}")
    typer.echo(f"  Positions Parquet: {summary.positions_path}")


def _latest_positions_file(username: str, processed_dir: Path) -> Path:
    safe_username = username.strip().lower()
    matches = sorted(processed_dir.glob(f"positions_{safe_username}_*.parquet"))
    if not matches:
        raise FileNotFoundError(
            f"No processed position dataset found for '{username}' in {processed_dir}"
        )
    return matches[-1]


@app.command("inspect-model")
def inspect_model(
    username: Annotated[str, typer.Argument(help="Player username in PositionRecords")],
    positions: Annotated[
        Path | None,
        typer.Option(
            help="PositionRecords Parquet file; defaults to the latest player batch"
        ),
    ] = None,
    processed_dir: Annotated[
        Path,
        typer.Option(hidden=True, help="Directory searched for processed positions"),
    ] = Path("data/processed"),
    examples: Annotated[
        int, typer.Option(min=0, help="Number of repeated-position examples to show")
    ] = 5,
) -> None:
    """Inspect a player's exact-position historical move model."""

    try:
        source_path = positions or _latest_positions_file(username, processed_dir)
        model = HistoricalMoveModel.from_parquet(source_path, username)
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(f"Model inspection failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    summary = model.summary
    typer.echo("Historical move model")
    typer.echo(f"  Player: {username}")
    typer.echo(f"  Dataset: {source_path}")
    typer.echo(f"  PositionRecords: {summary.total_observations}")
    typer.echo(f"  Unique positions: {summary.unique_positions}")
    typer.echo(f"  Repeated positions: {summary.repeated_positions}")
    typer.echo(
        "  Average observations per position: "
        f"{summary.average_observations_per_position:.3f}"
    )
    typer.echo(
        "  Max observations for one position: "
        f"{summary.max_observations_for_one_position}"
    )
    typer.echo(
        "  PositionRecords in repeated positions: "
        f"{summary.repeated_position_records_percentage:.2f}%"
    )

    repeated = model.get_repeated_positions(limit=examples)
    if repeated:
        typer.echo("Repeated position examples:")
        for item in repeated:
            typer.echo(
                f"  {item['position_key']} ({item['total_observations']} observations)"
            )
            for move in item["moves"]:
                typer.echo(
                    f"    {move['move_uci']}: {move['count']} "
                    f"({float(move['probability']):.3f})"
                )


@app.command("analyze-positions")
def analyze_positions(
    username: Annotated[str, typer.Argument(help="Player username in PositionRecords")],
    positions: Annotated[
        Path | None,
        typer.Option(
            help="PositionRecords Parquet file; defaults to the latest player batch"
        ),
    ] = None,
    stockfish_path: Annotated[
        str, typer.Option(help="Stockfish executable name or path")
    ] = "stockfish",
    nodes: Annotated[
        int, typer.Option(min=1, help="Node budget for each unique position")
    ] = 500,
    max_positions: Annotated[
        int, typer.Option(min=1, help="Maximum PositionRecords to analyze")
    ] = 10,
    multipv: Annotated[
        int, typer.Option(min=1, help="Principal variations per position")
    ] = 1,
    threads: Annotated[
        int, typer.Option(min=1, help="Stockfish threads per worker")
    ] = 1,
    hash_mb: Annotated[
        int, typer.Option(min=1, help="Stockfish hash size in MiB")
    ] = 16,
    cache_dir: Annotated[
        Path, typer.Option(help="Filesystem analysis-cache directory")
    ] = Path("data/cache/stockfish"),
    output: Annotated[
        Path | None, typer.Option(help="Analysis Parquet output path")
    ] = None,
    processed_dir: Annotated[
        Path,
        typer.Option(hidden=True, help="Directory searched for processed positions"),
    ] = Path("data/processed"),
) -> None:
    """Run a small, sequential, cache-aware Stockfish analysis batch."""

    try:
        source_path = positions or _latest_positions_file(username, processed_dir)
        output_path = output or default_analysis_output_path(username, processed_dir)
        settings = EngineSettings(
            nodes=nodes,
            multipv=multipv,
            threads=threads,
            hash_mb=hash_mb,
        )
        summary = analyze_position_dataset(
            source_path,
            username,
            analyzer=StockfishAnalyzer(stockfish_path),
            settings=settings,
            cache=FileAnalysisCache(cache_dir),
            output_path=output_path,
            max_positions=max_positions,
        )
    except (EngineAnalysisError, FileNotFoundError, ValueError) as exc:
        typer.echo(f"Engine analysis failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo("Engine analysis complete")
    typer.echo(f"  Player: {summary.username}")
    typer.echo(f"  Engine: {summary.engine_identity}")
    typer.echo(f"  PositionRecords: {summary.position_records}")
    typer.echo(f"  Unique positions: {summary.unique_positions}")
    typer.echo(f"  Engine calls: {summary.engine_calls}")
    typer.echo(f"  Cache hits: {summary.cache_hits}")
    typer.echo(f"  Output rows: {summary.output_rows}")
    typer.echo(f"  Analysis Parquet: {summary.output_path}")


if __name__ == "__main__":
    app()
