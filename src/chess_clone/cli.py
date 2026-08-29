"""Command-line interface for chess-clone ingestion."""

from pathlib import Path
from typing import Annotated

import typer

from chess_clone.ingestion import ingest_games
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


if __name__ == "__main__":
    app()
