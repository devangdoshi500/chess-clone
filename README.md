# chess-clone

Initial data-ingestion package for building a personalized chess-player model.
It downloads a player's public, rated, standard Lichess games, preserves the
original PGN, and creates normalized game and position Parquet datasets.

## Requirements

- Python 3.12

## Install

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

## Ingest games

```bash
python -m chess_clone.cli ingest USERNAME --max-games 100
```

`--since` and `--until` accept a Unix timestamp in milliseconds, an ISO-8601
date, or an ISO-8601 datetime. For example:

```bash
python -m chess_clone.cli ingest USERNAME \
  --max-games 100 \
  --since 2025-01-01 \
  --until 2025-12-31T23:59:59Z
```

Raw API bytes are written to `data/raw/`. Normalized `games_*.parquet` and
`positions_*.parquet` files are written to `data/processed/`.

## Test

```bash
pytest
```

The live integration test is opt-in:

```bash
RUN_LICHESS_INTEGRATION=1 pytest -m integration
```

## Inspect a historical move model

Build and inspect an exact-position move-frequency model from the latest
processed batch for a player:

```bash
python -m chess_clone.cli inspect-model USERNAME --examples 5
```

Use `--positions path/to/positions.parquet` to select a specific batch.

## Run a small Stockfish analysis

Install a Stockfish executable separately, then run a small sequential batch:

```bash
python -m chess_clone.cli analyze-positions USERNAME \
  --nodes 500 \
  --max-positions 10
```

Results are cached by canonical position, engine binary identity, and engine
settings under `data/cache/stockfish/`. Analysis rows are written as Parquet
under `data/processed/`. Use `--stockfish-path` when Stockfish is not on PATH.
