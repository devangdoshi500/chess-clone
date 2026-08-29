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

