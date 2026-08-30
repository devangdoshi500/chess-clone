# chess-clone

Data and analysis package for building a personalized chess-player model. It
downloads a player's public, rated, standard Lichess games, preserves the
original PGN, creates normalized game and position datasets, learns historical
exact-position move frequencies, runs cache-aware Stockfish analysis, and
derives one behavioral feature row per analyzed player decision.

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

Use `--perf-type blitz` to ask Lichess for rated standard blitz games only.

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

The model's key is the first four FEN fields: placement, side to move, castling
rights, and en-passant target. Move counters are intentionally ignored.

## Run a small Stockfish analysis

Install a Stockfish executable separately, then run a small sequential batch:

```bash
python -m chess_clone.cli analyze-positions USERNAME \
  --nodes 500 \
  --max-positions 10
```

Results are cached by canonical position, engine binary identity, and engine
settings under `data/cache/stockfish/`. Analysis rows are written as Parquet
under `data/processed/`. Each decision includes both the evaluation before the
move and a separately cached evaluation after the actual move. Scores are
normalized to the target player's perspective. Use `--stockfish-path` when
Stockfish is not on PATH.

## Build behavioral features

Join a PositionRecord dataset to its Stockfish analysis:

```bash
python -m chess_clone.cli build-features USERNAME \
  --positions data/processed/positions_USERNAME_BATCH.parquet \
  --analysis data/processed/analysis_USERNAME_BATCH.parquet
```

Both input options default to the latest matching player files. Feature rows
are saved under `data/processed/`. The build summary explicitly reports the
number of available, analyzed, emitted, and unanalyzed PositionRecords. Inspect
the aggregates with:

```bash
python -m chess_clone.cli feature-summary USERNAME
```

### Feature definitions

All board features are computed from the FEN before the player's move and from
that player's perspective. Material uses pawn=1, knight=3, bishop=3, rook=5,
queen=9. Move flags are also calculated before pushing the actual move, so
captures, checks, castling, promotion, and en passant retain their correct
meaning.

Game phase uses a small deterministic rule:

- `endgame`: no queens remain, or combined non-pawn/non-king material is at
  most 20 points;
- `opening`: otherwise, at least 28 pieces remain;
- `middlegame`: all other positions.

Lichess `%clk` values are post-move clocks. For every player move after their
first observed move, elapsed time is calculated as
`previous_post_move_clock + increment - current_post_move_clock`. The first
move, malformed time controls, absent clocks, and impossible negative results
remain null rather than being guessed. `low_time`/`time_pressure` is true below
either 30 seconds or 10% of initial time by default; both thresholds are CLI
options.

`centipawn_loss` is the non-negative difference between the best pre-move
evaluation and the post-actual-move evaluation, both in the moving player's
perspective. Mate evaluations use a documented ±100,000 centipawn sentinel;
post-move mate zero means that the target player just delivered checkmate and
is normalized to +100,000.
Winning chance is an approximation, not a reproduction of Lichess Insights:
`1 / (1 + exp(-0.00368208 * evaluation_cp))`.

The `BehaviorFeatureRecord` API groups fields conceptually into pre-decision
inference features, observed behavior, and post-move diagnostics. Actual-move
evaluation, centipawn loss, rank, top-N flags, and after-move winning chance are
labels/diagnostics and should not be supplied as inference-time inputs.
