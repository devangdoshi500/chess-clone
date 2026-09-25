"""Command-line interface for chess-clone data and analysis workflows."""

import json
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
from chess_clone.features import (
    TimePressureThresholds,
    build_behavior_features,
    default_feature_output_path,
    summarize_behavior_features,
)
from chess_clone.modeling import (
    DeepStylePredictor,
    HistoricalMoveModel,
    PredictionContext,
    default_model_artifact_dir,
    evaluate_saved_artifact,
    train_personalized_ranker,
)
from chess_clone.modeling.boosted_training import (
    default_boosted_artifact_dir,
    train_boosted_rankers,
)
from chess_clone.providers import LichessProvider, ProviderError

app = typer.Typer(no_args_is_help=True, help="Personalized chess data workflows.")


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
    perf_type: Annotated[
        str | None,
        typer.Option(
            "--perf-type",
            help="One standard Lichess speed to request (for example, blitz)",
        ),
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
            perf_type=perf_type,
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
    matches = list(processed_dir.glob(f"positions_{safe_username}_*.parquet"))
    if not matches:
        raise FileNotFoundError(
            f"No processed position dataset found for '{username}' in {processed_dir}"
        )
    return max(matches, key=lambda path: path.stat().st_mtime_ns)


def _latest_analysis_file(username: str, processed_dir: Path) -> Path:
    safe_username = username.strip().lower()
    matches = list(processed_dir.glob(f"analysis_{safe_username}_*.parquet"))
    if not matches:
        raise FileNotFoundError(
            f"No engine analysis dataset found for '{username}' in {processed_dir}"
        )
    return max(matches, key=lambda path: path.stat().st_mtime_ns)


def _latest_feature_file(username: str, processed_dir: Path) -> Path:
    safe_username = username.strip().lower()
    matches = list(processed_dir.glob(f"features_{safe_username}_*.parquet"))
    if not matches:
        raise FileNotFoundError(
            f"No behavioral feature dataset found for '{username}' in {processed_dir}"
        )
    return max(matches, key=lambda path: path.stat().st_mtime_ns)


def _latest_games_file(username: str, processed_dir: Path) -> Path:
    safe_username = username.strip().lower()
    matches = list(processed_dir.glob(f"games_{safe_username}_*.parquet"))
    if not matches:
        raise FileNotFoundError(
            f"No normalized game dataset found for '{username}' in {processed_dir}"
        )
    return max(matches, key=lambda path: path.stat().st_mtime_ns)


def _latest_rf_artifact(username: str, artifact_root: Path) -> Path:
    safe_username = username.strip().lower()
    matches = [
        path
        for path in artifact_root.glob(f"{safe_username}_*rf_baseline*")
        if (path / "split_metadata.json").is_file()
    ]
    if not matches:
        raise FileNotFoundError(
            f"No Random Forest artifact found for '{username}' in {artifact_root}"
        )
    return max(matches, key=lambda path: path.stat().st_mtime_ns)


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


@app.command("build-features")
def build_features(
    username: Annotated[str, typer.Argument(help="Player username in PositionRecords")],
    positions: Annotated[
        Path | None,
        typer.Option(help="PositionRecords Parquet; defaults to latest player batch"),
    ] = None,
    analysis: Annotated[
        Path | None,
        typer.Option(help="Engine analysis Parquet; defaults to latest player batch"),
    ] = None,
    output: Annotated[
        Path | None, typer.Option(help="Behavioral feature Parquet output path")
    ] = None,
    low_time_fraction: Annotated[
        float,
        typer.Option(
            min=0.0,
            max=1.0,
            help="Flag low time below this fraction of initial time",
        ),
    ] = 0.10,
    low_time_seconds: Annotated[
        float,
        typer.Option(min=0.0, help="Flag low time below this many seconds"),
    ] = 30.0,
    processed_dir: Annotated[
        Path, typer.Option(hidden=True, help="Processed Parquet directory")
    ] = Path("data/processed"),
) -> None:
    """Join positions and Stockfish output into behavioral decision features."""

    try:
        position_path = positions or _latest_positions_file(username, processed_dir)
        analysis_path = analysis or _latest_analysis_file(username, processed_dir)
        output_path = output or default_feature_output_path(username, processed_dir)
        summary = build_behavior_features(
            position_path,
            analysis_path,
            username,
            output_path=output_path,
            thresholds=TimePressureThresholds(
                fraction=low_time_fraction, seconds=low_time_seconds
            ),
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        typer.echo(f"Feature build failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo("Behavioral feature build complete")
    typer.echo(f"  Player: {summary.username}")
    typer.echo(f"  Available PositionRecords: {summary.available_position_records}")
    typer.echo(f"  Analyzed PositionRecords: {summary.analyzed_position_records}")
    typer.echo(f"  Feature rows: {summary.feature_rows}")
    typer.echo(f"  Unanalyzed PositionRecords: {summary.unanalyzed_position_records}")
    typer.echo(f"  Features Parquet: {summary.output_path}")


@app.command("feature-summary")
def feature_summary(
    username: Annotated[str, typer.Argument(help="Player username")],
    features: Annotated[
        Path | None,
        typer.Option(help="Behavioral feature Parquet; defaults to latest player batch"),
    ] = None,
    processed_dir: Annotated[
        Path, typer.Option(hidden=True, help="Processed Parquet directory")
    ] = Path("data/processed"),
) -> None:
    """Print Insight-style aggregate metrics for a behavioral dataset."""

    try:
        feature_path = features or _latest_feature_file(username, processed_dir)
        summary = summarize_behavior_features(feature_path)
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(f"Feature summary failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"Behavioral feature summary for {username}")
    typer.echo(f"  Dataset: {feature_path}")
    typer.echo(json.dumps(summary.to_dict(), indent=2, sort_keys=True))


@app.command("train-model")
def train_model(
    username: Annotated[str, typer.Argument(help="Player username")],
    features: Annotated[
        Path | None,
        typer.Option(help="BehaviorFeature Parquet; defaults to latest player batch"),
    ] = None,
    analysis: Annotated[
        Path | None,
        typer.Option(help="Stockfish analysis Parquet; defaults to latest player batch"),
    ] = None,
    games: Annotated[
        Path | None,
        typer.Option(help="Normalized games Parquet; defaults to latest player batch"),
    ] = None,
    artifact_dir: Annotated[
        Path | None,
        typer.Option(help="Model artifact directory; defaults under artifacts/models"),
    ] = None,
    processed_dir: Annotated[
        Path, typer.Option(hidden=True, help="Processed Parquet directory")
    ] = Path("data/processed"),
    candidate_k: Annotated[int, typer.Option(min=1, max=20, help="Fixed candidate width")] = 5,
    artifact_root: Annotated[
        Path, typer.Option(hidden=True, help="Default local model artifact root")
    ] = Path("artifacts/models"),
) -> None:
    """Train and evaluate the first personalized candidate-ranking baseline."""

    try:
        feature_path = features or _latest_feature_file(username, processed_dir)
        analysis_path = analysis or _latest_analysis_file(username, processed_dir)
        games_path = games or _latest_games_file(username, processed_dir)
        destination = artifact_dir or default_model_artifact_dir(
            username, artifact_root
        )
        summary = train_personalized_ranker(
            username,
            features_path=feature_path,
            analysis_path=analysis_path,
            games_path=games_path,
            artifact_dir=destination,
            candidate_k=candidate_k,
        )
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as exc:
        typer.echo(f"Model training failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo("Personalized candidate-ranking training complete")
    typer.echo(f"  Player: {summary.username}")
    typer.echo(f"  Total decisions: {summary.total_decisions}")
    typer.echo(f"  Inside top {candidate_k}: {summary.inside_top_5_decisions}")
    typer.echo(f"  Outside top {candidate_k}: {summary.outside_top_5_decisions}")
    typer.echo(f"  Runtime seconds: {summary.runtime_seconds:.3f}")
    typer.echo(f"  Artifact directory: {summary.artifact_dir}")
    typer.echo("Evaluation metrics:")
    typer.echo(json.dumps(summary.metrics, indent=2, sort_keys=True))


@app.command("evaluate-model")
def evaluate_model(
    artifact_path: Annotated[
        Path, typer.Argument(help="Saved model artifact directory")
    ],
) -> None:
    """Inspect the held-out evaluation stored with a trained ranker."""

    try:
        metrics = evaluate_saved_artifact(artifact_path)
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(f"Model evaluation failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Saved model evaluation: {artifact_path}")
    typer.echo(json.dumps(metrics, indent=2, sort_keys=True))


@app.command("predict-move")
def predict_move(
    fen: Annotated[str, typer.Argument(help="Full FEN for the position to score")],
    player: Annotated[
        str | None,
        typer.Option(help="Player whose sealed residual should be used when available"),
    ] = None,
    player_rating: Annotated[
        int | None,
        typer.Option(help="Mover rating; omit to use the model's missing-value path"),
    ] = None,
    opponent_rating: Annotated[
        int | None,
        typer.Option(help="Opponent rating; omit to use the model's missing-value path"),
    ] = None,
    speed: Annotated[
        str | None,
        typer.Option(help="Game speed, normally blitz for the validated cohort"),
    ] = None,
    time_control: Annotated[
        str | None,
        typer.Option(help="Lichess time control such as 180+0"),
    ] = None,
    policy: Annotated[
        str,
        typer.Option(help="auto, personal, shared, or population"),
    ] = "auto",
    top_k: Annotated[int, typer.Option(min=1, help="Number of ranked moves to print")] = 5,
    source: Annotated[
        Path,
        typer.Option(help="Completed sealed deep-style run directory"),
    ] = Path("artifacts/benchmarks/deep-style-v1"),
) -> None:
    """Rank every legal move with the frozen, label-free deep-style policy."""

    try:
        predictor = DeepStylePredictor(source)
        result = predictor.predict(
            fen,
            PredictionContext(
                player_username=player,
                player_rating=player_rating,
                opponent_rating=opponent_rating,
                speed=speed,
                time_control=time_control,
            ),
            policy=policy,  # type: ignore[arg-type]
            top_k=top_k,
        )
    except (FileNotFoundError, json.JSONDecodeError, RuntimeError, ValueError) as exc:
        typer.echo(f"Prediction failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    output = result.to_dict()
    output["load_latency_ms"] = predictor.load_latency_ms
    output["validated_scope"] = "rated standard blitz, mover rating 1300-1700"
    typer.echo(json.dumps(output, indent=2, sort_keys=True))


@app.command("train-boosted-model")
def train_boosted_model(
    username: Annotated[str, typer.Argument(help="Player username")],
    features: Annotated[
        Path | None,
        typer.Option(help="BehaviorFeature Parquet; defaults to latest player batch"),
    ] = None,
    analysis: Annotated[
        Path | None,
        typer.Option(help="Stockfish analysis Parquet; defaults to latest player batch"),
    ] = None,
    games: Annotated[
        Path | None,
        typer.Option(help="Normalized games Parquet; defaults to latest player batch"),
    ] = None,
    rf_artifact: Annotated[
        Path | None,
        typer.Option(help="Preserved Random Forest artifact used for split comparison"),
    ] = None,
    artifact_dir: Annotated[
        Path | None,
        typer.Option(help="New CatBoost artifact directory"),
    ] = None,
    processed_dir: Annotated[
        Path, typer.Option(hidden=True, help="Processed Parquet directory")
    ] = Path("data/processed"),
    artifact_root: Annotated[
        Path, typer.Option(hidden=True, help="Local model artifact root")
    ] = Path("artifacts/models"),
) -> None:
    """Train grouped CatBoost ablations on the preserved chronological split."""

    try:
        feature_path = features or _latest_feature_file(username, processed_dir)
        analysis_path = analysis or _latest_analysis_file(username, processed_dir)
        games_path = games or _latest_games_file(username, processed_dir)
        rf_path = rf_artifact or _latest_rf_artifact(username, artifact_root)
        destination = artifact_dir or default_boosted_artifact_dir(
            username, artifact_root
        )
        summary = train_boosted_rankers(
            username,
            features_path=feature_path,
            analysis_path=analysis_path,
            games_path=games_path,
            rf_artifact_dir=rf_path,
            artifact_dir=destination,
        )
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as exc:
        typer.echo(f"Boosted model training failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo("Grouped CatBoost training complete")
    typer.echo(f"  Player: {summary.username}")
    typer.echo(f"  Total decisions: {summary.total_decisions}")
    typer.echo(f"  Usable top-5 decisions: {summary.usable_decisions}")
    typer.echo(f"  Outside top 5: {summary.outside_top_5_decisions}")
    typer.echo(f"  Runtime seconds: {summary.runtime_seconds:.3f}")
    typer.echo(f"  Artifact directory: {summary.artifact_dir}")


@app.command("benchmark-candidate-coverage")
def benchmark_candidate_coverage(
    cohort: Annotated[Path, typer.Option(help="Explicit cohort JSON file")],
    multipv: Annotated[int, typer.Option(min=20, max=20)] = 20,
    nodes: Annotated[int, typer.Option(min=20000, max=20000)] = 20000,
    max_decisions_per_player: Annotated[int, typer.Option(min=1)] = 1000,
    all_decisions: Annotated[bool, typer.Option(help="Explicitly analyze every obtained decision")] = False,
    seed: Annotated[int, typer.Option()] = 42,
    stockfish_path: Annotated[str, typer.Option()] = "stockfish",
    output_dir: Annotated[Path | None, typer.Option(help="New directory; existing directories are refused")] = None,
    cache_dir: Annotated[Path, typer.Option()] = Path("data/cache/candidate-coverage"),
) -> None:
    """Measure every K=1–20 and K70 with Stockfish 18, without training."""
    from datetime import UTC, datetime
    from chess_clone.benchmark.runner import run_benchmark

    destination = output_dir or Path("artifacts/benchmarks") / datetime.now(UTC).strftime("coverage_%Y%m%dT%H%M%S%fZ")
    try:
        run_benchmark(
            cohort, output_dir=destination, cache_dir=cache_dir,
            stockfish_path=stockfish_path,
            settings=EngineSettings(nodes=nodes, multipv=multipv, threads=1),
            max_decisions_per_player=None if all_decisions else max_decisions_per_player,
            seed=seed, progress=typer.echo,
        )
    except (OSError, ValueError, RuntimeError, ProviderError) as exc:
        typer.echo(f"Benchmark failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("prepare-width-experiment")
def prepare_width_experiment(
    cohort: Annotated[Path, typer.Option(help="Explicit per-player histories; no pooling")],
    output_dir: Annotated[Path, typer.Option(help="New prepared-data directory")],
    stockfish_path: Annotated[str, typer.Option()] = "stockfish",
    cache_dir: Annotated[Path, typer.Option()] = Path("data/cache/candidate-coverage"),
) -> None:
    """Analyze all obtained games once at MultiPV=20 for width comparisons."""
    from chess_clone.experiments.width_data import prepare_width_data
    try:
        prepare_width_data(cohort, output_dir, cache_dir=cache_dir, stockfish_path=stockfish_path)
    except (OSError, ValueError, RuntimeError, ProviderError) as exc:
        typer.echo(f"Width preparation failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("compare-candidate-widths")
def compare_candidate_widths(
    prepared_dir: Annotated[Path, typer.Option(help="Complete output from prepare-width-experiment")],
    output_dir: Annotated[Path, typer.Option(help="New isolated model-experiment directory")],
) -> None:
    """Compare fixed K=5, K=10, and training-only K70 per player."""
    from chess_clone.experiments.width_training import train_width_comparison
    try:
        train_width_comparison(prepared_dir, output_dir)
    except (OSError, ValueError, RuntimeError) as exc:
        typer.echo(f"Width comparison failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("train-population-policy")
def train_population_policy_command(
    cohort: Annotated[
        Path, typer.Option(help="Prepared multi-player positions/games cohort JSON")
    ],
    output_dir: Annotated[
        Path, typer.Option(help="New model and evaluation artifact directory")
    ],
    rating_min: Annotated[
        int, typer.Option(help="Minimum observed mover rating, inclusive")
    ] = 1300,
    rating_max: Annotated[
        int, typer.Option(help="Maximum observed mover rating, inclusive")
    ] = 1700,
    max_decisions_per_player: Annotated[
        int, typer.Option(min=30, help="Deterministic per-player decision cap")
    ] = 1500,
) -> None:
    """Train population and personalized policies over every legal move."""

    from chess_clone.experiments.population_policy import train_population_policy

    try:
        summary = train_population_policy(
            cohort,
            output_dir,
            rating_min=rating_min,
            rating_max=rating_max,
            max_decisions_per_player=max_decisions_per_player,
        )
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as exc:
        typer.echo(f"Population-policy training failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo("All-legal population-policy training complete")
    typer.echo(f"  Players: {summary.players}")
    typer.echo(f"  Decisions: {summary.decisions}")
    typer.echo(f"  Runtime seconds: {summary.runtime_seconds:.3f}")
    typer.echo(f"  Artifact directory: {summary.artifact_dir}")


@app.command("benchmark-move-quality")
def benchmark_move_quality_command(
    cohort: Annotated[Path, typer.Option(help="Source positions/games cohort JSON")],
    artifact_dir: Annotated[Path, typer.Option(help="Directory of frozen test prediction Parquets")],
    output_dir: Annotated[Path, typer.Option(help="New benchmark destination")],
    stockfish_path: Annotated[str, typer.Option()] = "stockfish",
    cache_dir: Annotated[Path, typer.Option()] = Path("data/cache/candidate-coverage"),
) -> None:
    """Compare predicted and human move quality with Stockfish 18."""
    from chess_clone.experiments.move_quality import run_move_quality

    try:
        run_move_quality(cohort, artifact_dir, output_dir,
                         stockfish_path=stockfish_path, cache_dir=cache_dir)
    except (OSError, ValueError, RuntimeError) as exc:
        typer.echo(f"Move-quality benchmark failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Move-quality benchmark complete: {output_dir}")


@app.command("benchmark-winning-chance")
def benchmark_winning_chance_command(
    cohort: Annotated[Path, typer.Option()],
    artifact_dir: Annotated[Path, typer.Option()],
    test_quality_dir: Annotated[Path, typer.Option()],
    output_dir: Annotated[Path, typer.Option()],
    stockfish_path: Annotated[str, typer.Option()] = "stockfish",
    cache_dir: Annotated[Path, typer.Option()] = Path("data/cache/candidate-coverage"),
) -> None:
    """Select validation tolerances and benchmark frozen winning-chance differences."""
    from chess_clone.experiments.winning_chance import run_winning_chance
    try:
        run_winning_chance(cohort, artifact_dir, test_quality_dir, output_dir,
                           stockfish_path=stockfish_path, cache_dir=cache_dir)
    except (OSError, ValueError, RuntimeError) as exc:
        typer.echo(f"Winning-chance benchmark failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Winning-chance benchmark complete: {output_dir}")


@app.command("benchmark-sampled-policy")
def benchmark_sampled_policy_command(
    config: Annotated[
        Path,
        typer.Option(help="Predeclared sampled-policy protocol JSON"),
    ] = Path("configs/deep_style_sampled_policy_v2.json"),
    output_dir: Annotated[
        Path,
        typer.Option(help="New benchmark destination; existing paths are refused"),
    ] = Path("artifacts/benchmarks/deep-style-sampled-policy-v2"),
    stockfish_path: Annotated[str, typer.Option()] = "stockfish",
    cache_dir: Annotated[
        Path,
        typer.Option(help="Persistent Stockfish quality cache"),
    ] = Path("data/cache/candidate-coverage"),
) -> None:
    """Run the frozen sampled-policy strength/error guardrail."""

    from chess_clone.experiments.sampled_policy import run

    try:
        report = run(
            config,
            output_dir,
            stockfish_path=stockfish_path,
            cache_dir=cache_dir,
        )
    except (FileNotFoundError, FileExistsError, OSError, RuntimeError, ValueError) as exc:
        typer.echo(f"Sampled-policy benchmark failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo("Sampled-policy benchmark complete")
    typer.echo(f"  Decisions: {report['sampling']['decisions']}")
    typer.echo(f"  Games: {report['sampling']['games']}")
    typer.echo(f"  Runtime seconds: {report['runtime_seconds']:.3f}")
    typer.echo(f"  Output: {output_dir}")


@app.command("benchmark-generated-trajectories")
def benchmark_generated_trajectories_command(
    config: Annotated[
        Path,
        typer.Option(help="Predeclared generated-trajectory protocol JSON"),
    ] = Path("configs/deep_style_rollout_v1.json"),
    output_dir: Annotated[
        Path,
        typer.Option(help="New benchmark destination; existing paths are refused"),
    ] = Path("artifacts/benchmarks/deep-style-rollout-v1"),
) -> None:
    """Generate matched games for the four frozen policy arms."""

    from chess_clone.experiments.generated_rollout import run

    try:
        report = run(config, output_dir)
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as exc:
        typer.echo(f"Generated-trajectory benchmark failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo("Generated-trajectory benchmark complete")
    typer.echo(f"  Players: {report['players']}")
    typer.echo(f"  Games: {report['games']}")
    typer.echo(f"  Runtime seconds: {report['runtime_seconds']:.3f}")
    typer.echo(f"  Output: {output_dir}")


@app.command("audit-style-v2-capacity")
def audit_style_v2_capacity_command(
    config: Annotated[
        Path,
        typer.Option(help="Frozen Style Model v2 declaration"),
    ] = Path("configs/style_model_v2.json"),
    input_root: Annotated[
        Path,
        typer.Option(help="Local root recursively containing normalized game Parquets"),
    ] = Path("artifacts"),
    output: Annotated[
        Path,
        typer.Option(help="New JSON capacity report; existing files are refused"),
    ] = Path("artifacts/benchmarks/style-v2-capacity-v1/report.json"),
) -> None:
    """Audit local player/game capacity without network access or move labels."""

    from chess_clone.experiments.style_v2_capacity import run

    try:
        report = run(config, input_root, output)
    except (FileNotFoundError, FileExistsError, KeyError, OSError, ValueError) as exc:
        typer.echo(f"Style v2 capacity audit failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    summary = report["summary"]
    typer.echo("Style v2 capacity audit complete")
    typer.echo(f"  Players seen: {summary['players_seen']}")
    typer.echo(f"  Prototype scenario: {summary['selected_prototype_scenario']}")
    typer.echo(f"  Prototype capacity passed: {summary['prototype_capacity_passed']}")
    typer.echo(f"  Output: {output}")


@app.command("prepare-style-v2-prototype")
def prepare_style_v2_prototype_command(
    config: Annotated[Path, typer.Option()] = Path("configs/style_model_v2.json"),
    cohort: Annotated[Path, typer.Option()] = Path("configs/style_model_v2_prototype_cohort.json"),
    capacity_report: Annotated[Path, typer.Option()] = Path("artifacts/benchmarks/style-v2-capacity-v1/report.json"),
    artifact_dir: Annotated[Path, typer.Option()] = Path("artifacts/models/population_1500_safe_profile_v4"),
    output_dir: Annotated[Path, typer.Option()] = Path("artifacts/benchmarks/style-v2-prototype-data"),
) -> None:
    """Prepare all frozen 400/50/50 games; resume verified completed caches."""
    from chess_clone.experiments.style_v2_dataset import prepare_prototype_candidate_caches

    try:
        report = prepare_prototype_candidate_caches(config, cohort, capacity_report, artifact_dir, output_dir)
    except (FileNotFoundError, FileExistsError, KeyError, OSError, ValueError) as exc:
        typer.echo(f"Style v2 preparation failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(report["splits"], indent=2))


@app.command("prepare-style-v2-smoke")
def prepare_style_v2_smoke_command(
    cohort: Annotated[Path, typer.Option()] = Path("configs/style_model_v2_prototype_cohort.json"),
    cache_dir: Annotated[Path, typer.Option()] = Path("artifacts/benchmarks/deep-style-v1/residual/cache"),
    output_dir: Annotated[Path, typer.Option()] = Path("artifacts/benchmarks/style-v2-smoke-data"),
    players: Annotated[int, typer.Option(min=2)] = 2,
    decisions_per_player: Annotated[int, typer.Option(min=16)] = 256,
) -> None:
    """Prepare a bounded v1-cache engineering smoke test, not v2 evidence."""

    from chess_clone.experiments.style_v2_dataset import prepare_v1_cache_smoke

    try:
        manifest = prepare_v1_cache_smoke(
            cohort, cache_dir, output_dir, players_limit=players,
            max_decisions_per_player=decisions_per_player,
        )
    except (FileNotFoundError, FileExistsError, KeyError, OSError, ValueError) as exc:
        typer.echo(f"Style v2 smoke preparation failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo("Style v2 smoke data prepared")
    typer.echo(f"  Players: {len(manifest['players'])}")
    typer.echo(f"  Train decisions: {manifest['train_decisions']}")
    typer.echo(f"  Evaluation decisions: {manifest['evaluation_decisions']}")
    typer.echo(f"  Output: {output_dir}")


@app.command("benchmark-style-v2-prototype")
def benchmark_style_v2_prototype_command(
    config: Annotated[Path, typer.Option()] = Path("configs/style_model_v2_prototype_run.json"),
    data: Annotated[Path, typer.Option()] = Path("artifacts/benchmarks/style-v2-prototype-data"),
    output_dir: Annotated[Path, typer.Option()] = Path("artifacts/benchmarks/style-v2-smallest-candidate-v1"),
) -> None:
    """Fit the fixed embedding candidate and apply the necessary replay gate."""
    try:
        from chess_clone.experiments.style_v2_prototype import run
        report = run(config, data, output_dir)
    except (FileNotFoundError, FileExistsError, ImportError, KeyError, OSError, RuntimeError, ValueError) as exc:
        typer.echo(f"Style v2 prototype failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Style v2 prototype: {report['status']}")
    typer.echo(json.dumps(report["development"], indent=2))


@app.command("train-style-v2-smoke")
def train_style_v2_smoke_command(
    manifest: Annotated[Path, typer.Option()] = Path("artifacts/benchmarks/style-v2-smoke-data/train/manifest.json"),
    evaluation_manifest: Annotated[Path, typer.Option()] = Path("artifacts/benchmarks/style-v2-smoke-data/evaluation/manifest.json"),
    output_dir: Annotated[Path, typer.Option()] = Path("artifacts/benchmarks/style-v2-smoke-model"),
    epochs: Annotated[int, typer.Option(min=1)] = 3,
    batch_decisions: Annotated[int, typer.Option(min=1)] = 64,
) -> None:
    """Train the 32-dimensional zero-sequence-context engineering smoke model."""

    try:
        from chess_clone.modeling.player_embedding import train_embedding
        report = train_embedding(
            manifest, output_dir, evaluation_manifest_path=evaluation_manifest,
            embedding_dimension=32, epochs=epochs,
            batch_decisions=batch_decisions,
        )
    except (FileNotFoundError, FileExistsError, ImportError, OSError, RuntimeError, ValueError) as exc:
        typer.echo(f"Style v2 smoke training failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo("Style v2 smoke training complete")
    typer.echo(f"  Device: {report['device']}")
    typer.echo(f"  Decisions: {report['decisions']}")
    typer.echo(f"  Final loss: {report['history'][-1]['total']:.6f}")
    typer.echo(f"  Output: {output_dir}")


if __name__ == "__main__":
    app()
