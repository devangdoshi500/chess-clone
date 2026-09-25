"""Benchmark-only cache orchestration. Production top-five analysis is untouched."""

from dataclasses import asdict, dataclass, replace
import hashlib
import re
from time import perf_counter

import chess

from chess_clone.analysis.cache import FileAnalysisCache, build_analysis_cache_key
from chess_clone.analysis.pipeline import PositionAnalyzer
from chess_clone.analysis.schemas import EngineSettings


def benchmark_position(fen: str) -> str:
    # Preserve the halfmove clock for the 50-move rule, ignore only move number.
    return ' '.join(chess.Board(fen).fen(en_passant='fen').split()[:5]) + ' 1'


def benchmark_cache_key(fen: str, settings: EngineSettings, identity: str) -> str:
    base = build_analysis_cache_key(fen, settings, identity)
    return hashlib.sha256(f'coverage-v1:{base}:{benchmark_position(fen)}'.encode()).hexdigest()


@dataclass
class EngineStatistics:
    engine_calls: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    engine_seconds: float = 0.
    coverage_requests: int = 0
    quality_requests: int = 0


class CoverageEngine:
    def __init__(self, analyzer: PositionAnalyzer, cache: FileAnalysisCache, settings: EngineSettings):
        if settings.multipv != 20 or settings.nodes != 20_000 or settings.threads != 1 or settings.options:
            raise ValueError('benchmark requires MultiPV=20, nodes=20000, threads=1 and default engine options')
        if not re.match(r'^Stockfish 18(?:\||$)', analyzer.engine_identity):
            raise ValueError(f'benchmark requires Stockfish 18; found {analyzer.engine_identity}')
        self.analyzer, self.cache, self.settings = analyzer, cache, settings
        self.stats = EngineStatistics()
        self.pre_positions: set[str] = set()
        self.all_positions: set[str] = set()

    def snapshot(self) -> dict:
        return {**asdict(self.stats), 'unique_positions': len(self.pre_positions),
                'unique_positions_including_quality': len(self.all_positions),
                'average_time_per_miss_seconds': self.stats.engine_seconds / self.stats.cache_misses if self.stats.cache_misses else None}

    def _get(self, fen: str, width: int, purpose: str):
        fen = benchmark_position(fen)
        board = chess.Board(fen)
        expected = min(width, board.legal_moves.count())
        if expected < 1:
            raise ValueError('cannot rank a position with no legal moves')
        settings = replace(self.settings, multipv=expected)
        self.all_positions.add(fen)
        if purpose == 'coverage':
            self.pre_positions.add(fen)
            self.stats.coverage_requests += 1
        else:
            self.stats.quality_requests += 1
        identity = self.analyzer.engine_identity
        key = benchmark_cache_key(fen, settings, identity)
        cached = self.cache.get(key)
        hit = cached is not None
        if cached is None:
            started = perf_counter()
            lines = self.analyzer.analyze(fen, settings)
            self.stats.engine_seconds += perf_counter() - started
            self.stats.engine_calls += 1
            self.stats.cache_misses += 1
        else:
            self.stats.cache_hits += 1
            lines = list(cached.lines)
        lines = sorted(lines, key=lambda line: line.rank)
        moves = [line.best_move_uci for line in lines]
        # A partial MultiPV response cannot be mistaken for evidence of exclusion.
        if ([line.rank for line in lines] != list(range(1, expected + 1))
                or len(set(moves)) != expected
                or any(m not in {m.uci() for m in board.legal_moves} for m in moves)):
            raise ValueError(f'incomplete or invalid MultiPV response: expected {expected} legal unique lines')
        if not hit:
            self.cache.put(key, position_key=fen, engine_identity=identity, settings=settings, lines=lines)
        return lines, key, hit

    def analyze_candidates(self, fen: str):
        """Return the cached broad candidate set without post-move diagnostics."""
        return self._get(fen, 20, 'coverage')

    def analyze_quality(self, fen: str):
        """SinglePV evaluation with the existing clock-preserving quality cache."""
        return self._get(fen, 1, 'quality')

    def analyze_decision(self, row: dict) -> dict:
        board = chess.Board(row['fen'])
        move = chess.Move.from_uci(row['actual_move_uci'])
        if move not in board.legal_moves:
            raise ValueError(f'illegal actual move in {row["game_id"]} ply {row["ply"]}')
        lines, key, hit = self.analyze_candidates(board.fen(en_passant='fen'))
        rank = next((line.rank for line in lines if line.best_move_uci == move.uci()), None)
        board.push(move)
        post_key, post_hit = None, None
        # Finite CP loss uses a uniform separate post-move SinglePV search for
        # every decision, including moves outside top 20 (avoids selection bias).
        if board.is_checkmate():
            actual_cp, actual_mate = None, 0
        elif board.is_stalemate() or board.is_insufficient_material():
            actual_cp, actual_mate = 0, None
        else:
            after, post_key, post_hit = self._get(board.fen(en_passant='fen'), 1, 'quality')
            actual_cp = -after[0].score_cp if after[0].score_cp is not None else None
            actual_mate = -after[0].mate_in if after[0].mate_in is not None else None
        best = lines[0]
        loss = max(0, best.score_cp - actual_cp) if best.score_cp is not None and actual_cp is not None and best.mate_in is None and actual_mate is None else None
        return {
            'actual_move_rank': rank, 'effective_multipv': len(lines),
            'best_score_cp': best.score_cp, 'best_mate_in': best.mate_in,
            'actual_score_cp': actual_cp, 'actual_mate_in': actual_mate,
            'centipawn_loss': loss, 'cache_key': key, 'cache_hit': hit,
            'quality_cache_key': post_key, 'quality_cache_hit': post_hit,
        }
