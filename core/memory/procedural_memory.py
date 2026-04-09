from __future__ import annotations

import os
import sqlite3
import time
from dataclasses import dataclass
from typing import Any

from core.settings import get_settings


@dataclass(frozen=True)
class Strategy:
    hybrid_alpha: float
    rerank: bool
    top_k_boost: int
    bm25_k_boost: int


def _db_path() -> str:
    s = get_settings()
    return s.procedural_db_path


def _connect() -> sqlite3.Connection:
    path = _db_path()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS strategy_stats (
            user_id TEXT NOT NULL,
            query_type TEXT NOT NULL,
            strategy_key TEXT NOT NULL,
            n_runs INTEGER NOT NULL,
            avg_faithfulness REAL NOT NULL,
            avg_retrieval_retries REAL NOT NULL,
            avg_verification_retries REAL NOT NULL,
            updated_at REAL NOT NULL,
            PRIMARY KEY (user_id, query_type, strategy_key)
        );
        """
    )
    return conn


def _key(s: Strategy) -> str:
    return f"a={s.hybrid_alpha:.2f}|r={int(s.rerank)}|tk={s.top_k_boost}|bk={s.bm25_k_boost}"


def choose_strategy(*, user_id: str, query_type: str) -> Strategy:
    """
    Pick a retrieval strategy based on past outcomes.
    If no history, return a sensible default.
    """
    settings = get_settings()
    candidates = [
        Strategy(hybrid_alpha=settings.hybrid_alpha, rerank=True, top_k_boost=0, bm25_k_boost=0),
        Strategy(hybrid_alpha=min(0.75, settings.hybrid_alpha + 0.15), rerank=True, top_k_boost=2, bm25_k_boost=0),
        Strategy(hybrid_alpha=max(0.35, settings.hybrid_alpha - 0.15), rerank=True, top_k_boost=0, bm25_k_boost=4),
        Strategy(hybrid_alpha=settings.hybrid_alpha, rerank=False, top_k_boost=2, bm25_k_boost=4),
    ]

    conn = _connect()
    try:
        best = candidates[0]
        best_score = -1e9
        for c in candidates:
            row = conn.execute(
                "SELECT n_runs, avg_faithfulness, avg_retrieval_retries, avg_verification_retries FROM strategy_stats "
                "WHERE user_id=? AND query_type=? AND strategy_key=?",
                (user_id, query_type, _key(c)),
            ).fetchone()
            if not row:
                # Prefer strategies with some exploration depending on query type.
                prior = 0.05 if query_type == "multi-hop" else 0.0
                score = prior
            else:
                n_runs, avg_f, avg_rr, avg_vr = row
                score = float(avg_f) - 0.08 * float(avg_rr) - 0.12 * float(avg_vr) + 0.01 * float(min(n_runs, 20))
            if score > best_score:
                best_score = score
                best = c
        return best
    finally:
        conn.close()


def update_strategy_stats(
    *,
    user_id: str,
    query_type: str,
    strategy: Strategy,
    faithfulness: float,
    retrieval_retries: int,
    verification_retries: int,
) -> None:
    conn = _connect()
    try:
        k = _key(strategy)
        row = conn.execute(
            "SELECT n_runs, avg_faithfulness, avg_retrieval_retries, avg_verification_retries FROM strategy_stats "
            "WHERE user_id=? AND query_type=? AND strategy_key=?",
            (user_id, query_type, k),
        ).fetchone()

        now = time.time()
        if not row:
            conn.execute(
                "INSERT INTO strategy_stats(user_id, query_type, strategy_key, n_runs, avg_faithfulness, avg_retrieval_retries, "
                "avg_verification_retries, updated_at) VALUES(?,?,?,?,?,?,?,?)",
                (user_id, query_type, k, 1, float(faithfulness), float(retrieval_retries), float(verification_retries), now),
            )
        else:
            n_runs, avg_f, avg_rr, avg_vr = row
            n2 = int(n_runs) + 1
            new_avg_f = (float(avg_f) * int(n_runs) + float(faithfulness)) / n2
            new_avg_rr = (float(avg_rr) * int(n_runs) + float(retrieval_retries)) / n2
            new_avg_vr = (float(avg_vr) * int(n_runs) + float(verification_retries)) / n2
            conn.execute(
                "UPDATE strategy_stats SET n_runs=?, avg_faithfulness=?, avg_retrieval_retries=?, avg_verification_retries=?, updated_at=? "
                "WHERE user_id=? AND query_type=? AND strategy_key=?",
                (n2, new_avg_f, new_avg_rr, new_avg_vr, now, user_id, query_type, k),
            )

        conn.commit()
    finally:
        conn.close()


def clear_user_strategy_stats(*, user_id: str) -> int:
    conn = _connect()
    try:
        cur = conn.execute("DELETE FROM strategy_stats WHERE user_id=?", (user_id,))
        conn.commit()
        return int(cur.rowcount or 0)
    finally:
        conn.close()


def list_strategy_stats(*, user_id: str, limit: int = 50) -> list[dict[str, Any]]:
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT query_type, strategy_key, n_runs, avg_faithfulness, avg_retrieval_retries, "
            "avg_verification_retries, updated_at FROM strategy_stats WHERE user_id=? "
            "ORDER BY updated_at DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
        return [
            {
                "query_type": r[0],
                "strategy_key": r[1],
                "n_runs": int(r[2]),
                "avg_faithfulness": float(r[3]),
                "avg_retrieval_retries": float(r[4]),
                "avg_verification_retries": float(r[5]),
                "updated_at": float(r[6]),
            }
            for r in rows
        ]
    finally:
        conn.close()

