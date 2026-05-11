"""Checkpointer adapter for state persistence.

Supports multiple backends:
- memory: In-memory persistence (default for lab, good for CI/local testing)
- sqlite: SQLite database persistence (extension track, survives restart)
- postgres: Postgres database persistence (production, requires external DB)
"""

from __future__ import annotations

from typing import Any


def build_checkpointer(kind: str = "memory", database_url: str | None = None) -> Any | None:
    """Build a LangGraph checkpointer for workflow state persistence.

    Args:
        kind: Checkpointer type - "none", "memory", "sqlite", or "postgres"
        database_url: Database connection string (for sqlite/postgres)
            For sqlite: file path or connection string (default: "checkpoints.db")
            For postgres: full connection string (required)

    Returns:
        Checkpointer instance or None if kind=="none"

    Persistence supports crash-resume: interrupted runs can resume from same thread_id
    after restart, enabling state history and recovery.
    
    Extension track: Enable SQLite in lab.yaml (checkpointer: sqlite) to demonstrate
    persistence across process restarts.
    """
    if kind == "none":
        return None
    
    if kind == "memory":
        from langgraph.checkpoint.memory import MemorySaver
        return MemorySaver()
    
    if kind == "sqlite":
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
            import sqlite3
        except ImportError as exc:
            raise RuntimeError("SQLite checkpointer requires: pip install langgraph-checkpoint-sqlite") from exc
        db_path = database_url or "checkpoints.db"
        # Using connection object as recommended for version 3.x
        conn = sqlite3.connect(db_path, check_same_thread=False)
        return SqliteSaver(conn)
    
    if kind == "postgres":
        try:
            from langgraph.checkpoint.postgres import PostgresSaver
        except ImportError as exc:
            raise RuntimeError("Postgres checkpointer requires: pip install langgraph-checkpoint-postgres") from exc
        if not database_url:
            raise ValueError("Postgres checkpointer requires a database_url connection string")
        return PostgresSaver.from_conn_string(database_url)
    
    raise ValueError(f"Unknown checkpointer kind: {kind}. Choose from: none, memory, sqlite, postgres")