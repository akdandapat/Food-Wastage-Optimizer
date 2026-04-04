"""
Repository façade required by the architecture spec.

The operational implementation lives in ``backend.database``; this module
re-exports it so imports stay stable across documentation and tooling.
"""

from __future__ import annotations

from backend.database import SQLiteRepository, get_connection, initialize_database

__all__ = ["SQLiteRepository", "get_connection", "initialize_database"]
