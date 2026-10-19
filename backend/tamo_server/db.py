"""SQLite store for projects, actors, reference images, jobs and generations.

Files stay in the library folders; the database only records what exists and how it
was made, so every head can be traced back to its job, settings and source image.
"""

import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    id INTEGER PRIMARY KEY,
    slug TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    created TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS actors (
    id INTEGER PRIMARY KEY,
    project_id INTEGER NOT NULL REFERENCES projects(id),
    slug TEXT NOT NULL,
    name TEXT NOT NULL,
    created TEXT NOT NULL,
    preferred_generation_id INTEGER,
    UNIQUE (project_id, slug)
);
CREATE TABLE IF NOT EXISTS reference_images (
    id INTEGER PRIMARY KEY,
    actor_id INTEGER NOT NULL REFERENCES actors(id),
    view TEXT NOT NULL,              -- front, three_quarter, side, back, look
    path TEXT NOT NULL UNIQUE,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    created TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY,
    actor_id INTEGER NOT NULL REFERENCES actors(id),
    reference_id INTEGER NOT NULL REFERENCES reference_images(id),
    settings TEXT NOT NULL,          -- JSON
    status TEXT NOT NULL,            -- queued, running, done, failed, cancelled
    stage TEXT,
    progress REAL NOT NULL DEFAULT 0,
    error TEXT,
    run_dir TEXT,
    attempts INTEGER NOT NULL DEFAULT 0,
    created TEXT NOT NULL,
    started TEXT,
    finished TEXT
);
CREATE TABLE IF NOT EXISTS generations (
    id INTEGER PRIMARY KEY,
    actor_id INTEGER NOT NULL REFERENCES actors(id),
    job_id INTEGER REFERENCES jobs(id),
    run_dir TEXT NOT NULL UNIQUE,
    record TEXT NOT NULL,            -- run.json as written by run_phase1.py
    created TEXT NOT NULL
);
"""


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class Database:
    """One connection shared across threads, serialised by a lock."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(SCHEMA)
        self._lock = threading.Lock()

    def execute(self, sql: str, params=()) -> sqlite3.Cursor:
        with self._lock, self._conn:
            return self._conn.execute(sql, params)

    def transaction(self, statements: list[tuple[str, tuple]]) -> None:
        """Runs several statements as one change: all of them happen, or none."""
        with self._lock, self._conn:
            for sql, params in statements:
                self._conn.execute(sql, params)

    def one(self, sql: str, params=()) -> dict | None:
        with self._lock:
            row = self._conn.execute(sql, params).fetchone()
        return dict(row) if row else None

    def all(self, sql: str, params=()) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql, params).fetchall()]

    def insert(self, table: str, **values) -> int:
        columns = ", ".join(values)
        marks = ", ".join("?" for _ in values)
        return self.execute(f"INSERT INTO {table} ({columns}) VALUES ({marks})", tuple(values.values())).lastrowid

    def update(self, table: str, row_id: int, **values) -> None:
        sets = ", ".join(f"{k} = ?" for k in values)
        self.execute(f"UPDATE {table} SET {sets} WHERE id = ?", (*values.values(), row_id))


def job_view(job: dict) -> dict:
    return {**job, "settings": json.loads(job["settings"])}


def generation_view(generation: dict) -> dict:
    return {**generation, "record": json.loads(generation["record"])}
