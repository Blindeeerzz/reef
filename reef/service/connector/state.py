"""Private connector configuration and a durable record of dispatched HTTP requests."""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, cast


class ServeConfig(TypedDict):
    arguments: list[str]
    directory: str


class ConnectionConfig(TypedDict, total=False):
    platform_url: str
    reef_url: str
    instance_id: str
    name: str
    connector_token: str
    reef_token: str
    runtime_id: str
    http_policy: object
    serve: ServeConfig | None


@dataclass(frozen=True)
class HttpRequestState:
    fingerprint: str
    outcome: str


class ConnectorState:
    """Keep credentials on this machine and never re-execute a recorded mutation."""

    def __init__(self, directory: Path):
        self.directory = directory
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(directory, 0o700)
        self.config_path = directory / "connection.json"
        self.db = sqlite3.connect(directory / "requests.sqlite3")
        os.chmod(directory / "requests.sqlite3", 0o600)
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS http_requests "
            "(id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, outcome TEXT NOT NULL)"
        )
        self.db.commit()

    def load(self) -> ConnectionConfig:
        if not self.config_path.exists():
            return {}
        value = json.loads(self.config_path.read_text())
        if not isinstance(value, dict):
            raise ValueError("Connection configuration must be a JSON object")
        for key in ("platform_url", "reef_url", "instance_id", "name", "connector_token", "reef_token", "runtime_id"):
            if key in value and not isinstance(value[key], str):
                raise ValueError(f"Connection setting {key} must be a string")
        serve = value.get("serve")
        if serve is not None and (
            not isinstance(serve, dict)
            or not isinstance(serve.get("directory"), str)
            or not isinstance(serve.get("arguments"), list)
            or any(not isinstance(argument, str) for argument in serve["arguments"])
        ):
            raise ValueError("Invalid saved Reef service options")
        return cast(ConnectionConfig, value)

    def save(self, config: ConnectionConfig) -> None:
        temporary = self.directory / "connection.tmp"
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, "w") as stream:
            json.dump(config, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self.config_path)

    @contextmanager
    def lock(self):
        """OS locks release on crashes; a PID file alone cannot prove ownership."""
        stream = (self.directory / "connector.lock").open("a+b")
        try:
            if os.name == "nt":
                import msvcrt

                stream.write(b"0")
                stream.flush()
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            stream.close()
            raise RuntimeError("A connector is already running for this instance") from exc
        try:
            yield
        finally:
            stream.close()

    def close(self) -> None:
        self.db.close()

    def http_request(self, identity: str) -> HttpRequestState | None:
        row = self.db.execute("SELECT fingerprint,outcome FROM http_requests WHERE id=?", (identity,)).fetchone()
        if row is None:
            return None
        return HttpRequestState(row[0], row[1])

    def start_http_request(self, identity: str, fingerprint: str) -> None:
        """Persist dispatch before a write can reach Reef; keep no request bodies."""
        self.db.execute(
            "INSERT INTO http_requests (id,fingerprint,outcome) VALUES (?,?,'unknown')", (identity, fingerprint)
        )
        self.db.commit()

    def finish_http_request(self, identity: str, outcome: str) -> None:
        self.db.execute("UPDATE http_requests SET outcome=? WHERE id=?", (outcome, identity))
        self.db.commit()
