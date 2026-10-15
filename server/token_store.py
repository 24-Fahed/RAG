"""Locally managed opaque API tokens; only token hashes are persisted."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
import secrets
import sqlite3
import time
import uuid


TOKEN_PREFIX = "rag_pat_"
TOKEN_SCOPE = "rag:api"
_TOKEN_PATTERN = re.compile(r"rag_pat_([0-9a-f]{32})\.([A-Za-z0-9_-]{43})", re.ASCII)
_COLUMNS = "id, subject, created_at, expires_at, revoked_at"
_SCHEMA = """
CREATE TABLE IF NOT EXISTS access_tokens (
    id TEXT PRIMARY KEY,
    subject TEXT NOT NULL,
    token_hash TEXT NOT NULL UNIQUE,
    created_at INTEGER NOT NULL,
    expires_at INTEGER NOT NULL,
    revoked_at INTEGER
)
"""


class TokenStoreError(RuntimeError):
    """Configuration or storage failure, distinct from an invalid credential."""


@dataclass(frozen=True)
class TokenRecord:
    id: str
    subject: str
    created_at: int
    expires_at: int
    revoked_at: int | None

    def is_expired(self) -> bool:
        return time.time() >= self.expires_at

    def is_revoked(self) -> bool:
        return self.revoked_at is not None

    def get_scope(self) -> str:
        return TOKEN_SCOPE

    def status(self) -> str:
        if self.is_revoked():
            return "revoked"
        return "expired" if self.is_expired() else "active"

    def claims(self) -> dict:
        return {"sub": self.subject, "scope": TOKEN_SCOPE, "token_id": self.id,
                "iat": self.created_at, "exp": self.expires_at}


def configured_store() -> TokenStore:
    value = os.getenv("RAG_AUTH_TOKEN_DB_FILE", "")
    if not value:
        raise TokenStoreError("RAG_AUTH_TOKEN_DB_FILE is not configured")
    return TokenStore(Path(value))


class TokenStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        if not self.path.is_absolute():
            raise TokenStoreError("RAG_AUTH_TOKEN_DB_FILE must be an absolute path")

    @contextmanager
    def _connection(self, *, write: bool = False, create: bool = False):
        connection = None
        try:
            if create:
                self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                try:
                    descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                except FileExistsError:
                    pass
                else:
                    os.close(descriptor)
            # Validation and management reads must never create an empty store.
            mode = "rw" if write else "ro"
            connection = sqlite3.connect(f"{self.path.as_uri()}?mode={mode}", uri=True, timeout=5)
            connection.row_factory = sqlite3.Row
            yield connection
        except (sqlite3.Error, OSError) as exc:
            raise TokenStoreError("API token store is unavailable") from exc
        finally:
            if connection is not None:
                connection.close()

    @staticmethod
    def _record(row: sqlite3.Row) -> TokenRecord:
        return TokenRecord(**{column: row[column] for column in
                              ("id", "subject", "created_at", "expires_at", "revoked_at")})

    @staticmethod
    def _duration(days: int) -> int:
        if not isinstance(days, int) or isinstance(days, bool) or not 1 <= days <= 36500:
            raise ValueError("days must be an integer between 1 and 36500")
        return days * 86400

    def issue(self, subject: str, days: int = 30) -> tuple[str, TokenRecord]:
        subject = subject.strip()
        if not subject or len(subject) > 200 or any(c in subject for c in "\r\n\t"):
            raise ValueError("subject must contain 1-200 characters and no line breaks or tabs")
        duration = self._duration(days)
        now = int(time.time())
        token_id = uuid.uuid4().hex
        token = f"{TOKEN_PREFIX}{token_id}.{secrets.token_urlsafe(32)}"
        digest = hashlib.sha256(token.encode("ascii")).hexdigest()
        record = TokenRecord(token_id, subject, now, now + duration, None)
        with self._connection(write=True, create=True) as connection, connection:
            connection.execute(_SCHEMA)
            connection.execute(
                "INSERT INTO access_tokens VALUES (?, ?, ?, ?, ?, ?)",
                (record.id, record.subject, digest, record.created_at, record.expires_at, None),
            )
        return token, record

    def authenticate(self, token: str) -> TokenRecord | None:
        if len(token) != len(TOKEN_PREFIX) + 32 + 1 + 43:
            return None
        match = _TOKEN_PATTERN.fullmatch(token)
        if match is None:
            return None
        digest = hashlib.sha256(token.encode("ascii")).hexdigest()
        with self._connection() as connection:
            row = connection.execute(
                f"SELECT {_COLUMNS}, token_hash FROM access_tokens WHERE id = ?", (match[1],)
            ).fetchone()
        if row is None or not secrets.compare_digest(row["token_hash"], digest):
            return None
        return self._record(row)

    def list_tokens(self) -> list[TokenRecord]:
        with self._connection() as connection:
            rows = connection.execute(
                f"SELECT {_COLUMNS} FROM access_tokens ORDER BY created_at, id"
            ).fetchall()
        return [self._record(row) for row in rows]

    def renew(self, token_id: str, days: int = 30) -> TokenRecord:
        duration = self._duration(days)
        with self._connection(write=True) as connection, connection:
            # Serialize administrative changes: renewal cannot undo a racing revocation
            # or lose days when two administrators renew the same token at once.
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                f"SELECT {_COLUMNS} FROM access_tokens WHERE id = ?", (token_id,)
            ).fetchone()
            if row is None:
                raise ValueError("Token ID was not found")
            record = self._record(row)
            if record.is_revoked():
                raise ValueError("Revoked tokens cannot be renewed; issue a new token")
            expires_at = max(int(time.time()), record.expires_at) + duration
            connection.execute("UPDATE access_tokens SET expires_at = ? WHERE id = ?",
                               (expires_at, token_id))
        return TokenRecord(record.id, record.subject, record.created_at, expires_at, None)

    def revoke(self, token_id: str) -> TokenRecord:
        with self._connection(write=True) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                f"SELECT {_COLUMNS} FROM access_tokens WHERE id = ?", (token_id,)
            ).fetchone()
            if row is None:
                raise ValueError("Token ID was not found")
            record = self._record(row)
            revoked_at = record.revoked_at if record.is_revoked() else int(time.time())
            connection.execute("UPDATE access_tokens SET revoked_at = ? WHERE id = ?",
                               (revoked_at, token_id))
        return TokenRecord(record.id, record.subject, record.created_at, record.expires_at, revoked_at)
