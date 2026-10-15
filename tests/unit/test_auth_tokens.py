"""Opaque-token lifecycle and administration use a temporary SQLite store."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import sqlite3
import sys
from types import SimpleNamespace

import pytest

from server import auth, token_store


@pytest.fixture
def store(tmp_path, monkeypatch):
    clock = [1_800_000_000]
    monkeypatch.setattr(token_store, "time", SimpleNamespace(time=lambda: clock[0]))
    path = tmp_path / "auth" / "tokens.sqlite3"
    monkeypatch.setenv("RAG_AUTH_TOKEN_DB_FILE", str(path))
    return token_store.configured_store(), clock


def test_hash_only_persistence_and_independent_tokens(store):
    database, _ = store
    first, first_record = database.issue("老同学")
    second, second_record = database.issue("另一位熟人", 2)
    assert first != second and first_record.id != second_record.id
    assert first_record.expires_at - first_record.created_at == 30 * 86400
    assert database.authenticate(first) == first_record
    assert token_store.TokenStore(database.path).authenticate(second) == second_record
    assert {row.subject for row in database.list_tokens()} == {"老同学", "另一位熟人"}
    with sqlite3.connect(database.path) as connection:
        row = connection.execute("SELECT token_hash FROM access_tokens WHERE id = ?",
                                 (first_record.id,)).fetchone()
    assert row[0] == hashlib.sha256(first.encode("ascii")).hexdigest()
    persisted = database.path.read_bytes()
    assert first.encode() not in persisted and first.split(".")[1].encode() not in persisted


def test_expiry_boundary_renewal_and_terminal_revocation(store):
    database, clock = store
    token, issued = database.issue("friend", 1)
    clock[0] = issued.expires_at - 1
    assert database.authenticate(token).status() == "active"
    renewed = database.renew(issued.id, 2)
    assert renewed.expires_at == issued.expires_at + 2 * 86400
    clock[0] = renewed.expires_at
    assert database.authenticate(token).status() == "expired"
    clock[0] += 123
    renewed = database.renew(issued.id, 3)
    assert renewed.expires_at == clock[0] + 3 * 86400
    assert database.authenticate(token).status() == "active"
    revoked = database.revoke(issued.id)
    clock[0] += 10
    assert database.revoke(issued.id).revoked_at == revoked.revoked_at
    assert database.authenticate(token).status() == "revoked"
    with pytest.raises(ValueError, match="Revoked"):
        database.renew(issued.id)
    assert len(database.list_tokens()) == 1


def test_concurrent_renewals_do_not_lose_days(store):
    database, _ = store
    _, record = database.issue("friend", 1)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: database.renew(record.id, 1), range(2)))
    assert max(row.expires_at for row in results) == record.expires_at + 2 * 86400
    assert database.list_tokens()[0].expires_at == record.expires_at + 2 * 86400


def test_revoking_one_token_does_not_affect_another(store):
    database, _ = store
    first, record = database.issue("friend")
    second, _ = database.issue("friend")
    database.revoke(record.id)
    assert database.authenticate(first).is_revoked()
    assert not database.authenticate(second).is_revoked()


@pytest.mark.parametrize("days", [0, -1, 36501, 1.5, True])
def test_invalid_duration_does_not_create_store(store, days):
    database, _ = store
    with pytest.raises(ValueError, match="days"):
        database.issue("friend", days)
    assert not database.path.exists()


@pytest.mark.parametrize("subject", ["", "  ", "a\nb", "a\tb", "a" * 201])
def test_invalid_subject_does_not_create_store(store, subject):
    database, _ = store
    with pytest.raises(ValueError, match="subject"):
        database.issue(subject)
    assert not database.path.exists()


def test_malformed_or_tampered_tokens_are_rejected(store):
    database, _ = store
    token, _ = database.issue("friend")
    replacement = "A" if token[-1] != "A" else "B"
    assert database.authenticate(token[:-1] + replacement) is None
    for invalid in [token + "\n", "rag_pat_" + "x" * 100000, token.replace(".", "/"),
                    token.replace("rag_pat_", "RAG_PAT_"), token[:-1] + "中"]:
        assert database.authenticate(invalid) is None


def test_missing_store_is_never_created_by_reads_or_updates(store):
    database, _ = store
    for operation in [database.list_tokens, lambda: database.renew("absent"),
                      lambda: database.revoke("absent"),
                      lambda: database.authenticate("rag_pat_" + "a" * 32 + "." + "b" * 43)]:
        with pytest.raises(token_store.TokenStoreError):
            operation()
    assert not database.path.exists()


def test_unknown_id_and_sql_text_are_not_management_targets(store):
    database, _ = store
    database.issue("friend")
    for operation in [database.renew, database.revoke]:
        with pytest.raises(ValueError, match="not found"):
            operation("' OR 1=1 --")
    assert database.list_tokens()[0].status() == "active"


def test_cli_only_creation_outputs_secret(store, monkeypatch, capsys):
    database, _ = store

    def run(*arguments):
        monkeypatch.setattr(sys, "argv", ["server.auth", *arguments])
        auth.main()
        return capsys.readouterr()

    created = run("issue-token", "--subject", "同学")
    token = created.out.strip()
    summary = json.loads(created.err)
    assert database.authenticate(token).id == summary["id"]
    assert set(summary) == {"id", "subject", "expires_at", "status"}
    assert summary["expires_at"].endswith("Z")
    for arguments in [("list-tokens",), ("renew-token", "--id", summary["id"], "--days", "2"),
                      ("revoke-token", "--id", summary["id"])]:
        result = run(*arguments)
        assert token not in result.out + result.err
        assert "token_hash" not in result.out and "rag_pat_" not in result.out
        json.loads(result.out)
    with pytest.raises(SystemExit) as failure:
        run("renew-token", "--id", summary["id"])
    assert failure.value.code == 1
    assert "Revoked" in capsys.readouterr().err


def test_client_sends_opaque_token_only_in_header(store, tmp_path, monkeypatch):
    from client import client
    database, _ = store
    token, _ = database.issue("friend")
    token_file = tmp_path / "access.token"
    token_file.write_text(token + "\n", encoding="utf-8")
    seen = {}

    def post(url, **kwargs):
        import httpx
        seen.update(kwargs)
        return httpx.Response(200, json={}, request=httpx.Request("POST", url))

    monkeypatch.setattr(client.httpx, "post", post)
    client.query_single("question", token_file=str(token_file), collection="example")
    assert seen["headers"] == {"Authorization": "Bearer " + token}
    assert seen["json"] == {"query": "question", "collection": "example"}
