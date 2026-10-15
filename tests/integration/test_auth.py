"""Exercise the actual Server app's global auth dependency without inference."""

import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from joserfc import jwt
from joserfc.jwk import RSAKey

from server import auth, token_store


@pytest.fixture
def deployed_app(tmp_path, monkeypatch):
    # Import the real entrypoint, not a router-only FastAPI test substitute.
    from server.main import app
    monkeypatch.setenv("RAG_AUTH_TOKEN_DB_FILE", str(tmp_path / "auth" / "tokens.sqlite3"))
    monkeypatch.delenv("RAG_AUTH_PRIVATE_KEY_FILE", raising=False)
    monkeypatch.delenv("RAG_AUTH_PUBLIC_KEY_FILE", raising=False)
    auth.get_validator.cache_clear()
    yield TestClient(app), token_store.configured_store()
    auth.get_validator.cache_clear()


@pytest.mark.parametrize("path,method", [("/health", "GET"), ("/api/query", "POST"),
                                       ("/api/agent_query", "POST"), ("/api/index", "POST")])
def test_real_app_rejects_anonymous_requests(deployed_app, path, method):
    client, _ = deployed_app
    response = client.request(method, path)
    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"


def test_opaque_renewal_and_revocation_take_effect_without_restart(deployed_app, monkeypatch):
    client, store = deployed_app
    clock = [int(time.time())]
    monkeypatch.setattr(token_store, "time", SimpleNamespace(time=lambda: clock[0]))
    token, record = store.issue("friend", 1)
    headers = {"Authorization": "Bearer " + token}
    assert client.get("/health", headers=headers).status_code == 200
    clock[0] = record.expires_at
    assert client.get("/health", headers=headers).status_code == 401
    store.renew(record.id, 1)
    assert client.get("/health", headers=headers).status_code == 200
    store.revoke(record.id)
    assert client.get("/health", headers=headers).status_code == 401
    with pytest.raises(ValueError, match="Revoked"):
        store.renew(record.id)


def test_opaque_auth_gates_query_and_index(deployed_app):
    client, store = deployed_app
    token, _ = store.issue("friend")
    for path in ["/api/query", "/api/agent_query", "/api/index"]:
        # An authorized request reaches schema validation, not a model or storage.
        assert client.post(path, headers={"Authorization": "Bearer " + token}).status_code == 422
    bad = token[:-1] + ("A" if token[-1] != "A" else "B")
    assert client.get("/health", headers={"Authorization": "Bearer " + bad}).status_code == 401
    assert client.get("/health", headers={"Authorization": "Basic abc"}).status_code == 401


@pytest.mark.parametrize("configuration", ["missing", "relative", "absent", "corrupt", "empty"])
def test_bad_store_fails_closed(deployed_app, monkeypatch, tmp_path, configuration):
    client, store = deployed_app
    token, _ = store.issue("friend")
    if configuration == "missing":
        monkeypatch.delenv("RAG_AUTH_TOKEN_DB_FILE")
    elif configuration == "relative":
        monkeypatch.setenv("RAG_AUTH_TOKEN_DB_FILE", "relative.sqlite3")
    else:
        path = tmp_path / f"{configuration}.sqlite3"
        if configuration in {"corrupt", "empty"}:
            path.write_bytes(b"not a database" if configuration == "corrupt" else b"")
        monkeypatch.setenv("RAG_AUTH_TOKEN_DB_FILE", str(path))
    response = client.get("/health", headers={"Authorization": "Bearer " + token})
    assert response.status_code == 503
    assert response.json() == {"detail": "API authentication is not configured"}
    assert token not in response.text and str(tmp_path) not in response.text


@pytest.fixture
def jwt_keys(tmp_path, monkeypatch):
    directory = tmp_path / "rsa"
    auth.init_keys(directory)
    monkeypatch.setenv("RAG_AUTH_PRIVATE_KEY_FILE", str(directory / "private.pem"))
    monkeypatch.setenv("RAG_AUTH_PUBLIC_KEY_FILE", str(directory / "public.pem"))
    auth.get_validator.cache_clear()
    yield directory
    auth.get_validator.cache_clear()


def test_existing_jwt_works_without_token_store(deployed_app, jwt_keys, monkeypatch):
    client, _ = deployed_app
    monkeypatch.delenv("RAG_AUTH_TOKEN_DB_FILE")
    token = auth.generate_token("legacy-client", 300)
    assert client.get("/health", headers={"Authorization": "Bearer " + token}).status_code == 200
    claims = dict(jwt.decode(token, RSAKey.import_key(auth.load_key("RAG_AUTH_PUBLIC_KEY_FILE"))).claims)
    private = RSAKey.import_key(auth.load_key("RAG_AUTH_PRIVATE_KEY_FILE"))
    for update, expected in [({"exp": int(time.time()) - 120}, 401), ({"iss": "other"}, 401),
                             ({"aud": "other"}, 401), ({"scope": "other"}, 403),
                             ({"nbf": int(time.time()) + 600}, 401)]:
        invalid = jwt.encode({"alg": "RS256", "typ": "at+jwt"}, {**claims, **update}, private)
        assert client.get("/health", headers={"Authorization": "Bearer " + invalid}).status_code == expected
    pieces = token.split(".")
    pieces[2] = ("A" if pieces[2][0] != "A" else "B") + pieces[2][1:]
    for invalid in ["invalid", ".".join(pieces)]:
        assert client.get("/health", headers={"Authorization": "Bearer " + invalid}).status_code == 401


def test_jwt_missing_key_still_fails_closed(deployed_app):
    client, _ = deployed_app
    assert client.get("/health", headers={"Authorization": "Bearer invalid"}).status_code == 503


def test_public_docs_and_openapi_security_are_preserved(deployed_app):
    client, _ = deployed_app
    assert client.get("/docs").status_code == 200
    assert client.get("/redoc").status_code == 200
    schema = client.get("/openapi.json").json()
    assert schema["components"]["securitySchemes"]["RAGAccessToken"]["scheme"] == "bearer"
    for path in ["/health", "/api/query", "/api/agent_query", "/api/index"]:
        for operation in schema["paths"][path].values():
            assert operation["security"] == [{"RAGAccessToken": []}]
