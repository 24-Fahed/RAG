"""Locally managed opaque tokens and compatible Authlib JWT authentication.

CLI: python -m server.auth init-keys --directory /secure/rag-auth
     python -m server.auth generate-token --subject client --expires-in 86400
     python -m server.auth issue-token --subject friend --days 30
"""

import argparse
from datetime import datetime, timezone
import json
import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from authlib.oauth2 import OAuth2Error
from authlib.oauth2.rfc6750 import BearerTokenValidator
from authlib.oauth2.rfc9068 import JWTBearerTokenGenerator, JWTBearerTokenValidator
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from joserfc.jwk import RSAKey
from joserfc.errors import JoseError

from server.token_store import TOKEN_PREFIX, TOKEN_SCOPE, TokenStoreError, configured_store

ISSUER = "rag-server-cli"
AUDIENCE = "rag-server-api"
SCOPE = TOKEN_SCOPE
bearer = HTTPBearer(auto_error=False, scheme_name="RAGAccessToken")


class AuthConfigurationError(RuntimeError):
    pass


def load_key(variable: str) -> dict:
    path = os.getenv(variable, "")
    if not path:
        raise AuthConfigurationError(f"{variable} is not configured")
    try:
        key = RSAKey.import_key(Path(path).read_bytes(), parameters={"alg": "RS256", "use": "sig"})
        return key.as_dict(private=variable == "RAG_AUTH_PRIVATE_KEY_FILE")
    except (OSError, ValueError, JoseError) as exc:
        raise AuthConfigurationError(f"Cannot load {variable}") from exc


class AccessTokenGenerator(JWTBearerTokenGenerator):
    def get_jwks(self):
        return load_key("RAG_AUTH_PRIVATE_KEY_FILE")

    def get_audiences(self, client, user, scope):
        return AUDIENCE


class AccessTokenValidator(JWTBearerTokenValidator):
    def __init__(self, key: dict):
        super().__init__(issuer=ISSUER, resource_server=AUDIENCE)
        self.key = key

    def get_jwks(self):
        return self.key


class OpaqueTokenValidator(BearerTokenValidator):
    def authenticate_token(self, token_string):
        return configured_store().authenticate(token_string)


@lru_cache(maxsize=1)
def get_validator() -> AccessTokenValidator:
    return AccessTokenValidator(load_key("RAG_AUTH_PUBLIC_KEY_FILE"))


def require_access_token(
    credentials: HTTPAuthorizationCredentials | None = Security(bearer),
) -> dict:
    if credentials is None:
        raise HTTPException(401, "Bearer access token required", headers={"WWW-Authenticate": "Bearer"})
    try:
        opaque = credentials.credentials.startswith(TOKEN_PREFIX)
        validator = OpaqueTokenValidator() if opaque else get_validator()
        token = validator.authenticate_token(credentials.credentials)
        validator.validate_token(token, [SCOPE], None)
        return token.claims() if opaque else dict(token)
    except (AuthConfigurationError, TokenStoreError) as exc:
        # Missing configuration can never silently disable authentication.
        raise HTTPException(503, "API authentication is not configured") from exc
    except OAuth2Error as exc:
        raise HTTPException(exc.status_code, "Invalid or expired access token", headers={
            "WWW-Authenticate": "Bearer",
        }) from exc


@dataclass(frozen=True)
class CLIClient:
    subject: str

    def get_client_id(self):
        return self.subject


def generate_token(subject: str, expires_in: int = 86400) -> str:
    if not subject.strip() or expires_in <= 0:
        raise ValueError("subject must be nonempty and expires-in must be positive")
    generator = AccessTokenGenerator(issuer=ISSUER, alg="RS256", expires_generator=expires_in)
    return generator.access_token_generator(CLIClient(subject), "client_credentials", None, SCOPE)


def init_keys(directory: Path):
    private_path, public_path = directory / "private.pem", directory / "public.pem"
    if private_path.exists() or public_path.exists():
        raise ValueError("Key files already exist; refusing to overwrite existing keys")
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                serialization.NoEncryption())
    public = key.public_key().public_bytes(serialization.Encoding.PEM,
                                          serialization.PublicFormat.SubjectPublicKeyInfo)
    with os.fdopen(os.open(private_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as f:
        f.write(private)
    with public_path.open("xb") as f:
        f.write(public)


def main():
    parser = argparse.ArgumentParser(description="Manage RAG API access tokens (Authlib)")
    commands = parser.add_subparsers(dest="command", required=True)
    keys = commands.add_parser("init-keys", help="Create RSA keys without overwriting existing files")
    keys.add_argument("--directory", type=Path, required=True)
    issue = commands.add_parser("generate-token", help="Print a signed access token to stdout")
    issue.add_argument("--subject", default="rag-client")
    issue.add_argument("--expires-in", type=int, default=86400, help="Lifetime in seconds (default: 86400)")
    opaque_issue = commands.add_parser("issue-token", help="Create an opaque token; print its secret once")
    opaque_issue.add_argument("--subject", required=True, help="Friend or purpose, used as a management label")
    opaque_issue.add_argument("--days", type=int, default=30, help="Lifetime in 24-hour days (default: 30)")
    commands.add_parser("list-tokens", help="List token IDs, subjects, UTC expiry and status; never secrets")
    renew = commands.add_parser("renew-token", help="Extend an unrevoked token without changing its secret")
    renew.add_argument("--id", dest="token_id", required=True)
    renew.add_argument("--days", type=int, default=30, help="Additional 24-hour days (default: 30)")
    revoke = commands.add_parser("revoke-token", help="Permanently revoke a token by ID")
    revoke.add_argument("--id", dest="token_id", required=True)
    args = parser.parse_args()
    try:
        if args.command == "init-keys":
            init_keys(args.directory)
            print(f"Keys created in {args.directory}")
        elif args.command == "generate-token":
            print(generate_token(args.subject, args.expires_in))
        elif args.command == "issue-token":
            token, record = configured_store().issue(args.subject, args.days)
            # Keep stdout suitable for redirecting straight into a client token file.
            print(json.dumps(_token_summary(record), ensure_ascii=False), file=sys.stderr)
            print(token)
        elif args.command == "list-tokens":
            print(json.dumps([_token_summary(record) for record in configured_store().list_tokens()],
                             ensure_ascii=False, indent=2))
        elif args.command == "renew-token":
            print(json.dumps(_token_summary(configured_store().renew(args.token_id, args.days)), ensure_ascii=False))
        elif args.command == "revoke-token":
            print(json.dumps(_token_summary(configured_store().revoke(args.token_id)), ensure_ascii=False))
    except (AuthConfigurationError, TokenStoreError, ValueError, OSError) as exc:
        parser.exit(1, f"{exc}\n")


def _token_summary(record) -> dict:
    return {"id": record.id, "subject": record.subject,
            "expires_at": datetime.fromtimestamp(record.expires_at, timezone.utc).isoformat().replace("+00:00", "Z"),
            "status": record.status()}


if __name__ == "__main__":
    main()
