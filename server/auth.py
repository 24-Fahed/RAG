"""Authlib RFC 9068 access tokens and FastAPI Bearer authentication.

CLI: python -m server.auth init-keys --directory /secure/rag-auth
     python -m server.auth generate-token --subject client --expires-in 86400
"""

import argparse
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from authlib.oauth2 import OAuth2Error
from authlib.oauth2.rfc9068 import JWTBearerTokenGenerator, JWTBearerTokenValidator
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from joserfc.jwk import RSAKey
from joserfc.errors import JoseError

ISSUER = "rag-server-cli"
AUDIENCE = "rag-server-api"
SCOPE = "rag:api"
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


@lru_cache(maxsize=1)
def get_validator() -> AccessTokenValidator:
    return AccessTokenValidator(load_key("RAG_AUTH_PUBLIC_KEY_FILE"))


def require_access_token(
    credentials: HTTPAuthorizationCredentials | None = Security(bearer),
) -> dict:
    if credentials is None:
        raise HTTPException(401, "Bearer access token required", headers={"WWW-Authenticate": "Bearer"})
    try:
        validator = get_validator()
        token = validator.authenticate_token(credentials.credentials)
        validator.validate_token(token, [SCOPE], None)
        return dict(token)
    except AuthConfigurationError as exc:
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
    args = parser.parse_args()
    try:
        if args.command == "init-keys":
            init_keys(args.directory)
            print(f"Keys created in {args.directory}")
        else:
            print(generate_token(args.subject, args.expires_in))
    except (AuthConfigurationError, ValueError, OSError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
