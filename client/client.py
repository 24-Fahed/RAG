"""CLI client for querying and indexing the RAG knowledge base."""

import argparse
import os
from pathlib import Path

import httpx

_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument(
    "--mode",
    default="local",
    choices=["local", "staging", "production"],
    help="Deployment mode: local / staging / production (default: local)",
)
_pre_args, _ = _pre_parser.parse_known_args()

from client.config import load_config

load_config(_pre_args.mode)

from client.config import RAG_SERVER_URL


def _preview(text: str, limit: int = 200) -> str:
    text = text.replace("\n", "\\n").strip()
    return text[:limit] + ("..." if len(text) > limit else "")


def _print_documents(title: str, documents: list[dict], *, limit: int = 5) -> None:
    print(f"\n{title} ({len(documents)}):")
    if not documents:
        print("  <empty>")
        return

    for idx, doc in enumerate(documents[:limit], start=1):
        print(f"  [{idx}] score={doc.get('score', 0):.4f} metadata={doc.get('metadata', {})}")
        print(f"      {_preview(doc.get('content', ''))}")


def _auth_headers(token_file: str | None) -> dict[str, str]:
    if not token_file:
        return {}
    token = Path(token_file).read_text(encoding="utf-8").strip()
    if not token or any(c.isspace() for c in token):
        raise ValueError("Token file must contain one nonempty Bearer token")
    return {"Authorization": f"Bearer {token}"}


def query_single(query: str, *, token_file: str | None = None, **kwargs) -> None:
    """Send a single query to the RAG knowledge base service."""
    resp = httpx.post(f"{RAG_SERVER_URL}/api/query", json={"query": query, **kwargs}, headers=_auth_headers(token_file), timeout=120.0)
    resp.raise_for_status()
    data = resp.json()

    retrieved = data.get("retrieved_documents", [])
    reranked = data.get("reranked_documents", [])
    repacked_context = data.get("repacked_context", "")
    compressed_context = data.get("compressed_context", "")

    print(f"\n{'=' * 72}")
    print(f"Query: {query}")
    print(f"Retrieved documents: {len(retrieved)}")
    print(f"Reranked documents:  {len(reranked)}")
    print(f"Classification:      {data.get('classification_label')}")
    if data.get("hyde_document"):
        print(f"HyDE:                {_preview(data['hyde_document'], 120)}")
    print(f"Repacked chars:      {len(repacked_context)}")
    print(f"Compressed chars:    {len(compressed_context)}")
    print(f"{'=' * 72}")

    _print_documents("Retrieved", retrieved)
    _print_documents("Reranked", reranked)

    if repacked_context:
        print("\nRepacked context:")
        print(_preview(repacked_context, 500))

    if compressed_context:
        print("\nCompressed context:")
        print(_preview(compressed_context, 500))


def interactive_mode(**kwargs) -> None:
    """Interactive query loop."""
    print("=" * 72)
    print("RAG Knowledge Base Client")
    print(f"Server: {RAG_SERVER_URL}")
    print("Type 'quit' or 'exit' to stop.")
    print("=" * 72)

    while True:
        try:
            query = input("\nQuery> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if query.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break
        if not query:
            continue

        try:
            query_single(query, **kwargs)
        except httpx.ConnectError:
            print(f"ERROR: Cannot connect to {RAG_SERVER_URL}")
        except httpx.HTTPStatusError as exc:
            print(f"ERROR: Server returned {exc.response.status_code}: {exc.response.text[:200]}")
        except Exception as exc:
            print(f"ERROR: {exc}")


def index_documents(
    data_path: str,
    collection: str,
    embedding_model: str,
    embedding_dim: int,
    chunk_size: int = 512,
    chunk_overlap: int = 20,
    token_file: str | None = None,
) -> None:
    """Upload files to the RAG knowledge base for indexing."""
    if not os.path.exists(data_path):
        print(f"ERROR: Path not found: {data_path}")
        return

    files = []
    if os.path.isfile(data_path):
        files.append(open(data_path, "rb"))
    else:
        for fname in os.listdir(data_path):
            fpath = os.path.join(data_path, fname)
            if os.path.isfile(fpath):
                files.append(open(fpath, "rb"))

    if not files:
        print("ERROR: No files found to upload.")
        return

    try:
        resp = httpx.post(
            f"{RAG_SERVER_URL}/api/index",
            files=[("files", f) for f in files],
            headers=_auth_headers(token_file),
            data={
                "collection": collection,
                "embedding_model": embedding_model,
                "embedding_dim": embedding_dim,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
            },
            timeout=300.0,
        )
        resp.raise_for_status()
        data = resp.json()
        print(f"Status:      {data['status']}")
        print(f"Collection:  {data['collection']}")
        print(f"Chunks:      {data['document_count']}")
        if data.get("message"):
            print(f"Message:     {data['message']}")
    finally:
        for f in files:
            f.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG knowledge base client", parents=[_pre_parser])
    parser.add_argument("--token-file", required=True, help="File containing the Bearer token")
    sub = parser.add_subparsers(dest="command")

    q = sub.add_parser("query", help="Query the RAG knowledge base")
    q.add_argument("query", nargs="?", default=None, help="Query text")
    q.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    q.add_argument("--collection", required=True, help="Collection name")
    q.add_argument(
        "--search-method",
        required=True,
        choices=["original", "hyde", "hybrid", "hyde_with_hybrid", "bm25"],
        help="Retrieval method",
    )
    q.add_argument(
        "--rerank-method",
        required=True,
        choices=["monot5", "bge", "tilde", "rankllama"],
        help="Reranker model",
    )
    q.add_argument("--rerank-model", required=True, help="Model ID or Worker path")
    q.add_argument("--embedding-model")
    q.add_argument("--embedding-dim", type=int)
    q.add_argument("--hyde-provider", choices=["local", "siliconflow"])
    q.add_argument("--hyde-model")
    q.add_argument("--hyde-max-out-len", type=int, default=100)
    q.add_argument("--compression-model", required=True)
    q.add_argument("--classification-enabled", action="store_true")
    q.add_argument("--classification-model")
    q.add_argument("--classification-weights-path")
    q.add_argument("--top-k", type=int, default=10, help="Number of final documents to keep")
    q.add_argument(
        "--repack-method",
        default="sides",
        choices=["compact", "compact_reverse", "sides"],
        help="Context repacking method",
    )
    q.add_argument(
        "--compression-method",
        required=True,
        choices=["recomp_extractive", "recomp_abstractive", "llmlingua", "xprovence"],
        help="Context compression method",
    )
    q.add_argument("--compression-ratio", type=float, default=0.6, help="Compression ratio")
    q.add_argument("--hybrid-alpha", type=float, default=0.3, help="Sparse retrieval weight")
    q.add_argument("--search-k", type=int, default=100, help="Initial retrieval depth")

    idx = sub.add_parser("index", help="Upload documents into the knowledge base")
    idx.add_argument("path", help="File or directory path")
    idx.add_argument("--collection", required=True)
    idx.add_argument("--embedding-model", required=True)
    idx.add_argument("--embedding-dim", type=int, required=True)
    idx.add_argument("--chunk-size", type=int, default=512)
    idx.add_argument("--chunk-overlap", type=int, default=20)

    args = parser.parse_args()

    if args.command == "query":
        query_kwargs = {
            "token_file": args.token_file,
            "collection": args.collection,
            "search_method": args.search_method,
            "rerank_method": args.rerank_method,
            "rerank_model": args.rerank_model,
            "embedding_model": args.embedding_model,
            "embedding_dim": args.embedding_dim,
            "hyde_provider": args.hyde_provider,
            "hyde_model": args.hyde_model,
            "hyde_max_out_len": args.hyde_max_out_len,
            "compression_model": args.compression_model,
            "classification_enabled": args.classification_enabled,
            "classification_model": args.classification_model,
            "classification_weights_path": args.classification_weights_path,
            "top_k": args.top_k,
            "repack_method": args.repack_method,
            "compression_method": args.compression_method,
            "compression_ratio": args.compression_ratio,
            "hybrid_alpha": args.hybrid_alpha,
            "search_k": args.search_k,
        }
        query_kwargs = {key: value for key, value in query_kwargs.items() if value is not None}

        if args.interactive or args.query is None:
            interactive_mode(**query_kwargs)
        else:
            query_single(args.query, **query_kwargs)
    elif args.command == "index":
        index_documents(args.path, args.collection, args.embedding_model, args.embedding_dim, args.chunk_size, args.chunk_overlap, token_file=args.token_file)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
