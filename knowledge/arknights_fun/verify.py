"""Validate the corpus and exercise the project's real loader/splitter/BM25 offline.

No HTTP requests, model downloads, credentials, Milvus, or deployed collections.
Run from any directory: python knowledge/arknights_fun/verify.py
Use --report PATH to save the detailed, reproducible result.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[1]
sys.path.insert(0, str(ROOT))

from rag_langgraph.indexing.loader import load_from_directory, load_from_json
from rag_langgraph.indexing.splitter import split_documents
from server.services.bm25_search import BM25Index


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify() -> dict:
    corpus_path = PACKAGE / "corpus" / "documents.json"
    raw = read_json(corpus_path)
    source_list = read_json(PACKAGE / "sources.json")["sources"]
    queries = read_json(PACKAGE / "queries.json")
    sources = {s["id"]: s for s in source_list}
    checks = []

    def check(name: str, passed: bool):
        checks.append({"name": name, "passed": bool(passed)})

    ids = [d["id"] for d in raw]
    categories = Counter(d["category"] for d in raw)
    check("unique_document_ids", len(ids) == len(set(ids)))
    check("unique_titles", len(raw) == len({d["title"] for d in raw}))
    check("at_least_60_independent_entries", sum(d["category"] != "meta" for d in raw) >= 60)
    check("three_navigation_cards", categories["meta"] == 3)
    check("five_content_categories", set(categories) == {
        "meta", "character", "gameplay", "enemy_stage", "story", "community"})
    check("unique_source_ids", len(sources) == len(source_list))
    check("source_references_resolve", all(
        d["source_ids"] and set(d["source_ids"]) <= sources.keys() for d in raw))
    check("aliases_in_searchable_body", all(
        d["aliases"] and all(alias in d["content"] for alias in d["aliases"]) for d in raw))
    check("primary_source_in_searchable_body", all(
        d["category"] == "meta" or sources[d["source_ids"][0]]["url"] in d["content"]
        for d in raw))
    check("evidence_and_rating_fields", all(
        d["evidence_status"] and d["origin_status"] and d["checked_on"]
        and d["content_rating"] == "non_adult" for d in raw))

    loaded = load_from_json(str(corpus_path))
    directory_loaded = load_from_directory(str(PACKAGE / "corpus"))
    check("loader_preserves_all_documents", len(loaded) == len(raw))
    check("import_directory_contains_only_corpus", directory_loaded == loaded)
    check("flat_metadata_survives_loader", all(
        d["metadata"].get("id") == original["id"] and "metadata" not in d["metadata"]
        for d, original in zip(loaded, raw)))
    chunks = split_documents(loaded, chunk_size=512, chunk_overlap=20)
    per_doc = Counter(chunk["metadata"]["id"] for chunk in chunks)
    check("one_self_contained_chunk_per_card", len(chunks) == len(raw)
          and all(per_doc[doc_id] == 1 for doc_id in ids))
    check("unique_hybrid_dedup_prefixes", len(chunks) == len({
        chunk["content"][:200] for chunk in chunks}))
    check("query_expected_ids_resolve", all(
        set(case["expected_ids"]) <= set(ids)
        for suite in ("title_smoke", "challenge") for case in queries[suite]))

    # BM25Index creates its directory, but this check writes no pickle/index file.
    # Keep it within the package rather than using deployment-configured storage.
    index = BM25Index(index_dir=str(PACKAGE / "verification" / "bm25"))
    index.build([chunk["content"] for chunk in chunks])
    lookup = {chunk["content"]: chunk["metadata"]["id"] for chunk in chunks}
    top_k = queries["top_k"]
    results = {}
    for suite in ("title_smoke", "challenge"):
        suite_results = []
        for case in queries[suite]:
            hits = index.search(case["query"], top_k=top_k)
            retrieved = [lookup[hit["content"]] for hit in hits]
            expected = case["expected_ids"]
            ranks = {doc_id: retrieved.index(doc_id) + 1 if doc_id in retrieved else None
                     for doc_id in expected}
            suite_results.append({
                "group": case.get("group", "title_smoke"),
                "query": case["query"], "expected_ids": expected,
                "retrieved_ids": retrieved, "expected_ranks": ranks,
                "all_expected_in_top_k": all(rank is not None for rank in ranks.values()),
                "scores": [round(hit["score"], 6) for hit in hits],
            })
        passed = sum(r["all_expected_in_top_k"] for r in suite_results)
        results[suite] = {"passed": passed, "total": len(suite_results),
                          "all_expected_recall_at_k": passed / len(suite_results),
                          "cases": suite_results}

    structural_passed = all(c["passed"] for c in checks)
    all_queries_passed = all(s["passed"] == s["total"] for s in results.values())
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "corpus_sha256": hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
        "method": "Existing loader + 512/20 splitter + real rank_bm25 BM25Index, no mocks",
        "scope": "Offline structure and BM25 candidate recall only; no dense retrieval, reranking, compression, answer generation or deployment validation",
        "document_count": len(raw), "independent_entry_count": len(raw) - categories["meta"],
        "chunk_count": len(chunks), "source_count": len(sources),
        "categories": dict(sorted(categories.items())), "top_k": top_k,
        "structural_passed": structural_passed, "checks": checks,
        "retrieval": results, "passed": structural_passed and all_queries_passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, help="Save detailed JSON results at this path")
    args = parser.parse_args()
    report = verify()
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "passed": report["passed"], "structural_passed": report["structural_passed"],
        "documents": report["document_count"], "chunks": report["chunk_count"],
        "retrieval": {suite: {k: v for k, v in data.items() if k != "cases"}
                      for suite, data in report["retrieval"].items()},
        "failed_queries": [case for suite in report["retrieval"].values()
                           for case in suite["cases"] if not case["all_expected_in_top_k"]],
    }, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
