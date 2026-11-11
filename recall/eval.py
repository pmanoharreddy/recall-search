"""Retrieval evaluation.

    python -m recall.eval queries.json [recall.db] [--k 5] [--json results.json]

queries.json: [{"query": "...", "expected": "file.pdf" | ["a.pdf","b.pdf"],
                "pages": [2]        (optional, enables page accuracy),
                "type": "lexical"   (optional, results are broken down by type)}]

Metrics (file level, top-k files returned):
  Recall@k  fraction of queries whose expected file appears in the top k
  MRR       mean of 1/rank of the first expected file (0 if absent)
  Page@k    fraction of queries where, among top-k results, an expected file was returned WITH a
            matching page (only over queries that specify pages)
A mode that cannot run (e.g. semantic without sentence-transformers or without embeddings) is
reported as NOT MEASURED, never as 0%.
"""
import argparse
import json
import os
import sys
import time

from .db import connect
from .search import MODES, SemanticUnavailable, search


def _expected(item):
    e = item["expected"]
    return set(e) if isinstance(e, (list, tuple)) else {e}


def evaluate_full(con, queries, mode, k=5):
    """Returns dict of metrics, with per-type breakdown. Raises SemanticUnavailable if mode can't run."""
    def blank():
        return dict(n=0, hits=0, top1_hits=0, rr=0.0, page_n=0, page_hits=0)
    overall, by_type = blank(), {}
    t0 = time.perf_counter()
    for item in queries:
        res = search(con, item["query"], mode, top_files=k)
        exp = _expected(item)
        names = [os.path.basename(r["path"]) for r in res]
        rank = next((i + 1 for i, n in enumerate(names) if n in exp), None)
        page_ok = None
        if item.get("pages"):
            page_ok = any(os.path.basename(r["path"]) in exp and r["page"] in item["pages"] for r in res)
        for bucket in (overall, by_type.setdefault(item.get("type", "all"), blank())):
            bucket["n"] += 1
            if rank:
                bucket["hits"] += 1
                bucket["top1_hits"] += int(rank == 1)
                bucket["rr"] += 1 / rank
            if page_ok is not None:
                bucket["page_n"] += 1
                bucket["page_hits"] += int(page_ok)
    elapsed = time.perf_counter() - t0

    def fin(b):
        return dict(n=b["n"], recall_at_k=b["hits"] / b["n"] if b["n"] else None,
                    top1_accuracy=b["top1_hits"] / b["n"] if b["n"] else None,
                    mrr=b["rr"] / b["n"] if b["n"] else None,
                    page_at_k=b["page_hits"] / b["page_n"] if b["page_n"] else None)
    out = fin(overall)
    out["k"] = k
    out["ms_per_query"] = 1000 * elapsed / max(len(queries), 1)
    out["by_type"] = {t: fin(b) for t, b in by_type.items()}
    return out


def evaluate(con, queries, mode, k=5):
    """Backward-compatible (recall, mrr) tuple."""
    r = evaluate_full(con, queries, mode, k)
    return r["recall_at_k"], r["mrr"]


def _pct(x):
    return "  n/a " if x is None else f"{x:6.1%}"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="recall.eval")
    ap.add_argument("queries"); ap.add_argument("db", nargs="?", default="recall.db")
    ap.add_argument("--k", type=int, default=5); ap.add_argument("--json")
    a = ap.parse_args(argv)
    con = connect(a.db)
    try:
        with open(a.queries) as fh:
            qs = json.load(fh)
        n_chunks = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        n_emb = con.execute("SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL").fetchone()[0]
        print(f"{len(qs)} queries | index: {n_chunks} chunks, {n_emb} with embeddings | k={a.k}")
        print(f"{'mode':<10}{'Recall@k':>10}{'MRR':>8}{'Top-1':>8}{'Page@k':>9}{'ms/q':>8}")
        report = dict(k=a.k, n_queries=len(qs), chunks=n_chunks, chunks_with_embeddings=n_emb, modes={})
        for m in MODES:
            try:
                if m != "keyword" and n_emb == 0:
                    raise SemanticUnavailable("no embeddings in the index")
                r = evaluate_full(con, qs, m, a.k)
            except SemanticUnavailable as e:
                print(f"{m:<10}  NOT MEASURED ({e})")
                report["modes"][m] = dict(measured=False, reason=str(e))
                continue
            print(f"{m:<10}{_pct(r['recall_at_k']):>10}{r['mrr']:>8.3f}"
                  f"{_pct(r['top1_accuracy']):>8}{_pct(r['page_at_k']):>9}{r['ms_per_query']:>8.1f}")
            for t, b in r["by_type"].items():
                if len(r["by_type"]) > 1:
                    print(f"  {t:<8}{_pct(b['recall_at_k']):>10}{b['mrr']:>8.3f}"
                          f"{_pct(b['top1_accuracy']):>8}{_pct(b['page_at_k']):>9}   (n={b['n']})")
            report["modes"][m] = dict(measured=True, **r)
        if a.json:
            with open(a.json, "w") as fh:
                json.dump(report, fh, indent=1)
        return 0
    finally:
        con.close()


if __name__ == "__main__":
    sys.exit(main())
