import json

import pytest

from recall import eval as ev
from recall import sample_data
from recall.cli import main as cli_main
from recall.indexer import index_folder
from recall.search import SemanticUnavailable


def fake_search_factory(table):
    def fake_search(con, q, mode, top_files=5):
        return table[q]
    return fake_search


def r(name, page):
    return {"path": f"/x/{name}", "page": page, "snippet": "", "why": ""}


def test_metric_math(monkeypatch):
    table = {
        "q1": [r("a.pdf", 1)],                                   # rank 1, page ok
        "q2": [r("z.pdf", 1), r("b.pdf", 9)],                    # rank 2, page wrong
        "q3": [r("z.pdf", 1)],                                   # miss
        "q4": [r("c.pdf", 2), r("d.pdf", 1)],                    # rank 2 via list of expected, no pages
    }
    monkeypatch.setattr(ev, "search", fake_search_factory(table))
    qs = [{"query": "q1", "expected": "a.pdf", "pages": [1], "type": "lexical"},
          {"query": "q2", "expected": "b.pdf", "pages": [3], "type": "paraphrase"},
          {"query": "q3", "expected": "c.pdf", "type": "paraphrase"},
          {"query": "q4", "expected": ["d.pdf", "e.pdf"], "type": "lexical"}]
    out = ev.evaluate_full(None, qs, "keyword", k=5)
    assert out["n"] == 4 and out["recall_at_k"] == pytest.approx(3 / 4)
    assert out["mrr"] == pytest.approx((1 + 0.5 + 0 + 0.5) / 4)
    assert out["top1_accuracy"] == pytest.approx(1 / 4)
    assert out["page_at_k"] == pytest.approx(1 / 2)                      # only q1, q2 specify pages
    assert out["by_type"]["lexical"]["recall_at_k"] == 1.0
    assert out["by_type"]["lexical"]["top1_accuracy"] == pytest.approx(1 / 2)
    assert out["by_type"]["paraphrase"]["top1_accuracy"] == 0.0
    assert out["by_type"]["paraphrase"]["mrr"] == pytest.approx(0.25)
    assert ev.evaluate(None, qs, "keyword") == (out["recall_at_k"], out["mrr"])


def test_sample_dataset_is_internally_consistent(tmp_path):
    n_pdf, n_q = sample_data.generate(str(tmp_path / "pdfs"), str(tmp_path / "q.json"))
    qs = json.load(open(tmp_path / "q.json"))
    assert n_q == len(qs) >= 30 and n_pdf == len(sample_data.DOCS)
    for item in qs:
        pages = sample_data.DOCS[item["expected"]]
        assert all(1 <= p <= len(pages) for p in item["pages"])
        assert item["type"] in ("lexical", "paraphrase")


def test_broad_sample_dataset_covers_project_skills_and_contents(tmp_path):
    n_pdf, n_q = sample_data.generate_broad(str(tmp_path / "pdfs"), str(tmp_path / "q.json"))
    with open(tmp_path / "q.json") as fh:
        queries = json.load(fh)

    assert n_pdf == len(sample_data.BROAD_DOCS)
    assert n_q == len(queries) == 8
    assert {item["type"] for item in queries} == {
        "projects", "technologies", "projects+technologies", "skills", "document-contents"
    }
    assert all(item["expected"] in sample_data.BROAD_DOCS for item in queries)


def test_eval_cli_reports_semantic_as_not_measured_without_embeddings(tmp_path, capsys, no_embedder):
    sample_data.generate(str(tmp_path / "pdfs"), str(tmp_path / "q.json"))
    db = str(tmp_path / "i.db")
    assert cli_main(["--db", db, "index", str(tmp_path / "pdfs"), "--no-embed"]) == 0
    capsys.readouterr()
    out_json = tmp_path / "res.json"
    ev.main([str(tmp_path / "q.json"), db, "--json", str(out_json)])
    text = capsys.readouterr().out
    assert "NOT MEASURED" in text and "keyword" in text
    rep = json.load(open(out_json))
    assert rep["modes"]["keyword"]["measured"] and not rep["modes"]["semantic"]["measured"]
    assert not rep["modes"]["hybrid"]["measured"]
    assert 0 <= rep["modes"]["keyword"]["recall_at_k"] <= 1


def test_cli_embed_backfills_without_reindexing_pdfs(tmp_path, docs, make_pdf, capsys, fake):
    make_pdf(docs / "a.pdf", ["A short passage about database indexing"])
    db = str(tmp_path / "repair.db")
    assert cli_main(["--db", db, "index", str(docs), "--no-embed"]) == 0
    capsys.readouterr()

    assert cli_main(["--db", db, "embed"]) == 0

    out = capsys.readouterr().out
    assert "chunks embedded: 1" in out
    assert "chunks still pending: 0" in out
    assert cli_main(["--db", db, "search", "database indexing", "--mode", "semantic"]) == 0
    assert "a.pdf" in capsys.readouterr().out


def test_cli_embed_reports_missing_model(tmp_path, capsys, no_embedder):
    rc = cli_main(["--db", str(tmp_path / "no-model.db"), "embed"])
    assert rc == 2
    assert "sentence-transformers is not installed" in capsys.readouterr().err


def test_eval_runs_all_modes_with_embedder(tmp_path, capsys, fake):
    sample_data.generate(str(tmp_path / "pdfs"), str(tmp_path / "q.json"))
    db = str(tmp_path / "i.db")
    cli_main(["--db", db, "index", str(tmp_path / "pdfs")])
    capsys.readouterr()
    out_json = tmp_path / "res.json"
    ev.main([str(tmp_path / "q.json"), db, "--json", str(out_json)])
    rep = json.load(open(out_json))
    assert all(rep["modes"][m]["measured"] for m in ("keyword", "semantic", "hybrid"))


# ---------------------------------------------------------------- CLI
def test_cli_index_search_status(tmp_path, docs, make_pdf, capsys, no_embedder):
    make_pdf(docs / "a.pdf", ["the lost update problem explained"])
    db = str(tmp_path / "c.db")
    assert cli_main(["--db", db, "index", str(docs), "--no-embed"]) == 0
    assert cli_main(["--db", db, "search", "lost update", "--mode", "keyword"]) == 0
    out = capsys.readouterr().out
    assert "a.pdf" in out and "page 1" in out
    assert cli_main(["--db", db, "search", "zzzzqqq"]) == 0
    assert "No results." in capsys.readouterr().out
    assert cli_main(["--db", db, "status"]) == 0
    assert "files: 1" in capsys.readouterr().out


def test_cli_exit_codes_for_failures_and_errors(tmp_path, docs, make_pdf, capsys, no_embedder):
    db = str(tmp_path / "c.db")
    (docs / "bad.pdf").write_bytes(b"junk")
    assert cli_main(["--db", db, "index", str(docs), "--no-embed"]) == 1
    err = capsys.readouterr().err
    assert "failed to index" in err
    cli_main(["--db", db, "status"])
    assert "bad.pdf" in capsys.readouterr().out
    assert cli_main(["--db", db, "index", str(tmp_path / "nope"), "--no-embed"]) == 2
    assert "left alone" in capsys.readouterr().err
    assert cli_main(["--db", db, "search", "x", "--mode", "semantic"]) == 2
    assert "semantic search unavailable" in capsys.readouterr().err


def test_paraphrase_queries_really_avoid_the_target_vocabulary():
    """Guards the benchmark's value: if paraphrase queries copied the page's words, keyword search
    would trivially hit 100% and the comparison would say nothing."""
    import re
    stop = set("the a an of to in and or is are was for with that this it as on by from then each".split())
    toks = lambda t: {w[:5] for w in re.findall(r"[a-z0-9]+", t.lower()) if w not in stop and len(w) > 2}
    for q, f, pg, kind in sample_data.QUERIES:
        shared = toks(q) & toks(sample_data.DOCS[f][pg - 1])
        if kind == "paraphrase":
            assert len(shared) <= 1, (q, shared)
        else:
            assert len(shared) >= 1, (q, shared)


def test_sample_dataset_includes_natural_language_project_query():
    assert any(q == "What student-facing tool was built, and what programming stack powers it?"
               and expected == "portfolio.pdf" and kind == "paraphrase"
               for q, expected, _, kind in sample_data.QUERIES)
