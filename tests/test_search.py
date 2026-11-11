import logging
import json
import math
import os

import numpy as np
import pytest

import recall.search as search_module
from recall import embed, vectors
from recall.eval import evaluate_full
from recall.indexer import index_folder
from recall.sample_data import generate_broad
from recall.search import SemanticUnavailable, rrf, search


@pytest.fixture
def corpus(con, docs, make_pdf, fake, quiet):
    make_pdf(docs / "cars.pdf", ["An automobile engine needs fuel and oil to run smoothly"])
    make_pdf(docs / "pets.pdf", ["The puppy chased the ball across the garden"])
    make_pdf(docs / "db.pdf", ["Transactions must be atomic.", "Lost update happens when writes overwrite each other"])
    index_folder(con, docs, log=quiet)
    return con


def names(results):
    return [os.path.basename(r["path"]) for r in results]


def test_keyword_search_finds_correct_file_and_page(corpus):
    r = search(corpus, "lost update overwrite", "keyword")
    assert names(r)[0] == "db.pdf" and r[0]["page"] == 2 and "keyword" in r[0]["why"]
    assert "Lost update" in r[0]["snippet"]


def test_keyword_does_not_match_synonyms_but_semantic_does(corpus):
    assert search(corpus, "car", "keyword") == []
    sem = search(corpus, "car", "semantic")
    assert names(sem)[0] == "cars.pdf" and "semantic" in sem[0]["why"]


def test_semantic_ranks_synonym_over_unrelated(corpus):
    assert names(search(corpus, "canine", "semantic"))[0] == "pets.pdf"


def test_natural_language_project_query_retrieves_synthetic_portfolio(con, docs, make_pdf, fake, quiet):
    make_pdf(docs / "portfolio.pdf", [
        "Developed a project named Cedar Ledger, a budgeting application using Python, Flask, and SQLite"])
    make_pdf(docs / "unrelated.pdf", ["College activities and engineering entrance exam cut-off ranks"])
    index_folder(con, docs, log=quiet)

    result = search(con, "Which projects did this person build, and what tools did they use?", "semantic")

    assert names(result)[0] == "portfolio.pdf"


def test_specific_project_query_keeps_its_original_embedding_text():
    query = "InterviewIQ AI-powered mock interview platform React Node Express MongoDB Gemini"
    assert search_module._semantic_query_text(query) == query


def test_broad_project_intent_and_multiple_passages_improve_retrieval_metrics(
        con, docs, make_pdf, fake, quiet, monkeypatch):
    queries = [
        "Which projects has this student developed, and what technologies were used?",
        "What projects did the portfolio describe and which technologies power them?",
        "Which applications and software stack are described in the project portfolio?",
    ]
    query = queries[0]
    assert search_module._semantic_query_text(query).startswith("student ")
    intent_queries = {search_module._semantic_query_text(text) for text in queries}
    intent_query = search_module._semantic_query_text(query)
    target_texts = [
        "Aster is a practice environment that lets candidates rehearse job interviews.",
        "The system combines a browser interface, a server, a database, and a generative AI service.",
    ]
    distractor_texts = [
        f"Engineering admission tables and student activities for department {n}."
        for n in range(5)
    ]
    make_pdf(docs / "portfolio.pdf", target_texts)
    for n, text in enumerate(distractor_texts):
        make_pdf(docs / f"college_{n}.pdf", [text])

    distractor_vector = np.array([0.5, 0.82, math.sqrt(1 - 0.5**2 - 0.82**2)],
                                 dtype=np.float32)
    encoded = {
        "dimension probe": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        **{text: np.array([1.0, 0.0, 0.0], dtype=np.float32) for text in queries},
        **{text: np.array([0.0, 1.0, 0.0], dtype=np.float32) for text in intent_queries},
        target_texts[0]: np.array([0.3, 0.8, math.sqrt(1 - 0.3**2 - 0.8**2)],
                                  dtype=np.float32),
        target_texts[1]: np.array([0.3, 0.8, -math.sqrt(1 - 0.3**2 - 0.8**2)],
                                  dtype=np.float32),
        **{text: distractor_vector for text in distractor_texts},
    }
    monkeypatch.setattr(embed, "encode", lambda texts: np.vstack([encoded[t.strip()] for t in texts]))
    index_folder(con, docs, log=quiet)
    labels = [{"query": text, "expected": "portfolio.pdf", "type": "paraphrase"}
              for text in queries]

    with monkeypatch.context() as m:
        m.setattr(search_module, "_semantic_query_text", lambda text: text)
        baseline = evaluate_full(con, labels, "semantic", k=5)
    assert baseline["n"] == 3
    assert baseline["recall_at_k"] == 0.0
    assert baseline["mrr"] == 0.0

    with monkeypatch.context() as m:
        m.setattr(search_module, "_semantic_query_text", lambda text: intent_query)
        m.setattr(search_module, "_document_evidence_score", lambda passage_scores: passage_scores[0])
        without_passage_evidence = evaluate_full(con, labels, "semantic", k=5)
    assert without_passage_evidence["recall_at_k"] == 0.0

    semantic = evaluate_full(con, labels, "semantic", k=5)
    hybrid = evaluate_full(con, labels, "hybrid", k=5)
    keyword = evaluate_full(con, labels, "keyword", k=5)
    assert semantic["recall_at_k"] == hybrid["recall_at_k"] == 1.0
    assert semantic["mrr"] == hybrid["mrr"] == 1.0
    assert keyword["recall_at_k"] == keyword["mrr"] == 0.0
    assert names(search(con, query, "semantic"))[0] == "portfolio.pdf"
    assert names(search(con, query, "hybrid"))[0] == "portfolio.pdf"


def test_broad_eval_questions_retrieve_labeled_synthetic_documents(
        con, tmp_path, fake, quiet):
    pdf_dir = tmp_path / "broad-pdfs"
    query_file = tmp_path / "broad-queries.json"
    generate_broad(str(pdf_dir), str(query_file))
    index_folder(con, str(pdf_dir), log=quiet)
    with open(query_file) as source:
        labels = json.load(source)

    assert len(labels) == 8
    for mode in ("semantic", "hybrid"):
        result = evaluate_full(con, labels, mode, k=5)
        assert result["recall_at_k"] == 1.0


def test_hybrid_combines_both_signals(corpus):
    r = search(corpus, "car fuel", "hybrid")
    assert names(r)[0] == "cars.pdf" and r[0]["why"] == "keyword match + semantic match"
    r2 = search(corpus, "car", "hybrid")                              # semantic-only evidence still surfaces
    assert names(r2)[0] == "cars.pdf" and r2[0]["why"] == "semantic match"


def test_rrf_prefers_items_ranked_well_in_both_lists():
    order, scores = rrf([1, 2, 3], [3, 1, 4])
    assert order[:2] == [1, 3] and set(order) == {1, 2, 3, 4}
    assert scores[1] > scores[2] and scores[3] > scores[4]


def test_results_are_one_per_file_and_limited(corpus):
    assert len(search(corpus, "the", "keyword")) <= 5
    r = search(corpus, "transactions lost update writes atomic", "keyword")
    assert len(names(r)) == len(set(names(r)))


@pytest.mark.parametrize("mode", ["keyword", "semantic", "hybrid"])
@pytest.mark.parametrize("q", ["", "   ", None])
def test_empty_queries_return_nothing(corpus, mode, q):
    assert search(corpus, q, mode) == []


def test_stopword_only_and_no_match_queries(corpus):
    assert search(corpus, "the and of what", "keyword") == []
    assert search(corpus, "zzzzqqqq", "keyword") == []


@pytest.mark.parametrize("q", ['"unbalanced', "a AND OR NOT", "foo*", "x:y", "(((", "'; DROP TABLE chunks;--"])
def test_fts_special_characters_do_not_crash(corpus, q):
    search(corpus, q, "keyword")
    assert corpus.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 4


def test_search_on_empty_index(con, fake):
    for mode in ("keyword", "semantic", "hybrid"):
        assert search(con, "anything", mode) == []


def test_unknown_mode_is_rejected(corpus):
    with pytest.raises(ValueError, match="unknown mode"):
        search(corpus, "x", "fuzzy")


def test_semantic_mode_raises_clear_error_when_unavailable(corpus, no_embedder):
    with pytest.raises(SemanticUnavailable, match="sentence-transformers"):
        search(corpus, "car", "semantic")


def test_hybrid_falls_back_to_keyword_with_warning(corpus, no_embedder, caplog):
    with caplog.at_level(logging.WARNING, logger="recall"):
        r = search(corpus, "lost update", "hybrid")
    assert names(r)[0] == "db.pdf" and "keyword results only" in caplog.text


def test_query_embedding_failure_is_reported_not_swallowed(corpus, fake):
    fake.fail_all = True
    with pytest.raises(SemanticUnavailable, match="could not embed the query"):
        search(corpus, "car", "semantic")
    assert search(corpus, "lost update", "hybrid")                    # hybrid still serves keyword


def test_semantic_with_no_embeddings_warns(con, docs, make_pdf, fake, quiet, caplog):
    make_pdf(docs / "a.pdf", ["some content about sorting"])
    index_folder(con, docs, use_embeddings=False, log=quiet)
    with caplog.at_level(logging.WARNING, logger="recall"):
        assert search(con, "sorting", "semantic") == []
    assert "run `index` without --no-embed" in caplog.text
    assert "python -m recall.cli embed" in caplog.text
    assert search(con, "sorting", "hybrid")                           # keyword half still works


def test_partial_embeddings_warn_and_only_map_to_embedded_chunks(con, docs, make_pdf, fake, quiet, caplog):
    make_pdf(docs / "target.pdf", ["a vehicle engine needs fuel"])
    make_pdf(docs / "other.pdf", ["a puppy chased the ball"])
    index_folder(con, docs, log=quiet)
    con.execute("UPDATE chunks SET embedding=NULL WHERE file_id=(SELECT id FROM files WHERE path LIKE '%target.pdf')")
    con.commit()
    vectors.clear_cache()

    with caplog.at_level(logging.WARNING, logger="recall"):
        results = search(con, "car", "semantic")

    assert results and "other.pdf" in names(results)
    assert "semantic index is incomplete: 1 of 2 chunks" in caplog.text
    assert "python -m recall.cli embed" in caplog.text


# ---------------------------------------------------------------- vector cache
def test_vector_matrix_is_cached_between_queries_and_refreshed_after_indexing(corpus, docs, make_pdf, fake, quiet, monkeypatch):
    loads = []
    real = vectors.load_index
    monkeypatch.setattr(vectors, "load_index", lambda con, dim: loads.append(1) or real(con, dim))
    for q in ("car", "puppy", "canine", "automobile"):
        search(corpus, q, "semantic")
    assert len(loads) == 1                                            # not reloaded per query

    make_pdf(docs / "new.pdf", ["a brand new vehicle document"])
    index_folder(corpus, docs, log=quiet)
    r = search(corpus, "car", "semantic")
    assert len(loads) == 2 and "new.pdf" in names(r)                  # new data visible after indexing


def test_deleted_files_disappear_from_cached_semantic_results(corpus, docs, quiet):
    assert names(search(corpus, "car", "semantic"))[0] == "cars.pdf"
    (docs / "cars.pdf").unlink()
    index_folder(corpus, docs, log=quiet)
    assert "cars.pdf" not in names(search(corpus, "car", "semantic"))
    assert "cars.pdf" not in names(search(corpus, "car", "hybrid"))


# ---------------------------------------------------------------- keyword precision (regression)
@pytest.fixture
def noisy_corpus(con, docs, make_pdf, fake, quiet):
    make_pdf(docs / "cars.pdf", ["An automobile engine needs fuel"])
    make_pdf(docs / "noise.pdf", ["The node restarted after the long maintenance window yesterday evening while "
                                  "nobody was watching the dashboard every way up"])
    make_pdf(docs / "both.pdf", ["Wire each node to the switch"])
    index_folder(con, docs, log=quiet)
    return con


def test_single_incidental_word_is_not_a_keyword_match_for_longer_queries(noisy_corpus):
    q = "cheapest way to wire up every node together without loops"      # meaningful words: cheapest wire node loops
    found = names(search(noisy_corpus, q, "keyword"))
    assert "both.pdf" in found                                            # contains two of the words
    assert "noise.pdf" not in found                                       # contains only "node"


def test_short_queries_still_match_on_one_word(noisy_corpus):
    assert "noise.pdf" in names(search(noisy_corpus, "node", "keyword"))
    assert "both.pdf" in names(search(noisy_corpus, "wire node", "keyword"))


def test_filler_words_are_not_searched(noisy_corpus):
    assert search(noisy_corpus, "way up every without together", "keyword") == []


def test_hybrid_does_not_let_one_incidental_keyword_outrank_the_semantic_hit(noisy_corpus):
    # keyword leg: nothing has two of {car, garage, node, cheapest}; semantic leg: cars.pdf via synonym
    r = search(noisy_corpus, "car garage node cheapest", "hybrid")
    assert names(r)[0] == "cars.pdf"


# ---------------------------------------------------------------- similarity scores and optional cutoff
def test_semantic_results_report_similarity_scores_best_first(corpus):
    r = search(corpus, "car", "semantic")
    scores = [x["semantic_score"] for x in r]
    assert all(isinstance(s, float) for s in scores) and scores == sorted(scores, reverse=True)
    assert names(r)[0] == "cars.pdf" and scores[0] > scores[-1]
    assert all(x["semantic_score"] is None for x in search(corpus, "lost update", "keyword"))


def test_min_similarity_env_drops_weak_semantic_matches(corpus, monkeypatch):
    assert len(search(corpus, "car", "semantic")) == 3                    # default: no cutoff, always fills up
    monkeypatch.setenv("RECALL_MIN_SIMILARITY", "0.4")
    assert names(search(corpus, "car", "semantic")) == ["cars.pdf"]       # only the genuinely similar chunk
    assert names(search(corpus, "car", "hybrid")) == ["cars.pdf"]
    monkeypatch.setenv("RECALL_MIN_SIMILARITY", "0.99")
    assert search(corpus, "car", "semantic") == []                        # nothing is that similar: say so


def test_invalid_min_similarity_is_ignored_with_a_warning(corpus, monkeypatch, caplog):
    monkeypatch.setenv("RECALL_MIN_SIMILARITY", "abc")
    with caplog.at_level(logging.WARNING, logger="recall"):
        assert len(search(corpus, "car", "semantic")) == 3
    assert "invalid RECALL_MIN_SIMILARITY" in caplog.text


# ---------------------------------------------------------------- non-English queries
def test_query_terms_keep_non_ascii_words():
    from recall.search import _query_terms
    assert _query_terms("résumé café naïve") == ["résumé", "café", "naïve"]
    assert _query_terms("ಮರುಪಾವತಿ refund") == ["ಮರುಪಾವತಿ", "refund"]        # Kannada vowel signs stay attached


def test_keyword_search_finds_non_ascii_text(con):
    fid = con.execute("INSERT INTO files(path, sha256, mtime) VALUES('/x/t.pdf','h',0)").lastrowid
    con.execute("INSERT INTO chunks(file_id, page, text) VALUES(?,?,?)", (fid, 1, "No refund ಮರುಪಾವತಿ for the café résumé"))
    con.commit()
    for q in ("résumé", "café", "ಮರುಪಾವತಿ"):
        assert [r["page"] for r in search(con, q, "keyword")] == [1], q
