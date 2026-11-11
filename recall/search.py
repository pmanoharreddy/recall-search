import logging
import os
import unicodedata

from . import embed, vectors

log = logging.getLogger("recall")

STOP = set("""
the a an of to in and or is are was were be been being am for with that this these those it its as on by from at into
onto over under up out off about above below between through during before after again further then once here there
when while if than so such not no nor only own same too very just also all any both each few more most other some every
i you he she we they them their his her our your me my who which what how where why whom
has have had do does did will would shall should can could may might must
without within because until against much many get got way ways together something anything somewhere
pdf file document explains explain one find show example
""".split())
RRF_K = 60
MODES = ("keyword", "semantic", "hybrid")
SEMANTIC_CANDIDATES = 200
PASSAGE_SUPPORT_WEIGHT = 0.05


class SemanticUnavailable(RuntimeError):
    """Semantic search was requested but cannot run (missing dependency, model failure)."""


CANDIDATES = 500          # keyword candidates examined before the coverage filter

PROJECT_INTENT_TERMS = {
    "project", "projects", "application", "applications", "product", "products",
    "build", "built", "develop", "developed", "create", "created",
}
TECHNOLOGY_INTENT_TERMS = {
    "technology", "technologies", "tool", "tools", "stack", "stacks", "framework",
    "frameworks", "language", "languages", "software", "programming",
}
QUESTION_OPENERS = {"what", "which", "who", "where", "how", "describe", "list"}
INTENT_ACTION_TERMS = {
    "build", "built", "develop", "developed", "create", "created", "implement",
    "implemented", "use", "used", "describe", "described", "power", "powers",
}
INTENT_CONTEXT_TERMS = {"portfolio", "facing"}


def _tokenize(text):
    """Lower-cased words made of letters, digits and combining marks, in ANY script. This matches how
    SQLite's unicode61 tokenizer splits text (marks are needed for e.g. Kannada vowel signs and accents)."""
    words, cur = [], []
    for ch in text.lower():
        if unicodedata.category(ch)[0] in "LNM":
            cur.append(ch)
        elif cur:
            words.append("".join(cur)); cur = []
    if cur:
        words.append("".join(cur))
    return words


def _query_terms(q):
    terms = []
    for t in _tokenize(q):
        if t not in STOP and len(t) > 1 and t not in terms:
            terms.append(t)
    return terms


def _semantic_query_text(q):
    """Normalize broad questions about projects and their technology stacks to both facets."""
    tokens = _tokenize(q)
    terms = set(tokens)
    if tokens and tokens[0] in QUESTION_OPENERS:
        if terms & PROJECT_INTENT_TERMS and terms & TECHNOLOGY_INTENT_TERMS:
            excluded = (PROJECT_INTENT_TERMS | TECHNOLOGY_INTENT_TERMS |
                        INTENT_ACTION_TERMS | INTENT_CONTEXT_TERMS)
            subjects = [term for term in _query_terms(q) if term not in excluded]
            return " ".join(subjects + ["project portfolio applications software stacks technologies tools"])
    return q


def _fts_query(q):
    return " OR ".join(f'"{t}"' for t in _query_terms(q))


def keyword_ranks(con, q, n=50):
    """Chunk ids ranked by BM25. A chunk matches if it contains any query word, but when the query has
    3+ meaningful words it must contain at least 2 of them: a single incidental word ("nodes", "speed up")
    is not evidence, and in hybrid search such noise would outrank a strong semantic-only hit."""
    terms = _query_terms(q)
    if not terms:
        return []
    ids = [r[0] for r in con.execute(
        "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts) LIMIT ?",
        (_fts_query(q), CANDIDATES))]
    need = 1 if len(terms) <= 2 else 2
    if need == 1 or not ids:
        return ids[:n]
    marks = ",".join("?" * len(ids))
    covered = {}
    for t in terms:       # counted with FTS5's own tokenizer/stemmer, so "nodes" matches "node"
        for (rid,) in con.execute(
                f"SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? AND rowid IN ({marks})", (f'"{t}"', *ids)):
            covered[rid] = covered.get(rid, 0) + 1
    return [i for i in ids if covered.get(i, 0) >= need][:n]


def _min_similarity():
    """Optional cutoff from env RECALL_MIN_SIMILARITY (e.g. 0.3). Unset = no cutoff (default)."""
    raw = os.environ.get("RECALL_MIN_SIMILARITY", "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        log.warning("ignoring invalid RECALL_MIN_SIMILARITY=%r (expected a number such as 0.3)", raw)
        return None


def semantic_scored(con, q, n=50):
    """[(chunk_id, cosine similarity)] best first. Raises SemanticUnavailable if it cannot run.
    The vector matrix is cached (see vectors.py); it is not rebuilt per query."""
    if not q or not q.strip():
        return []
    if not embed.available():
        raise SemanticUnavailable("sentence-transformers is not installed; run: pip install sentence-transformers")
    try:
        qv = embed.encode([_semantic_query_text(q)])[0]
    except Exception as e:
        raise SemanticUnavailable(f"could not embed the query: {type(e).__name__}: {e}") from e
    index = vectors.get_index(con, len(qv))
    if not len(index):
        log.warning("no embeddings of dimension %d are indexed; run `index` without --no-embed or "
                    "`python -m recall.cli embed` to backfill stored chunks", len(qv))
        return []
    missing = index.total_chunks - len(index)
    if missing:
        log.warning(
            "semantic index is incomplete: %d of %d chunks have no compatible embedding; "
            "run `python -m recall.cli embed` to backfill them",
            missing, index.total_chunks)
    scored = index.search_scored(qv, n)
    floor = _min_similarity()
    return [(c, sc) for c, sc in scored if sc >= floor] if floor is not None else scored


def semantic_ranks(con, q, n=50):
    """Chunk ids ranked by cosine similarity (see semantic_scored)."""
    return [cid for cid, _ in semantic_scored(con, q, n)]


def rrf(*rankings):
    s = {}
    for r in rankings:
        for pos, cid in enumerate(r):
            s[cid] = s.get(cid, 0) + 1 / (RRF_K + pos + 1)
    return sorted(s, key=s.get, reverse=True), s


def _document_evidence_score(passage_scores):
    """Blend a document's best passage with support from up to two other passages."""
    if len(passage_scores) < 2:
        return passage_scores[0]
    supporting = [max(0.0, score) for score in passage_scores[1:3]]
    return passage_scores[0] + PASSAGE_SUPPORT_WEIGHT * sum(supporting) / len(supporting)


def _document_candidates(con, ranked_chunks, scored=False):
    """Group ranked passages by file, retaining its best passage and document-level evidence."""
    if not ranked_chunks:
        return [], {}
    chunk_ids = [item[0] if scored else item for item in ranked_chunks]
    marks = ",".join("?" for _ in chunk_ids)
    rows = con.execute(
        f"SELECT c.id, f.id, f.path, c.page, c.text "
        f"FROM chunks c JOIN files f ON f.id=c.file_id WHERE c.id IN ({marks})",
        chunk_ids).fetchall()
    row_by_chunk = {row[0]: row[1:] for row in rows}
    documents = {}
    for rank, item in enumerate(ranked_chunks):
        cid, score = item if scored else (item, None)
        row = row_by_chunk.get(cid)
        if row is None:
            continue
        fid, path, page, text = row
        passage = dict(chunk_id=cid, rank=rank, score=score, path=path, page=page, text=text)
        documents.setdefault(fid, {"passages": []})["passages"].append(passage)

    for document in documents.values():
        if not scored:
            document["best"] = document["passages"][0]
            continue
        distinct, seen = [], set()
        for passage in document["passages"]:
            normalized = " ".join(passage["text"].casefold().split())
            if normalized not in seen:
                seen.add(normalized)
                distinct.append(passage)
        document["passages"] = distinct
        document["best"] = distinct[0]
        document["score"] = _document_evidence_score([p["score"] for p in distinct])
    order = (sorted(documents, key=lambda fid: documents[fid]["score"], reverse=True)
             if scored else list(documents))
    return order, documents


def snippet(text, q, width=220):
    terms = [t for t in _tokenize(q) if t not in STOP]
    low = text.lower()
    pos = min((low.find(t) for t in terms if low.find(t) >= 0), default=0)
    start = max(0, pos - width // 3)
    s = text[start:start + width].strip()
    return ("…" if start else "") + s + "…"


def search(con, q, mode="hybrid", top_files=5):
    if mode not in MODES:
        raise ValueError(f"unknown mode '{mode}'; choose one of {', '.join(MODES)}")
    if not q or not q.strip():
        return []
    kw = keyword_ranks(con, q) if mode in ("keyword", "hybrid") else []
    if mode == "keyword":
        results, seen = [], set()
        for cid in kw:
            row = con.execute(
                "SELECT f.id, f.path, c.page, c.text FROM chunks c JOIN files f ON f.id=c.file_id WHERE c.id=?",
                (cid,)).fetchone()
            if row is None:
                continue
            fid, path, page, text = row
            if fid in seen:
                continue
            seen.add(fid)
            results.append(dict(path=path, page=page, snippet=snippet(text, q),
                                why="keyword match", semantic_score=None))
            if len(results) == top_files:
                break
        return results

    sem_scored = []
    if mode == "semantic":
        sem_scored = semantic_scored(con, q, SEMANTIC_CANDIDATES)
    else:
        try:
            sem_scored = semantic_scored(con, q, SEMANTIC_CANDIDATES)
        except SemanticUnavailable as e:
            log.warning("semantic search unavailable (%s); hybrid is using keyword results only", e)
    sem_order, sem_documents = _document_candidates(con, sem_scored, scored=True)
    kw_order, kw_documents = _document_candidates(con, kw)
    if mode == "semantic":
        order = sem_order
    else:
        order, _ = rrf(kw_order, sem_order)

    sim = dict(sem_scored)
    results = []
    for fid in order:
        sem_document = sem_documents.get(fid)
        kw_document = kw_documents.get(fid)
        if sem_document and kw_document:
            sem_rank = sem_document["best"]["rank"]
            kw_rank = kw_document["best"]["rank"]
            document = (kw_document if 1 / (RRF_K + kw_rank + 1) >
                        1 / (RRF_K + sem_rank + 1) else sem_document)
        else:
            document = sem_document or kw_document
        passage = document["best"]
        why = [label for label, present in (
            ("keyword match", kw_document is not None),
            ("semantic match", sem_document is not None)) if present]
        results.append(dict(path=passage["path"], page=passage["page"],
                            snippet=snippet(passage["text"], q), why=" + ".join(why),
                            semantic_score=sim.get(passage["chunk_id"])))
        if len(results) == top_files:
            break
    return results
