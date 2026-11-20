import logging
import os
import unicodedata
from collections import defaultdict

from . import embed, vectors
from .db import get_meta, transaction

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
# At UI-sized result limits, early ranks should carry substantially more weight than the tail.
RRF_K = 3
MODES = ("keyword", "semantic", "hybrid")
SEMANTIC_CANDIDATES = 200
RERANK_WINDOW = 8
PASSAGE_SUPPORT_WEIGHT = 0.05


class SemanticUnavailable(RuntimeError):
    """Semantic search was requested but cannot run (missing dependency, model failure)."""


KEYWORD_BATCH = 500
# An unverified legacy row gets its own integer key. SHA-256 keys are text.
_CONTENT_KEY = "COALESCE(NULLIF(f.sha256, ''), f.id)"

PROJECT_INTENT_TERMS = {
    "project", "projects", "application", "applications", "product", "products",
    "build", "built", "develop", "developed", "create", "created",
}
TECHNOLOGY_INTENT_TERMS = {
    "technology", "technologies", "tool", "tools", "stack", "stacks", "framework",
    "frameworks", "language", "languages", "software", "programming",
}
QUESTION_OPENERS = {"what", "which", "who", "where", "how", "describe", "list"}
DESCRIPTION_OPENERS = QUESTION_OPENERS | {"a", "an", "the", "find", "show", "i", "my", "when"}
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


def keyword_ranks(con, q, n=50, *, distinct_files=False, distinct_contents=False, phrase=False):
    """Chunk ids ranked by BM25. A chunk matches if it contains any query word, but when the query has
    3+ meaningful words it must contain at least 2 of them: a single incidental word ("nodes", "speed up")
    is not evidence, and in hybrid search such noise would outrank a strong semantic-only hit."""
    terms = _query_terms(q)
    if not terms or n <= 0 or (phrase and len(terms) < 2):
        return []
    query = ('"' + " ".join(_tokenize(q)) + '"' if phrase
             else " OR ".join(f'"{term}"' for term in terms))
    key = _CONTENT_KEY if distinct_contents else "c.file_id"
    ranked = con.execute(
        f"SELECT c.id, {key} FROM chunks_fts JOIN chunks c ON c.id=chunks_fts.rowid "
        "JOIN files f ON c.file_id=f.id "
        "WHERE chunks_fts MATCH ? ORDER BY rank, c.id", (query,))
    result, seen_files = [], set()
    while batch := ranked.fetchmany(KEYWORD_BATCH):
        if not phrase and len(terms) >= 3:
            ids = [cid for cid, _ in batch]
            eligible = _covered_chunks(con, ids, terms)
            batch = [(cid, fid) for cid, fid in batch if cid in eligible]
        for cid, fid in batch:
            if (distinct_files or distinct_contents) and fid in seen_files:
                continue
            seen_files.add(fid)
            result.append(cid)
            if len(result) == n:
                return result
    return result


def _covered_chunks(con, ids, terms):
    # Count matches with the same FTS tokenizer and stemmer used for ranking.
    marks = ",".join("?" for _ in ids)
    matches, covered = set(), set()
    for term in terms:
        for (cid,) in con.execute(
                f"SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? AND rowid IN ({marks})",
                (f'"{term}"', *ids)):
            if cid in matches:
                covered.add(cid)
            matches.add(cid)
    return covered


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


def semantic_scored(con, q, n=50, *, require_index=False):
    """[(chunk_id, cosine similarity)] best first. Raises SemanticUnavailable if it cannot run.
    The vector matrix is cached (see vectors.py); it is not rebuilt per query."""
    if not q or not q.strip():
        return []
    if not embed.available():
        raise SemanticUnavailable("sentence-transformers is not installed; run: pip install sentence-transformers")
    stored_model = get_meta(con, "embedding_model")
    if stored_model is not None and stored_model != embed.MODEL_NAME:
        raise SemanticUnavailable(f"index uses {stored_model}, but search uses {embed.MODEL_NAME}; "
                                  "run `python -m recall.cli embed` to rebuild the embeddings")
    has_vectors = con.execute("SELECT 1 FROM chunks WHERE embedding IS NOT NULL LIMIT 1").fetchone()
    if not has_vectors:
        message = ("no embeddings are indexed; run `index` without --no-embed or "
                   "`python -m recall.cli embed` to backfill stored chunks")
        if require_index:
            raise SemanticUnavailable(message)
        log.warning(message)
        return []
    try:
        qv = embed.encode([_semantic_query_text(q)])[0]
    except Exception as e:
        raise SemanticUnavailable(f"could not embed the query: {type(e).__name__}: {e}") from e
    if embed.fragmented_identifier(q.strip()) and not keyword_ranks(con, q, 1):
        return []
    index = vectors.get_index(con, len(qv))
    if not len(index):
        message = (f"no embeddings of dimension {len(qv)} are indexed; "
                   "run `python -m recall.cli embed` to rebuild the embeddings")
        if require_index:
            raise SemanticUnavailable(message)
        log.warning(message)
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
        f"SELECT c.id, {_CONTENT_KEY}, f.path, c.page, c.text "
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
    pos = low.find(q.strip().lower())
    if pos < 0:
        pos = min((low.find(t) for t in terms if low.find(t) >= 0), default=0)
    start = max(0, pos - width // 3)
    s = text[start:start + width].strip()
    return ("…" if start else "") + s + ("…" if start + width < len(text) else "")


def _filename_key(stem):
    folded = unicodedata.normalize("NFKD", stem.casefold())
    return _tokenize("".join(ch for ch in folded if not unicodedata.combining(ch)))


def _filename_hits(con, q, limit, catalog):
    query = q.strip().casefold()
    stem = query[:-4] if query.endswith(".pdf") else query
    key = _filename_key(stem)
    matches = []
    for path, (fid, group) in catalog.items():
        name = os.path.basename(path)
        file_stem = os.path.splitext(name)[0]
        literal = query in (name.casefold(), file_stem.casefold())
        if literal or (key and _filename_key(file_stem) == key):
            matches.append((not literal, path, fid, group))
    hits = {}
    for _, path, fid, group in sorted(matches):
        if group in hits:
            continue
        row = con.execute("SELECT page, text FROM chunks WHERE file_id=? ORDER BY page, id LIMIT 1",
                          (fid,)).fetchone()
        hits[group] = dict(path=path, page=row[0] if row else 1,
                         snippet=snippet(row[1], q) if row else "No searchable text in this PDF.",
                         why="filename match", semantic_score=None)
        if len(hits) == limit:
            break
    return list(hits.values())


def _unique_full_match(con, q, passage):
    """Promote the leading keyword passage when only its file contains all terms together."""
    terms = _query_terms(q)
    if len(terms) < 2:
        return False
    expr = " AND ".join(f'"{term}"' for term in terms)
    if not con.execute("SELECT 1 FROM chunks_fts WHERE rowid=? AND chunks_fts MATCH ?",
                       (passage["chunk_id"], expr)).fetchone():
        return False
    files = con.execute(f"SELECT DISTINCT {_CONTENT_KEY} FROM chunks_fts JOIN chunks c ON c.id=chunks_fts.rowid "
                        "JOIN files f ON f.id=c.file_id "
                        "WHERE chunks_fts MATCH ? LIMIT 2", (expr,)).fetchall()
    return len(files) == 1


def search(con, q, mode="hybrid", top_files=5, *, require_semantic=False):
    if mode not in MODES:
        raise ValueError(f"unknown mode '{mode}'; choose one of {', '.join(MODES)}")
    if not q or not q.strip() or top_files <= 0:
        return []
    # Keep rankings and their copy locations on the same SQLite snapshot during re-indexing.
    with transaction(con, "recall_search"):
        return _search_snapshot(con, q, mode, top_files, require_semantic)


def _search_snapshot(con, q, mode, top_files, require_semantic):
    catalog = {path: (fid, group) for fid, path, group in
               con.execute(f"SELECT f.id, f.path, {_CONTENT_KEY} FROM files f")}
    named = _filename_hits(con, q, top_files, catalog)
    name_only = bool(named) and (q.strip().casefold().endswith(".pdf") or len(named) == top_files)
    name_only = name_only or all(t in STOP for t in _tokenize(q))
    passages = []
    if not name_only or require_semantic:
        try:
            passages = _search_passages(con, q, mode, top_files, require_semantic)
        except SemanticUnavailable:
            if not named or require_semantic:
                raise
    locations = defaultdict(list)
    for path, (fid, group) in catalog.items():
        locations[group].append((fid, path))
    results = {}
    for hit in named + ([] if name_only else passages):
        path = hit["path"]
        file_id, group = catalog[path]
        if group in results:
            continue
        hit["file_id"] = file_id
        hit["copies"] = [dict(file_id=fid, path=p) for fid, p in locations[group] if p != path]
        results[group] = hit
        if len(results) == top_files:
            break
    return list(results.values())


def _search_passages(con, q, mode, top_files, require_semantic):
    kw = []
    phrase_ids = []
    if mode in ("keyword", "hybrid"):
        limit = max(50 if mode == "hybrid" else RERANK_WINDOW, top_files)
        kw = keyword_ranks(con, q, limit, distinct_contents=True)
        phrase_ids = keyword_ranks(con, q, limit, distinct_contents=True, phrase=True)
    sem_scored = []
    if mode != "keyword":
        try:
            sem_scored = semantic_scored(con, q, SEMANTIC_CANDIDATES, require_index=require_semantic)
        except SemanticUnavailable as e:
            if mode == "semantic" or require_semantic:
                raise
            log.warning("semantic search unavailable (%s); hybrid is using keyword results only", e)
    sem_order, sem_documents = _document_candidates(con, sem_scored, scored=True)
    kw_order, kw_documents = _document_candidates(con, kw)
    full_match = None
    if mode == "hybrid":
        order, scores = rrf(kw_order, sem_order)
        words = _tokenize(q)
        if (sem_order and words and words[0] in DESCRIPTION_OPENERS and len(_query_terms(q)) >= 3
                and not phrase_ids):
            # Use meaning to order the shortlist for a description. Keep the lexical candidates
            # that earned a place in it, even when the language model disagrees.
            for pos, fid in enumerate(sem_order, 1):
                scores[fid] += 1 / (RRF_K + pos)
            order[:RERANK_WINDOW] = sorted(order[:RERANK_WINDOW], key=scores.get, reverse=True)
        if kw_order and sem_order and _unique_full_match(con, q, kw_documents[kw_order[0]]["best"]):
            full_match = kw_order[0]
            order.remove(full_match)
            order.insert(0, full_match)
    else:
        order = kw_order if mode == "keyword" else sem_order

    phrase_order, phrase_documents = _document_candidates(con, phrase_ids)
    if phrase_order:
        # Retain fusion order among the strongest phrase hits, including the BM25
        # phrase candidates that a bag-of-words or semantic search can miss.
        leading_phrases = [fid for fid in order[:RERANK_WINDOW] if fid in phrase_documents]
        order = list(dict.fromkeys(leading_phrases + phrase_order + order))
    sim = dict(sem_scored)
    results = []
    for fid in order:
        sem_document = sem_documents.get(fid)
        kw_document = kw_documents.get(fid)
        if fid in phrase_documents:
            document = phrase_documents[fid]
        elif fid == full_match:
            document = kw_document
        elif sem_document and kw_document:
            sem_rank = sem_document["best"]["rank"]
            kw_rank = kw_document["best"]["rank"]
            document = kw_document if kw_rank < sem_rank else sem_document
        else:
            document = sem_document or kw_document
        passage = document["best"]
        why = [label for label, present in (
            ("keyword phrase match" if fid in phrase_documents else "keyword match",
             kw_document is not None or fid in phrase_documents),
            ("semantic match", sem_document is not None)) if present]
        results.append(dict(path=passage["path"], page=passage["page"],
                            snippet=snippet(passage["text"], q), why=" + ".join(why),
                            semantic_score=sim.get(passage["chunk_id"])))
        if len(results) == top_files:
            break
    return results
