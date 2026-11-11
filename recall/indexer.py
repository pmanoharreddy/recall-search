"""PDF indexing.

Design (see README "Indexing guarantees"):
  1. Text extraction happens BEFORE the database is touched, so an extraction failure can never
     damage the last known-good index of a file.
  2. Each file is written in one transaction (chunks + hash + error bookkeeping). The new SHA-256
     is stored only if everything in that transaction succeeded.
  3. Chunks are always written with embedding=NULL. Embedding is a separate backfill phase
     (`sync_embeddings`) that fills every NULL embedding by chunk id. This makes
     `--no-embed` -> normal run, partial embedding failures and retries all the same code path.
  4. Failures are recorded in `index_errors` and retried on the next run.
"""
import hashlib
import os
import re
import time

import numpy as np

try:
    import pymupdf
except ImportError:  # PyMuPDF < 1.24.3
    import fitz as pymupdf

from . import embed
from .db import bump_generation, get_meta, set_meta, transaction

CHUNK_WORDS, OVERLAP = 120, 30
EMBED_BATCH = 64
MAX_CONSECUTIVE_EMBED_FAILURES = 3
_HAS_LETTERS = re.compile(r"[^\W\d_]{2,}")  # keeps short titles ("Lost update"), drops bare page numbers


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def chunk_words(text):
    words = text.split()
    step = CHUNK_WORDS - OVERLAP
    for i in range(0, max(len(words), 1), step):
        piece = " ".join(words[i:i + CHUNK_WORDS])
        if _HAS_LETTERS.search(piece):
            yield piece
        if i + CHUNK_WORDS >= len(words):
            break


def extract_chunks(path):
    """Page-aware chunking: a chunk never spans pages, so page numbers stay exact."""
    with pymupdf.open(path) as doc:
        if doc.needs_pass:
            raise ValueError("PDF is password-protected")
        if doc.page_count == 0:
            raise ValueError("PDF contains no pages")
        for pno, page in enumerate(doc, start=1):
            for c in chunk_words(page.get_text()):
                yield pno, c


def _insert_chunks(con, file_id, chunks):
    con.executemany("INSERT INTO chunks(file_id, page, text, embedding) VALUES(?,?,?,NULL)",
                    [(file_id, page, text) for page, text in chunks])


def _is_under(path, folder):
    """True if `path` is strictly inside `folder`. Works for Windows and POSIX paths."""
    try:
        folder_n = os.path.normcase(os.path.normpath(folder))
        path_n = os.path.normcase(os.path.normpath(path))
        return os.path.commonpath([folder_n, path_n]) == folder_n and path_n != folder_n
    except ValueError:  # e.g. different drives on Windows
        return False


def _record_failure(con, path, digest, msg, log, stats):
    log(f"  ! {os.path.basename(path)}: {msg}")
    with transaction(con):
        con.execute("INSERT INTO index_errors(path, error, sha256, failed_at) VALUES(?,?,?,?) "
                    "ON CONFLICT(path) DO UPDATE SET error=excluded.error, sha256=excluded.sha256, "
                    "failed_at=excluded.failed_at", (path, msg, digest, time.time()))
    stats["failed"] += 1
    stats["errors"].append(f"{path}: {msg}")


def _index_file(con, p, path_set, log, stats):
    try:
        digest = sha256(p)
    except OSError as e:
        _record_failure(con, p, None, f"cannot read file: {e}", log, stats)
        return
    row = con.execute("SELECT id, sha256 FROM files WHERE path=?", (p,)).fetchone()

    if row and row[1] == digest:                       # unchanged
        stats["skipped"] += 1
        if con.execute("SELECT 1 FROM index_errors WHERE path=?", (p,)).fetchone():
            with transaction(con):
                con.execute("DELETE FROM index_errors WHERE path=?", (p,))
        return

    if not row:                                        # moved/renamed? same content, old path gone
        for fid, old in con.execute("SELECT id, path FROM files WHERE sha256=?", (digest,)).fetchall():
            if old not in path_set and not os.path.exists(old):
                with transaction(con):
                    con.execute("UPDATE files SET path=?, mtime=? WHERE id=?", (p, os.path.getmtime(p), fid))
                    con.execute("DELETE FROM index_errors WHERE path IN (?,?)", (p, old))
                    bump_generation(con)
                stats["moved"] += 1
                log(f"  moved: {os.path.basename(old)} -> {p}")
                return

    try:                                               # extract first; DB untouched if this fails
        chunks = list(extract_chunks(p))
    except Exception as e:
        _record_failure(con, p, digest, f"text extraction failed: {type(e).__name__}: {e}", log, stats)
        return

    try:
        with transaction(con):
            if row:
                fid = row[0]
                con.execute("DELETE FROM chunks WHERE file_id=?", (fid,))
                con.execute("UPDATE files SET sha256=?, mtime=? WHERE id=?",
                            (digest, os.path.getmtime(p), fid))
            else:
                fid = con.execute("INSERT INTO files(path, sha256, mtime) VALUES(?,?,?)",
                                  (p, digest, os.path.getmtime(p))).lastrowid
            _insert_chunks(con, fid, chunks)
            con.execute("DELETE FROM index_errors WHERE path=?", (p,))
            bump_generation(con)
    except Exception as e:                             # transaction rolled back: last good index intact
        _record_failure(con, p, digest, f"database write failed (rolled back): {type(e).__name__}: {e}",
                        log, stats)
        return

    key = "updated" if row else "added"
    stats[key] += 1
    if not chunks:
        stats["no_text"] += 1
        log(f"  {key}: {os.path.basename(p)} (0 chunks: no extractable text; scanned PDF? OCR is not implemented)")
    else:
        log(f"  {key}: {os.path.basename(p)} ({len(chunks)} chunks)")


def _remove_stale(con, folder, path_set, unreadable_dirs, log, stats):
    for fid, p in con.execute("SELECT id, path FROM files").fetchall():
        if not _is_under(p, folder) or p in path_set:
            continue
        if any(p == d or _is_under(p, d) for d in unreadable_dirs):
            continue                                   # can't tell whether it was deleted
        if os.path.exists(p):
            continue                                   # present but not seen: don't risk deleting
        with transaction(con):
            con.execute("DELETE FROM chunks WHERE file_id=?", (fid,))
            con.execute("DELETE FROM files WHERE id=?", (fid,))
            con.execute("DELETE FROM index_errors WHERE path=?", (p,))
            bump_generation(con)
        stats["removed"] += 1
        log(f"  removed: {p}")
    with transaction(con):                             # errors for files that no longer exist
        for (p,) in con.execute("SELECT path FROM index_errors").fetchall():
            if _is_under(p, folder) and p not in path_set and not os.path.exists(p):
                con.execute("DELETE FROM index_errors WHERE path=?", (p,))


def _encode(texts):
    vecs = np.asarray(embed.encode(texts), dtype=np.float32)
    if vecs.ndim != 2 or vecs.shape[0] != len(texts) or not np.isfinite(vecs).all():
        raise ValueError(f"embedder returned invalid output shape {vecs.shape} for {len(texts)} texts")
    return vecs


def _embed_batch(texts, log):
    """Returns (vectors-or-None per text, first error). A failing batch is retried chunk by chunk
    so one bad chunk does not block its neighbours."""
    try:
        return list(_encode(texts)), None
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        if len(texts) == 1:
            return [None], err
        out = []
        for t in texts:
            try:
                out.append(_encode([t])[0])
            except Exception:
                out.append(None)
        return out, err


def sync_embeddings(con, log=print, stats=None):
    """Fill every chunk whose embedding is NULL, matching by chunk id. Idempotent and safe to retry.

    Already-embedded chunks are never recomputed unless the embedding model changed or the stored
    vector has the wrong size for the current model (both detected and reported).
    """
    stats = stats if stats is not None else {}
    stats.setdefault("embedded", 0), stats.setdefault("embed_failed", 0), stats.setdefault("errors", [])
    model = embed.MODEL_NAME
    stored_model, stored_dim = get_meta(con, "embedding_model"), get_meta(con, "embedding_dim")
    pending = con.execute("SELECT COUNT(*) FROM chunks WHERE embedding IS NULL").fetchone()[0]
    wrong_size = 0
    if stored_dim:
        wrong_size = con.execute("SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL "
                                 "AND length(embedding)!=?", (int(stored_dim) * 4,)).fetchone()[0]
    if pending == 0 and wrong_size == 0 and stored_model == model and stored_dim:
        return stats                                   # nothing to do; model never loaded

    try:
        dim = int(_encode(["dimension probe"]).shape[1])
    except Exception as e:
        msg = f"embedding model unavailable ({type(e).__name__}: {e}); keyword search still works, " \
              f"{pending} chunks remain pending and will be retried on the next run"
        log(f"  ! {msg}")
        stats["errors"].append(msg)
        return stats

    with transaction(con):
        if stored_model not in (None, model):
            n = con.execute("UPDATE chunks SET embedding=NULL WHERE embedding IS NOT NULL").rowcount
            log(f"  embedding model changed ({stored_model} -> {model}); re-embedding {n} chunks")
        else:
            n = con.execute("UPDATE chunks SET embedding=NULL WHERE embedding IS NOT NULL "
                            "AND length(embedding)!=?", (dim * 4,)).rowcount
            if n:
                log(f"  discarded {n} embeddings with wrong dimension (expected {dim}); re-embedding")
        set_meta(con, "embedding_model", model)
        set_meta(con, "embedding_dim", dim)
        bump_generation(con)

    rows = con.execute("SELECT id, text FROM chunks WHERE embedding IS NULL ORDER BY id").fetchall()
    consecutive_failures = 0
    for i in range(0, len(rows), EMBED_BATCH):
        batch = rows[i:i + EMBED_BATCH]
        vecs, err = _embed_batch([t for _, t in batch], log)
        good = [(v.tobytes(), cid) for (cid, _), v in zip(batch, vecs) if v is not None]
        failed = len(batch) - len(good)
        if good:
            with transaction(con):
                con.executemany("UPDATE chunks SET embedding=? WHERE id=?", good)
                bump_generation(con)
            stats["embedded"] += len(good)
        if failed:
            stats["embed_failed"] += failed
            msg = f"{failed} chunk(s) not embedded ({err}); will be retried on the next run"
            log(f"  ! {msg}")
            stats["errors"].append(msg)
        consecutive_failures = consecutive_failures + 1 if not good else 0
        if consecutive_failures >= MAX_CONSECUTIVE_EMBED_FAILURES:
            log("  ! too many consecutive embedding failures; stopping (remaining chunks stay pending)")
            break
    return stats


def index_folder(con, folder, use_embeddings=True, log=print):
    folder = os.path.abspath(folder)
    if not os.path.isdir(folder):
        raise NotADirectoryError(
            f"'{folder}' is not an accessible directory. Nothing was changed; the existing index was "
            f"left alone (an unmounted drive or a typo must not wipe it).")
    stats = dict(added=0, updated=0, skipped=0, removed=0, moved=0, failed=0, no_text=0,
                 embedded=0, embed_failed=0, pending_embeddings=0, semantic_available=embed.available(),
                 unreadable_dirs=[], errors=[])

    unreadable = []
    paths = []
    for root, _, names in os.walk(folder, onerror=lambda e: unreadable.append(os.path.abspath(e.filename))):
        for n in sorted(names):
            if n.lower().endswith(".pdf"):
                paths.append(os.path.abspath(os.path.join(root, n)))
    for d in unreadable:
        msg = f"cannot list directory {d}; its files were left untouched"
        log(f"  ! {msg}")
        stats["errors"].append(msg)
    stats["unreadable_dirs"] = unreadable
    path_set = set(paths)

    for p in sorted(paths):
        _index_file(con, p, path_set, log, stats)
    _remove_stale(con, folder, path_set, unreadable, log, stats)

    if use_embeddings:
        if stats["semantic_available"]:
            sync_embeddings(con, log, stats)
        else:
            log("  ! sentence-transformers is not installed: semantic indexing skipped, keyword search "
                "still works. Install it (pip install sentence-transformers) and re-run to add embeddings.")
    stats["pending_embeddings"] = con.execute(
        "SELECT COUNT(*) FROM chunks WHERE embedding IS NULL").fetchone()[0]
    con.commit()
    return stats
