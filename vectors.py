"""In-memory vector index for semantic search.

The NumPy backend is an exact brute-force dot product: perfectly adequate up to roughly
100k chunks. Its limitation is that the whole embedding matrix lives in RAM
(chunks x dim x 4 bytes, e.g. 100k x 384 ~ 150 MB) and every query is O(chunks x dim).

Two things keep it cheap:
  * the matrix is cached per database and only reloaded when the index `generation`
    stored in the DB changes (every indexing write bumps it), not on every query;
  * an optional FAISS backend (exact IndexFlatIP, so identical results, just faster) is used
    when faiss is installed and the collection is large. It is never required.

RECALL_VECTOR_BACKEND = numpy (default) | faiss | auto
RECALL_FAISS_MIN_ROWS = row count at which `auto` switches to FAISS (default 100000)

Measured (synthetic 384-d vectors, this repo's tests/benchmark): exact FAISS IndexFlatIP was only
~15-20% faster per query than NumPy at 100k vectors (15 ms vs 18 ms), so FAISS is opt-in rather
than automatic. The next real step for very large collections would be an approximate index
(IVF/HNSW), which trades recall for speed and is deliberately not enabled here.
"""
import logging
import os

import numpy as np

from .db import get_generation

log = logging.getLogger("recall")
_CACHE = {}  # db key -> VectorIndex


def _faiss():
    try:
        import faiss
        return faiss
    except ImportError:
        return None


class VectorIndex:
    def __init__(self, ids, matrix, generation, backend="numpy", total_chunks=None):
        self.ids, self.matrix, self.generation = ids, matrix, generation
        self.total_chunks = len(ids) if total_chunks is None else total_chunks
        self.dim = matrix.shape[1] if matrix.size else 0
        self.backend, self._faiss_index = "numpy", None
        if len(ids) and backend == "faiss":
            faiss = _faiss()
            if faiss is None:
                log.warning("RECALL_VECTOR_BACKEND=faiss but faiss is not installed; using NumPy")
            else:
                self._faiss_index = faiss.IndexFlatIP(self.dim)
                self._faiss_index.add(np.ascontiguousarray(matrix))
                self.backend = "faiss"

    def __len__(self):
        return len(self.ids)

    def search_scored(self, qvec, n=50):
        """[(chunk_id, similarity)] best first. Similarity is the dot product, i.e. cosine for
        the normalised vectors this project stores."""
        if not len(self.ids):
            return []
        q = np.ascontiguousarray(qvec, dtype=np.float32).reshape(1, -1)
        n = min(n, len(self.ids))
        if self._faiss_index is not None:
            sims, idx = self._faiss_index.search(q, n)
            return [(int(self.ids[i]), float(sc)) for sc, i in zip(sims[0], idx[0]) if i >= 0]
        scores = self.matrix @ q[0]
        top = np.argpartition(-scores, n - 1)[:n]
        top = top[np.argsort(-scores[top], kind="stable")]
        return [(int(self.ids[i]), float(scores[i])) for i in top]

    def search(self, qvec, n=50):
        return [cid for cid, _ in self.search_scored(qvec, n)]


def _choose_backend(rows):
    pref = os.environ.get("RECALL_VECTOR_BACKEND", "numpy").lower()
    if pref in ("numpy", "faiss"):
        return pref
    if pref != "auto":
        log.warning("unknown RECALL_VECTOR_BACKEND=%r; using numpy", pref)
        return "numpy"
    threshold = int(os.environ.get("RECALL_FAISS_MIN_ROWS", "100000"))
    return "faiss" if rows >= threshold and _faiss() is not None else "numpy"


def _db_key(con):
    for _, name, path in con.execute("PRAGMA database_list"):
        if name == "main" and path:
            return os.path.realpath(path)
    return id(con)  # in-memory database


def load_index(con, dim):
    rows = con.execute("SELECT id, embedding FROM chunks WHERE embedding IS NOT NULL "
                       "AND length(embedding)=?", (dim * 4,)).fetchall()
    total_chunks = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    if not rows:
        return VectorIndex(np.empty(0, dtype=np.int64), np.empty((0, dim), dtype=np.float32),
                           get_generation(con), total_chunks=total_chunks)
    ids = np.fromiter((r[0] for r in rows), dtype=np.int64, count=len(rows))
    mat = np.frombuffer(b"".join(r[1] for r in rows), dtype=np.float32).reshape(len(rows), dim)
    return VectorIndex(ids, mat, get_generation(con), _choose_backend(len(rows)), total_chunks)


def get_index(con, dim):
    """Cached index; rebuilt only if the DB generation or vector dimension changed."""
    key, gen = (_db_key(con), dim), get_generation(con)
    cached = _CACHE.get(key)
    if cached is None or cached.generation != gen:
        cached = _CACHE[key] = load_index(con, dim)
    return cached


def clear_cache():
    _CACHE.clear()
