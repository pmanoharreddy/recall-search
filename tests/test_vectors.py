import numpy as np
import pytest

from recall import vectors
from recall.vectors import VectorIndex


def _random_index(n=500, dim=32, seed=0):
    rng = np.random.default_rng(seed)
    m = rng.normal(size=(n, dim)).astype(np.float32)
    m /= np.linalg.norm(m, axis=1, keepdims=True)
    return np.arange(1000, 1000 + n, dtype=np.int64), m


def test_faiss_backend_returns_same_ranking_as_numpy():
    pytest.importorskip("faiss", reason="faiss-cpu not installed (optional dependency)")
    ids, m = _random_index()
    np_idx, fa_idx = VectorIndex(ids, m, 1, "numpy"), VectorIndex(ids, m, 1, "faiss")
    assert fa_idx.backend == "faiss" and np_idx.backend == "numpy"
    rng = np.random.default_rng(1)
    for _ in range(20):
        q = rng.normal(size=32).astype(np.float32)
        assert np_idx.search(q, 10) == fa_idx.search(q, 10)


def test_n_larger_than_collection_and_empty_index():
    ids, m = _random_index(n=3)
    assert len(VectorIndex(ids, m, 1, "numpy").search(m[0], 50)) == 3
    if vectors._faiss() is not None:
        assert len(VectorIndex(ids, m, 1, "faiss").search(m[0], 50)) == 3
    empty = VectorIndex(np.empty(0, dtype=np.int64), np.empty((0, 8), dtype=np.float32), 1)
    assert empty.search(np.zeros(8, dtype=np.float32)) == []


def test_auto_backend_switches_on_threshold(monkeypatch):
    monkeypatch.delenv("RECALL_VECTOR_BACKEND", raising=False)
    assert vectors._choose_backend(10**7) == "numpy"                  # default never uses faiss
    monkeypatch.setenv("RECALL_VECTOR_BACKEND", "auto")
    monkeypatch.setenv("RECALL_FAISS_MIN_ROWS", "100")
    assert vectors._choose_backend(99) == "numpy"
    assert vectors._choose_backend(100) == ("faiss" if vectors._faiss() is not None else "numpy")
    monkeypatch.setenv("RECALL_VECTOR_BACKEND", "numpy")
    assert vectors._choose_backend(10**6) == "numpy"


def test_faiss_requested_but_missing_falls_back(monkeypatch, caplog):
    monkeypatch.setattr(vectors, "_faiss", lambda: None)
    ids, m = _random_index(n=5)
    idx = VectorIndex(ids, m, 1, "faiss")
    assert idx.backend == "numpy" and idx.search(m[0], 2)[0] == ids[0]
    assert "faiss is not installed" in caplog.text


def test_end_to_end_search_with_faiss_backend(con, docs, make_pdf, fake, quiet, monkeypatch):
    from recall.indexer import index_folder
    from recall.search import search
    monkeypatch.setenv("RECALL_VECTOR_BACKEND", "faiss")
    make_pdf(docs / "cars.pdf", ["An automobile engine needs fuel"])
    make_pdf(docs / "pets.pdf", ["The puppy chased the ball"])
    index_folder(con, docs, log=quiet)
    assert search(con, "car", "semantic")[0]["path"].endswith("cars.pdf")
    expected_backend = "faiss" if vectors._faiss() is not None else "numpy"
    assert vectors.get_index(con, 64).backend == expected_backend


def test_numpy_fallback_path_works_without_faiss(monkeypatch):
    monkeypatch.setattr(vectors, "_faiss", lambda: None)
    assert vectors._choose_backend(10**6) == "numpy"


def test_vector_ids_stay_mapped_when_chunk_ids_are_not_contiguous():
    ids = np.array([11, 89], dtype=np.int64)
    matrix = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    index = VectorIndex(ids, matrix, 1)

    assert index.search(np.array([0.0, 1.0], dtype=np.float32), 1) == [89]
