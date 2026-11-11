import re
import zlib

import numpy as np
import pymupdf
import pytest

from recall import embed, vectors
from recall.db import connect

DIM = 64
CONCEPTS = {  # tiny synonym table so the fake embedder behaves "semantically"
    "car": "vehicle", "automobile": "vehicle", "vehicle": "vehicle",
    "cancelled": "rollback", "aborted": "rollback", "rollback": "rollback",
    "puppy": "dog", "canine": "dog", "dog": "dog",
    "projects": "project", "project": "project", "application": "project",
    "build": "create", "built": "create", "developed": "create",
    "tools": "technology", "technology": "technology", "stack": "technology",
    "use": "use", "used": "use", "using": "use",
}


def fake_vec(text):
    v = np.zeros(DIM, dtype=np.float32)
    for tok in re.findall(r"[a-z]+", text.lower()):
        v[zlib.crc32(CONCEPTS.get(tok, tok).encode()) % DIM] += 1
    n = np.linalg.norm(v)
    return v / n if n else v


class FakeEmbedder:
    """Stands in for sentence-transformers: deterministic, offline, records every call."""
    def __init__(self):
        self.calls, self.fail_on, self.fail_all, self.bad_shape = [], set(), False, False

    def encode(self, texts):
        self.calls.append(list(texts))
        if self.fail_all:
            raise RuntimeError("embedder offline")
        if any(f in t for t in texts for f in self.fail_on):
            raise RuntimeError("poison chunk")
        if self.bad_shape:
            return np.zeros((len(texts) + 1, DIM), dtype=np.float32)
        return np.vstack([fake_vec(t) for t in texts])

    @property
    def texts_embedded(self):  # every real chunk text sent for embedding (probe excluded)
        return [t for batch in self.calls for t in batch if t != "dimension probe"]


@pytest.fixture(autouse=True)
def _clear_vector_cache():
    vectors.clear_cache()
    yield
    vectors.clear_cache()


@pytest.fixture
def fake(monkeypatch):
    f = FakeEmbedder()
    monkeypatch.setattr(embed, "available", lambda: True)
    monkeypatch.setattr(embed, "encode", f.encode)
    monkeypatch.setattr(embed, "MODEL_NAME", "fake-model")
    return f


@pytest.fixture
def no_embedder(monkeypatch):
    monkeypatch.setattr(embed, "available", lambda: False)


@pytest.fixture
def con(tmp_path):
    c = connect(str(tmp_path / "index.db"))
    yield c
    c.close()


@pytest.fixture
def docs(tmp_path):
    d = tmp_path / "docs"
    d.mkdir()
    return d


@pytest.fixture
def make_pdf():
    def _make(path, pages):
        doc = pymupdf.open()
        for text in pages:
            page = doc.new_page()
            if text:
                page.insert_textbox(pymupdf.Rect(50, 50, 550, 790), text, fontsize=11)
        doc.save(str(path))
        doc.close()
        return path
    return _make


@pytest.fixture
def quiet():
    return lambda *a, **k: None


@pytest.fixture
def check_consistent():
    def _check(con):
        assert con.execute("SELECT COUNT(*) FROM chunks WHERE file_id NOT IN (SELECT id FROM files)").fetchone()[0] == 0, "orphan chunks"
        assert con.execute("SELECT COUNT(*) FROM chunks c JOIN files f ON f.id=c.file_id").fetchone()[0] == \
            con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        # FTS index must match the chunks table exactly (raises if out of sync)
        con.execute("INSERT INTO chunks_fts(chunks_fts, rank) VALUES('integrity-check', 1)")
        assert con.execute("SELECT COUNT(*) FROM (SELECT file_id, page, text FROM chunks GROUP BY file_id, page, text "
                           "HAVING COUNT(*)>1)").fetchone()[0] == 0, "duplicate chunks"
    return _check
