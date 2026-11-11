import os
import shutil
import sqlite3

import numpy as np
import pytest

from recall import embed, indexer, vectors
from recall.db import connect
from recall.indexer import index_folder, sync_embeddings
from recall.search import search
from conftest import fake_vec

TEXT_A = "lost update problem two transactions overwrite the balance"
TEXT_B = "round robin scheduling time quantum for processes"


def nulls(con):
    return con.execute("SELECT COUNT(*) FROM chunks WHERE embedding IS NULL").fetchone()[0]


def count(con, table):
    return con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


@pytest.fixture
def two_docs(docs, make_pdf):
    make_pdf(docs / "a.pdf", [TEXT_A, "second page of a about indexes"])
    make_pdf(docs / "b.pdf", [TEXT_B])
    return docs


# ---------------------------------------------------------------- --no-embed then normal run
def test_no_embed_then_normal_run_generates_embeddings(con, two_docs, fake, quiet, check_consistent):
    s1 = index_folder(con, two_docs, use_embeddings=False, log=quiet)
    assert s1["added"] == 2 and fake.calls == []                      # 1. no model touched
    assert nulls(con) == count(con, "chunks") == 3 and s1["pending_embeddings"] == 3

    s2 = index_folder(con, two_docs, log=quiet)                       # 2. PDFs unmodified
    assert s2["skipped"] == 2 and s2["added"] == s2["updated"] == 0
    assert s2["embedded"] == 3 and nulls(con) == 0                    # 3. embeddings generated
    assert s2["pending_embeddings"] == 0

    n_calls = len(fake.calls)
    s3 = index_folder(con, two_docs, log=quiet)                       # 4. nothing recomputed
    assert s3["embedded"] == 0 and len(fake.calls) == n_calls
    assert count(con, "chunks") == 3
    check_consistent(con)

    for mode in ("keyword", "semantic", "hybrid"):                    # 5. all modes return results
        res = search(con, "lost update problem", mode)
        assert res and res[0]["path"].endswith("a.pdf") and res[0]["page"] == 1


def test_embeddings_are_attached_to_the_right_chunks(con, two_docs, fake, quiet):
    index_folder(con, two_docs, use_embeddings=False, log=quiet)
    index_folder(con, two_docs, log=quiet)
    for text, blob in con.execute("SELECT text, embedding FROM chunks"):
        np.testing.assert_array_equal(np.frombuffer(blob, dtype=np.float32), fake_vec(text))


def test_embedded_chunks_are_never_recomputed_when_only_new_chunks_pending(con, two_docs, fake, quiet, make_pdf):
    index_folder(con, two_docs, log=quiet)
    fake.calls.clear()
    make_pdf(two_docs / "c.pdf", ["brand new document about heap sort"])
    index_folder(con, two_docs, log=quiet)
    assert fake.texts_embedded == ["brand new document about heap sort"]


def test_keyword_search_works_without_embedder(con, two_docs, no_embedder, quiet, caplog):
    logs = []
    s = index_folder(con, two_docs, log=logs.append)
    assert s["added"] == 2 and s["semantic_available"] is False
    assert any("sentence-transformers" in m for m in logs)            # actionable, not silent
    assert search(con, "round robin quantum", "keyword")[0]["path"].endswith("b.pdf")
    hybrid = search(con, "round robin quantum", "hybrid")             # degrades to keyword
    assert hybrid and hybrid[0]["path"].endswith("b.pdf")
    assert "keyword results only" in caplog.text


# ---------------------------------------------------------------- partial / total embedding failure
def test_partial_embedding_failure_is_isolated_and_retried(con, docs, make_pdf, fake, quiet):
    make_pdf(docs / "a.pdf", ["good chunk about sorting", "POISON chunk about hashing"])
    fake.fail_on = {"POISON"}
    s = index_folder(con, docs, log=quiet)
    assert s["embedded"] == 1 and s["embed_failed"] == 1 and s["pending_embeddings"] == 1
    assert any("retried on the next run" in e for e in s["errors"])
    assert search(con, "POISON hashing", "keyword")                   # still keyword searchable
    assert search(con, "sorting", "semantic")                         # embedded neighbour searchable

    fake.fail_on, fake.calls = set(), []
    s2 = index_folder(con, docs, log=quiet)                           # file unchanged
    assert s2["skipped"] == 1 and s2["embedded"] == 1 and s2["pending_embeddings"] == 0
    assert fake.texts_embedded == ["POISON chunk about hashing"]      # only the missing one


def test_embedder_down_keeps_keyword_search_and_retries_later(con, two_docs, fake, quiet):
    fake.fail_all = True
    s = index_folder(con, two_docs, log=quiet)                        # must not raise
    assert s["added"] == 2 and s["embedded"] == 0 and s["pending_embeddings"] == 3
    assert any("keyword search still works" in e for e in s["errors"])
    assert search(con, "time quantum", "keyword")
    assert search(con, "time quantum", "hybrid")                      # hybrid falls back, no crash
    fake.fail_all = False
    s2 = index_folder(con, two_docs, log=quiet)
    assert s2["embedded"] == 3 and s2["pending_embeddings"] == 0


def test_invalid_embedder_output_is_treated_as_failure_not_stored(con, two_docs, fake, quiet):
    fake.bad_shape = True
    s = index_folder(con, two_docs, log=quiet)
    assert s["embedded"] == 0 and nulls(con) == 3


def test_consecutive_embedding_failures_stop_the_run(con, docs, make_pdf, fake, monkeypatch):
    monkeypatch.setattr(indexer, "EMBED_BATCH", 2)
    make_pdf(docs / "a.pdf", [f"page{i} content here" for i in range(12)])
    fake.fail_on = {"content"}                                        # probe passes, every chunk fails
    logs = []
    s = index_folder(con, docs, log=logs.append)
    assert any("too many consecutive" in m for m in logs)
    assert s["embedded"] == 0 and s["pending_embeddings"] == 12


# ---------------------------------------------------------------- failure handling / transactions
def test_malformed_pdf_does_not_stop_others_and_recovers_when_fixed(con, two_docs, make_pdf, no_embedder, quiet):
    bad = two_docs / "bad.pdf"
    bad.write_bytes(b"garbage, not a pdf")
    s = index_folder(con, two_docs, use_embeddings=False, log=quiet)
    assert s["added"] == 2 and s["failed"] == 1
    assert con.execute("SELECT path FROM files WHERE path LIKE '%bad.pdf'").fetchone() is None
    assert con.execute("SELECT COUNT(*) FROM index_errors").fetchone()[0] == 1

    s_again = index_folder(con, two_docs, use_embeddings=False, log=quiet)
    assert s_again["failed"] == 1                                     # retried every run, not skipped

    make_pdf(bad, ["now a valid document about compilers"])           # file becomes readable
    s2 = index_folder(con, two_docs, use_embeddings=False, log=quiet)
    assert s2["added"] == 1 and s2["failed"] == 0
    assert count(con, "index_errors") == 0
    assert search(con, "compilers", "keyword")[0]["path"].endswith("bad.pdf")


def test_empty_pdf_is_indexed_once_with_no_chunks(con, docs, make_pdf, no_embedder, quiet):
    make_pdf(docs / "blank.pdf", ["", ""])
    s = index_folder(con, docs, use_embeddings=False, log=quiet)
    assert s["added"] == 1 and s["no_text"] == 1 and s["failed"] == 0
    assert count(con, "chunks") == 0
    assert index_folder(con, docs, use_embeddings=False, log=quiet)["skipped"] == 1


def test_password_protected_pdf_is_recorded_as_failure(con, docs, no_embedder, quiet):
    import pymupdf
    d = pymupdf.open(); d.new_page().insert_text((50, 50), "secret words"); 
    d.save(str(docs / "locked.pdf"), encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="u", owner_pw="o")
    s = index_folder(con, docs, use_embeddings=False, log=quiet)
    assert s["failed"] == 1 and "password" in s["errors"][0]


def test_unreadable_file_is_isolated_and_retried(con, two_docs, monkeypatch, no_embedder, quiet):
    real = indexer.sha256
    def deny(p):
        if p.endswith("a.pdf"):
            raise PermissionError("denied")
        return real(p)
    monkeypatch.setattr(indexer, "sha256", deny)
    s = index_folder(con, two_docs, use_embeddings=False, log=quiet)
    assert s["added"] == 1 and s["failed"] == 1 and "cannot read file" in s["errors"][0]
    monkeypatch.setattr(indexer, "sha256", real)
    s2 = index_folder(con, two_docs, use_embeddings=False, log=quiet)
    assert s2["added"] == 1 and s2["failed"] == 0 and s2["skipped"] == 1


@pytest.mark.skipif(os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0),
                    reason="file permissions are not enforced for root")
def test_real_permission_denied_file(con, two_docs, no_embedder, quiet):
    os.chmod(two_docs / "a.pdf", 0)
    try:
        s = index_folder(con, two_docs, use_embeddings=False, log=quiet)
        assert s["failed"] == 1 and s["added"] == 1
    finally:
        os.chmod(two_docs / "a.pdf", 0o644)


def test_extraction_failure_on_modified_file_keeps_last_good_index(con, two_docs, make_pdf, monkeypatch, no_embedder, quiet, check_consistent):
    index_folder(con, two_docs, use_embeddings=False, log=quiet)
    old_hash = con.execute("SELECT sha256 FROM files WHERE path LIKE '%a.pdf'").fetchone()[0]
    old_chunks = con.execute("SELECT id, page, text FROM chunks ORDER BY id").fetchall()

    make_pdf(two_docs / "a.pdf", ["completely different mergesort content"])
    real = indexer.extract_chunks
    def boom(p):
        if p.endswith("a.pdf"):
            raise RuntimeError("simulated extraction failure")
        return real(p)
    monkeypatch.setattr(indexer, "extract_chunks", boom)
    s = index_folder(con, two_docs, use_embeddings=False, log=quiet)
    assert s["failed"] == 1 and s["updated"] == 0
    assert con.execute("SELECT sha256 FROM files WHERE path LIKE '%a.pdf'").fetchone()[0] == old_hash
    assert con.execute("SELECT id, page, text FROM chunks ORDER BY id").fetchall() == old_chunks
    assert search(con, "lost update", "keyword")                      # old content still searchable
    check_consistent(con)

    monkeypatch.setattr(indexer, "extract_chunks", real)              # recovery
    s2 = index_folder(con, two_docs, use_embeddings=False, log=quiet)
    assert s2["updated"] == 1 and count(con, "index_errors") == 0
    assert search(con, "mergesort", "keyword") and not search(con, "lost update", "keyword")
    check_consistent(con)


def test_db_write_failure_rolls_back_new_file(con, two_docs, monkeypatch, no_embedder, quiet, check_consistent):
    real = indexer._insert_chunks
    def partial_then_fail(c, fid, chunks):
        real(c, fid, chunks[:1])
        raise sqlite3.OperationalError("disk I/O error (simulated)")
    monkeypatch.setattr(indexer, "_insert_chunks", partial_then_fail)
    s = index_folder(con, two_docs, use_embeddings=False, log=quiet)
    assert s["failed"] == 2 and s["added"] == 0
    assert count(con, "files") == 0 and count(con, "chunks") == 0     # nothing half-written
    assert "rolled back" in s["errors"][0]
    check_consistent(con)
    monkeypatch.setattr(indexer, "_insert_chunks", real)
    s2 = index_folder(con, two_docs, use_embeddings=False, log=quiet)
    assert s2["added"] == 2 and count(con, "index_errors") == 0
    check_consistent(con)


def test_db_write_failure_rolls_back_modified_file_and_keeps_old_chunks(con, two_docs, make_pdf, monkeypatch, no_embedder, quiet, check_consistent):
    index_folder(con, two_docs, use_embeddings=False, log=quiet)
    before = con.execute("SELECT f.path, f.sha256, c.id, c.text FROM chunks c JOIN files f ON f.id=c.file_id ORDER BY c.id").fetchall()
    make_pdf(two_docs / "a.pdf", ["new text one", "new text two", "new text three"])
    real = indexer._insert_chunks
    def partial_then_fail(c, fid, chunks):
        real(c, fid, chunks[:1])
        raise sqlite3.OperationalError("simulated")
    monkeypatch.setattr(indexer, "_insert_chunks", partial_then_fail)
    s = index_folder(con, two_docs, use_embeddings=False, log=quiet)
    assert s["failed"] == 1
    after = con.execute("SELECT f.path, f.sha256, c.id, c.text FROM chunks c JOIN files f ON f.id=c.file_id ORDER BY c.id").fetchall()
    assert after == before
    check_consistent(con)


# ---------------------------------------------------------------- lifecycle
def test_added_unchanged_modified_deleted(con, two_docs, make_pdf, fake, quiet, check_consistent):
    s = index_folder(con, two_docs, log=quiet)
    assert (s["added"], s["skipped"]) == (2, 0)
    assert index_folder(con, two_docs, log=quiet)["skipped"] == 2     # unchanged
    assert count(con, "chunks") == 3

    make_pdf(two_docs / "a.pdf", ["replacement content about dijkstra shortest path"])
    fake.calls.clear()
    s = index_folder(con, two_docs, log=quiet)
    assert s["updated"] == 1 and s["skipped"] == 1
    assert fake.texts_embedded == ["replacement content about dijkstra shortest path"]
    assert count(con, "chunks") == 2                                  # old a.pdf chunks gone
    assert not search(con, "lost update", "keyword")
    for mode in ("semantic", "hybrid"):                               # nearest neighbours, but never stale text
        assert not [r for r in search(con, "lost update", mode) if "overwrite" in r["snippet"]]
    assert search(con, "dijkstra", "hybrid")[0]["path"].endswith("a.pdf")
    check_consistent(con)

    os.remove(two_docs / "b.pdf")
    s = index_folder(con, two_docs, log=quiet)
    assert s["removed"] == 1 and count(con, "files") == 1
    assert not search(con, "round robin quantum", "keyword")
    assert not [r for r in search(con, "round robin quantum", "semantic") if r["path"].endswith("b.pdf")]
    assert not [r for r in search(con, "round robin quantum", "hybrid") if r["path"].endswith("b.pdf")]
    check_consistent(con)


def test_rename_and_move_keep_chunks_and_embeddings(con, two_docs, fake, quiet, check_consistent):
    index_folder(con, two_docs, log=quiet)
    chunk_ids = con.execute("SELECT id FROM chunks ORDER BY id").fetchall()
    (two_docs / "sub").mkdir()
    shutil.move(str(two_docs / "a.pdf"), str(two_docs / "sub" / "renamed.pdf"))
    fake.calls.clear()
    s = index_folder(con, two_docs, log=quiet)
    assert (s["moved"], s["added"], s["removed"]) == (1, 0, 0)
    assert fake.calls == []                                           # no re-extraction / re-embedding
    assert con.execute("SELECT id FROM chunks ORDER BY id").fetchall() == chunk_ids
    res = search(con, "lost update problem", "hybrid")
    assert res[0]["path"].endswith(os.path.join("sub", "renamed.pdf"))
    assert not any(r["path"].endswith(os.path.sep + "a.pdf") for r in res)
    check_consistent(con)


def test_copy_of_a_file_is_indexed_as_a_new_file_not_a_move(con, two_docs, no_embedder, quiet):
    index_folder(con, two_docs, use_embeddings=False, log=quiet)
    shutil.copy(two_docs / "a.pdf", two_docs / "a_copy.pdf")
    s = index_folder(con, two_docs, use_embeddings=False, log=quiet)
    assert (s["added"], s["moved"], s["removed"]) == (1, 0, 0) and count(con, "files") == 3


def test_missing_folder_never_wipes_the_index(con, two_docs, no_embedder, quiet):
    index_folder(con, two_docs, use_embeddings=False, log=quiet)
    shutil.move(str(two_docs), str(two_docs) + "_gone")               # e.g. unmounted drive
    with pytest.raises(NotADirectoryError, match="left alone"):
        index_folder(con, two_docs, use_embeddings=False, log=quiet)
    assert count(con, "files") == 2 and count(con, "chunks") == 3


def test_sibling_folder_with_common_prefix_is_not_touched(con, tmp_path, make_pdf, no_embedder, quiet):
    (tmp_path / "notes").mkdir(); (tmp_path / "notes2").mkdir()
    make_pdf(tmp_path / "notes" / "a.pdf", ["alpha content in notes"])
    make_pdf(tmp_path / "notes2" / "b.pdf", ["beta content in notes two"])
    index_folder(con, tmp_path / "notes2", use_embeddings=False, log=quiet)
    s = index_folder(con, tmp_path / "notes", use_embeddings=False, log=quiet)
    assert s["removed"] == 0 and count(con, "files") == 2


def test_unlistable_subdirectory_does_not_delete_its_files(con, docs, make_pdf, monkeypatch, no_embedder, quiet):
    (docs / "locked").mkdir()
    make_pdf(docs / "top.pdf", ["top level file content"])
    make_pdf(docs / "locked" / "inner.pdf", ["inner file content"])
    index_folder(con, docs, use_embeddings=False, log=quiet)
    real_walk = os.walk
    def walk_denied(top, onerror=None, **kw):
        for root, dirs, files in real_walk(top, **kw):
            dirs[:] = []
            onerror(OSError(13, "Permission denied", str(docs / "locked")))
            yield root, dirs, files
    monkeypatch.setattr(indexer.os, "walk", walk_denied)
    s = index_folder(con, docs, use_embeddings=False, log=quiet)
    assert s["removed"] == 0 and count(con, "files") == 2 and s["unreadable_dirs"]


def test_repeated_indexing_creates_no_duplicates(con, two_docs, fake, quiet, check_consistent):
    for _ in range(4):
        index_folder(con, two_docs, log=quiet)
    assert count(con, "files") == 2 and count(con, "chunks") == 3 and nulls(con) == 0
    check_consistent(con)


# ---------------------------------------------------------------- model / compatibility
def test_model_change_triggers_full_reembed(con, two_docs, fake, monkeypatch, quiet):
    index_folder(con, two_docs, log=quiet)
    monkeypatch.setattr(embed, "MODEL_NAME", "other-model")
    fake.calls.clear()
    s = index_folder(con, two_docs, log=quiet)
    assert s["embedded"] == 3 and len(fake.texts_embedded) == 3 and nulls(con) == 0


def test_wrong_size_embeddings_are_replaced(con, two_docs, fake, quiet):
    index_folder(con, two_docs, log=quiet)
    cid = con.execute("SELECT id FROM chunks ORDER BY id LIMIT 1").fetchone()[0]
    con.execute("UPDATE chunks SET embedding=zeroblob(20) WHERE id=?", (cid,)); con.commit()
    fake.calls.clear()
    s = index_folder(con, two_docs, log=quiet)
    assert s["embedded"] == 1 and len(fake.texts_embedded) == 1
    assert con.execute("SELECT length(embedding) FROM chunks WHERE id=?", (cid,)).fetchone()[0] == 64 * 4


def test_legacy_database_from_previous_version_is_usable(tmp_path, two_docs, fake, monkeypatch, quiet):
    """A DB created before meta/index_errors existed: same files/chunks/FTS layout, no migration."""
    legacy_schema = """
    CREATE TABLE files(id INTEGER PRIMARY KEY, path TEXT UNIQUE, sha256 TEXT, mtime REAL);
    CREATE TABLE chunks(id INTEGER PRIMARY KEY, file_id INTEGER REFERENCES files(id) ON DELETE CASCADE,
                        page INTEGER, text TEXT, embedding BLOB);
    CREATE VIRTUAL TABLE chunks_fts USING fts5(text, content='chunks', content_rowid='id', tokenize='porter unicode61');
    CREATE TRIGGER chunks_ai AFTER INSERT ON chunks BEGIN INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text); END;
    CREATE TRIGGER chunks_ad AFTER DELETE ON chunks BEGIN INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES('delete', old.id, old.text); END;
    """
    path = str(tmp_path / "legacy.db")
    raw = sqlite3.connect(path); raw.executescript(legacy_schema)
    embedded_text = None
    for f in sorted(two_docs.iterdir()):
        fid = raw.execute("INSERT INTO files(path, sha256, mtime) VALUES(?,?,0)", (str(f), indexer.sha256(str(f)))).lastrowid
        for page, text in indexer.extract_chunks(str(f)):
            blob = None
            if f.name == "a.pdf" and page == 1:               # one chunk already embedded by the old version
                blob, embedded_text = fake_vec(text).tobytes(), text
            raw.execute("INSERT INTO chunks(file_id, page, text, embedding) VALUES(?,?,?,?)", (fid, page, text, blob))
    raw.commit(); raw.close()

    con = connect(path)
    monkeypatch.setattr(indexer, "extract_chunks", lambda p: (_ for _ in ()).throw(AssertionError("must not re-extract")))
    s = index_folder(con, two_docs, log=quiet)
    assert s["skipped"] == 2 and s["embedded"] == 2 and nulls(con) == 0
    assert embedded_text not in fake.texts_embedded                   # valid legacy embedding kept
    assert search(con, "round robin quantum", "hybrid")[0]["path"].endswith("b.pdf")


def test_index_errors_table_is_cleaned_when_failing_file_is_deleted(con, docs, make_pdf, no_embedder, quiet):
    (docs / "bad.pdf").write_bytes(b"junk")
    index_folder(con, docs, use_embeddings=False, log=quiet)
    assert count(con, "index_errors") == 1
    os.remove(docs / "bad.pdf")
    index_folder(con, docs, use_embeddings=False, log=quiet)
    assert count(con, "index_errors") == 0


def test_is_under_is_path_aware_not_string_prefix():
    assert indexer._is_under("/a/notes/x.pdf", "/a/notes")
    assert indexer._is_under("/a/notes/sub/x.pdf", "/a/notes")
    assert not indexer._is_under("/a/notes2/x.pdf", "/a/notes")
    assert not indexer._is_under("/a/notes", "/a/notes")


def test_deleted_file_in_sibling_folder_is_only_cleaned_when_that_folder_is_indexed(con, tmp_path, make_pdf, no_embedder, quiet):
    (tmp_path / "notes").mkdir(); (tmp_path / "notes2").mkdir()
    make_pdf(tmp_path / "notes" / "a.pdf", ["alpha content in notes"])
    make_pdf(tmp_path / "notes2" / "b.pdf", ["beta content in notes two"])
    index_folder(con, tmp_path / "notes", use_embeddings=False, log=quiet)
    index_folder(con, tmp_path / "notes2", use_embeddings=False, log=quiet)
    (tmp_path / "notes2" / "b.pdf").unlink()
    assert index_folder(con, tmp_path / "notes", use_embeddings=False, log=quiet)["removed"] == 0
    assert count(con, "files") == 2                                   # b.pdf's folder wasn't scanned
    assert index_folder(con, tmp_path / "notes2", use_embeddings=False, log=quiet)["removed"] == 1
    assert count(con, "files") == 1
