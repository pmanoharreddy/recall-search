import pymupdf
import pytest

from recall import indexer


def test_page_numbers_are_exact(tmp_path, make_pdf):
    p = make_pdf(tmp_path / "a.pdf", ["alpha page one text", "beta page two text", "gamma page three text"])
    chunks = list(indexer.extract_chunks(str(p)))
    assert [pg for pg, _ in chunks] == [1, 2, 3]
    assert "beta" in chunks[1][1]


def test_long_page_splits_into_overlapping_chunks_on_same_page(tmp_path, make_pdf):
    words = " ".join(f"word{i}" for i in range(300))
    p = make_pdf(tmp_path / "a.pdf", [words])
    chunks = list(indexer.extract_chunks(str(p)))
    assert len(chunks) >= 3 and {pg for pg, _ in chunks} == {1}
    first, second = chunks[0][1].split(), chunks[1][1].split()
    assert set(first) & set(second), "chunks should overlap"
    assert {w for _, c in chunks for w in c.split()} == {f"word{i}" for i in range(300)}


def test_short_titles_are_kept_but_bare_page_numbers_are_not(tmp_path, make_pdf):
    p = make_pdf(tmp_path / "a.pdf", ["Lost update", "7"])
    assert [(pg, t) for pg, t in indexer.extract_chunks(str(p))] == [(1, "Lost update")]


def test_blank_pdf_yields_no_chunks(tmp_path, make_pdf):
    p = make_pdf(tmp_path / "blank.pdf", ["", ""])
    assert list(indexer.extract_chunks(str(p))) == []


def test_malformed_pdf_raises(tmp_path):
    p = tmp_path / "bad.pdf"
    p.write_bytes(b"this is not a pdf at all")
    with pytest.raises(Exception):
        list(indexer.extract_chunks(str(p)))


def test_password_protected_pdf_raises_clear_error(tmp_path):
    doc = pymupdf.open()
    doc.new_page().insert_text((50, 50), "secret")
    p = tmp_path / "locked.pdf"
    doc.save(str(p), encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="u", owner_pw="o")
    with pytest.raises(ValueError, match="password"):
        list(indexer.extract_chunks(str(p)))
