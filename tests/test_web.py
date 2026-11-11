import pytest

pytest.importorskip("flask", reason="optional web interface dependency")

from recall.db import connect
from recall.indexer import index_folder
from recall.web import create_app


def fail_search(*_args, **_kwargs):
    from recall.search import SemanticUnavailable
    raise SemanticUnavailable("model unavailable")


def test_web_search_and_pdf_download(tmp_path, docs, make_pdf, quiet):
    pdf_path = make_pdf(docs / "guide.pdf", ["Database indexing improves lookup speed."])
    db_path = tmp_path / "web.db"
    con = connect(str(db_path))
    try:
        index_folder(con, str(docs), use_embeddings=False, log=quiet)
    finally:
        con.close()

    client = create_app(str(db_path)).test_client()
    home = client.get("/")
    assert home.status_code == 200 and b"Describe what you remember" in home.data
    assert home.headers["X-Content-Type-Options"] == "nosniff"
    assert home.headers["Referrer-Policy"] == "no-referrer"

    response = client.get("/api/search?q=database%20indexing&mode=keyword")
    assert response.status_code == 200
    result = response.get_json()["results"][0]
    assert result["name"] == "guide.pdf" and result["exists"]
    assert result["path"] == str(pdf_path)

    inline = client.get(result["url"])
    download = client.get(result["download_url"])
    assert inline.status_code == download.status_code == 200
    assert inline.mimetype == download.mimetype == "application/pdf"
    assert "attachment" not in inline.headers.get("Content-Disposition", "")
    assert "attachment" in download.headers["Content-Disposition"]


def test_web_status_empty_query_and_request_validation(tmp_path):
    db_path = tmp_path / "empty.db"
    client = create_app(str(db_path)).test_client()

    status = client.get("/api/status")
    assert status.status_code == 200
    assert status.get_json()["files"] == 0
    assert isinstance(status.get_json()["semantic_available"], bool)

    empty = client.get("/api/search?q=%20%20")
    assert empty.status_code == 200 and empty.get_json()["results"] == []
    assert client.get("/api/search?q=x&mode=unknown").status_code == 400
    assert client.get("/api/search?q=x&limit=not-a-number").status_code == 400
    assert client.get("/file/999").status_code == 404


def test_web_reports_semantic_unavailable_as_service_unavailable(tmp_path, monkeypatch):
    import recall.web as web

    monkeypatch.setattr(web, "search", fail_search)
    client = create_app(str(tmp_path / "empty.db")).test_client()

    response = client.get("/api/search?q=meaning&mode=semantic")
    assert response.status_code == 503
    assert "model unavailable" in response.get_json()["error"]
