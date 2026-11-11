"""Local web interface for recall: search by content, then open or download the matching PDF.

Run with:  python -m recall.cli serve
"""
import os
import time

from flask import Flask, abort, g, jsonify, request, send_file, send_from_directory

from . import embed
from .db import connect
from .search import MODES, SemanticUnavailable, search

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
MAX_RESULTS = 20
DEFAULT_RESULTS = 8


def create_app(db_path="recall.db"):
    app = Flask(__name__, static_folder=None)
    app.config["RECALL_DB"] = db_path

    def db():
        if "db" not in g:
            g.db = connect(app.config["RECALL_DB"])
        return g.db

    @app.teardown_appcontext
    def close_db(_exc):
        con = g.pop("db", None)
        if con is not None:
            con.close()

    @app.after_request
    def harden(resp):
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["Referrer-Policy"] = "no-referrer"
        return resp

    @app.get("/")
    def home():
        return send_from_directory(STATIC_DIR, "index.html", max_age=0)

    @app.get("/api/status")
    def status():
        con = db()
        one = lambda sql: con.execute(sql).fetchone()[0]
        return jsonify(
            files=one("SELECT COUNT(*) FROM files"),
            chunks=one("SELECT COUNT(*) FROM chunks"),
            pending_embeddings=one("SELECT COUNT(*) FROM chunks WHERE embedding IS NULL"),
            failed_files=one("SELECT COUNT(*) FROM index_errors"),
            semantic_available=embed.available(),
        )

    @app.get("/api/search")
    def api_search():
        q = (request.args.get("q") or "").strip()
        mode = request.args.get("mode", "hybrid")
        if mode not in MODES:
            return jsonify(error=f"mode must be one of: {', '.join(MODES)}"), 400
        try:
            limit = max(1, min(int(request.args.get("limit", DEFAULT_RESULTS)), MAX_RESULTS))
        except ValueError:
            return jsonify(error="limit must be a number"), 400
        if not q:
            return jsonify(query=q, mode=mode, results=[], took_ms=0)

        con = db()
        started = time.perf_counter()
        try:
            found = search(con, q, mode, top_files=limit)
        except SemanticUnavailable as e:
            return jsonify(error=f"Search by meaning is unavailable: {e}"), 503

        results = []
        for r in found:
            row = con.execute("SELECT id FROM files WHERE path=?", (r["path"],)).fetchone()
            if row is None:
                continue
            fid = row[0]
            results.append(dict(
                file_id=fid,
                name=os.path.basename(r["path"]),
                folder=os.path.dirname(r["path"]),
                path=r["path"],
                page=r["page"],
                snippet=r["snippet"],
                why=r["why"],
                score=r.get("semantic_score"),
                exists=os.path.isfile(r["path"]),
                url=f"/file/{fid}",
                download_url=f"/file/{fid}?download=1",
            ))
        took = round((time.perf_counter() - started) * 1000)
        return jsonify(query=q, mode=mode, results=results, took_ms=took)

    @app.get("/file/<int:file_id>")
    def serve_file(file_id):
        row = db().execute("SELECT path FROM files WHERE id=?", (file_id,)).fetchone()
        if row is None or not os.path.isfile(row[0]):
            abort(404)
        path = row[0]
        return send_file(path, mimetype="application/pdf", conditional=True, max_age=0,
                         as_attachment=request.args.get("download") == "1",
                         download_name=os.path.basename(path))

    @app.errorhandler(404)
    def not_found(_e):
        if request.path.startswith("/api/") or request.path.startswith("/file/"):
            return jsonify(error="Not found. The file may have been moved or deleted; re-run `index`."), 404
        return "Not found", 404

    return app


def serve(db_path, host="127.0.0.1", port=8000):
    app = create_app(db_path)
    app.run(host=host, port=port, threaded=True, debug=False)