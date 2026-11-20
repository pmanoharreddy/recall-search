import argparse
import ipaddress
import logging
import sys

from .db import connect
from . import embed
from .indexer import index_folder, sync_embeddings
from .search import SemanticUnavailable, search


def _status(con):
    q = lambda sql: con.execute(sql).fetchone()[0]
    print(f"files: {q('SELECT COUNT(*) FROM files')}")
    print(f"chunks: {q('SELECT COUNT(*) FROM chunks')}")
    print(f"chunks with embeddings: {q('SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL')}")
    print(f"chunks pending embeddings: {q('SELECT COUNT(*) FROM chunks WHERE embedding IS NULL')}")
    errs = con.execute("SELECT path, error FROM index_errors ORDER BY path").fetchall()
    print(f"files failing to index: {len(errs)}")
    for p, e in errs:
        print(f"  ! {p}: {e}")


def _port(value):
    try:
        port = int(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError("port must be an integer from 1 to 65535") from e
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be an integer from 1 to 65535")
    return port


def _host(value):
    if not value.strip():
        raise argparse.ArgumentTypeError("host must not be empty")
    return value


def _is_loopback_host(host):
    if host.casefold() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _serve(a):
    try:
        from .web import serve
    except ModuleNotFoundError as e:
        if e.name != "flask":
            raise
        print("error: the web interface needs Flask; run: pip install flask", file=sys.stderr)
        return 2
    if not _is_loopback_host(a.host):
        print(f"warning: --host {a.host} lets other devices on your network search and download your "
              f"indexed PDFs. There is no login.", file=sys.stderr)
    display_host = f"[{a.host}]" if ":" in a.host else a.host
    print(f"starting recall at http://{display_host}:{a.port}  (Ctrl+C to stop)")
    try:
        serve(a.db, a.host, a.port)
    except KeyboardInterrupt:
        return 130
    except OSError as e:
        print(f"error: could not start web server: {e}", file=sys.stderr)
        return 2
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="recall")
    ap.add_argument("--db", default="recall.db")
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("index"); i.add_argument("folder"); i.add_argument("--no-embed", action="store_true")
    sub.add_parser("embed", help="backfill missing or stale embeddings without reading PDFs")
    s = sub.add_parser("search"); s.add_argument("query")
    s.add_argument("--mode", default="hybrid", choices=["keyword", "semantic", "hybrid"])
    sub.add_parser("status", help="show index counts, pending embeddings and failing files")
    w = sub.add_parser("serve", help="start the local web interface (search, then open or download PDFs)")
    w.add_argument("--host", type=_host, default="127.0.0.1", help="default 127.0.0.1 = this computer only")
    w.add_argument("--port", type=_port, default=8000)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    if a.cmd == "serve":
        return _serve(a)

    con = connect(a.db)
    try:
        if a.cmd == "index":
            try:
                stats = index_folder(con, a.folder, use_embeddings=not a.no_embed)
            except NotADirectoryError as e:
                print(f"error: {e}", file=sys.stderr)
                return 2
            print({k: v for k, v in stats.items() if k not in ("errors", "unreadable_dirs")})
            if stats["failed"] or stats["unreadable_dirs"]:
                print(f"{stats['failed']} file(s) failed to index; they will be retried on the next run "
                      f"(see `status`).", file=sys.stderr)
                return 1
        elif a.cmd == "status":
            _status(con)
        elif a.cmd == "embed":
            if not embed.available():
                print("error: sentence-transformers is not installed; run: pip install sentence-transformers",
                      file=sys.stderr)
                return 2
            stats = sync_embeddings(con, log=lambda message: print(message, file=sys.stderr))
            con.commit()
            pending = con.execute("SELECT COUNT(*) FROM chunks WHERE embedding IS NULL").fetchone()[0]
            print(f"chunks embedded: {stats['embedded']}")
            print(f"chunks still pending: {pending}")
            if stats["errors"] or pending:
                return 1
        else:
            try:
                results = search(con, a.query, a.mode)
            except SemanticUnavailable as e:
                print(f"error: semantic search unavailable: {e}", file=sys.stderr)
                return 2
            if not results:
                print("No results.")
            for n, r in enumerate(results, 1):
                sim = f"  sim={r['semantic_score']:.2f}" if r.get("semantic_score") is not None else ""
                print(f"{n}. {r['path']}  (page {r['page']})  [{r['why']}]{sim}\n   {r['snippet']}\n")
                for copy in r.get("copies", []):
                    print(f"   Other location: {copy['path']}")
        return 0
    finally:
        con.close()


if __name__ == "__main__":
    sys.exit(main())
