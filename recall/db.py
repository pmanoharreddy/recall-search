import contextlib
import sqlite3

# Tables below the original three are additive (CREATE IF NOT EXISTS), so databases
# created by earlier versions open unchanged and need no migration.
SCHEMA = """
CREATE TABLE IF NOT EXISTS files(
  id INTEGER PRIMARY KEY, path TEXT UNIQUE, sha256 TEXT, mtime REAL);
CREATE TABLE IF NOT EXISTS chunks(
  id INTEGER PRIMARY KEY, file_id INTEGER REFERENCES files(id) ON DELETE CASCADE,
  page INTEGER, text TEXT, embedding BLOB);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
  text, content='chunks', content_rowid='id', tokenize='porter unicode61');
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
  INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text); END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
  INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES('delete', old.id, old.text); END;

-- key/value store: embedding_model, embedding_dim, generation (bumped on every index write)
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
-- files whose last (re)index attempt failed; cleared when a later attempt succeeds
CREATE TABLE IF NOT EXISTS index_errors(
  path TEXT PRIMARY KEY, error TEXT NOT NULL, sha256 TEXT, failed_at REAL);
CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_id);
CREATE INDEX IF NOT EXISTS idx_files_sha ON files(sha256);
"""


def connect(path="recall.db"):
    con = sqlite3.connect(path)
    con.execute("PRAGMA foreign_keys=ON")
    con.executescript(SCHEMA)
    return con


@contextlib.contextmanager
def transaction(con, name="recall_txn"):
    """Atomic block: commits on success, rolls back everything on any exception.

    Uses SAVEPOINT so it works regardless of the connection's isolation mode and nests safely.
    """
    con.execute(f"SAVEPOINT {name}")
    try:
        yield con
    except BaseException:
        con.execute(f"ROLLBACK TO {name}")
        con.execute(f"RELEASE {name}")
        raise
    else:
        con.execute(f"RELEASE {name}")


def get_meta(con, key, default=None):
    row = con.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return row[0] if row else default


def set_meta(con, key, value):
    con.execute("INSERT INTO meta(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, str(value)))


def bump_generation(con):
    """Called inside every write transaction so cached vector indexes know they are stale."""
    con.execute("INSERT INTO meta(key, value) VALUES('generation', '1') "
                "ON CONFLICT(key) DO UPDATE SET value=CAST(value AS INTEGER)+1")


def get_generation(con):
    return int(get_meta(con, "generation", 0))
