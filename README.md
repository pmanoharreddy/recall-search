# recall — "I remember the information, not the file."

Local-first semantic + keyword search over your PDFs. Returns the best file, page and passage.

## Install, index, search
    pip install -r requirements.txt                 # pymupdf, numpy, sentence-transformers
    python -m recall.cli index ~/College/Notes      # add --no-embed for keyword-only (fast, no model)
    python -m recall.cli --db recall.db embed        # backfill missing embeddings from stored chunks; no PDF scan
    python -m recall.cli search "the pdf that explains the lost update problem with an example"
    python -m recall.cli search "..." --mode keyword|semantic|hybrid
    python -m recall.cli status                     # counts, pending embeddings, files that failed to index
Use `--db path.db` to choose the database (default `recall.db`). Exit codes: 0 ok, 1 some files failed to
index (they are retried next run), 2 error (bad folder, semantic search unavailable).

## Tests
    pip install -r requirements-dev.txt
    python -m pytest -q          # offline: embedding model is mocked, no downloads

## Evaluation
    python -m recall.sample_data demo_pdfs queries.json      # deterministic synthetic corpus + 35 labelled queries
    python -m recall.cli index demo_pdfs
    python -m recall.eval queries.json recall.db --json eval_results.json
    python -m recall.sample_data --broad broad_demo_pdfs broad_queries.json  # focused 8-query broad-intent corpus
    python -m recall.cli --db broad.db index broad_demo_pdfs
    python -m recall.eval broad_queries.json broad.db --json broad_eval_results.json
Query file format: `[{"query": "...", "expected": "file.pdf", "pages": [2], "type": "lexical|paraphrase"}]`.
Metrics: Recall@k, MRR, Top-1 accuracy, Page@k (right file *and* right page), per-type breakdown.
A mode that cannot run is printed as `NOT MEASURED`, never as 0%. For real conclusions, write your own
50-100 queries about your own notes; the synthetic corpus only shows the pipeline works.

### Measured results
Measured locally with `all-MiniLM-L6-v2` on the synthetic corpus (13 PDFs / 33 chunks / 35 labelled queries),
at k=5. All modes use the same query and expected-file labels.

| mode | Recall@5 | MRR | Top-1 accuracy | Page@5 |
|---|---:|---:|---:|---:|
| keyword | 62.9% | 0.595 | 57.1% | 54.3% |
| semantic | 100.0% | 0.943 | 88.6% | 94.3% |
| hybrid | 100.0% | 0.914 | 82.9% | 91.4% |

On the 17 paraphrase queries, Recall@5 / MRR / Top-1 were 23.5% / 0.167 / 11.8% for keyword,
100.0% / 0.882 / 76.5% for semantic, and 100.0% / 0.824 / 64.7% for hybrid. This small synthetic
benchmark characterizes this corpus; it does not establish a performance improvement on personal PDFs.
Use 30-50 labelled queries about your own documents before drawing conclusions about their retrieval quality.

The focused broad-intent corpus has 9 synthetic documents and 8 labeled questions covering project discovery,
technology stacks, candidate skills, research-report contents, and study-note topics. It is intended to expose
ranking tradeoffs for broad questions, not to represent personal-document accuracy.

Measured with the cached `all-MiniLM-L6-v2` model on 13 passages from this corpus at k=5. The baseline disables
only the broad semantic query normalizer; indexing, labels, document scoring, and the embedding model are
otherwise identical.

| mode | baseline Recall@5 | baseline MRR | current Recall@5 | current MRR |
|---|---:|---:|---:|---:|
| keyword | 25.0% | 0.250 | 25.0% | 0.250 |
| semantic | 100.0% | 0.688 | 100.0% | 0.719 |
| hybrid | 100.0% | 0.719 | 100.0% | 0.719 |

The current broad project/technology normalizer raises semantic MRR by 0.031 on this small corpus, without
changing Recall@5; hybrid metrics are unchanged. This is a small measured difference, not evidence of general
improvement. Additional generic skills/content rewrites lowered MRR in exploratory runs, so they were not
adopted. The existing 35-query synthetic benchmark remains at keyword 62.9% / 0.595, semantic 100% / 0.943,
and hybrid 100% / 0.914 for Recall@5 / MRR.

### Keyword matching rule
Filler words (the, way, up, every, ...) are ignored. A query with 1-2 meaningful words matches any chunk containing
one of them; with 3+ meaningful words a chunk must contain at least 2. Semantic search is unaffected.

### Broad semantic questions
Broad questions that ask about both projects and their technologies are normalized to a general project/portfolio
and software-stack search phrase while retaining other meaningful query terms. No external service or extra model
is used, and specific named-project queries keep their original wording. Semantic retrieval considers up to 200
candidate passages, scores files using their best distinct passage plus a small corroboration bonus from up to
two other passages, and returns one result per file. Hybrid RRF combines file-level keyword and semantic
rankings; keyword-only ranking is unchanged.

## Indexing guarantees
- Text is extracted *before* the database is touched; each file is then written in one transaction
  (chunks + hash). The new hash is stored only if everything succeeded, so a failed re-index leaves the last
  good index of that file intact, and the file is retried on every later run (`index_errors` table / `status`).
- Chunks are written without embeddings; a separate backfill step embeds every chunk whose embedding is NULL,
  keyed by chunk id. So `--no-embed` followed by a normal run, an interrupted run, or a failed batch are all
  healed by simply running `index` again. Already-embedded chunks are never recomputed; if nothing is pending
  the model is not even loaded. A failing batch is retried chunk by chunk so one bad chunk can't block others.
- `python -m recall.cli embed` runs that same retryable backfill on stored chunk text without scanning or
  re-extracting PDFs. Semantic search warns when its cached vector index contains only a subset of chunks.
- Changing the embedding model (or finding wrong-sized vectors) re-embeds; the vectors are derived data.
- Lifecycle: added / unchanged (skipped by SHA-256) / modified (replaced) / deleted (removed) / moved or
  renamed (same hash, old path gone: path updated, chunks and embeddings kept). Indexing a folder that does
  not exist or cannot be listed never deletes anything from the index.
- Existing databases open unchanged: the only schema change is two new tables (`meta`, `index_errors`) created
  automatically; no data is migrated or deleted.

## Semantic search scalability
Vectors are held in one in-memory matrix cached per database and reloaded only when an indexing write bumps
the stored `generation` (not per query). NumPy brute force is exact; measured on synthetic 384-d vectors:
10k vectors 0.9 ms/query (15 MB), 100k vectors 18 ms/query (154 MB, ~0.5 s one-time load). Optional exact FAISS
backend (`pip install faiss-cpu`, `RECALL_VECTOR_BACKEND=faiss`) was only ~15-20% faster at 100k, so it is opt-in.

## Known limitations
- Semantic search has no similarity cutoff: it always returns nearest neighbours, so in `semantic`/`hybrid`
  mode an unrelated query still returns (low-quality) files. A threshold needs tuning on the real model.
- Scanned PDFs yield no text (reported as `no_text`); OCR is not implemented.
- Every run re-hashes every PDF (correct, but O(total bytes) on big folders).
- Single-process use assumed; no file watcher yet. Only `.pdf` is supported.
