# Recall

Recall searches local PDFs by filename, exact terms, or meaning and opens the matching page.
It combines SQLite FTS5 keyword search with MiniLM passage embeddings and document-level rank fusion.
The command line and local Flask app share the same index.

## Try the bundled demo

From the project folder, run these commands inside a Python virtual environment.

```sh
python -m pip install -r requirements-web.txt
python -m recall.cli --db demo.db index demo_pdfs
python -m recall.cli --db demo.db serve
```

Open [Recall on this computer](http://127.0.0.1:8000/) and search for
`lost update` to find a passage in `dbms_transactions.pdf`. Then try `dbms_transactions.pdf`
to look up the file directly. The first semantic index run downloads the embedding model.
Use `index demo_pdfs --no-embed` for a filename and keyword demo without a model download.

## Install, index, search
    pip install -r requirements.txt                 # pymupdf, numpy, sentence-transformers
    python -m recall.cli index ~/College/Notes      # add --no-embed for keyword-only (fast, no model)
    python -m recall.cli --db recall.db embed        # backfill missing embeddings from stored chunks; no PDF scan
    python -m recall.cli search "the pdf that explains the lost update problem with an example"
    python -m recall.cli search "..." --mode keyword|semantic|hybrid
    python -m recall.cli status                     # counts, pending embeddings, files that failed to index
    python -m recall.cli --db recall.db serve       # optional local browser interface at 127.0.0.1:8000
Use `--db path.db` to choose the database (default `recall.db`). Exit codes: 0 ok, 1 some files failed to
index (they are retried next run), 2 error (bad folder, semantic search unavailable).

The optional web interface requires Flask: `pip install -r requirements-web.txt`. It binds to this computer
only by default. Setting `serve --host` to a non-loopback address exposes indexed PDF search and downloads on
the network; the server has no login, so only do this on a trusted network.

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
| hybrid | 100.0% | 0.914 | 82.9% | 94.3% |

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
one of them; with 3+ meaningful words a chunk must contain at least 2. Each file contributes its best
matching passage. Repeated hits in a long PDF leave other matching files eligible for the results.

In Exact words and Both, passages containing the full sequence of query words take priority
for queries with at least two meaningful terms. Phrase matching uses the same tokenizer and
stemming as keyword search. The result opens at a page containing the phrase.

Recall ranks byte-identical PDFs together using their stored SHA-256 digest. **Other locations**
opens the same page from each indexed copy. Separate files that share a name or a passage
keep their own results.

### Filename lookup and hybrid ranking

A complete PDF filename takes priority in every mode. A filename stem such as `annual report`
also finds `Annual_Report.pdf`. Matching ignores case, accent and separator differences, with
literal filenames ranked first. This works for indexed scans that have no extracted text.
Filename results open at the first indexed passage, or page 1 for a scan.

Both mode combines the keyword and semantic file rankings using reciprocal rank fusion with
`k=3` so the first few ranks carry more weight. When the leading keyword passage contains every
meaningful query term and only one file has such a passage, Recall puts that file and passage
first. For a descriptive question, Both gives semantic evidence extra weight within the
first eight candidates. Phrase and filename matches retain priority.

On a local index of 762 PDFs and 47,919 passages, 16 content clues targeted the same résumé.
These were development cases used to tune the earlier filename and rank-fusion update.
Before that update, the hybrid ranker returned the file in the top eight for 13/16 clues and
first for 11/16. After that update, it appeared
in the top eight for 16/16 and first for 15/16. Two additional filename queries also ranked
it first. On the two synthetic corpora above, hybrid search retrieved the expected file in
the top five for 35/35 and 8/8 questions respectively.

In that earlier evaluation, 33 of the study-note questions had
their labelled demo PDF present among the 762 files. Counting a byte-identical copy as a hit,
the expected PDF appeared in the top eight for 21/33 questions with the previous ranking and
24/33 with the update. First place stayed at 16/33. These questions reuse the bundled labels.

### Desktop retrieval benchmark

The next evaluation used 250 fixed queries targeting 100 different PDFs in the 762-file
collection. Fifty queries used filenames. The other 200 used phrases, keywords, specific
paraphrases and broad descriptions. A byte-identical copy counted as the intended file.
These cases informed development. Both versions ran against the same index.

| Both mode | Correct file first | Correct file in top eight |
|---|---:|---:|
| Before phrase matching and copy grouping | 121/250 (48.4%) | 189/250 (75.6%) |
| With phrase matching, copy grouping and descriptive reranking | 138/250 (55.2%) | 199/250 (79.6%) |

On the 200 content clues alone, top-eight retrieval improved from 139 to 149.
Remembered phrases reached 45/50. Broad descriptions reached 24/50.
In a separate check using five-word excerpts from 50 additional PDFs, the expected
file appeared in the top eight for 46/50 queries, compared with 40/50 before the update.

### Broad semantic questions
Broad questions that ask about both projects and their technologies are normalized to a general project/portfolio
and software-stack search phrase while retaining other meaningful query terms. No external service or extra model
is used, and specific named-project queries keep their original wording. Semantic retrieval considers up to 200
candidate passages, scores files using their best distinct passage plus a small corroboration bonus from up to
two other passages, and returns one result per distinct PDF. Hybrid RRF combines the keyword and semantic
rankings. For a lone ASCII identifier that the embedding tokenizer splits into many short fragments,
semantic search requires a literal indexed term before returning neighbours.

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
- A fluent but unrelated question can still return weak semantic matches.
  Set `RECALL_MIN_SIMILARITY` to a cosine similarity threshold to filter the semantic results.
  Tune it against labelled queries from your own PDFs.
- Scanned PDFs yield no text (reported as `no_text`); OCR is not implemented.
- Every run re-hashes every PDF (correct, but O(total bytes) on big folders).
- Single-process use assumed; no file watcher yet. Only `.pdf` is supported.
