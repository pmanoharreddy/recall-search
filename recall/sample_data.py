"""Deterministic sample corpus + labelled queries for evaluation.

    python -m recall.sample_data demo_pdfs queries.json

The corpus is SYNTHETIC (short study-note style pages written for this repo), so results on it
show the pipeline works; they are not evidence about performance on real notes. Queries are tagged:
  lexical    - share distinctive words with the target page
  paraphrase - describe the idea with little vocabulary overlap (what semantic search should help with)
"""
import json
import os
import sys

import pymupdf

DOCS = {
 "dbms_transactions.pdf": [
  "ACID properties. Atomicity means a transaction either completes fully or has no effect. Consistency keeps the database valid. Isolation hides concurrent transactions from each other. Durability means committed data survives a crash.",
  "Lost update problem. Two transactions T1 and T2 both read the balance 100. T1 adds 50 and writes 150. T2 subtracts 30 and writes 70. The write of T1 is overwritten, so the final balance is wrong. Locking the data item prevents this.",
  "Dirty read and unrepeatable read. A dirty read reads data written by an uncommitted transaction that later rolls back. An unrepeatable read gets different values when the same row is read twice in one transaction.",
  "Two phase locking has a growing phase where locks are acquired and a shrinking phase where locks are released. Strict two phase locking holds exclusive locks until commit. Deadlock between transactions is detected with a wait for graph."],
 "dbms_normalization.pdf": [
  "Functional dependency. X determines Y if two rows with equal X always have equal Y. Armstrong axioms are reflexivity, augmentation and transitivity. The closure of an attribute set finds every attribute it determines.",
  "First normal form requires atomic values. Second normal form removes partial dependency on part of a composite key. Third normal form removes transitive dependency so every non key attribute depends only on the key.",
  "Boyce Codd normal form is stricter than third normal form. For every functional dependency X to Y, X must be a superkey. Decomposition into BCNF may lose dependency preservation."],
 "dbms_indexing.pdf": [
  "A B plus tree keeps all records in the leaf nodes, which are linked for range scans. Internal nodes only guide the search. Height stays small so lookups need few disk reads.",
  "Hash indexing maps a key through a hash function to a bucket. It gives constant time equality lookups but cannot answer range queries. Collisions are handled with overflow chains."],
 "os_scheduling.pdf": [
  "First come first served runs processes in arrival order. A long process at the front makes the others wait, called the convoy effect. Shortest job first minimizes average waiting time but needs the burst length in advance.",
  "Round robin gives every process a fixed time quantum and then moves it to the back of the ready queue. A very large quantum behaves like FCFS and a tiny quantum wastes time on context switches.",
  "Priority scheduling runs the highest priority process first. Low priority processes may suffer starvation. Aging gradually raises the priority of processes that have waited a long time."],
 "os_memory.pdf": [
  "Paging splits memory into fixed size frames and programs into pages. The page table maps each virtual page to a physical frame. A TLB caches recent translations to speed up access.",
  "Page replacement chooses a victim when no frame is free. FIFO evicts the oldest page and can show Belady anomaly. LRU evicts the page unused for the longest time. The optimal algorithm evicts the page needed furthest in the future.",
  "Thrashing happens when processes spend more time swapping pages than executing. The working set model keeps the pages a process actively uses in memory to avoid it."],
 "os_deadlock.pdf": [
  "Four conditions are needed for deadlock: mutual exclusion, hold and wait, no preemption and circular wait. Processes are blocked forever, each holding a resource another one needs.",
  "Banker algorithm avoids deadlock. Before granting a request it checks whether the system stays in a safe state where some order lets every process finish with the available resources."],
 "networks_tcp.pdf": [
  "The TCP three way handshake opens a connection. The client sends SYN, the server replies SYN ACK, and the client answers ACK. Sequence numbers are synchronized during this exchange.",
  "TCP congestion control starts with slow start, where the congestion window doubles every round trip. After reaching the threshold it grows linearly, and a packet loss cuts the window.",
  "UDP is connectionless and unreliable. It has no retransmission or ordering, so it suits streaming, DNS lookups and games where low latency matters more than delivery."],
 "networks_osi.pdf": [
  "The OSI model has seven layers: physical, data link, network, transport, session, presentation and application. Routers work at the network layer using IP addresses. Switches work at the data link layer using MAC addresses."],
 "dsa_sorting.pdf": [
  "Quicksort picks a pivot and partitions the array so smaller elements go left and larger go right, then recurses on both sides. Average time is n log n but a bad pivot gives quadratic worst case.",
  "Merge sort splits the array in half, sorts each half and merges the sorted halves. It is stable and always n log n but needs extra memory for merging.",
  "Heap sort builds a max heap and repeatedly swaps the root with the last element. It sorts in place in n log n time but is not stable."],
 "dsa_graphs.pdf": [
  "Breadth first search explores a graph level by level using a queue. It finds the fewest edges path in an unweighted graph. Depth first search uses a stack or recursion.",
  "Dijkstra algorithm finds shortest paths from a source in a graph with non negative edge weights. It repeatedly picks the unvisited vertex with the smallest distance using a priority queue and relaxes its edges.",
  "A minimum spanning tree connects all vertices with the least total edge weight. Kruskal sorts edges and adds them unless they form a cycle, using union find. Prim grows a tree from one vertex."],
 "ml_basics.pdf": [
  "Overfitting means the model memorizes training noise and performs poorly on new data. Regularization such as L2 penalty, dropout, early stopping and more training data reduce it.",
  "Gradient descent minimizes a loss by stepping against the gradient. The learning rate controls the step size: too large diverges, too small is slow. Stochastic versions use mini batches.",
  "K means clustering assigns each point to the nearest centroid, then moves each centroid to the mean of its points. It repeats until assignments stop changing and needs the number of clusters in advance."],
 "compilers.pdf": [
  "Lexical analysis converts source characters into tokens such as identifiers, keywords and numbers using regular expressions and finite automata. Whitespace and comments are discarded.",
  "Parsing checks that tokens follow the grammar and builds a parse tree. Top down LL parsers expand from the start symbol. Bottom up LR parsers reduce the input to the start symbol using a stack."],
 "portfolio.pdf": [
  "Campus Queue is a web application that lets students reserve a place in the cafeteria line. It uses Python, FastAPI, and SQLite, with token based sign in."],
}

# (query, expected file, expected page, type)
QUERIES = [
 ("the pdf that explains the lost update problem with an example", "dbms_transactions.pdf", 2, "lexical"),
 ("concurrent clients clobbering each other's changes to a shared bank account figure", "dbms_transactions.pdf", 2, "paraphrase"),
 ("dirty read versus unrepeatable read", "dbms_transactions.pdf", 3, "lexical"),
 ("seeing somebody else's tentative changes that then get undone", "dbms_transactions.pdf", 3, "paraphrase"),
 ("strict two phase locking and wait for graph", "dbms_transactions.pdf", 4, "lexical"),
 ("third normal form transitive dependency", "dbms_normalization.pdf", 2, "lexical"),
 ("the determining side of each constraint must uniquely identify rows", "dbms_normalization.pdf", 3, "paraphrase"),
 ("Armstrong axioms and attribute closure", "dbms_normalization.pdf", 1, "lexical"),
 ("balanced multiway structure where the bottom level is chained for sequential traversal over intervals", "dbms_indexing.pdf", 1, "paraphrase"),
 ("hash index bucket collisions overflow chain", "dbms_indexing.pdf", 2, "lexical"),
 ("convoy effect in first come first served", "os_scheduling.pdf", 1, "lexical"),
 ("each job gets a short equal turn on the processor then waits for the next lap", "os_scheduling.pdf", 2, "paraphrase"),
 ("important tasks always win so unimportant ones may never execute unless their rank slowly improves", "os_scheduling.pdf", 3, "paraphrase"),
 ("page table maps virtual pages to frames TLB", "os_memory.pdf", 1, "lexical"),
 ("Belady anomaly FIFO LRU optimal replacement", "os_memory.pdf", 2, "lexical"),
 ("machine busy moving data to and from disk instead of doing useful work", "os_memory.pdf", 3, "paraphrase"),
 ("four necessary conditions for deadlock circular wait", "os_deadlock.pdf", 1, "lexical"),
 ("simulating if everyone could still complete when approving an allocation", "os_deadlock.pdf", 2, "paraphrase"),
 ("SYN SYN ACK ACK connection setup", "networks_tcp.pdf", 1, "lexical"),
 ("sending rate ramps up exponentially at first then cautiously after hitting a ceiling", "networks_tcp.pdf", 2, "paraphrase"),
 ("fire and forget transport good for video calls and name resolution", "networks_tcp.pdf", 3, "paraphrase"),
 ("seven layers routers switches MAC addresses", "networks_osi.pdf", 1, "lexical"),
 ("pivot partition worst case quadratic", "dsa_sorting.pdf", 1, "lexical"),
 ("recursive bisect, order both parts, then zip them back, keeping equal keys in original order but using scratch buffer", "dsa_sorting.pdf", 2, "paraphrase"),
 ("max heap swap root with last element in place", "dsa_sorting.pdf", 3, "lexical"),
 ("visit neighbours ring by ring from a source with a FIFO container", "dsa_graphs.pdf", 1, "paraphrase"),
 ("shortest path with non negative weights priority queue relax edges", "dsa_graphs.pdf", 2, "lexical"),
 ("cheapest way to wire up every node together without loops", "dsa_graphs.pdf", 3, "paraphrase"),
 ("network that clings too tightly to its examples so fails on fresh inputs", "ml_basics.pdf", 1, "paraphrase"),
 ("learning rate too large diverges", "ml_basics.pdf", 2, "lexical"),
 ("unsupervised grouping where samples join the closest prototype and prototypes drift to group averages", "ml_basics.pdf", 3, "paraphrase"),
 ("chopping program text into meaningful words via pattern matching machines", "compilers.pdf", 1, "paraphrase"),
 ("LL versus LR parsers", "compilers.pdf", 2, "lexical"),
 ("Campus Queue Python FastAPI SQLite", "portfolio.pdf", 1, "lexical"),
 ("What student-facing tool was built, and what programming stack powers it?", "portfolio.pdf", 1, "paraphrase"),
]

BROAD_DOCS = {
 "portfolio.pdf": [
  "Aster is a project that helps applicants practice job interviews using a conversational mock interviewer.",
  "The application uses a browser interface with React, a Node and Express service, MongoDB storage, and the Gemini API."],
 "candidate_profile.pdf": [
  "The candidate has expertise in Python programming, backend API design, data analysis, and cloud deployment.",
  "Professional strengths include clear written communication, collaborative problem solving, mentoring, and project planning."],
 "research_report.pdf": [
  "This report investigates urban heat islands and tree canopy coverage in three neighborhoods, examining evidence from each area.",
  "It compares satellite temperature measurements with land-use surveys and recommends targeted planting corridors."],
 "course_notes.pdf": [
  "The notes explain database normalization, transaction isolation, indexing, and query planning with worked examples.",
  "They compare relational storage choices and show how constraints protect data integrity."],
 "college_calendar.pdf": [
  "The college calendar lists examination dates, cultural events, holidays, and department activities."],
 "cutoff_tables.pdf": [
  "Engineering admission cut-off ranks by institute, course, category, and counseling round."],
 "booking_analysis.pdf": [
  "Duplicate booking rows were removed before analysis; unique records were retained for occupancy calculations."],
 "network_guide.pdf": [
  "A guide compares network routing, transport protocols, packet loss, and congestion management."],
 "job_application.pdf": [
  "An applicant describes leadership responsibilities, customer service, and scheduling experience."],
}

BROAD_QUERIES = [
 ("Which projects has this student developed?", "portfolio.pdf", "projects"),
 ("What technologies power the candidate applications?", "portfolio.pdf", "technologies"),
 ("Which projects and technologies were used to build them?", "portfolio.pdf", "projects+technologies"),
 ("What technical skills does this candidate possess?", "candidate_profile.pdf", "skills"),
 ("Which strengths and capabilities are described for this applicant?", "candidate_profile.pdf", "skills"),
 ("What does the report cover and which evidence supports its findings?", "research_report.pdf", "document-contents"),
 ("What subjects are explained in these course materials?", "course_notes.pdf", "document-contents"),
 ("What are the main ideas in the provided study material?", "course_notes.pdf", "document-contents"),
]


def _write_pdf_corpus(pdf_dir, corpus):
    os.makedirs(pdf_dir, exist_ok=True)
    for name, pages in corpus.items():
        doc = pymupdf.open()
        for text in pages:
            page = doc.new_page()
            page.insert_textbox(pymupdf.Rect(50, 50, 550, 790), text, fontsize=11)
        doc.save(os.path.join(pdf_dir, name))
        doc.close()


def generate(pdf_dir, queries_path):
    _write_pdf_corpus(pdf_dir, DOCS)
    data = [{"query": q, "expected": f, "pages": [p], "type": t} for q, f, p, t in QUERIES]
    with open(queries_path, "w") as fh:
        json.dump(data, fh, indent=1)
    return len(DOCS), len(data)


def generate_broad(pdf_dir, queries_path):
    """Generate a separate, labeled corpus for broad project, skills, and content questions."""
    _write_pdf_corpus(pdf_dir, BROAD_DOCS)
    data = [{"query": q, "expected": filename, "type": kind}
            for q, filename, kind in BROAD_QUERIES]
    with open(queries_path, "w") as fh:
        json.dump(data, fh, indent=1)
    return len(BROAD_DOCS), len(data)


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--broad":
        pdfs, queries = generate_broad(*(args[1:3] or ["broad_demo_pdfs", "broad_queries.json"]))
    else:
        pdfs, queries = generate(*(args[:2] or ["demo_pdfs", "queries.json"]))
    print(f"wrote {pdfs} PDFs and {queries} labelled queries")
