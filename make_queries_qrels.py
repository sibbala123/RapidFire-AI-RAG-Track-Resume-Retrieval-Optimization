import json
import re
from pathlib import Path
from collections import defaultdict

CORPUS_PATH = Path("dataset/corpus.jsonl")
OUT_QUERIES = Path("dataset/queries.jsonl")
OUT_QRELS = Path("dataset/qrels.tsv")

# ---- 1) Define your query concepts (edit freely) ----
CONCEPTS = [
    # data/engineering
    ("python", "Which candidate has Python experience?"),
    ("sql", "Which candidate has SQL experience?"),
    ("etl", "Which candidate has worked on ETL pipelines?"),
    ("spark", "Which candidate has Apache Spark experience?"),
    ("kafka", "Which candidate has Kafka experience?"),
    ("airflow", "Which candidate has Apache Airflow experience?"),
    ("databricks", "Which candidate has Databricks experience?"),
    ("snowflake", "Which candidate has Snowflake experience?"),
    ("aws", "Which candidate has AWS experience?"),
    ("azure", "Which candidate has Azure experience?"),
    ("gcp", "Which candidate has GCP experience?"),
    ("kubernetes", "Which candidate has Kubernetes experience?"),
    ("docker", "Which candidate has Docker experience?"),

    # HR / business (since your sample is HR-heavy)
    ("recruitment", "Which candidate has recruitment experience?"),
    ("benefits", "Which candidate has benefits administration experience?"),
    ("payroll", "Which candidate has payroll experience?"),
    ("hris", "Which candidate has HRIS experience?"),
    ("employee relations", "Which candidate has employee relations experience?"),
    ("project management", "Which candidate has project management experience?"),
    ("sharepoint", "Which candidate has SharePoint experience?"),

    # education keywords
    ("master", "Which candidate has a master's degree?"),
    ("bachelor", "Which candidate has a bachelor's degree?"),
]

# Limits to keep eval set sane & balanced
MAX_RELEVANT_PER_QUERY = 50   # cap positives per query (avoid overly broad like "sql")
MIN_RELEVANT_PER_QUERY = 5    # skip concepts that match too few resumes

def load_corpus():
    docs = []
    with CORPUS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            docs.append(rec)
    return docs

def normalize_text(s: str) -> str:
    s = s.lower()
    # collapse whitespace
    s = re.sub(r"\s+", " ", s)
    return s

def keyword_match(text: str, keyword: str) -> bool:
    """
    Word-boundary-ish match:
    - for multi-word keywords, do substring after normalization
    - for single word, use boundary regex
    """
    text_n = normalize_text(text)
    kw_n = normalize_text(keyword)

    if " " in kw_n:
        return kw_n in text_n

    # single token
    return re.search(rf"\b{re.escape(kw_n)}\b", text_n) is not None

docs = load_corpus()
print("Loaded corpus docs:", len(docs))

# Build relevance lists per concept
concept_to_hits = defaultdict(list)

for rec in docs:
    doc_id = str(rec["doc_id"])
    # prefer plain text, fallback to html-stripped
    text = rec.get("text_plain") or rec.get("text_html_stripped") or rec.get("text") or ""
    for kw, _ in CONCEPTS:
        if keyword_match(text, kw):
            concept_to_hits[kw].append(doc_id)

# Write queries + qrels
OUT_QUERIES.parent.mkdir(parents=True, exist_ok=True)

qid = 0
queries_written = 0
qrels_rows = []

with OUT_QUERIES.open("w", encoding="utf-8") as qf:
    for kw, question in CONCEPTS:
        hits = concept_to_hits.get(kw, [])
        if len(hits) < MIN_RELEVANT_PER_QUERY:
            print(f"Skipping '{kw}' (only {len(hits)} relevant)")
            continue

        # cap overly broad queries for cleaner evaluation
        hits = hits[:MAX_RELEVANT_PER_QUERY]

        qid += 1
        query_id = str(qid)

        qf.write(json.dumps({"query_id": query_id, "query": question}, ensure_ascii=False) + "\n")
        queries_written += 1

        for doc_id in hits:
            qrels_rows.append((query_id, doc_id, 1))

# Write qrels.tsv
with OUT_QRELS.open("w", encoding="utf-8") as rf:
    rf.write("query-id\tcorpus-id\tscore\n")
    for query_id, doc_id, rel in qrels_rows:
        rf.write(f"{query_id}\t{doc_id}\t{rel}\n")

print(f"✅ Wrote {queries_written} queries to {OUT_QUERIES}")
print(f"✅ Wrote {len(qrels_rows)} qrels rows to {OUT_QRELS}")
