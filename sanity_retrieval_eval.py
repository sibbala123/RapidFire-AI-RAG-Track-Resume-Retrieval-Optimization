#!/usr/bin/env python3
"""
01_sanity_retrieval_eval.py

First script to run after you have:
- dataset/corpus.jsonl
- dataset/queries.jsonl
- dataset/qrels.tsv

What it does:
1) Load resumes (corpus)
2) Chunk resumes (2 chunk configs)
3) Embed chunks
4) Build FAISS
5) Retrieve top-K chunks for each query
6) Collapse chunks -> parent resume doc_ids
7) Compute retrieval metrics (Precision/Recall/MRR/NDCG@5)

Run:
  source .venv/bin/activate
  python 01_sanity_retrieval_eval.py
"""

import json
import math
from pathlib import Path
from typing import List, Dict, Any, Tuple

import pandas as pd

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


# -----------------------------
# Paths
# -----------------------------
DATASET_DIR = Path("dataset")
CORPUS_PATH = DATASET_DIR / "corpus.jsonl"
QUERIES_PATH = DATASET_DIR / "queries.jsonl"
QRELS_PATH = DATASET_DIR / "qrels.tsv"


# -----------------------------
# Metrics
# -----------------------------
def compute_rr(ranked_docs: List[int], expected: set) -> float:
    for i, d in enumerate(ranked_docs):
        if d in expected:
            return 1.0 / (i + 1)
    return 0.0


def compute_ndcg_at_k(ranked_docs: List[int], expected: set, k: int = 5) -> float:
    # Binary relevance: 1 if in expected else 0
    rels = [1 if d in expected else 0 for d in ranked_docs[:k]]
    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(rels))

    # Ideal DCG: all relevant first
    ideal_len = min(k, len(expected))
    ideal_rels = [1] * ideal_len + [0] * (k - ideal_len)
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_rels))

    return dcg / idcg if idcg > 0 else 0.0


def compute_metrics(
    retrieved_ranked: List[List[int]],
    ground_truth: List[List[int]],
    k_for_precision_recall: int = 5,
) -> Dict[str, float]:
    assert len(retrieved_ranked) == len(ground_truth)

    precisions, recalls, rrs, ndcgs = [], [], [], []

    for pred, gt in zip(retrieved_ranked, ground_truth):
        expected = set(gt)
        ranked = pred[:k_for_precision_recall]

        if len(ranked) == 0:
            precisions.append(0.0)
        else:
            tp = len(set(ranked) & expected)
            precisions.append(tp / len(ranked))

        recalls.append((len(set(ranked) & expected) / len(expected)) if expected else 0.0)
        rrs.append(compute_rr(pred, expected))
        ndcgs.append(compute_ndcg_at_k(pred, expected, k=5))

    n = len(ground_truth)
    return {
        f"Precision@{k_for_precision_recall}": sum(precisions) / n,
        f"Recall@{k_for_precision_recall}": sum(recalls) / n,
        "MRR": sum(rrs) / n,
        "NDCG@5": sum(ndcgs) / n,
    }


# -----------------------------
# Loading helpers
# -----------------------------
def load_corpus_docs() -> List[Document]:
    if not CORPUS_PATH.exists():
        raise FileNotFoundError(f"Missing: {CORPUS_PATH}")

    docs: List[Document] = []
    with CORPUS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            doc_id = int(rec["doc_id"])
            text = rec.get("text") or rec.get("text_plain") or ""
            meta = {
                "corpus_id": doc_id,  # IMPORTANT: parent resume id
                "category": rec.get("category"),
            }
            docs.append(Document(page_content=text, metadata=meta))
    return docs


def load_queries() -> Tuple[List[int], List[str]]:
    if not QUERIES_PATH.exists():
        raise FileNotFoundError(f"Missing: {QUERIES_PATH}")

    qids, queries = [], []
    with QUERIES_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            qids.append(int(rec["query_id"]))
            queries.append(rec["query"])
    return qids, queries


def load_qrels() -> pd.DataFrame:
    if not QRELS_PATH.exists():
        raise FileNotFoundError(f"Missing: {QRELS_PATH}")

    qrels = pd.read_csv(QRELS_PATH, sep="\t")
    qrels = qrels.rename(
        columns={"query-id": "query_id", "corpus-id": "corpus_id", "score": "relevance"}
    )
    # Ensure ints for matching
    qrels["query_id"] = qrels["query_id"].astype(int)
    qrels["corpus_id"] = qrels["corpus_id"].astype(int)
    return qrels


def build_ground_truth_lists(qids: List[int], qrels: pd.DataFrame) -> List[List[int]]:
    gt = []
    for qid in qids:
        gt.append(qrels[qrels["query_id"] == qid]["corpus_id"].tolist())
    return gt


# -----------------------------
# Retrieval eval
# -----------------------------
def collapse_chunks_to_doc_ids(retrieved_docs: List[Document]) -> List[int]:
    """
    retrieved_docs are chunk Documents.
    Each chunk has metadata['corpus_id'] = parent resume id.
    We dedup while preserving order.
    """
    ordered = []
    seen = set()
    for d in retrieved_docs:
        cid = int(d.metadata["corpus_id"])
        if cid not in seen:
            seen.add(cid)
            ordered.append(cid)
    return ordered


def run_one_config(
    base_docs: List[Document],
    splitter: RecursiveCharacterTextSplitter,
    embeddings: HuggingFaceEmbeddings,
    queries: List[str],
    top_k_chunks: int = 8,
) -> List[List[int]]:
    # 1) chunk the corpus
    chunked_docs = splitter.split_documents(base_docs)

    # 2) build FAISS over chunk embeddings
    vs = FAISS.from_documents(chunked_docs, embeddings)

    # 3) retrieve for each query: top-k chunks
    retrieved_ranked_doc_ids: List[List[int]] = []
    for q in queries:
        chunks = vs.similarity_search(q, k=top_k_chunks)
        doc_ids = collapse_chunks_to_doc_ids(chunks)
        retrieved_ranked_doc_ids.append(doc_ids)

    return retrieved_ranked_doc_ids


def main():
    print("== Loading files ==")
    corpus_docs = load_corpus_docs()
    qids, queries = load_queries()
    qrels = load_qrels()
    ground_truth = build_ground_truth_lists(qids, qrels)

    print(f"Corpus resumes: {len(corpus_docs)}")
    print(f"Queries: {len(queries)}")
    print(f"Qrels rows: {len(qrels)}")

    print("\n== Embeddings ==")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},  # set "cuda:0" if your local has GPU
        encode_kwargs={"normalize_embeddings": True, "batch_size": 64},
    )

    # Two starter chunking configs (you'll add HTML-heading later)
    configs = [
        ("chunk256_overlap32", RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="gpt2", chunk_size=256, chunk_overlap=32
        )),
        ("chunk128_overlap32", RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="gpt2", chunk_size=128, chunk_overlap=32
        )),
    ]

    # Retrieval settings
    top_k_chunks = 8
    k_for_metrics = 5  # Precision@5 / Recall@5

    print("\n== Running retrieval sanity eval ==")
    results = []

    for name, splitter in configs:
        print(f"\n--- Config: {name} ---")
        retrieved = run_one_config(
            base_docs=corpus_docs,
            splitter=splitter,
            embeddings=embeddings,
            queries=queries,
            top_k_chunks=top_k_chunks,
        )
        metrics = compute_metrics(retrieved, ground_truth, k_for_precision_recall=k_for_metrics)
        print(metrics)
        results.append((name, metrics))

    print("\n== Summary ==")
    for name, metrics in results:
        print(name, "=>", metrics)

    # Optional: print one debug example
    idx = 0
    print("\n== Debug one query ==")
    print("Query:", queries[idx])
    print("Retrieved doc_ids:", results[0][1], "(metrics dict shown above)")
    print("Ground truth doc_ids:", ground_truth[idx])


if __name__ == "__main__":
    main()
