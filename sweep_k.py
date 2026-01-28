#!/usr/bin/env python3
import json, math
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

DATASET_DIR = Path("dataset")
CORPUS_PATH = DATASET_DIR / "corpus.jsonl"
QUERIES_PATH = DATASET_DIR / "queries.jsonl"
QRELS_PATH = DATASET_DIR / "qrels.tsv"

def compute_rr(ranked_docs: List[int], expected: set) -> float:
    for i, d in enumerate(ranked_docs):
        if d in expected:
            return 1.0 / (i + 1)
    return 0.0

def compute_ndcg_at_k(ranked_docs: List[int], expected: set, k: int = 5) -> float:
    rels = [1 if d in expected else 0 for d in ranked_docs[:k]]
    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(rels))
    ideal_len = min(k, len(expected))
    ideal_rels = [1] * ideal_len + [0] * (k - ideal_len)
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_rels))
    return dcg / idcg if idcg > 0 else 0.0

def compute_metrics(retrieved_ranked: List[List[int]], ground_truth: List[List[int]], k: int = 5) -> Dict[str, float]:
    precisions, recalls, rrs, ndcgs = [], [], [], []
    for pred, gt in zip(retrieved_ranked, ground_truth):
        expected = set(gt)
        topk = pred[:k]
        tp = len(set(topk) & expected)
        precisions.append(tp / len(topk) if topk else 0.0)
        recalls.append(tp / len(expected) if expected else 0.0)
        rrs.append(compute_rr(pred, expected))
        ndcgs.append(compute_ndcg_at_k(pred, expected, k=5))
    n = len(ground_truth)
    return {"Precision@5": sum(precisions)/n, "Recall@5": sum(recalls)/n, "MRR": sum(rrs)/n, "NDCG@5": sum(ndcgs)/n}

def load_corpus_docs() -> List[Document]:
    docs = []
    with CORPUS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            doc_id = int(rec["doc_id"])
            text = rec.get("text") or rec.get("text_plain") or ""
            docs.append(Document(page_content=text, metadata={"corpus_id": doc_id, "category": rec.get("category")}))
    return docs

def load_queries() -> Tuple[List[int], List[str]]:
    qids, queries = [], []
    with QUERIES_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            qids.append(int(rec["query_id"]))
            queries.append(rec["query"])
    return qids, queries

def load_qrels() -> pd.DataFrame:
    qrels = pd.read_csv(QRELS_PATH, sep="\t").rename(columns={"query-id":"query_id","corpus-id":"corpus_id","score":"relevance"})
    qrels["query_id"] = qrels["query_id"].astype(int)
    qrels["corpus_id"] = qrels["corpus_id"].astype(int)
    return qrels

def build_ground_truth(qids: List[int], qrels: pd.DataFrame) -> List[List[int]]:
    return [qrels[qrels["query_id"] == qid]["corpus_id"].tolist() for qid in qids]

def collapse_chunks_to_doc_ids(chunks: List[Document]) -> List[int]:
    ordered, seen = [], set()
    for c in chunks:
        cid = int(c.metadata["corpus_id"])
        if cid not in seen:
            seen.add(cid)
            ordered.append(cid)
    return ordered

def main():
    corpus_docs = load_corpus_docs()
    qids, queries = load_queries()
    qrels = load_qrels()
    gt = build_ground_truth(qids, qrels)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},  # change to cuda:0 if you have it
        encode_kwargs={"normalize_embeddings": True, "batch_size": 64},
    )

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="gpt2", chunk_size=128, chunk_overlap=32
    )

    # Build index once for this splitter
    chunked = splitter.split_documents(corpus_docs)
    vs = FAISS.from_documents(chunked, embeddings)

    for top_k_chunks in [5, 10, 20]:
        retrieved = []
        for q in queries:
            chunks = vs.similarity_search(q, k=top_k_chunks)
            retrieved.append(collapse_chunks_to_doc_ids(chunks))
        metrics = compute_metrics(retrieved, gt, k=5)
        print(f"\n--- chunk128_overlap32 | retrieve_chunks={top_k_chunks} ---")
        print(metrics)

    # Debug one query
    idx = 0
    print("\n== Debug one query ==")
    print("Query:", queries[idx])
    print("Ground truth count:", len(gt[idx]))
    print("Retrieved doc_ids (k=10 chunks):", collapse_chunks_to_doc_ids(vs.similarity_search(queries[idx], k=10)))

if __name__ == "__main__":
    main()
