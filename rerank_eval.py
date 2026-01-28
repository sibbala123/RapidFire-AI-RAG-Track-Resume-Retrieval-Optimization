#!/usr/bin/env python3
import json, math
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from sentence_transformers import CrossEncoder  # <-- reranker


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

def rerank_chunks(query: str, chunks: List[Document], cross_encoder: CrossEncoder, top_n: int) -> List[Document]:
    pairs = [(query, c.page_content) for c in chunks]
    scores = cross_encoder.predict(pairs)  # higher is better
    ranked = [c for _, c in sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)]
    return ranked[:top_n]

def main():
    corpus_docs = load_corpus_docs()
    qids, queries = load_queries()
    qrels = load_qrels()
    gt = build_ground_truth(qids, qrels)

    # Baseline settings you just found
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="gpt2", chunk_size=128, chunk_overlap=32
    )
    retrieve_chunks = 10

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},  # change to cuda:0 if you have it
        encode_kwargs={"normalize_embeddings": True, "batch_size": 64},
    )

    chunked = splitter.split_documents(corpus_docs)
    vs = FAISS.from_documents(chunked, embeddings)

    # Reranker (CPU is fine)
    cross = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")

    configs = [
        ("no_rerank", None),
        ("rerank_top2", 2),
        ("rerank_top5", 5),
    ]

    for name, top_n in configs:
        retrieved_all = []
        for q in queries:
            chunks = vs.similarity_search(q, k=retrieve_chunks)

            if top_n is not None:
                chunks = rerank_chunks(q, chunks, cross, top_n=top_n)

            retrieved_all.append(collapse_chunks_to_doc_ids(chunks))

        metrics = compute_metrics(retrieved_all, gt, k=5)
        print(f"\n--- {name} ---")
        print(metrics)

if __name__ == "__main__":
    main()
