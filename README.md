# Retrieval-First RAG Optimization for Resume Search

This repository contains my submission for the **RapidFire AI RAG Track**, focused on
**retrieval-first experimentation** for resume-based question answering.

Unlike generation-centric RAG demos, this project systematically evaluates how
**chunking, indexing, and retrieval choices** affect retrieval quality using
Precision, Recall, NDCG, and MRR — without relying on LLM-as-judge evaluation.

---

##  Use Case
Given a corpus of resumes, answer skill-based queries such as:
- *Which candidate has Python experience?*
- *Which candidate has worked on ETL pipelines?*

The goal is to retrieve the most relevant resume evidence, not to optimize generation.

---

## Experiment Dimensions
- **Chunking**
  - Fixed token chunking (128 / 256 tokens)
  - HTML-aware section chunking
  - Hybrid HTML → token sub-chunking
- **Embeddings**
  - all-MiniLM-L6-v2
- **Retrieval**
  - Similarity search
  - Top-K = 5 / 10
- **Indexing**
  - FAISS Flat
  - FAISS IVF

~20 configurations were evaluated using RapidFire AI’s hyperparallel evaluation API.

---

## Key Results
Best configuration:
HTML sections → 128-token sub-chunks
all-MiniLM-L6-v2
FAISS Flat
similarity search (k=10)

**Metrics**
- Precision@5: **0.3867**
- NDCG@5: **0.3812**
- MRR: **0.5711**

This hybrid strategy outperformed both pure token chunking and pure HTML section chunking.

---

##  How to Run
```bash
pip install -r requirements.txt
jupyter notebook notebook/rag_track_resume_retrieval.ipynb
