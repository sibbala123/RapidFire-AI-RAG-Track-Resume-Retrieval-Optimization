# ============================================================
# RapidFire AI — RAG Track Notebook (Resume Retrieval)
# End-to-end: Load dataset -> Define knob space -> Sample ~20 configs
# -> run_evals() hyperparallel -> log retrieval metrics (Precision/Recall/MRR/NDCG)
#
# Dataset expected layout:
# datasets/resume_rag/
#   corpus.jsonl
#   queries.jsonl
#   qrels.tsv
# ============================================================

# ---------------------------
# 0) Install dependencies
# ---------------------------

import os
import re
import math
import random
from pathlib import Path
from typing import Dict, Any, List as PyList, Optional

import pandas as pd
from datasets import load_dataset

# RapidFire
try:
    from rapidfireai import Experiment
except Exception as e:
    raise ImportError(
        "Could not import Experiment from rapidfireai. "
        "Please confirm rapidfireai is installed and your version matches the starter kit."
    ) from e

try:
   from rapidfireai.evals.automl import RFLangChainRagSpec
except Exception as e:
    raise ImportError(
        "Could not import RFLangChainRagSpec. Your rapidfireai install may be missing RAG modules."
    ) from e

try:
    from rapidfireai.prompts import RFPromptManager
except Exception:
    RFPromptManager = None

# LangChain
from langchain_community.document_loaders import DirectoryLoader, JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

try:
    # LangChain >= 0.1 / 0.2+
    from langchain_core.documents import Document
except ImportError:
    # Older LangChain fallback
    from langchain.schema import Document

try:
    # for newer langchain versions
    from langchain_core.documents import Document as CoreDocument
    Document = CoreDocument
except Exception:
    pass

from bs4 import BeautifulSoup

# Optional reranker
RERANK_AVAILABLE = True
try:
    from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
except Exception:
    RERANK_AVAILABLE = False
    CrossEncoderReranker = None

print("RERANK_AVAILABLE =", RERANK_AVAILABLE)


# ---------------------------
# 1) Dataset paths + loading
# ---------------------------
dataset_dir = Path("dataset")
assert (dataset_dir / "corpus.jsonl").exists(), f"Missing: {dataset_dir / 'corpus.jsonl'}"
assert (dataset_dir / "queries.jsonl").exists(), f"Missing: {dataset_dir / 'queries.jsonl'}"
assert (dataset_dir / "qrels.tsv").exists(), f"Missing: {dataset_dir / 'qrels.tsv'}"

queries_ds = load_dataset("json", data_files=str(dataset_dir / "queries.jsonl"), split="train")
queries_ds = queries_ds.rename_columns({"query": "query", "query_id": "query_id"})
queries_ds = queries_ds.map(lambda x: {"query_id": int(x["query_id"])})

qrels = pd.read_csv(dataset_dir / "qrels.tsv", sep="\t")
qrels = qrels.rename(columns={"query-id": "query_id", "corpus-id": "corpus_id", "score": "relevance"})
qrels["query_id"] = qrels["query_id"].astype(int)
qrels["corpus_id"] = qrels["corpus_id"].astype(int)

print("Queries:", len(queries_ds))
print("Qrels rows:", len(qrels))


# ---------------------------------------------------------
# 2) HTML-aware splitting utilities
# ---------------------------------------------------------
SECTION_KEYWORDS = [
    "summary", "objective", "experience", "work experience", "employment",
    "projects", "project experience", "skills", "technical skills",
    "education", "certifications", "certification", "achievements", "awards",
    "publications", "leadership", "activities", "volunteer", "interests"
]

def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def looks_like_heading(line: str) -> bool:
    """
    Heuristic heading detector for stripped HTML/plain text:
    - short line
    - mostly alphabetic / title-like
    - matches common resume section keywords
    """
    s = normalize_ws(line).lower()
    if not s:
        return False
    if len(s) > 60:
        return False
    # keyword match
    for kw in SECTION_KEYWORDS:
        if s == kw or s.startswith(kw + " "):
            return True
    # "ALL CAPS" headings
    if line.strip().isupper() and 4 <= len(line.strip()) <= 40:
        return True
    return False

def html_to_blocks(html: str) -> PyList[Dict[str, str]]:
    """
    Convert HTML into ordered blocks like:
    [{"title": "Experience", "text": "..."},
     {"title": "Skills", "text": "..."}]
    We try to use actual tags first; fallback to heuristic line parsing.
    """
    if not html or not isinstance(html, str):
        return [{"title": "Document", "text": ""}]

    soup = BeautifulSoup(html, "lxml")

    # Remove scripts/styles
    for tag in soup(["script", "style"]):
        tag.decompose()

    # Prefer explicit headings if present (h1-h6, sectiontitle, etc.)
    heading_tags = soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])
    # Also pick some resume-template-like headings
    heading_tags += soup.find_all(class_=re.compile(r"(sectiontitle|heading|sectname)", re.I))

    # If we found real-ish headings, segment by them
    if heading_tags:
        # Build a linear sequence of text nodes, splitting at headings
        blocks = []
        current_title = "Document"
        current_text = []

        # Walk all descendants in order and split on headings
        for el in soup.descendants:
            if getattr(el, "name", None) in ["h1","h2","h3","h4","h5","h6"] or (
                hasattr(el, "get") and el.get("class") and any(re.search(r"(sectiontitle|heading|sectname)", c, re.I) for c in el.get("class"))
            ):
                # flush previous
                if current_text:
                    blocks.append({"title": current_title, "text": normalize_ws(" ".join(current_text))})
                    current_text = []
                # new title
                t = normalize_ws(el.get_text(" ", strip=True))
                current_title = t if t else "Section"
                continue

            # collect paragraph-like text
            if isinstance(el, str):
                continue
            if getattr(el, "name", None) in ["p", "li", "div", "span"]:
                txt = normalize_ws(el.get_text(" ", strip=True))
                if txt:
                    current_text.append(txt)

        if current_text:
            blocks.append({"title": current_title, "text": normalize_ws(" ".join(current_text))})

        # cleanup tiny/empty blocks
        blocks = [b for b in blocks if b["text"] and len(b["text"]) > 30]
        if blocks:
            return blocks

    # Fallback: use stripped text with heuristic headings
    full_text = soup.get_text("\n", strip=True)
    lines = [normalize_ws(x) for x in full_text.split("\n")]
    lines = [x for x in lines if x]

    blocks = []
    current_title = "Document"
    buf = []

    for line in lines:
        if looks_like_heading(line):
            if buf:
                blocks.append({"title": current_title, "text": normalize_ws(" ".join(buf))})
                buf = []
            current_title = line.strip()
        else:
            buf.append(line)

    if buf:
        blocks.append({"title": current_title, "text": normalize_ws(" ".join(buf))})

    blocks = [b for b in blocks if b["text"] and len(b["text"]) > 30]
    return blocks if blocks else [{"title": "Document", "text": normalize_ws(full_text)}]


class HTMLSectionTextSplitter:
    """
    LangChain-compatible splitter:
    input: list[Document]
    output: list[Document] (chunked)
    Strategy:
      1) split doc into HTML-derived "sections" (Experience / Skills / ...)
      2) within each section, optionally token-chunk using RecursiveCharacterTextSplitter

    This gives you a realistic "HTML-aware" chunking knob for RAG Track.
    """

    def __init__(self, inner_splitter: Optional[RecursiveCharacterTextSplitter] = None, min_section_chars: int = 200):
        self.inner_splitter = inner_splitter
        self.min_section_chars = min_section_chars

    def split_documents(self, docs: PyList[Document]) -> PyList[Document]:
        out_docs = []
        for d in docs:
            md = dict(d.metadata or {})
            html = md.get("raw_html") or md.get("resume_html") or md.get("text_html") or md.get("html")
            # If loader didn't store raw html in metadata, fallback to page_content as "html-ish"
            html = html if html else d.page_content

            blocks = html_to_blocks(html)
            for b in blocks:
                title = b["title"]
                text = b["text"]
                if not text or len(text) < self.min_section_chars:
                    continue

                # create section doc
                section_md = {**md, "section_title": title, "chunking": "html_section"}
                section_doc = Document(page_content=f"{title}\n\n{text}", metadata=section_md)

                if self.inner_splitter:
                    out_docs.extend(self.inner_splitter.split_documents([section_doc]))
                else:
                    out_docs.append(section_doc)

        return out_docs


# ---------------------------------------------------------
# 3) Retrieval metrics (Precision@K, Recall@K, MRR, NDCG@K)
# ---------------------------------------------------------
def _ndcg_at_k(ranked_ids: PyList[int], relevant_set: set, k: int) -> float:
    rels = [1 if doc_id in relevant_set else 0 for doc_id in ranked_ids[:k]]
    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(rels))
    ideal = [1] * min(k, len(relevant_set)) + [0] * max(0, k - len(relevant_set))
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal))
    return (dcg / idcg) if idcg > 0 else 0.0

def _rr(ranked_ids: PyList[int], relevant_set: set) -> float:
    for i, doc_id in enumerate(ranked_ids):
        if doc_id in relevant_set:
            return 1.0 / (i + 1)
    return 0.0

def compute_metrics_fn(batch: Dict[str, PyList[Any]]) -> Dict[str, Dict[str, Any]]:
    ks = [5, 10]
    total = len(batch["query_id"])

    sums = {f"Precision@{k}": 0.0 for k in ks}
    sums |= {f"Recall@{k}": 0.0 for k in ks}
    sums |= {f"NDCG@{k}": 0.0 for k in ks}
    sums["MRR"] = 0.0

    for retrieved, gt in zip(batch["retrieved_documents"], batch["ground_truth_documents"]):
        rel = set(gt)
        for k in ks:
            topk = retrieved[:k]
            tp = sum(1 for x in topk if x in rel)
            precision = tp / len(topk) if len(topk) else 0.0
            recall = tp / len(rel) if len(rel) else 0.0
            sums[f"Precision@{k}"] += precision
            sums[f"Recall@{k}"] += recall
            sums[f"NDCG@{k}"] += _ndcg_at_k(retrieved, rel, k)
        sums["MRR"] += _rr(retrieved, rel)

    for k in sums:
        sums[k] /= total

    metrics = {name: {"value": float(val), "value_range": (0.0, 1.0)} for name, val in sums.items()}
    metrics["Total"] = {"value": int(total)}
    return metrics


def accumulate_metrics_fn(aggregated_metrics: Dict[str, PyList[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    totals = [m["value"] for m in aggregated_metrics.get("Total", [])]
    total_q = int(sum(totals)) if totals else 0
    if total_q == 0:
        return {"Total": {"value": 0}}

    def _wavg(metric_name: str) -> float:
        return float(sum(m["value"] * t for m, t in zip(aggregated_metrics[metric_name], totals)) / total_q)

    metric_names = [k for k in aggregated_metrics.keys() if k != "Total"]
    out = {"Total": {"value": total_q}}
    for mn in metric_names:
        out[mn] = {"value": _wavg(mn), "value_range": (0.0, 1.0), "is_algebraic": True}
    return out


# ---------------------------------------------------------
# 4) Preprocess: retrieve -> attach retrieved + ground truth
# ---------------------------------------------------------
def _ground_truth_for_query_ids(query_ids: PyList[int]) -> PyList[PyList[int]]:
    return [
        qrels[qrels["query_id"] == qid]["corpus_id"].tolist()
        for qid in query_ids
    ]

def preprocess_fn(batch: Dict[str, PyList[Any]], rag: RFLangChainRagSpec, prompt_manager=None) -> Dict[str, PyList[Any]]:
    contexts = rag.get_context(batch_queries=batch["query"], serialize=False)

    retrieved_documents = []
    for docs in contexts:
        ids = [int(d.metadata["corpus_id"]) for d in docs]
        retrieved_documents.append(ids)

    query_ids = [int(qid) for qid in batch["query_id"]]
    ground_truth_documents = _ground_truth_for_query_ids(query_ids)

    return {
        **batch,
        "retrieved_documents": retrieved_documents,
        "ground_truth_documents": ground_truth_documents,
    }


# ---------------------------------------------------------
# 5) Define knob space (including HTML-aware splitter) and sample ~20 configs
# ---------------------------------------------------------
SEED = 42
random.seed(SEED)

BATCH_SIZE = 64

# Standard token-based splitters
base_splitters = [
    ("chunk128_overlap32",
     RecursiveCharacterTextSplitter.from_tiktoken_encoder("gpt2", chunk_size=128, chunk_overlap=32)),
    ("chunk256_overlap32",
     RecursiveCharacterTextSplitter.from_tiktoken_encoder("gpt2", chunk_size=256, chunk_overlap=32)),
    ("chunk256_overlap64",
     RecursiveCharacterTextSplitter.from_tiktoken_encoder("gpt2", chunk_size=256, chunk_overlap=64)),
]

# HTML-aware splitters:
# We first split into HTML "sections", then optionally sub-chunk within each section.
html_section_splitters = [
    ("html_sections_only",
     HTMLSectionTextSplitter(inner_splitter=None, min_section_chars=200)),
    ("html_sections_then_chunk256o32",
     HTMLSectionTextSplitter(
         inner_splitter=RecursiveCharacterTextSplitter.from_tiktoken_encoder("gpt2", chunk_size=256, chunk_overlap=32),
         min_section_chars=200,
     )),
]

SPLITTERS = base_splitters + html_section_splitters

# Embedding knobs
EMBEDDERS = [
    ("all-MiniLM-L6-v2", {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "model_kwargs": {"device": "cuda:0"},
        "encode_kwargs": {"normalize_embeddings": True, "batch_size": BATCH_SIZE},
    }),
    ("bge-small-en-v1.5", {
        "model_name": "BAAI/bge-small-en-v1.5",
        "model_kwargs": {"device": "cuda:0"},
        "encode_kwargs": {"normalize_embeddings": True, "batch_size": BATCH_SIZE},
    }),
]

SEARCH_TYPES = ["similarity", "mmr"]
TOPK = [5, 10, 20]
GPU_SEARCH = [False, True]

RERANK_TOPN = [None, 3, 5] if RERANK_AVAILABLE else [None]

def build_rag_spec(
    splitter_name: str,
    splitter_obj,
    embed_name: str,
    embed_kwargs: Dict[str, Any],
    search_type: str,
    k: int,
    enable_gpu_search: bool,
    rerank_top_n: Optional[int],
) -> RFLangChainRagSpec:

    reranker_cls = None
    reranker_kwargs = None
    if rerank_top_n is not None and rerank_top_n > 0 and RERANK_AVAILABLE:
        reranker_cls = CrossEncoderReranker
        reranker_kwargs = {
            "model_name": "cross-encoder/ms-marco-MiniLM-L6-v2",
            "model_kwargs": {"device": "cpu"},
            "top_n": rerank_top_n,
        }

    # IMPORTANT for HTML-aware splitter:
    # We want the loader to provide raw HTML in metadata so our splitter can access it.
    # We'll store Resume_html into metadata key "raw_html" by using content_key="text_plain"
    # and metadata_func to attach raw_html if your corpus has it.
    # If your corpus.jsonl does NOT include raw html, the splitter will fallback to page_content.
    def _metadata_func(rec, md):
        return {
            "corpus_id": int(rec["doc_id"]),
            "person_id": str(rec.get("person_id", rec.get("doc_id"))),
            "category": rec.get("category", None),
            "doc_type": rec.get("doc_type", "resume"),
            # optional fields for HTML-aware chunking:
            "raw_html": rec.get("resume_html") or rec.get("text_html") or rec.get("Resume_html") or rec.get("html"),
        }

    spec = RFLangChainRagSpec(
        document_loader=DirectoryLoader(
            path=str(dataset_dir),
            glob="corpus.jsonl",
            loader_cls=JSONLoader,
            loader_kwargs={
                "jq_schema": ".",
                "content_key": "text_plain",   # plain text for embedding, HTML stored in metadata
                "metadata_func": _metadata_func,
                "json_lines": True,
                "text_content": False,
            },
        ),
        text_splitter=splitter_obj,
        embedding_cls=HuggingFaceEmbeddings,
        embedding_kwargs=embed_kwargs,
        vector_store=None,  # default (usually FAISS)
        search_type=search_type,
        search_kwargs={"k": int(k)},
        enable_gpu_search=bool(enable_gpu_search),
        reranker_cls=reranker_cls,
        reranker_kwargs=reranker_kwargs,
    )
    return spec


# Build cartesian space, then sample ~20 unique configs
all_candidates = []
for splitter_name, splitter_obj in SPLITTERS:
    for embed_name, embed_kwargs in EMBEDDERS:
        for stype in SEARCH_TYPES:
            for k in TOPK:
                for gs in GPU_SEARCH:
                    for topn in RERANK_TOPN:
                        all_candidates.append((splitter_name, splitter_obj, embed_name, embed_kwargs, stype, k, gs, topn))

print("Total possible combos:", len(all_candidates))

TARGET_CONFIGS = 20
sampled = random.sample(all_candidates, k=min(TARGET_CONFIGS, len(all_candidates)))
print("Sampled configs:", len(sampled))

config_group = []
for i, (splitter_name, splitter_obj, embed_name, embed_kwargs, stype, k, gs, topn) in enumerate(sampled, start=1):
    spec = build_rag_spec(splitter_name, splitter_obj, embed_name, embed_kwargs, stype, k, gs, topn)
    config_name = f"cfg{i:02d}__{splitter_name}__emb={embed_name}__{stype}__k={k}__gpu={gs}__rerank={topn or 0}"
    cfg = {
        "name": config_name,
        "rag": spec,
        "preprocess_fn": preprocess_fn,
        "compute_metrics_fn": compute_metrics_fn,
        "accumulate_metrics_fn": accumulate_metrics_fn,
    }
    config_group.append(cfg)

for c in config_group:
    print("-", c["name"])


# ---------------------------------------------------------
# 6) Run RapidFire experiment with run_evals() (hyperparallel)
# ---------------------------------------------------------
experiment = Experiment(experiment_name="exp-resume-rag-track", mode="evals")

results = experiment.run_evals(
    config_group=config_group,
    dataset=queries_ds,
    num_actors=1,
    num_shards=4,
    seed=SEED,
)

results


# ---------------------------------------------------------
# 7) Locate artifacts/logs for plots (MLflow/TensorBoard)
# ---------------------------------------------------------
possible_roots = [
    Path("rapidfireai"),
    Path("/content/rapidfireai"),
    Path.home() / "rapidfireai",
]

found_root = None
for root in possible_roots:
    if root.exists():
        found_root = root
        print("Found RapidFire root:", root)
        break

if not found_root:
    print("Could not auto-detect RapidFire artifact root. Check your experiment output banner.")
else:
    # show likely artifact folders
    for p in found_root.rglob("*"):
        name = p.name.lower()
        if p.is_dir() and (("mlruns" in name) or ("tensorboard" in name) or ("logs" in name) or ("experiments" in name)):
            print(" -", p)
