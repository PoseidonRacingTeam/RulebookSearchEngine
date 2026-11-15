"""Central configuration (paths, model names, limits)."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PDF_DIR = DATA_DIR / "pdf"
INDEX_DIR = DATA_DIR / "index"
MODELS_DIR = ROOT / "models"
STATIC_DIR = ROOT / "static"

# the rulebook that is indexed (copied into data/pdf by setup / rebuild)
PDF_PATH = Path(os.environ.get("FS_RULES_PDF", PDF_DIR / "FS_Rules_2027_v1.0.pdf"))

# Models. These are cached under models/ by setup.sh and loaded offline.
EMBEDDING_MODEL = os.environ.get("FS_EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
RERANKER_MODEL = os.environ.get("FS_RERANKER_MODEL", "mixedbread-ai/mxbai-rerank-base-v1")
EMBEDDING_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

INDEX_FORMAT_VERSION = "4"

HOST = os.environ.get("FS_HOST", "127.0.0.1")
PORT = int(os.environ.get("FS_PORT", "8765"))

# retrieval sizes / limits
BM25_TOP_K = 80
DENSE_TOP_K = 80
RERANK_MAX_CANDIDATES = 110
MAX_RESULTS = 8
SEARCH_BUDGET_SECONDS = 8.0        # hard budget: a search over this is reported as failed
RERANK_BUDGET_SECONDS = 6.0        # if retrieval already used more than this, skip rerank and say so
MAX_QUERY_CHARS = 300
MAX_PENDING_SEARCHES = 2           # beyond this the server answers 429 instead of queueing work


def offline_env() -> None:
    """Force Hugging Face libraries to use the local cache only."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HOME", str(MODELS_DIR / "hf_home"))
