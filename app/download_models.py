"""Download (or verify) the embedding and reranker models into ./models.

  uv run python -m app.download_models            # download if missing
  uv run python -m app.download_models --check    # offline check only
"""
from __future__ import annotations

import os
import sys

from . import config


def main(argv: list[str]) -> int:
    check_only = "--check" in argv
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if check_only:
        config.offline_env()
    from sentence_transformers import CrossEncoder, SentenceTransformer

    ok = True
    for kind, name in (("embedding", config.EMBEDDING_MODEL), ("reranker", config.RERANKER_MODEL)):
        try:
            if kind == "embedding":
                m = SentenceTransformer(name, cache_folder=str(config.MODELS_DIR), device="cpu", local_files_only=check_only)
                dim = m.get_sentence_embedding_dimension()
                print(f"   {kind}: {name} ok (dim {dim})")
            else:
                CrossEncoder(name, cache_folder=str(config.MODELS_DIR), device="cpu", local_files_only=check_only)
                print(f"   {kind}: {name} ok")
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"   {kind}: {name} FAILED: {type(e).__name__}: {e}")
            if check_only:
                print("   -> run ./setup.sh (or `uv run python -m app.download_models`) while online")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
