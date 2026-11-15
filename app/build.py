"""Build (or rebuild) the persistent index: `uv run python -m app.build [pdf]`."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

from . import config
from .indexer import build_index


def main(argv: list[str]) -> int:
    pdf = Path(argv[1]).expanduser() if len(argv) > 1 else config.PDF_PATH
    if not pdf.exists():
        print(f"PDF not found: {pdf}")
        return 2
    config.PDF_DIR.mkdir(parents=True, exist_ok=True)
    target = config.PDF_PATH
    if pdf.resolve() != target.resolve():
        shutil.copyfile(pdf, target)   # the original PDF is never modified
        print(f"Copied {pdf} -> {target}")
    if config.INDEX_DIR.exists():
        shutil.rmtree(config.INDEX_DIR)
    build_index(target, config.INDEX_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
