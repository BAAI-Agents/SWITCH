#!/usr/bin/env python3
from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    runpy.run_path(str(repo_root / "scripts" / "apply_versioned_qa_refinements.py"), run_name="__main__")


if __name__ == "__main__":
    main()
