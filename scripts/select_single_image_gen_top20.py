#!/usr/bin/env python3
"""
Select a curated set of 20 representative samples from the single-image
generation dataset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


SELECTED_SAMPLE_IDS = [
    "003_state_transition_video_002",
    "003_final_state_video_003",
    "003_verification_state_video_005",
    "004_state_transition_video_002",
    "004_final_state_video_008",
    "005_state_transition_video_001",
    "005_state_transition_video_011",
    "005_final_state_video_012",
    "006_state_transition_video_002",
    "006_final_state_video_008",
    "007_state_transition_video_002",
    "007_final_state_video_003",
    "008_state_transition_video_003",
    "008_final_state_video_005",
    "008_verification_state_video_008",
    "009_state_transition_video_003",
    "009_final_state_video_004",
    "010_state_transition_video_002",
    "010_final_state_video_004",
    "010_recovery_video_008",
]


def load_jsonl(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_md(path: Path, rows: List[Dict]) -> None:
    lines: List[str] = []
    lines.append("# Curated 20 Single-Image Generation Tasks")
    lines.append("")
    lines.append(f"- Selected samples: `{len(rows)}`")
    lines.append("")
    for idx, row in enumerate(rows, start=1):
        lines.append(f"## {idx}. {row['sample_id']}")
        lines.append("")
        lines.append(f"- Task type: `{row['task_type']}`")
        lines.append(f"- Source video: `{row['source_video']}`")
        lines.append(f"- Anchor frame: [{row['anchor_frame_path']}](/d:/Search/BAAI/SWITCH/{row['anchor_frame_path']})")
        lines.append(f"- Goal: `{row['goal_text']}`")
        lines.append(f"- Prompt EN: {row['prompt_en']}")
        lines.append(f"- Expected final state: `{row['expected_final_state']}`")
        if row.get("quality_flags"):
            lines.append(f"- Quality flags: `{', '.join(row['quality_flags'])}`")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Select 20 curated generation tasks.")
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=Path("30fps") / "single_image_gen_dataset.jsonl",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path("30fps") / "single_image_gen_dataset_top20.jsonl",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("30fps") / "single_image_gen_dataset_top20.md",
    )
    args = parser.parse_args()

    rows = load_jsonl(args.input_jsonl)
    by_id = {row["sample_id"]: row for row in rows}
    selected = [by_id[sid] for sid in SELECTED_SAMPLE_IDS if sid in by_id]

    missing = [sid for sid in SELECTED_SAMPLE_IDS if sid not in by_id]
    if missing:
        raise KeyError(f"Missing sample ids: {missing}")

    write_jsonl(args.output_jsonl, selected)
    write_md(args.output_md, selected)
    print(f"Wrote JSONL: {args.output_jsonl}")
    print(f"Wrote Markdown: {args.output_md}")
    print(f"Selected samples: {len(selected)}")


if __name__ == "__main__":
    main()
