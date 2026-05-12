#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import cv2

from split_prompt_v2_by_input_mode import (
    SOURCE_PROMPT_VERSION,
    TASK_TYPES,
    build_video_sample,
    clean_text,
    infer_sample_fps,
    load_jsonl,
    prune_assets,
    relative_repo_path,
    safe_name,
    span_kind,
    validate_video_rows,
    write_jsonl,
    MediaExporter,
)


STATE_LABELS = {"ui_state", "physical_world_state"}
STATE_AUGMENTED_IMAGE_PROMPT_VERSION = "single_image_gen_prompt_v2_state_augmented"


def is_single_frame_span(span: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(span, dict):
        return False
    start = span.get("start")
    end = span.get("end")
    if start is None or end is None:
        return False
    return int(start) == int(end)


def load_candidate_map(annotation_root: Path) -> Dict[str, Dict[str, Any]]:
    payload = json.loads((annotation_root / "switch_all.qa_candidates.json").read_text(encoding="utf-8"))
    candidate_map: Dict[str, Dict[str, Any]] = {}
    for video in payload["videos"]:
        for qa in video["qa_candidates"]:
            qa_id = qa["qa_id"]
            if qa_id in candidate_map:
                raise RuntimeError(f"Duplicate qa_id in inventory: {qa_id}")
            source_span = qa.get("source_span") or {}
            candidate_map[qa_id] = {
                "task_family": qa["task_family"],
                "source_label": qa["source_label"],
                "source_span": {
                    "start": int(source_span["start"]) if source_span.get("start") is not None else None,
                    "end": int(source_span["end"]) if source_span.get("end") is not None else None,
                },
                "answer": clean_text(qa["answer"]),
            }
    return candidate_map


def insert_state_input_note(prompt_en: str, prompt_zh: str) -> tuple[str, str]:
    en_lines = prompt_en.splitlines()
    zh_lines = prompt_zh.splitlines()
    en_note = "Input note: this image is the annotated single-frame state cue for the task."
    zh_note = "输入说明：这张图像来自任务中标注的单帧状态证据，请将其作为视觉输入参考。"
    if en_note not in prompt_en:
        en_lines = [en_lines[0], en_note, *en_lines[1:]] if en_lines else [en_note]
    if zh_note not in prompt_zh:
        zh_lines = [zh_lines[0], zh_note, *zh_lines[1:]] if zh_lines else [zh_note]
    return "\n".join(en_lines), "\n".join(zh_lines)


def extract_frame(video_path: Path, frame_index: int, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        return
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    target_frame = frame_index
    if total_frames > 0:
        target_frame = max(0, min(frame_index, total_frames - 1))
    capture.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ok, frame = capture.read()
    if not ok and target_frame > 0:
        capture.set(cv2.CAP_PROP_POS_FRAMES, target_frame - 1)
        ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Unable to read frame {frame_index} from {video_path}")
    if not cv2.imwrite(str(output_path), frame):
        raise RuntimeError(f"Unable to write image: {output_path}")


def build_anchor_image_variant(
    sample: Dict[str, Any],
    repo_root: Path,
    image_root: Path,
    exporter: MediaExporter,
) -> Dict[str, Any]:
    source_frame_path = repo_root / sample["anchor_frame_path"]
    output_frame_path = image_root / "frames" / f"{safe_name(sample['sample_id'])}.jpg"
    exporter.copy_frame(source_frame_path, output_frame_path)
    updated = dict(sample)
    updated.pop("_line", None)
    updated["prompt_version"] = STATE_AUGMENTED_IMAGE_PROMPT_VERSION
    updated["source_prompt_version"] = sample.get("prompt_version")
    updated["split_bucket"] = "single_image"
    updated["classification_reason"] = "anchor_source_span_is_single_frame_or_missing"
    updated["routing_mode"] = "state_augmented"
    updated["input_modality"] = "image"
    updated["output_modality"] = "video"
    updated["input_variant"] = "anchor_frame"
    updated["input_frame_path"] = relative_repo_path(output_frame_path, repo_root)
    updated["anchor_frame_path"] = updated["input_frame_path"]
    return updated


def build_state_image_variant(
    sample: Dict[str, Any],
    candidate_info: Dict[str, Any],
    repo_root: Path,
    image_root: Path,
    exporter: MediaExporter,
) -> Dict[str, Any]:
    source_video_path = repo_root / sample["source_video"]
    frame_index = int(candidate_info["source_span"]["start"])
    output_frame_path = image_root / "frames" / f"{safe_name(sample['sample_id'])}.jpg"
    extract_frame(source_video_path, frame_index, output_frame_path)

    updated = dict(sample)
    updated.pop("_line", None)
    updated["source_prompt_version"] = sample.get("prompt_version")
    updated["prompt_version"] = STATE_AUGMENTED_IMAGE_PROMPT_VERSION
    updated["prompt_en"], updated["prompt_zh"] = insert_state_input_note(sample["prompt_en"], sample["prompt_zh"])
    updated["split_bucket"] = "single_image"
    updated["classification_reason"] = "candidate_source_state_is_single_frame"
    updated["routing_mode"] = "state_augmented"
    updated["input_modality"] = "image"
    updated["output_modality"] = "video"
    updated["input_variant"] = "state_source_frame"
    updated["input_frame_path"] = relative_repo_path(output_frame_path, repo_root)

    updated["legacy_anchor_frame"] = sample.get("anchor_frame")
    updated["legacy_anchor_frame_time"] = sample.get("anchor_frame_time")
    updated["legacy_anchor_frame_path"] = sample.get("anchor_frame_path")
    updated["legacy_anchor_source_type"] = sample.get("anchor_source_type")
    updated["legacy_anchor_source_span"] = sample.get("anchor_source_span")

    fps = infer_sample_fps(sample)
    updated["anchor_frame"] = frame_index
    updated["anchor_frame_time"] = round(frame_index / fps, 3)
    updated["anchor_frame_path"] = updated["input_frame_path"]
    updated["anchor_source_type"] = "source_state"
    updated["anchor_source_span"] = {
        "label": candidate_info["source_label"],
        "text": candidate_info["answer"],
        "start": frame_index,
        "end": frame_index,
    }
    updated["state_input_source_label"] = candidate_info["source_label"]
    updated["state_input_source_span"] = dict(updated["anchor_source_span"])
    updated["input_state_text"] = candidate_info["answer"]
    return updated


def should_route_to_state_image(sample: Dict[str, Any], candidate_info: Optional[Dict[str, Any]]) -> bool:
    if candidate_info is None:
        return False
    if sample["task_type"] not in {"verification_state_video", "final_state_video"}:
        return False
    if candidate_info["source_label"] not in STATE_LABELS:
        return False
    return is_single_frame_span(candidate_info["source_span"])


def write_bucket_summary(
    path: Path,
    bucket_name: str,
    rows: Sequence[Dict[str, Any]],
    asset_pattern: str,
) -> None:
    task_counts = Counter(row["task_type"] for row in rows)
    lines = [
        f"# {bucket_name} Summary",
        "",
        f"- Total samples: `{len(rows)}`",
        f"- Asset pattern: `{asset_pattern}`",
        "",
        "## Task Counts",
        "",
    ]
    for task_type in TASK_TYPES:
        lines.append(f"- `{task_type}`: `{task_counts.get(task_type, 0)}`")
    if rows and bucket_name == "Single Image Bucket":
        variant_counts = Counter(row["input_variant"] for row in rows)
        lines.extend(["", "## Input Variants", ""])
        for variant, count in sorted(variant_counts.items()):
            lines.append(f"- `{variant}`: `{count}`")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_bucket_readme(path: Path, bucket_name: str, rows: Sequence[Dict[str, Any]]) -> None:
    task_counts = Counter(row["task_type"] for row in rows)
    lines = [
        f"# {bucket_name}",
        "",
        f"- Source dataset: `annotations/0421/switch/{SOURCE_PROMPT_VERSION}/dataset.jsonl`",
        f"- Samples: `{len(rows)}`",
        "",
        "## Task Counts",
        "",
    ]
    for task_type in TASK_TYPES:
        lines.append(f"- `{task_type}`: `{task_counts.get(task_type, 0)}`")
    lines.extend(
        [
            "",
            "## Key Fields",
            "",
            "- `routing_mode`: fixed as `state_augmented` in this split",
            "- `input_variant`: either `anchor_frame` or `state_source_frame`",
            "- `classification_reason`: why the sample was routed here",
        ]
    )
    if rows and rows[0]["split_bucket"] == "single_image":
        lines.extend(
            [
                "- `input_frame_path`: frame asset used as model input",
                "- `legacy_anchor_*`: original anchor metadata from the full dataset when the input was rewritten from a state frame",
            ]
        )
    else:
        lines.extend(
            [
                "- `input_clip_path`: clip asset used as model input",
                "- `source_prompt_en` / `source_prompt_zh`: preserved source prompts from the full dataset",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_root_summary(path: Path, image_rows: Sequence[Dict[str, Any]], video_rows: Sequence[Dict[str, Any]]) -> None:
    image_task_counts = Counter(row["task_type"] for row in image_rows)
    video_task_counts = Counter(row["task_type"] for row in video_rows)
    variant_counts = Counter(row["input_variant"] for row in image_rows)
    lines = [
        "# State-Augmented Input-Mode Split Summary",
        "",
        f"- Source prompt version: `{SOURCE_PROMPT_VERSION}`",
        f"- Total samples scanned: `{len(image_rows) + len(video_rows)}`",
        f"- `single_image`: `{len(image_rows)}`",
        f"- `video_to_video`: `{len(video_rows)}`",
        "",
        "## Image Input Variants",
        "",
    ]
    for variant, count in sorted(variant_counts.items()):
        lines.append(f"- `{variant}`: `{count}`")
    lines.extend(["", "## Task Breakdown", ""])
    for task_type in TASK_TYPES:
        lines.append(
            f"- `{task_type}`: image=`{image_task_counts.get(task_type, 0)}`, "
            f"video=`{video_task_counts.get(task_type, 0)}`"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_root_readme(path: Path) -> None:
    lines = [
        "# State-Augmented Input-Mode Split",
        "",
        "This directory keeps the existing `by_input_mode/` output untouched and adds an alternative split that allows state-based samples to enter the single-image bucket.",
        "",
        "## Routing Rule",
        "",
        "- If a `verification_state_video` or `final_state_video` sample comes from a single-frame `ui_state` or `physical_world_state`, it is routed to `single_image/` using that state frame as input.",
        "- Otherwise, if the original `anchor_source_span` is already single-frame or missing, it is routed to `single_image/` using the original anchor frame.",
        "- All remaining samples are routed to `video_to_video/`.",
        "",
        "## Buckets",
        "",
        "- `single_image/`: image-input, video-output samples with anchor-frame or state-frame inputs.",
        "- `video_to_video/`: segment-input, video-output samples preserved from the current clip-based routing logic.",
        "",
        "## Notes",
        "",
        "- The existing `single_image_gen_prompt_v2_full/by_input_mode/` directory is not modified.",
        "- State-derived image samples keep the original prompt content but add an explicit input note and preserve the original anchor metadata under `legacy_anchor_*`.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def validate_image_rows(rows: Sequence[Dict[str, Any]], repo_root: Path, image_root: Path) -> None:
    for row in rows:
        input_path = repo_root / row["input_frame_path"]
        if not input_path.exists():
            raise RuntimeError(f"Missing input frame: {input_path}")
        if not input_path.is_relative_to(image_root):
            raise RuntimeError(f"Input frame escapes image bucket: {row['sample_id']}")
        if row["input_variant"] == "state_source_frame":
            span = row.get("anchor_source_span")
            if row.get("anchor_source_type") != "source_state":
                raise RuntimeError(f"State image missing source_state anchor type: {row['sample_id']}")
            if not is_single_frame_span(span):
                raise RuntimeError(f"State image anchor span is not single-frame: {row['sample_id']}")
            if (span or {}).get("label") not in STATE_LABELS:
                raise RuntimeError(f"State image anchor label is not a state label: {row['sample_id']}")
            if "Input note:" not in row["prompt_en"]:
                raise RuntimeError(f"State image prompt missing input note: {row['sample_id']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a state-augmented split for prompt v2.")
    parser.add_argument(
        "--annotation-root",
        type=Path,
        default=Path("annotations") / "0421" / "switch",
        help="Root directory for SWITCH annotations.",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("annotations") / "0421" / "switch" / SOURCE_PROMPT_VERSION,
        help="Source directory that contains the full prompt dataset.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("annotations") / "0421" / "switch" / SOURCE_PROMPT_VERSION / "by_input_mode_with_state_single_image",
        help="Output directory for the state-augmented split.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    annotation_root = (repo_root / args.annotation_root).resolve()
    source_root = (repo_root / args.source_root).resolve()
    output_root = (repo_root / args.output_root).resolve()
    image_root = output_root / "single_image"
    video_root = output_root / "video_to_video"

    candidate_map = load_candidate_map(annotation_root)
    rows = load_jsonl(source_root / "dataset.jsonl")
    exporter = MediaExporter()

    image_rows: List[Dict[str, Any]] = []
    video_rows: List[Dict[str, Any]] = []

    for sample in rows:
        candidate_info = candidate_map.get(sample.get("source_qa_id"))
        if should_route_to_state_image(sample, candidate_info):
            image_rows.append(build_state_image_variant(sample, candidate_info, repo_root, image_root, exporter))
            continue
        if span_kind(sample) == "frame":
            image_rows.append(build_anchor_image_variant(sample, repo_root, image_root, exporter))
            continue
        video_rows.append(build_video_sample(sample, repo_root, video_root, exporter))

    image_rows.sort(key=lambda row: row["sample_id"])
    video_rows.sort(key=lambda row: row["sample_id"])

    write_jsonl(image_root / "dataset.jsonl", image_rows)
    write_jsonl(video_root / "dataset.jsonl", video_rows)
    write_bucket_summary(image_root / "summary.md", "Single Image Bucket", image_rows, "frames/{sample_id}.jpg")
    write_bucket_summary(video_root / "summary.md", "Video-to-Video Bucket", video_rows, "clips/{sample_id}.mp4")
    write_bucket_readme(image_root / "README.md", "Single Image Bucket", image_rows)
    write_bucket_readme(video_root / "README.md", "Video-to-Video Bucket", video_rows)
    write_root_summary(output_root / "summary.md", image_rows, video_rows)
    write_root_readme(output_root / "README.md")

    prune_assets(image_root / "frames", [f"{safe_name(row['sample_id'])}.jpg" for row in image_rows])
    prune_assets(video_root / "clips", [f"{safe_name(row['sample_id'])}.mp4" for row in video_rows])

    validate_image_rows(image_rows, repo_root, image_root)
    validate_video_rows(video_rows, repo_root, video_root)

    print(f"Source rows: {len(rows)}")
    print(f"Single-image rows: {len(image_rows)}")
    print(f"Video-to-video rows: {len(video_rows)}")
    print(f"Wrote: {output_root}")


if __name__ == "__main__":
    main()
