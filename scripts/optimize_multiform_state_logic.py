#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2


SOURCE_DATASET_DIRNAME = "hf_innovative_qa_v2_multiform"
TARGET_DATASET_DIRNAME = "hf_innovative_qa_v2_multiform_state_aligned_v1"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text).strip("._") or "item"


def get_label_text(result: Dict[str, Any]) -> str:
    meta = result.get("meta") or {}
    value = meta.get("text")
    if isinstance(value, list) and value:
        return str(value[0]).strip()
    if isinstance(value, str):
        return value.strip()
    return ""


def get_first_range(result: Dict[str, Any]) -> Dict[str, Optional[int]]:
    ranges = (result.get("value") or {}).get("ranges") or [{}]
    first = ranges[0]
    start = first.get("start")
    end = first.get("end")
    return {
        "start": int(start) if start is not None else None,
        "end": int(end) if end is not None else None,
    }


def build_video_path_index(annotation_root: Path) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for video_path in sorted(annotation_root.glob("*.mp4")):
        index[video_path.name] = video_path
    for subdir in sorted(annotation_root.glob("SWITCHAction_*")):
        if not subdir.is_dir():
            continue
        for video_path in subdir.glob("*.mp4"):
            index[video_path.name] = video_path
    return index


def iter_raw_annotation_jsons(annotation_root: Path) -> List[Path]:
    source_files: List[Path] = []
    for json_path in sorted(annotation_root.glob("*.json")):
        if any(
            json_path.name.endswith(suffix)
            for suffix in (".mcq.json", ".openqa.json", ".qa_candidates.json")
        ):
            continue
        if json_path.name.endswith("_all.json"):
            continue
        source_files.append(json_path)
    return source_files


def resolve_candidate_payload_path(annotation_root: Path) -> Path:
    preferred = annotation_root / "switch_all.qa_candidates.json"
    if preferred.exists():
        return preferred
    candidates = sorted(annotation_root.glob("*_all.qa_candidates.json"))
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        return candidates[0]
    direct_candidates = sorted(annotation_root.glob("*.qa_candidates.json"))
    if len(direct_candidates) == 1:
        return direct_candidates[0]
    raise FileNotFoundError(f"Unable to resolve combined qa_candidates file under {annotation_root}")


def load_raw_video_index(annotation_root: Path) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    json_paths = iter_raw_annotation_jsons(annotation_root)
    for json_path in json_paths:
        items = load_json(json_path)
        for item in items:
            annotations = item.get("annotations") or []
            if not annotations:
                continue
            data = item.get("data") or {}
            video_name = next(
                (
                    Path(value).name
                    for key, value in data.items()
                    if key != "meta" and isinstance(value, str) and value.lower().endswith(".mp4")
                ),
                None,
            )
            if not video_name:
                continue
            record: Dict[str, Any] = {
                "source_file": json_path.name,
                "video_name": video_name,
                "data_id": None,
                "overall_requirement": "",
                "overall_verification": "",
                "is_final_state_frame": None,
            }
            for result in annotations[0].get("result", []):
                labels = (result.get("value") or {}).get("timelinelabels") or []
                if not labels:
                    continue
                label = labels[0]
                frame_range = get_first_range(result)
                text = get_label_text(result)
                if label == "data_id":
                    record["data_id"] = text
                elif label == "overall_requirement" and not record["overall_requirement"]:
                    record["overall_requirement"] = text
                elif label == "overall_verification" and not record["overall_verification"]:
                    record["overall_verification"] = text
                elif label == "is_final_state":
                    record["is_final_state_frame"] = frame_range["start"]
            if record["data_id"]:
                index[str(record["data_id"])] = record
    return index


def load_candidate_index(annotation_root: Path) -> Dict[str, Dict[str, Any]]:
    candidate_payload = load_json(resolve_candidate_payload_path(annotation_root))
    index: Dict[str, Dict[str, Any]] = {}
    for video in candidate_payload["videos"]:
        for qa in video["qa_candidates"]:
            index[qa["qa_id"]] = qa
    return index


def extract_frame_to_path(video_path: Path, frame_index: int, output_path: Path) -> None:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    candidate_frames: List[int] = []
    if total_frames > 0:
        candidate_frames.append(min(int(frame_index), total_frames - 1))
    else:
        candidate_frames.append(int(frame_index))
    for offset in range(1, 8):
        fallback = max(0, candidate_frames[0] - offset)
        if fallback not in candidate_frames:
            candidate_frames.append(fallback)
    ok = False
    frame = None
    for candidate_frame in candidate_frames:
        capture.set(cv2.CAP_PROP_POS_FRAMES, candidate_frame)
        ok, frame = capture.read()
        if ok:
            break
    capture.release()
    if not ok:
        raise RuntimeError(f"Unable to read frame {frame_index} from {video_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), frame):
        raise RuntimeError(f"Unable to write image: {output_path}")


def append_note(note: Optional[str], extra: str) -> str:
    base = (note or "").strip()
    if extra in base:
        return base
    if not base:
        return extra
    return f"{base} {extra}"


def correct_index(gt_value: str) -> int:
    return max(0, min(3, ord(str(gt_value).strip()[0].upper()) - ord("A")))


def enrich_row_metadata(
    row: Dict[str, Any],
    candidate: Optional[Dict[str, Any]],
    raw_info: Optional[Dict[str, Any]],
    *,
    family: str,
) -> None:
    if candidate is not None:
        row["source_label"] = candidate.get("source_label")
        row["candidate_task_family"] = candidate.get("task_family")
        row["candidate_qa_type"] = candidate.get("qa_type")
    if raw_info is not None:
        row["raw_overall_requirement"] = raw_info.get("overall_requirement")
        row["raw_overall_verification"] = raw_info.get("overall_verification")
        row["is_final_state_frame"] = raw_info.get("is_final_state_frame")
    if family == "verification_state":
        row["verification_logic_version"] = "post_change_state_aligned_v1"
        row["verification_target_mode"] = "post_change_state"
        row["notes"] = append_note(
            row.get("notes"),
            "Optimized with post-change state logic: single-image verification targets should use resulting state rather than change point.",
        )
    if family == "final_state":
        row["final_state_logic_version"] = "is_final_state_frame_aligned_v1"
        row["final_state_gt_mode"] = "is_final_state_frame"
        row["notes"] = append_note(
            row.get("notes"),
            "Optimized with state-aligned logic: final-state GT frame is overridden to the annotated is_final_state frame.",
        )


def process_final_state_visual_form(
    dataset_root: Path,
    form_relpath: str,
    raw_index: Dict[str, Dict[str, Any]],
    candidate_index: Dict[str, Dict[str, Any]],
    video_index: Dict[str, Path],
    report: Dict[str, Any],
) -> None:
    form_path = dataset_root / form_relpath / "vqa.json"
    payload = load_json(form_path)
    changed_count = 0
    missing_final_frame: List[str] = []
    for row in payload["data"]:
        origin_qa_id = str(row["origin_qa_id"])
        data_id = origin_qa_id.split("_", 1)[0]
        raw_info = raw_index.get(data_id)
        candidate = candidate_index.get(origin_qa_id)
        enrich_row_metadata(row, candidate, raw_info, family="final_state")
        if raw_info is None or raw_info.get("is_final_state_frame") is None:
            missing_final_frame.append(origin_qa_id)
            continue
        video_name = str(row.get("origin_qa_id", "")).split("_", 1)[0]
        del video_name
        frame_index = int(raw_info["is_final_state_frame"])
        gt_idx = correct_index(str(row["GT"]))
        option_paths = row.get("option_imgs_path") or []
        if gt_idx >= len(option_paths):
            missing_final_frame.append(origin_qa_id)
            continue
        video_path = video_index.get(str(row.get("origin_qa_id", "")).split("_", 1)[0] + ".mp4")
        if video_path is None and raw_info is not None:
            video_path = video_index.get(str(raw_info["video_name"]))
        if video_path is None:
            missing_final_frame.append(origin_qa_id)
            continue
        option_relpath = Path(option_paths[gt_idx])
        extract_frame_to_path(video_path, frame_index, dataset_root / form_relpath / option_relpath)
        row["gt_frame_source_label"] = "is_final_state"
        row["gt_frame_index"] = frame_index
        row["gt_frame_overrides_candidate_span"] = True
        row["gt_frame_override_reason"] = "Use explicit is_final_state frame as whole-video final-state GT."
        if "option_source_frames" in row and gt_idx < len(row["option_source_frames"]):
            row["option_source_frames"][gt_idx] = frame_index
        if "option_origin_qa_ids" in row and gt_idx < len(row["option_origin_qa_ids"]):
            row["option_origin_qa_ids"][gt_idx] = f"{data_id}_is_final_state_frame"
        if "option_source_types" in row and gt_idx < len(row["option_source_types"]):
            row["option_source_types"][gt_idx] = "is_final_state_frame"
        changed_count += 1
    write_json(form_path, payload)
    report["forms"][form_relpath] = {
        "changed_rows": changed_count,
        "missing_is_final_state_rows": missing_final_frame,
    }


def process_json_form_rows(
    json_path: Path,
    family: str,
    raw_index: Dict[str, Dict[str, Any]],
    candidate_index: Dict[str, Dict[str, Any]],
) -> int:
    payload = load_json(json_path)
    changed = 0
    for row in payload.get("data", []):
        origin_qa_id = str(row.get("origin_qa_id") or "")
        candidate = candidate_index.get(origin_qa_id)
        data_id = origin_qa_id.split("_", 1)[0] if "_" in origin_qa_id else None
        raw_info = raw_index.get(data_id) if data_id else None
        enrich_row_metadata(row, candidate, raw_info, family=family)
        changed += 1
    write_json(json_path, payload)
    return changed


def update_manifest_and_readme(dataset_root: Path, report: Dict[str, Any]) -> None:
    manifest_path = dataset_root / "dataset_manifest.json"
    manifest = load_json(manifest_path)
    manifest["schema_version"] = "switch-hf-innovative-qa-v2-multiform-state-aligned-v1"
    manifest["state_alignment_summary"] = report
    notes = manifest.get("notes") or []
    note = "State-aligned optimization: final_state visual GT uses is_final_state; single-image verification is audited as post-change state centric."
    if note not in notes:
        notes.append(note)
    manifest["notes"] = notes
    write_json(manifest_path, manifest)

    readme_path = dataset_root / "README.md"
    readme = readme_path.read_text(encoding="utf-8")
    if "State-Aligned Optimization" not in readme:
        readme = readme.rstrip() + "\n\n## State-Aligned Optimization\n\n- `final_state` visual forms now override the correct GT frame to the annotated `is_final_state` frame.\n- Single-image `verification_state` forms are explicitly audited as post-change state targets rather than change-point targets.\n- `video -> video` change-style verification remains a future extension; this pass focuses on fixing state-oriented supervision in the existing multiform package.\n"
        readme_path.write_text(readme, encoding="utf-8")


def choose_balanced_rows(rows: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row.get("scenario_family") or "other")].append(row)
    chosen: List[Dict[str, Any]] = []
    scenarios = sorted(buckets.keys())
    while len(chosen) < limit and scenarios:
        next_round: List[str] = []
        for scenario in scenarios:
            bucket = buckets[scenario]
            if bucket:
                chosen.append(bucket.pop(0))
                if len(chosen) >= limit:
                    break
            if bucket:
                next_round.append(scenario)
        scenarios = next_round
    return chosen


def build_review_batch(dataset_root: Path, raw_index: Dict[str, Dict[str, Any]], candidate_index: Dict[str, Dict[str, Any]]) -> Path:
    batch_root = dataset_root / "review_batches" / "batch_01_visual_state_alignment"
    if batch_root.exists():
        shutil.rmtree(batch_root)
    batch_root.mkdir(parents=True, exist_ok=True)

    selection_specs = [
        ("final_state/img2img/vqa.json", 4),
        ("final_state/video2img/vqa.json", 3),
        ("verification_state/img2img/vqa.json", 3),
    ]

    selected_samples: List[Dict[str, Any]] = []
    for relpath, limit in selection_specs:
        payload = load_json(dataset_root / relpath)
        rows = choose_balanced_rows(payload["data"], limit)
        form_name = relpath.replace("/vqa.json", "")
        for row in rows:
            origin_qa_id = str(row["origin_qa_id"])
            data_id = origin_qa_id.split("_", 1)[0]
            candidate = candidate_index.get(origin_qa_id) or {}
            raw_info = raw_index.get(data_id) or {}
            sample = {
                "bundle_id": safe_name(f"{form_name}_{origin_qa_id}"),
                "form": form_name,
                "origin_qa_id": origin_qa_id,
                "scenario_family": row.get("scenario_family"),
                "query": row.get("query"),
                "GT": row.get("GT"),
                "canonical_answer": row.get("canonical_answer"),
                "source_label": row.get("source_label") or candidate.get("source_label"),
                "source_span": row.get("source_span") or candidate.get("source_span"),
                "raw_overall_requirement": row.get("raw_overall_requirement") or raw_info.get("overall_requirement"),
                "raw_overall_verification": row.get("raw_overall_verification") or raw_info.get("overall_verification"),
                "is_final_state_frame": row.get("is_final_state_frame") or raw_info.get("is_final_state_frame"),
                "gt_frame_source_label": row.get("gt_frame_source_label"),
                "gt_frame_index": row.get("gt_frame_index"),
                "query_img_path": row.get("query_img_path"),
                "query_video_path": row.get("query_video_path"),
                "option_imgs_path": row.get("option_imgs_path"),
                "notes": row.get("notes"),
                "review_focus": (
                    "Check whether final-state GT really matches the annotated is_final_state frame."
                    if str(row.get("task_family")) == "final_state"
                    else "Check whether verification target is a resulting state rather than a change point."
                ),
            }
            selected_samples.append(sample)

    payload = {
        "batch_id": "batch_01_visual_state_alignment",
        "dataset_root": str(dataset_root),
        "selection_logic": "Balanced first-pass review focusing on changed visual forms after state-aligned optimization.",
        "samples": selected_samples,
    }
    write_json(batch_root / "selected_samples.json", payload)

    md_lines = [
        "# Batch 01 Visual State Alignment Review",
        "",
        f"- Dataset root: `{dataset_root}`",
        "- Focus: inspect changed visual forms after state-aligned optimization",
        "",
    ]
    for sample in selected_samples:
        md_lines.append(f"## {sample['bundle_id']}")
        md_lines.append("")
        md_lines.append(f"- form: `{sample['form']}`")
        md_lines.append(f"- origin_qa_id: `{sample['origin_qa_id']}`")
        md_lines.append(f"- scenario_family: `{sample['scenario_family']}`")
        md_lines.append(f"- query: {sample['query']}")
        md_lines.append(f"- GT: `{sample['GT']}`")
        md_lines.append(f"- canonical_answer: `{sample['canonical_answer']}`")
        md_lines.append(f"- source_label: `{sample['source_label']}`")
        md_lines.append(f"- source_span: `{sample['source_span']}`")
        md_lines.append(f"- raw_overall_requirement: `{sample['raw_overall_requirement']}`")
        md_lines.append(f"- raw_overall_verification: `{sample['raw_overall_verification']}`")
        md_lines.append(f"- is_final_state_frame: `{sample['is_final_state_frame']}`")
        if sample.get("gt_frame_source_label"):
            md_lines.append(f"- gt_frame_source_label: `{sample['gt_frame_source_label']}`")
            md_lines.append(f"- gt_frame_index: `{sample['gt_frame_index']}`")
        if sample.get("query_img_path"):
            md_lines.append(f"- query_img_path: `{sample['query_img_path']}`")
        if sample.get("query_video_path"):
            md_lines.append(f"- query_video_path: `{sample['query_video_path']}`")
        if sample.get("option_imgs_path"):
            md_lines.append(f"- option_imgs_path: `{sample['option_imgs_path']}`")
        md_lines.append(f"- review_focus: {sample['review_focus']}")
        md_lines.append(f"- notes: {sample['notes']}")
        md_lines.append("")
    (batch_root / "selected_samples.md").write_text("\n".join(md_lines), encoding="utf-8")
    return batch_root


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply state-aligned post-processing to a generated SWITCH multiform dataset."
    )
    parser.add_argument(
        "--annotation-root",
        type=Path,
        default=Path("annotations") / "0421" / "switch",
        help="Directory containing raw annotation JSON, videos, qa_candidates, and the source dataset.",
    )
    parser.add_argument(
        "--source-dirname",
        type=str,
        default=SOURCE_DATASET_DIRNAME,
        help="Existing multiform dataset directory name under annotation-root.",
    )
    parser.add_argument(
        "--target-dirname",
        type=str,
        default=TARGET_DATASET_DIRNAME,
        help="Target optimized dataset directory name under annotation-root.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    annotation_root = args.annotation_root
    if not annotation_root.is_absolute():
        annotation_root = repo_root / annotation_root
    source_root = annotation_root / args.source_dirname
    target_root = annotation_root / args.target_dirname
    if target_root.exists():
        shutil.rmtree(target_root)
    shutil.copytree(source_root, target_root)

    raw_index = load_raw_video_index(annotation_root)
    candidate_index = load_candidate_index(annotation_root)
    video_index = build_video_path_index(annotation_root)

    report: Dict[str, Any] = {
        "target_root": str(target_root),
        "forms": {},
        "metadata_updates": {},
        "warnings": [],
    }

    for form_relpath in ["final_state/img2img", "final_state/video2img"]:
        process_final_state_visual_form(target_root, form_relpath, raw_index, candidate_index, video_index, report)

    metadata_forms = [
        ("final_state/video2txt/openqa.json", "final_state"),
        ("final_state/video2txt/vqa.json", "final_state"),
        ("final_state/img2txt/openqa.json", "final_state"),
        ("final_state/img2txt/vqa.json", "final_state"),
        ("verification_state/video2txt/openqa.json", "verification_state"),
        ("verification_state/video2txt/vqa.json", "verification_state"),
        ("verification_state/img2txt/openqa.json", "verification_state"),
        ("verification_state/img2txt/vqa.json", "verification_state"),
        ("verification_state/img2img/vqa.json", "verification_state"),
    ]
    for relpath, family in metadata_forms:
        changed = process_json_form_rows(target_root / relpath, family, raw_index, candidate_index)
        report["metadata_updates"][relpath] = changed

    report["final_state_visual_rows_changed"] = sum(
        entry["changed_rows"] for key, entry in report["forms"].items() if key.startswith("final_state/")
    )
    report["verification_state_rows_audited"] = (
        report["metadata_updates"]["verification_state/video2txt/openqa.json"]
        + report["metadata_updates"]["verification_state/video2txt/vqa.json"]
        + report["metadata_updates"]["verification_state/img2txt/openqa.json"]
        + report["metadata_updates"]["verification_state/img2txt/vqa.json"]
        + report["metadata_updates"]["verification_state/img2img/vqa.json"]
    )
    update_manifest_and_readme(target_root, report)
    write_json(target_root / "state_alignment_report.json", report)
    batch_root = build_review_batch(target_root, raw_index, candidate_index)
    print(f"Wrote optimized dataset to: {target_root}")
    print(f"Wrote report to: {target_root / 'state_alignment_report.json'}")
    print(f"Wrote first review batch to: {batch_root}")


if __name__ == "__main__":
    main()
