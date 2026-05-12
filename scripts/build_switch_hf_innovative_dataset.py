#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


TASK_FAMILIES = [
    "vqa_task",
    "action",
    "final_state",
    "verification_action",
    "verification_state",
    "recovery",
]

CAPABILITY_LEVEL = {
    "vqa_task": "L2",
    "action": "L2",
    "final_state": "L3",
    "verification_action": "L4",
    "verification_state": "L4",
    "recovery": "L4",
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def sentence_case(text: str) -> str:
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return ""
    if text[0].isalpha():
        text = text[0].upper() + text[1:]
    return text


def ensure_period(text: str) -> str:
    text = sentence_case(text)
    if not text:
        return text
    if text[-1] not in ".?!":
        text += "."
    return text


def parse_step_number(question_zh: str) -> Optional[int]:
    match = re.search(r"第\s*(\d+)", question_zh or "")
    if not match:
        return None
    return int(match.group(1))


def format_option_block(options: List[str]) -> str:
    letters = ["A", "B", "C", "D"]
    return "".join(f"{letters[i]}. {ensure_period(option)}\n" for i, option in enumerate(options))


def make_hardlink_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def infer_video_source_root(annotation_root: Path) -> Dict[str, Path]:
    video_paths: Dict[str, Path] = {}
    for video in annotation_root.rglob("*.mp4"):
        video_paths.setdefault(video.name, video)
    return video_paths


def build_video_meta(annotation_root: Path) -> Dict[str, Dict[str, Any]]:
    video_meta: Dict[str, Dict[str, Any]] = {}
    for json_path in sorted(annotation_root.glob("*.json")):
        if any(
            json_path.name.endswith(suffix)
            for suffix in (
                ".mcq.json",
                ".openqa.json",
                ".qa_candidates.json",
            )
        ):
            continue
        data = load_json(json_path)
        for item in data:
            annotations = item.get("annotations") or []
            if not annotations:
                continue
            data_id = ""
            for result in annotations[0].get("result", []):
                value = result.get("value") or {}
                labels = value.get("timelinelabels") or []
                if "data_id" not in labels:
                    continue
                meta_text = (result.get("meta") or {}).get("text") or []
                if meta_text:
                    data_id = str(meta_text[0]).strip()
                    break
            meta_block = item.get("data", {}).get("meta", {})
            media_meta = None
            for value in meta_block.values():
                if isinstance(value, dict) and "duration" in value:
                    media_meta = value
                    break
            if data_id and media_meta:
                video_meta[data_id] = {
                    "duration": media_meta.get("duration"),
                    "total_frames": media_meta.get("total_frames"),
                    "fps": media_meta.get("fps"),
                    "source_file": json_path.name,
                }
    return video_meta


def build_video_profiles(
    candidate_payload: Dict[str, Any],
    video_meta: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    profiles: Dict[str, Dict[str, Any]] = {}
    for video in candidate_payload["videos"]:
        data_id = video["data_id"]
        qas = video["qa_candidates"]
        action_spans = [
            qa["source_span"]["end"]
            for qa in qas
            if qa["task_family"] == "action" and qa["source_span"]["end"] is not None
        ]
        final_starts = [
            qa["source_span"]["start"]
            for qa in qas
            if qa["task_family"] == "final_state" and qa["source_span"]["start"] is not None
        ]
        action_count = sum(1 for qa in qas if qa["task_family"] == "action")
        verification_count = sum(
            1 for qa in qas if qa["task_family"] in {"verification_action", "verification_state"}
        )
        has_recovery = any(qa["task_family"] == "recovery" for qa in qas)
        last_action_end = max(action_spans) if action_spans else None
        first_final_start = min(final_starts) if final_starts else None
        delayed_frames = None
        if last_action_end is not None and first_final_start is not None:
            delayed_frames = max(0, first_final_start - last_action_end)

        meta = video_meta.get(data_id, {})
        duration = meta.get("duration")
        total_frames = meta.get("total_frames")
        slice_tags: List[str] = []
        if action_count >= 4:
            slice_tags.append("multi_step")
        if verification_count >= 3:
            slice_tags.append("verification_heavy")
        if has_recovery:
            slice_tags.append("recovery")
        if duration is not None and duration >= 45:
            slice_tags.append("long_horizon")
        if delayed_frames is not None and delayed_frames >= 90:
            slice_tags.append("delayed_effect")
        if not slice_tags:
            slice_tags.append("clean_success")

        profiles[data_id] = {
            "data_id": data_id,
            "video_name": video["video_name"],
            "video_local_path": video["video_local_path"],
            "main_task": sentence_case(video["main_task"]),
            "main_verification": sentence_case(video.get("main_verification") or ""),
            "scenario_family": video["scenario_family"],
            "duration": duration,
            "total_frames": total_frames,
            "action_count": action_count,
            "verification_count": verification_count,
            "has_recovery": has_recovery,
            "delayed_frames": delayed_frames,
            "slice_tags": slice_tags,
            "origin_task_counts": video.get("task_family_counts") or {},
        }
    return profiles


def build_recovery_chain_items(
    candidate_payload: Dict[str, Any],
    profiles: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    chain_items: List[Dict[str, Any]] = []
    for video in candidate_payload["videos"]:
        recovery_qas = [qa for qa in video["qa_candidates"] if qa["task_family"] == "recovery"]
        if not recovery_qas:
            continue
        errors = [qa for qa in recovery_qas if qa["qa_type"] == "error_action"]
        fixes = [qa for qa in recovery_qas if qa["qa_type"] == "correction_action"]
        if not errors or not fixes:
            continue
        data_id = video["data_id"]
        answer = f"Wrong action: {errors[0]['answer']}; Fix: {' -> '.join(fix['answer'] for fix in fixes)}"
        start_values = [qa["source_span"]["start"] for qa in recovery_qas if qa["source_span"]["start"] is not None]
        end_values = [qa["source_span"]["end"] for qa in recovery_qas if qa["source_span"]["end"] is not None]
        chain_items.append(
            {
                "id": f"{data_id}_recovery_chain",
                "origin_qa_ids": [qa["qa_id"] for qa in recovery_qas],
                "source_file": video["source_file"],
                "video_name": video["video_name"],
                "video_local_path": video["video_local_path"],
                "data_id": data_id,
                "item_id": video["item_id"],
                "scenario_family": video["scenario_family"],
                "task_family": "recovery",
                "qa_type": "recovery_chain",
                "capability_level": CAPABILITY_LEVEL["recovery"],
                "slice_tags": profiles[data_id]["slice_tags"],
                "question_zh": "这段视频里先出现了什么错误动作，之后又如何修正？",
                "answer": answer,
                "answer_explanation_zh": "该答案同时覆盖错误动作识别和修复动作，贴合 SWITCH v2 的 recovery 闭环评测。",
                "source_span": {
                    "start": min(start_values) if start_values else None,
                    "end": max(end_values) if end_values else None,
                },
                "notes": "Synthetic recovery_chain item derived from the same recovery trajectory.",
            }
        )
    return chain_items


def build_mcq_query(item: Dict[str, Any], profile: Dict[str, Any]) -> str:
    options = [item["option_a"], item["option_b"], item["option_c"], item["option_d"]]
    qa_type = item["qa_type"]
    main_task = profile["main_task"]
    step_number = parse_step_number(item.get("question_zh", ""))

    if qa_type == "task":
        prompt = "Which of the following options best describes the task being performed in the whole video?"
    elif qa_type == "action_step":
        if step_number is None:
            prompt = f'In the workflow of "{main_task}", which option best describes a key action shown in the video?'
        else:
            prompt = f'In the workflow of "{main_task}", which option best describes key action step {step_number} shown in the video?'
    elif qa_type == "final_state":
        if "delayed_effect" in profile["slice_tags"]:
            prompt = f'After the workflow of "{main_task}" eventually takes effect, which option best describes the resulting state shown in the video?'
        else:
            prompt = f'After the workflow of "{main_task}" is completed, which option best describes the resulting state shown in the video?'
    elif qa_type == "verification_action":
        prompt = (
            f'Following the content of the video input, which of the four provided options correctly '
            f'represents the action for outcome verification in the flow of "{main_task}"?'
        )
    elif qa_type == "verification_state":
        prompt = (
            f'Following the content of the video input, which of the four provided options best describes '
            f'the success signal that should be observed in the flow of "{main_task}"?'
        )
    elif qa_type == "correction_action":
        prompt = (
            f'After the earlier mistake in the video, which of the following options best describes the '
            f'correction action used to recover in the flow of "{main_task}"?'
        )
    else:
        prompt = sentence_case(item.get("question_zh") or "Answer the question based on the video.")

    return prompt + "\n" + format_option_block(options)


def build_openqa_query(item: Dict[str, Any], profile: Dict[str, Any]) -> str:
    qa_type = item["qa_type"]
    main_task = profile["main_task"]
    step_number = parse_step_number(item.get("question_zh", ""))

    if qa_type == "task":
        return "What is the main task in the whole video? Answer with a short phrase."
    if qa_type == "action_step":
        if step_number is None:
            return f'What key action is shown in the workflow of "{main_task}"? Answer with a short phrase.'
        return f'What is key action step {step_number} in the workflow of "{main_task}"? Answer with a short phrase.'
    if qa_type == "final_state":
        return f'What final result is shown after "{main_task}" is completed? Answer with a short phrase.'
    if qa_type == "verification_action":
        return f'How is success verified in the workflow of "{main_task}"? Answer with a short phrase.'
    if qa_type == "verification_state":
        return f'What success signal should be observed in the workflow of "{main_task}"? Answer with a short phrase.'
    if qa_type == "error_action":
        return "What incorrect action is shown in the recovery trajectory? Answer with a short phrase."
    if qa_type == "correction_action":
        return "What correction action is used to recover from the earlier mistake? Answer with a short phrase."
    if qa_type == "recovery_chain":
        return "What mistake happens in the video, and how is it corrected? Answer in the form 'Wrong action: ...; Fix: ...'."
    return sentence_case(item.get("question_zh") or "Answer the question based on the video.")


def build_task_entries(
    *,
    mcq_items: List[Dict[str, Any]],
    openqa_items: List[Dict[str, Any]],
    recovery_chain_items: List[Dict[str, Any]],
    profiles: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    task_entries: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
        task: {"mcq": [], "openqa": []} for task in TASK_FAMILIES
    }

    for item in mcq_items:
        task_family = item["task_family"]
        profile = profiles[item["data_id"]]
        task_entries[task_family]["mcq"].append(
            {
                "origin_qa_id": item["id"],
                "video_name": item["video_name"],
                "data_id": item["data_id"],
                "task_family": task_family,
                "qa_type": item["qa_type"],
                "capability_level": CAPABILITY_LEVEL[task_family],
                "scenario_family": item["scenario_family"],
                "slice_tags": profile["slice_tags"],
                "query": build_mcq_query(item, profile),
                "GT": item["correct_option"],
                "correct_answer": sentence_case(item["correct_answer"]),
                "question_zh": item["question_zh"],
                "answer_explanation_zh": item["answer_explanation_zh"],
                "query_video_name": item["video_name"],
                "source_file": item["source_file"],
                "source_span": item["source_span"],
                "option_a": ensure_period(item["option_a"]),
                "option_b": ensure_period(item["option_b"]),
                "option_c": ensure_period(item["option_c"]),
                "option_d": ensure_period(item["option_d"]),
                "notes": item.get("notes"),
            }
        )

    for item in openqa_items:
        task_family = item["task_family"]
        profile = profiles[item["data_id"]]
        task_entries[task_family]["openqa"].append(
            {
                "origin_qa_id": item["id"],
                "video_name": item["video_name"],
                "data_id": item["data_id"],
                "task_family": task_family,
                "qa_type": item["qa_type"],
                "capability_level": CAPABILITY_LEVEL[task_family],
                "scenario_family": item["scenario_family"],
                "slice_tags": profile["slice_tags"],
                "query": build_openqa_query(item, profile),
                "GT": sentence_case(item["answer"]),
                "answer_type": "structured_short_answer" if task_family == "recovery" else "short_answer",
                "question_zh": item["question_zh"],
                "answer_explanation_zh": item["answer_explanation_zh"],
                "query_video_name": item["video_name"],
                "source_file": item["source_file"],
                "source_span": item["source_span"],
                "notes": item.get("notes"),
            }
        )

    for item in recovery_chain_items:
        task_entries["recovery"]["openqa"].append(
            {
                "origin_qa_id": item["id"],
                "origin_qa_ids": item["origin_qa_ids"],
                "video_name": item["video_name"],
                "data_id": item["data_id"],
                "task_family": "recovery",
                "qa_type": "recovery_chain",
                "capability_level": CAPABILITY_LEVEL["recovery"],
                "scenario_family": item["scenario_family"],
                "slice_tags": item["slice_tags"],
                "query": build_openqa_query(item, profiles[item["data_id"]]),
                "GT": sentence_case(item["answer"]),
                "answer_type": "structured_short_answer",
                "question_zh": item["question_zh"],
                "answer_explanation_zh": item["answer_explanation_zh"],
                "query_video_name": item["video_name"],
                "source_file": item["source_file"],
                "source_span": item["source_span"],
                "notes": item.get("notes"),
            }
        )

    for task_family in TASK_FAMILIES:
        task_entries[task_family]["mcq"].sort(key=lambda row: (row["data_id"], row["origin_qa_id"]))
        task_entries[task_family]["openqa"].sort(key=lambda row: (row["data_id"], row["origin_qa_id"]))

    return task_entries


def materialize_task_directory(
    *,
    output_root: Path,
    task_family: str,
    entries: Dict[str, List[Dict[str, Any]]],
    video_source_map: Dict[str, Path],
) -> Dict[str, int]:
    task_dir = output_root / task_family / "video2txt"
    videos_dir = task_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    unique_videos = sorted({row["video_name"] for group in entries.values() for row in group})
    for video_name in unique_videos:
        source = video_source_map.get(video_name)
        if source is None:
            continue
        make_hardlink_or_copy(source, videos_dir / video_name)

    counts: Dict[str, int] = {}

    mcq_payload = {"data": []}
    for idx, row in enumerate(entries["mcq"]):
        mcq_payload["data"].append(
            {
                "id": idx,
                "query_video_path": f"videos/{row['query_video_name']}",
                "query": row["query"],
                "GT": row["GT"],
                "correct_answer": row["correct_answer"],
                "task_family": row["task_family"],
                "qa_type": row["qa_type"],
                "capability_level": row["capability_level"],
                "scenario_family": row["scenario_family"],
                "slice_tags": row["slice_tags"],
                "question_zh": row["question_zh"],
                "answer_explanation_zh": row["answer_explanation_zh"],
                "origin_qa_id": row["origin_qa_id"],
                "source_file": row["source_file"],
                "source_span": row["source_span"],
                "option_a": row["option_a"],
                "option_b": row["option_b"],
                "option_c": row["option_c"],
                "option_d": row["option_d"],
                "notes": row.get("notes"),
            }
        )
    write_json(task_dir / "vqa.json", mcq_payload)
    counts["mcq"] = len(mcq_payload["data"])

    openqa_payload = {"data": []}
    for idx, row in enumerate(entries["openqa"]):
        openqa_payload["data"].append(
            {
                "id": idx,
                "query_video_path": f"videos/{row['query_video_name']}",
                "query": row["query"],
                "GT": row["GT"],
                "answer_type": row["answer_type"],
                "task_family": row["task_family"],
                "qa_type": row["qa_type"],
                "capability_level": row["capability_level"],
                "scenario_family": row["scenario_family"],
                "slice_tags": row["slice_tags"],
                "question_zh": row["question_zh"],
                "answer_explanation_zh": row["answer_explanation_zh"],
                "origin_qa_id": row["origin_qa_id"],
                "source_file": row["source_file"],
                "source_span": row["source_span"],
                "notes": row.get("notes"),
                "origin_qa_ids": row.get("origin_qa_ids"),
            }
        )
    write_json(task_dir / "openqa.json", openqa_payload)
    counts["openqa"] = len(openqa_payload["data"])

    return counts


def build_closed_loop_bundles(
    task_entries: Dict[str, Dict[str, List[Dict[str, Any]]]],
    profiles: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    by_video: Dict[str, Dict[str, Any]] = {}
    for data_id, profile in profiles.items():
        by_video[data_id] = {
            "data_id": data_id,
            "video_name": profile["video_name"],
            "main_task": profile["main_task"],
            "scenario_family": profile["scenario_family"],
            "capability_layers_present": [],
            "slice_tags": profile["slice_tags"],
            "duration": profile["duration"],
            "total_frames": profile["total_frames"],
            "qas": {task: {"mcq": [], "openqa": []} for task in TASK_FAMILIES},
        }

    for task_family, entries in task_entries.items():
        for fmt_name, rows in entries.items():
            for row in rows:
                data_id = row["data_id"]
                by_video[data_id]["qas"][task_family][fmt_name].append(row["origin_qa_id"])
                level = CAPABILITY_LEVEL[task_family]
                if level not in by_video[data_id]["capability_layers_present"]:
                    by_video[data_id]["capability_layers_present"].append(level)

    return sorted(by_video.values(), key=lambda row: row["data_id"])


def build_manifest(
    task_entries: Dict[str, Dict[str, List[Dict[str, Any]]]],
    profiles: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    task_stats: Dict[str, Dict[str, int]] = {}
    for task_family in TASK_FAMILIES:
        task_stats[task_family] = {
            "mcq": len(task_entries[task_family]["mcq"]),
            "openqa": len(task_entries[task_family]["openqa"]),
        }

    slice_counter = Counter()
    for profile in profiles.values():
        for tag in profile["slice_tags"]:
            slice_counter[tag] += 1

    return {
        "schema_version": "switch-hf-innovative-qa-v1",
        "num_videos": len(profiles),
        "task_stats": task_stats,
        "slice_stats": dict(slice_counter),
        "task_families": TASK_FAMILIES,
        "notes": [
            "Dataset is organized in HF-style task/video2txt folders.",
            "MCQ uses vqa.json to stay compatible with SWITCH-Basic v1 public format.",
            "OpenQA is provided as a parallel openqa.json file.",
            "Recovery adds synthetic recovery_chain items to better support L4 evaluation.",
            "Current package reuses full source videos because frame-accurate clip extraction tools are unavailable in this environment.",
        ],
    }


def write_summary_md(
    output_path: Path,
    manifest: Dict[str, Any],
    bundles: List[Dict[str, Any]],
) -> None:
    lines: List[str] = []
    lines.append("# SWITCH HF-like Innovative QA Summary")
    lines.append("")
    lines.append(f"- Videos: `{manifest['num_videos']}`")
    lines.append("")
    lines.append("## Task Stats")
    lines.append("")
    for task_family, stats in manifest["task_stats"].items():
        lines.append(f"- `{task_family}`: mcq=`{stats['mcq']}`, openqa=`{stats['openqa']}`")
    lines.append("")
    lines.append("## Slice Stats")
    lines.append("")
    for tag, count in manifest["slice_stats"].items():
        lines.append(f"- `{tag}`: `{count}`")
    lines.append("")
    lines.append("## Bundle Preview")
    lines.append("")
    for bundle in bundles[:10]:
        lines.append(f"### {bundle['data_id']} | {bundle['video_name']}")
        lines.append("")
        lines.append(f"- Main task: `{bundle['main_task']}`")
        lines.append(f"- Layers: `{', '.join(bundle['capability_layers_present'])}`")
        lines.append(f"- Slice tags: `{', '.join(bundle['slice_tags'])}`")
        for task_family, fmt_block in bundle["qas"].items():
            total = len(fmt_block["mcq"]) + len(fmt_block["openqa"])
            if total == 0:
                continue
            lines.append(
                f"- `{task_family}`: mcq=`{len(fmt_block['mcq'])}`, openqa=`{len(fmt_block['openqa'])}`"
            )
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    annotation_root = Path("annotations") / "0421" / "switch"
    output_root = annotation_root / "hf_innovative_qa_v1"

    candidate_payload = load_json(annotation_root / "switch_all.qa_candidates.json")
    mcq_payload = load_json(annotation_root / "switch_all.mcq.json")
    openqa_payload = load_json(annotation_root / "switch_all.openqa.json")

    video_meta = build_video_meta(annotation_root)
    profiles = build_video_profiles(candidate_payload, video_meta)
    recovery_chain_items = build_recovery_chain_items(candidate_payload, profiles)
    task_entries = build_task_entries(
        mcq_items=mcq_payload["data"],
        openqa_items=openqa_payload["data"],
        recovery_chain_items=recovery_chain_items,
        profiles=profiles,
    )
    video_source_map = infer_video_source_root(annotation_root)

    task_counts: Dict[str, Dict[str, int]] = {}
    for task_family in TASK_FAMILIES:
        task_counts[task_family] = materialize_task_directory(
            output_root=output_root,
            task_family=task_family,
            entries=task_entries[task_family],
            video_source_map=video_source_map,
        )

    bundles = build_closed_loop_bundles(task_entries, profiles)
    manifest = build_manifest(task_entries, profiles)
    manifest["task_stats"] = {
        task: {
            "mcq": task_counts.get(task, {}).get("mcq", 0),
            "openqa": task_counts.get(task, {}).get("openqa", 0),
        }
        for task in TASK_FAMILIES
    }

    write_json(output_root / "dataset_manifest.json", manifest)
    write_json(output_root / "video_profiles.json", profiles)
    write_json(output_root / "closed_loop_bundles.json", bundles)
    write_summary_md(output_root / "SUMMARY.md", manifest, bundles)
    shutil.copy2(output_root / "SUMMARY.md", output_root / "README.md")

    print(f"Wrote HF-like innovative QA dataset to: {output_root}")
    for task_family in TASK_FAMILIES:
        counts = manifest["task_stats"][task_family]
        print(f"  - {task_family}: mcq={counts['mcq']}, openqa={counts['openqa']}")


if __name__ == "__main__":
    main()
