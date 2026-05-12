#!/usr/bin/env python3
"""
Extract QA/OpenQA candidates from the current Label Studio timeline annotation
export used by SWITCH `Action_1`.

The script is intentionally conservative:
- it only extracts what the current annotation can support relatively stably
- it marks every auto-generated MCQ option set as `needs_manual_review`
- it keeps source spans so annotators can trace each QA back to the original
  timeline segment

Example:
    python scripts/extract_action1_qa.py \
        --input 30fps/SWITCH_帧_Action_1.json \
        --output-json 30fps/SWITCH_帧_Action_1.qa_candidates.json \
        --output-md 30fps/SWITCH_帧_Action_1.qa_summary.md
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


KNOWN_LABELS = {
    "Target object",
    "Demand",
    "Initial position",
    "Execute action",
    "Verification action",
    "Environmental status",
    "Physical environmental condition",
    "Incorrect action",
    "Correction action",
}


AUTO_MCQ_ELIGIBLE_TYPES = {
    "task",
    "target_object",
    "initial_state",
    "action_step",
    "verification_action",
    "intermediate_state",
    "final_state",
    "error_action",
    "correction_action",
}


@dataclass
class Segment:
    label: str
    text: str
    start: Optional[float]
    end: Optional[float]


def split_payload(raw_text: str, label: str) -> str:
    raw_text = (raw_text or "").strip()
    if not raw_text:
        return ""

    # Common forms in the file:
    #   Target object：Elevator button
    #   Demand:Query Project
    #   Demand;Arrive at the sixth floor
    parts = re.split(r"[:：;；]\s*", raw_text, maxsplit=1)
    if len(parts) == 2:
        return parts[1].strip()
    return raw_text.replace(label, "", 1).strip()


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def parse_segments(item: Dict[str, Any]) -> List[Segment]:
    segments: List[Segment] = []
    annotations = item.get("annotations") or []
    if not annotations:
        return segments

    for result in annotations[0].get("result", []):
        value = result.get("value", {})
        labels = value.get("timelinelabels") or []
        if not labels:
            continue
        label = labels[0]
        raw_texts = result.get("meta", {}).get("text", []) or []
        raw_text = raw_texts[0] if raw_texts else ""
        payload = normalize_spaces(split_payload(raw_text, label))
        ranges = value.get("ranges", [])
        start = ranges[0].get("start") if ranges else None
        end = ranges[0].get("end") if ranges else None
        segments.append(Segment(label=label, text=payload, start=start, end=end))

    segments.sort(key=lambda s: ((s.start or 0), (s.end or 0), s.label))
    return segments


def infer_summary_labels(segments: List[Segment]) -> List[str]:
    return [seg.label for seg in segments if seg.label not in KNOWN_LABELS]


def choose_main_task(demands: List[str], summaries: List[str]) -> str:
    if summaries:
        return summaries[0]
    if not demands:
        return ""
    return max(demands, key=len)


def make_candidate(
    video_name: str,
    qa_id: str,
    qa_type: str,
    question_zh: str,
    answer: str,
    source_label: str,
    start: Optional[float],
    end: Optional[float],
    notes: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "qa_id": qa_id,
        "video_name": video_name,
        "qa_type": qa_type,
        "question_zh": question_zh,
        "answer": answer,
        "supported_formats": ["openqa"],
        "source_label": source_label,
        "source_span": {"start": start, "end": end},
        "notes": notes,
    }


def build_candidates_for_item(item: Dict[str, Any]) -> Dict[str, Any]:
    video_name = Path(item.get("data", {}).get("Action_1", "")).name or f"{item.get('id')}.mp4"
    segments = parse_segments(item)

    by_label: Dict[str, List[Segment]] = defaultdict(list)
    for seg in segments:
        by_label[seg.label].append(seg)

    summaries = infer_summary_labels(segments)
    demands = [seg.text for seg in by_label["Demand"] if seg.text]
    main_task = choose_main_task(demands, summaries)

    candidates: List[Dict[str, Any]] = []
    counter = 1

    def next_id(prefix: str) -> str:
        nonlocal counter
        qa_id = f"{prefix}_{counter:03d}"
        counter += 1
        return qa_id

    target_objects = [seg.text for seg in by_label["Target object"] if seg.text]
    initial_positions = [seg.text for seg in by_label["Initial position"] if seg.text]
    execute_actions = by_label["Execute action"]
    verification_actions = by_label["Verification action"]
    env_statuses = by_label["Environmental status"]
    final_conditions = by_label["Physical environmental condition"]
    incorrect_actions = by_label["Incorrect action"]
    correction_actions = by_label["Correction action"]

    if main_task:
        candidates.append(
            make_candidate(
                video_name=video_name,
                qa_id=next_id("task"),
                qa_type="task",
                question_zh="这段视频的主要任务是什么？",
                answer=main_task,
                source_label="Demand/Summary",
                start=segments[0].start if segments else None,
                end=segments[-1].end if segments else None,
                notes="优先使用整段 summary label；若缺失，则回退到最长 Demand。",
            )
        )

    if target_objects:
        seg = by_label["Target object"][0]
        candidates.append(
            make_candidate(
                video_name=video_name,
                qa_id=next_id("target"),
                qa_type="target_object",
                question_zh="这段视频的目标设备或交互对象是什么？",
                answer=target_objects[0],
                source_label="Target object",
                start=seg.start,
                end=seg.end,
            )
        )

    for idx, seg in enumerate(by_label["Initial position"], start=1):
        if not seg.text:
            continue
        candidates.append(
            make_candidate(
                video_name=video_name,
                qa_id=next_id("init"),
                qa_type="initial_state",
                question_zh=f"这段视频在初始阶段的关键状态 {idx} 是什么？",
                answer=seg.text,
                source_label="Initial position",
                start=seg.start,
                end=seg.end,
                notes="当前文件的 Initial position 数量较少，通常可作为初始状态问题来源。",
            )
        )

    for idx, seg in enumerate(execute_actions, start=1):
        if not seg.text:
            continue
        notes = None
        if "click the button" in seg.text.lower():
            notes = "动作文本过泛，建议人工改写成更明确的 canonical action。"
        candidates.append(
            make_candidate(
                video_name=video_name,
                qa_id=next_id("action"),
                qa_type="action_step",
                question_zh=f"第 {idx} 个关键动作是什么？",
                answer=seg.text,
                source_label="Execute action",
                start=seg.start,
                end=seg.end,
                notes=notes,
            )
        )

    for idx, seg in enumerate(verification_actions, start=1):
        if not seg.text:
            continue
        notes = None
        if "screen has switched" in seg.text.lower():
            notes = "验证动作过泛，建议人工细化成具体检查哪个目标页面/状态。"
        candidates.append(
            make_candidate(
                video_name=video_name,
                qa_id=next_id("verify"),
                qa_type="verification_action",
                question_zh=f"第 {idx} 个关键步骤应该如何验证是否成功？",
                answer=seg.text,
                source_label="Verification action",
                start=seg.start,
                end=seg.end,
                notes=notes,
            )
        )

    for idx, seg in enumerate(env_statuses, start=1):
        if not seg.text:
            continue
        candidates.append(
            make_candidate(
                video_name=video_name,
                qa_id=next_id("intermediate"),
                qa_type="intermediate_state",
                question_zh=f"中间过程状态 {idx} 是什么？",
                answer=seg.text,
                source_label="Environmental status",
                start=seg.start,
                end=seg.end,
                notes="当前文件里的 Environmental status 更像 intermediate_state 候选，而不是最终结果。",
            )
        )

    for idx, seg in enumerate(final_conditions, start=1):
        if not seg.text:
            continue
        candidates.append(
            make_candidate(
                video_name=video_name,
                qa_id=next_id("final"),
                qa_type="final_state",
                question_zh="这段视频最终完成后的结果是什么？",
                answer=seg.text,
                source_label="Physical environmental condition",
                start=seg.start,
                end=seg.end,
                notes="当前文件里的 Physical environmental condition 可直接作为 final_state 候选。",
            )
        )

    for idx, seg in enumerate(incorrect_actions, start=1):
        if not seg.text:
            continue
        candidates.append(
            make_candidate(
                video_name=video_name,
                qa_id=next_id("error"),
                qa_type="error_action",
                question_zh=f"这段视频中的错误动作 {idx} 是什么？",
                answer=seg.text,
                source_label="Incorrect action",
                start=seg.start,
                end=seg.end,
            )
        )

    for idx, seg in enumerate(correction_actions, start=1):
        if not seg.text:
            continue
        candidates.append(
            make_candidate(
                video_name=video_name,
                qa_id=next_id("fix"),
                qa_type="correction_action",
                question_zh=f"这段视频中的修正动作 {idx} 是什么？",
                answer=seg.text,
                source_label="Correction action",
                start=seg.start,
                end=seg.end,
            )
        )

    # Additional goal-parameter-like candidates when multiple Demand labels exist.
    if len(demands) > 1:
        for extra_idx, demand_text in enumerate(demands[1:], start=1):
            if not demand_text:
                continue
            source_seg = by_label["Demand"][extra_idx]
            candidates.append(
                make_candidate(
                    video_name=video_name,
                    qa_id=next_id("goalparam"),
                    qa_type="goal_parameter",
                    question_zh=f"这段视频中额外给出的目标参数 {extra_idx} 是什么？",
                    answer=demand_text,
                    source_label="Demand",
                    start=source_seg.start,
                    end=source_seg.end,
                    notes="该项通常对应 goal_slot 候选，建议人工判定是目标站点、楼层还是其他参数。",
                )
            )

    return {
        "video_name": video_name,
        "item_id": item.get("id"),
        "main_task": main_task,
        "label_counts": dict(Counter(seg.label for seg in segments)),
        "qa_candidates": candidates,
    }


def attach_auto_mcq_options(all_video_results: List[Dict[str, Any]], seed: int = 7) -> None:
    rng = random.Random(seed)
    answer_pool: Dict[str, List[str]] = defaultdict(list)
    for video in all_video_results:
        for qa in video["qa_candidates"]:
            answer = qa["answer"]
            if answer and answer not in answer_pool[qa["qa_type"]]:
                answer_pool[qa["qa_type"]].append(answer)

    for video in all_video_results:
        for qa in video["qa_candidates"]:
            qa_type = qa["qa_type"]
            if qa_type not in AUTO_MCQ_ELIGIBLE_TYPES:
                continue

            pool = [x for x in answer_pool[qa_type] if x != qa["answer"]]
            if len(pool) < 3:
                continue

            distractors = rng.sample(pool, 3)
            options = [qa["answer"], *distractors]
            rng.shuffle(options)

            qa["supported_formats"].append("mcq")
            qa["mcq_options_auto"] = options
            qa["mcq_answer"] = qa["answer"]
            qa["mcq_needs_manual_review"] = True


def write_markdown(output_path: Path, all_video_results: List[Dict[str, Any]]) -> None:
    lines: List[str] = []
    total_qas = sum(len(v["qa_candidates"]) for v in all_video_results)
    lines.append("# SWITCH Action_1 QA Extraction Summary")
    lines.append("")
    lines.append(f"- Videos parsed: `{len(all_video_results)}`")
    lines.append(f"- QA/OpenQA candidates extracted: `{total_qas}`")
    lines.append("- Note: all MCQ options are auto-generated weak distractors and need manual review.")
    lines.append("")

    type_counter = Counter()
    for video in all_video_results:
        for qa in video["qa_candidates"]:
            type_counter[qa["qa_type"]] += 1

    lines.append("## QA Type Counts")
    lines.append("")
    for qa_type, count in type_counter.most_common():
        lines.append(f"- `{qa_type}`: `{count}`")
    lines.append("")

    for video in all_video_results:
        lines.append(f"## {video['video_name']}")
        lines.append("")
        if video["main_task"]:
            lines.append(f"- Main task: `{video['main_task']}`")
        lines.append(f"- Candidate count: `{len(video['qa_candidates'])}`")
        lines.append("")

        for qa in video["qa_candidates"]:
            lines.append(f"### {qa['qa_id']} · `{qa['qa_type']}`")
            lines.append("")
            lines.append(f"- Question: {qa['question_zh']}")
            lines.append(f"- Answer: `{qa['answer']}`")
            lines.append(f"- Source: `{qa['source_label']}`")
            lines.append(
                f"- Span: `{qa['source_span']['start']} -> {qa['source_span']['end']}`"
            )
            lines.append(f"- Formats: `{', '.join(qa['supported_formats'])}`")
            if qa.get("mcq_options_auto"):
                lines.append("- Auto MCQ options:")
                for idx, opt in enumerate(qa["mcq_options_auto"], start=1):
                    lines.append(f"  {idx}. `{opt}`")
            if qa.get("notes"):
                lines.append(f"- Notes: {qa['notes']}")
            lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract QA/OpenQA candidates from SWITCH Action_1 Label Studio export."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("30fps") / "SWITCH_帧_Action_1.json",
        help="Path to the Label Studio JSON export.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("30fps") / "SWITCH_帧_Action_1.qa_candidates.json",
        help="Where to write the structured QA candidate JSON.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("30fps") / "SWITCH_帧_Action_1.qa_summary.md",
        help="Where to write the human-readable markdown summary.",
    )
    args = parser.parse_args()

    with args.input.open("r", encoding="utf-8") as f:
        data = json.load(f)

    all_video_results = [build_candidates_for_item(item) for item in data]
    attach_auto_mcq_options(all_video_results)

    output_payload = {
        "input_file": str(args.input),
        "num_videos": len(all_video_results),
        "num_qa_candidates": sum(len(v["qa_candidates"]) for v in all_video_results),
        "videos": all_video_results,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(output_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_markdown(args.output_md, all_video_results)

    print(f"Wrote JSON: {args.output_json}")
    print(f"Wrote Markdown: {args.output_md}")
    print(f"Videos parsed: {output_payload['num_videos']}")
    print(f"QA candidates: {output_payload['num_qa_candidates']}")


if __name__ == "__main__":
    main()
