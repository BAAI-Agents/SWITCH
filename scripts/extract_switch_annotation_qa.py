#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


ACTION_LABELS = {
    "action-type",
    "action_requirement",
    "action_description",
    "action_step_id",
}

STATE_LABELS = {
    "ui_state",
    "physical_world_state",
}

MCQ_ELIGIBLE_TYPES = {
    "task",
    "action_step",
    "verification_action",
    "verification_state",
    "final_state",
}

SKIP_OUTPUT_SUFFIXES = (
    ".qa_candidates.json",
    ".openqa.json",
    ".mcq.json",
    ".qa_summary.md",
)

TEXT_REPLACEMENTS = {
    "floorr": "floor",
    "The elevator door close": "The elevator door closes",
    " The elevator door close": "The elevator door closes",
    "Whether successfully got the subway ticket": "Successfully got the subway ticket",
    "Successfully got a subway ticket": "Successfully got the subway ticket",
    "The button for the eighth floorr lights up": "The button for the eighth floor lights up",
}

ACTION_TYPE_MAP = {
    "execute action": "execute action",
    "verification action": "verification action",
    "wrong action": "wrong action",
    "recovery action": "recovery action",
}

QA_TYPE_TO_FAMILY = {
    "task": "vqa_task",
    "action_step": "action",
    "verification_action": "verification_action",
    "verification_state": "verification_state",
    "final_state": "final_state",
    "error_action": "recovery",
    "correction_action": "recovery",
}


@dataclass
class Segment:
    label: str
    text: str
    start: Optional[float]
    end: Optional[float]


@dataclass
class ActionEvent:
    action_type: str
    action_requirement: str
    action_description: str
    step_id: str
    start: Optional[float]
    end: Optional[float]


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def sentence_case(text: str) -> str:
    if not text:
        return text
    if text[0].isalpha():
        return text[0].upper() + text[1:]
    return text


def clean_text(text: str) -> str:
    cleaned = normalize_spaces(text)
    if not cleaned:
        return ""
    for src, dst in TEXT_REPLACEMENTS.items():
        cleaned = cleaned.replace(src, dst)
    cleaned = cleaned.replace("  ", " ")
    cleaned = re.sub(r"\s+\.", ".", cleaned)
    return sentence_case(cleaned.strip())


def get_meta_text(result: Dict[str, Any]) -> str:
    meta = result.get("meta") or {}
    value = meta.get("text")
    if isinstance(value, list) and value:
        return str(value[0])
    if isinstance(value, str):
        return value
    return ""


def get_first_range(value: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    ranges = value.get("ranges") or []
    if not ranges:
        return None, None
    first = ranges[0]
    return first.get("start"), first.get("end")


def normalize_action_type(text: str) -> str:
    normalized = normalize_spaces(text).lower()
    return ACTION_TYPE_MAP.get(normalized, normalized)


def infer_video_name(item: Dict[str, Any], data_id: str) -> str:
    data = item.get("data") or {}
    for key, value in data.items():
        if key == "meta":
            continue
        if isinstance(value, str) and value.lower().endswith(".mp4"):
            return Path(value).name
    if data_id:
        return f"{data_id}.mp4"
    return f"{item.get('id')}.mp4"


def build_video_index(root_dir: Path) -> Dict[str, str]:
    index: Dict[str, str] = {}
    for video_path in root_dir.rglob("*.mp4"):
        index.setdefault(video_path.name, video_path.relative_to(root_dir).as_posix())
    return index


def infer_scenario_family(task_text: str) -> str:
    lowered = (task_text or "").lower()
    if "elevator" in lowered:
        return "elevator"
    if "appointment" in lowered or "medical" in lowered or "doctor" in lowered or "machine" in lowered:
        return "medical_kiosk"
    if "ticket" in lowered or "subway" in lowered:
        return "subway_ticket"
    return "other"


def parse_segments(item: Dict[str, Any]) -> List[Segment]:
    segments: List[Segment] = []
    annotations = item.get("annotations") or []
    if not annotations:
        return segments

    for result in annotations[0].get("result", []):
        value = result.get("value") or {}
        labels = value.get("timelinelabels") or []
        if not labels:
            continue
        label = labels[0]
        raw_text = get_meta_text(result)
        text = normalize_spaces(raw_text) if label == "data_id" else clean_text(raw_text)
        start, end = get_first_range(value)
        segments.append(Segment(label=label, text=text, start=start, end=end))

    segments.sort(key=lambda seg: ((seg.start or 0), (seg.end or 0), seg.label))
    return segments


def build_action_events(segments: Iterable[Segment]) -> List[ActionEvent]:
    grouped: Dict[Tuple[Optional[float], Optional[float]], Dict[str, List[str]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for segment in segments:
        if segment.label not in ACTION_LABELS:
            continue
        grouped[(segment.start, segment.end)][segment.label].append(segment.text)

    events: List[ActionEvent] = []
    for (start, end), payload in grouped.items():
        action_type = normalize_action_type((payload.get("action-type") or [""])[0])
        action_requirement = clean_text((payload.get("action_requirement") or [""])[0])
        action_description = clean_text((payload.get("action_description") or [""])[0])
        step_id = normalize_spaces((payload.get("action_step_id") or [""])[0])
        events.append(
            ActionEvent(
                action_type=action_type,
                action_requirement=action_requirement,
                action_description=action_description or action_requirement,
                step_id=step_id,
                start=start,
                end=end,
            )
        )

    def sort_key(event: ActionEvent) -> Tuple[float, float, int]:
        try:
            step_num = int(event.step_id)
        except (TypeError, ValueError):
            step_num = 10**9
        return (event.start or 0, event.end or 0, step_num)

    events.sort(key=sort_key)
    return events


def explain_candidate(qa_type: str, source_label: str) -> str:
    if qa_type == "task":
        return "标注中的 overall_requirement 明确给出了整段视频的主要目标。"
    if qa_type == "action_step":
        return "该时间段的 action_description 描述了完成任务所需的关键动作。"
    if qa_type == "verification_action":
        return "该时间段被标为 verification action，表示应如何检查任务是否成功。"
    if qa_type == "verification_state":
        return f"{source_label} 提供了可直接观察到的成功信号。"
    if qa_type == "final_state":
        return "末段状态信号或 overall_verification 给出了任务成功完成后的结果。"
    if qa_type == "error_action":
        return "标注中显式出现 wrong action，可作为错误轨迹的依据。"
    if qa_type == "correction_action":
        return "标注中显式出现 recovery action，可作为修正轨迹的依据。"
    return "答案直接来自当前时间轴标注。"


def make_candidate(
    *,
    source_file: Path,
    video_name: str,
    video_local_path: Optional[str],
    data_id: str,
    item_id: Any,
    scenario_family: str,
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
        "source_file": source_file.name,
        "video_name": video_name,
        "video_local_path": video_local_path,
        "data_id": data_id,
        "item_id": item_id,
        "scenario_family": scenario_family,
        "task_family": QA_TYPE_TO_FAMILY[qa_type],
        "qa_type": qa_type,
        "question_zh": question_zh,
        "answer": answer,
        "answer_explanation_zh": explain_candidate(qa_type, source_label),
        "supported_formats": ["openqa"],
        "source_label": source_label,
        "source_span": {"start": start, "end": end},
        "notes": notes,
    }


def choose_final_state_candidate(
    state_segments: List[Segment],
    overall_verification: str,
    frame_end: Optional[float],
) -> Tuple[str, str, Optional[float], Optional[float], Optional[str]]:
    if state_segments:
        threshold = (frame_end or 0) * 0.6 if frame_end else None
        tail_states = [
            segment
            for segment in state_segments
            if threshold is None or (segment.end is not None and segment.end >= threshold)
        ]
        chosen = tail_states[-1] if tail_states else state_segments[-1]
        return (
            chosen.text,
            chosen.label,
            chosen.start,
            chosen.end,
            "优先使用末段可观察状态，便于和 verification/final-state 任务对齐。",
        )

    return overall_verification, "overall_verification", None, None, None


def build_candidates_for_item(
    item: Dict[str, Any],
    *,
    source_file: Path,
    video_index: Dict[str, str],
) -> Dict[str, Any]:
    segments = parse_segments(item)
    by_label: Dict[str, List[Segment]] = defaultdict(list)
    for segment in segments:
        by_label[segment.label].append(segment)

    data_id = (by_label.get("data_id") or [Segment("data_id", "", None, None)])[0].text
    frame_end = None
    if by_label.get("frame_end"):
        frame_end = by_label["frame_end"][0].end
    video_name = infer_video_name(item, data_id)
    video_local_path = video_index.get(video_name)
    overall_requirement = (by_label.get("overall_requirement") or [Segment("", "", None, None)])[0].text
    overall_verification = (by_label.get("overall_verification") or [Segment("", "", None, None)])[0].text
    scenario_family = infer_scenario_family(overall_requirement)
    action_events = build_action_events(segments)
    state_segments = [
        segment for segment in segments if segment.label in STATE_LABELS and segment.text
    ]

    candidates: List[Dict[str, Any]] = []
    counter = 1

    def next_id(prefix: str) -> str:
        nonlocal counter
        qa_id = f"{data_id}_{prefix}_{counter:03d}"
        counter += 1
        return qa_id

    if overall_requirement:
        candidates.append(
            make_candidate(
                source_file=source_file,
                video_name=video_name,
                video_local_path=video_local_path,
                data_id=data_id,
                item_id=item.get("id"),
                scenario_family=scenario_family,
                qa_id=next_id("task"),
                qa_type="task",
                question_zh="这段视频的主要任务是什么？",
                answer=overall_requirement,
                source_label="overall_requirement",
                start=1,
                end=frame_end,
                notes="对应 processed_code_v1/v2 中的 vqa_task。",
            )
        )

    final_answer, final_label, final_start, final_end, final_notes = choose_final_state_candidate(
        state_segments,
        overall_verification,
        frame_end,
    )
    if final_answer:
        candidates.append(
            make_candidate(
                source_file=source_file,
                video_name=video_name,
                video_local_path=video_local_path,
                data_id=data_id,
                item_id=item.get("id"),
                scenario_family=scenario_family,
                qa_id=next_id("final"),
                qa_type="final_state",
                question_zh="这段视频最终完成后的结果是什么？",
                answer=final_answer,
                source_label=final_label,
                start=final_start,
                end=final_end,
                notes=final_notes,
            )
        )

    execute_index = 0
    verify_index = 0
    wrong_index = 0
    recovery_index = 0

    for event in action_events:
        description = event.action_description or event.action_requirement
        if not description:
            continue

        if event.action_type == "execute action":
            execute_index += 1
            note_parts = ["对应 processed_code_v1/v2 中的 action。"]
            if event.step_id and event.step_id != str(execute_index):
                note_parts.append(f"原始 action_step_id={event.step_id}，这里按时间顺序重排为第 {execute_index} 步。")
            candidates.append(
                make_candidate(
                    source_file=source_file,
                    video_name=video_name,
                    video_local_path=video_local_path,
                    data_id=data_id,
                    item_id=item.get("id"),
                    scenario_family=scenario_family,
                    qa_id=next_id("action"),
                    qa_type="action_step",
                    question_zh=f"第 {execute_index} 个关键动作是什么？",
                    answer=description,
                    source_label="action_description",
                    start=event.start,
                    end=event.end,
                    notes=" ".join(note_parts),
                )
            )
        elif event.action_type == "verification action":
            verify_index += 1
            note_parts = ["对应 processed_code_v1/v2 中的 verification_action。"]
            if event.step_id and event.step_id != str(verify_index):
                note_parts.append(f"原始 action_step_id={event.step_id}，这里按验证动作顺序重排为第 {verify_index} 步。")
            candidates.append(
                make_candidate(
                    source_file=source_file,
                    video_name=video_name,
                    video_local_path=video_local_path,
                    data_id=data_id,
                    item_id=item.get("id"),
                    scenario_family=scenario_family,
                    qa_id=next_id("verify"),
                    qa_type="verification_action",
                    question_zh=f"为了确认任务是否成功，第 {verify_index} 个验证动作是什么？",
                    answer=description,
                    source_label="action_description",
                    start=event.start,
                    end=event.end,
                    notes=" ".join(note_parts),
                )
            )
        elif event.action_type == "wrong action":
            wrong_index += 1
            candidates.append(
                make_candidate(
                    source_file=source_file,
                    video_name=video_name,
                    video_local_path=video_local_path,
                    data_id=data_id,
                    item_id=item.get("id"),
                    scenario_family=scenario_family,
                    qa_id=next_id("wrong"),
                    qa_type="error_action",
                    question_zh=f"这段视频中的错误动作 {wrong_index} 是什么？",
                    answer=description,
                    source_label="action_description",
                    start=event.start,
                    end=event.end,
                    notes="对应 processed_code_v1/v2 中的 recovery 错误链条。",
                )
            )
        elif event.action_type == "recovery action":
            recovery_index += 1
            candidates.append(
                make_candidate(
                    source_file=source_file,
                    video_name=video_name,
                    video_local_path=video_local_path,
                    data_id=data_id,
                    item_id=item.get("id"),
                    scenario_family=scenario_family,
                    qa_id=next_id("fix"),
                    qa_type="correction_action",
                    question_zh=f"针对前面的错误，修正动作 {recovery_index} 是什么？",
                    answer=description,
                    source_label="action_description",
                    start=event.start,
                    end=event.end,
                    notes="对应 processed_code_v1/v2 中的 recovery 修正链条。",
                )
            )

    ui_state_index = 0
    physical_state_index = 0
    for segment in state_segments:
        if segment.label == "ui_state":
            ui_state_index += 1
            question = f"界面成功信号 {ui_state_index} 是什么？"
        else:
            physical_state_index += 1
            question = f"物理世界成功信号 {physical_state_index} 是什么？"

        candidates.append(
            make_candidate(
                source_file=source_file,
                video_name=video_name,
                video_local_path=video_local_path,
                data_id=data_id,
                item_id=item.get("id"),
                scenario_family=scenario_family,
                qa_id=next_id("state"),
                qa_type="verification_state",
                question_zh=question,
                answer=segment.text,
                source_label=segment.label,
                start=segment.start,
                end=segment.end,
                notes="对应 processed_code_v1/v2 中的 verification_state。",
            )
        )

    task_family_counts = Counter(candidate["task_family"] for candidate in candidates)
    qa_type_counts = Counter(candidate["qa_type"] for candidate in candidates)

    return {
        "source_file": source_file.name,
        "item_id": item.get("id"),
        "data_id": data_id,
        "video_name": video_name,
        "video_local_path": video_local_path,
        "scenario_family": scenario_family,
        "main_task": overall_requirement,
        "main_verification": overall_verification,
        "task_family_counts": dict(task_family_counts),
        "qa_type_counts": dict(qa_type_counts),
        "qa_candidates": candidates,
    }


def attach_auto_mcq_options(video_results: List[Dict[str, Any]], seed: int = 7) -> None:
    rng = random.Random(seed)
    answer_pool: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))

    for video in video_results:
        scenario = video["scenario_family"]
        for candidate in video["qa_candidates"]:
            if candidate["qa_type"] not in MCQ_ELIGIBLE_TYPES:
                continue
            answer = candidate["answer"]
            qa_type = candidate["qa_type"]
            if answer and answer not in answer_pool[qa_type]["all"]:
                answer_pool[qa_type]["all"].append(answer)
            if answer and answer not in answer_pool[qa_type][scenario]:
                answer_pool[qa_type][scenario].append(answer)

    for video in video_results:
        scenario = video["scenario_family"]
        for candidate in video["qa_candidates"]:
            qa_type = candidate["qa_type"]
            if qa_type not in MCQ_ELIGIBLE_TYPES:
                continue

            same_scenario = [
                answer
                for answer in answer_pool[qa_type].get(scenario, [])
                if answer != candidate["answer"]
            ]
            cross_scenario = [
                answer
                for answer in answer_pool[qa_type]["all"]
                if answer != candidate["answer"] and answer not in same_scenario
            ]

            distractors: List[str] = []
            if len(same_scenario) >= 3:
                distractors.extend(rng.sample(same_scenario, 3))
            else:
                distractors.extend(rng.sample(same_scenario, min(len(same_scenario), 3)))
                remaining = 3 - len(distractors)
                if len(cross_scenario) < remaining:
                    continue
                distractors.extend(rng.sample(cross_scenario, remaining))

            options = [candidate["answer"], *distractors]
            rng.shuffle(options)
            correct_option = "ABCD"[options.index(candidate["answer"])]

            candidate["supported_formats"].append("mcq")
            candidate["mcq_options_auto"] = options
            candidate["mcq_answer"] = candidate["answer"]
            candidate["mcq_correct_option"] = correct_option
            candidate["mcq_needs_manual_review"] = True


def flatten_openqa(video_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    flat: List[Dict[str, Any]] = []
    for video in video_results:
        for candidate in video["qa_candidates"]:
            flat.append(
                {
                    "id": candidate["qa_id"],
                    "source_file": candidate["source_file"],
                    "video_name": candidate["video_name"],
                    "video_local_path": candidate["video_local_path"],
                    "data_id": candidate["data_id"],
                    "item_id": candidate["item_id"],
                    "scenario_family": candidate["scenario_family"],
                    "task_family": candidate["task_family"],
                    "qa_type": candidate["qa_type"],
                    "question_zh": candidate["question_zh"],
                    "answer": candidate["answer"],
                    "answer_explanation_zh": candidate["answer_explanation_zh"],
                    "source_label": candidate["source_label"],
                    "source_span": candidate["source_span"],
                    "notes": candidate.get("notes"),
                }
            )
    return flat


def flatten_mcq(video_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    flat: List[Dict[str, Any]] = []
    for video in video_results:
        for candidate in video["qa_candidates"]:
            options = candidate.get("mcq_options_auto")
            if not options:
                continue
            flat.append(
                {
                    "id": candidate["qa_id"],
                    "source_file": candidate["source_file"],
                    "video_name": candidate["video_name"],
                    "video_local_path": candidate["video_local_path"],
                    "data_id": candidate["data_id"],
                    "item_id": candidate["item_id"],
                    "scenario_family": candidate["scenario_family"],
                    "task_family": candidate["task_family"],
                    "qa_type": candidate["qa_type"],
                    "question_type": "single_choice",
                    "question_zh": candidate["question_zh"],
                    "option_a": options[0],
                    "option_b": options[1],
                    "option_c": options[2],
                    "option_d": options[3],
                    "correct_option": candidate["mcq_correct_option"],
                    "correct_answer": candidate["mcq_answer"],
                    "answer_explanation_zh": candidate["answer_explanation_zh"],
                    "needs_review": True,
                    "source_label": candidate["source_label"],
                    "source_span": candidate["source_span"],
                    "notes": candidate.get("notes"),
                }
            )
    return flat


def summarize_counts(video_results: List[Dict[str, Any]]) -> Tuple[Counter, Counter]:
    task_family_counter = Counter()
    qa_type_counter = Counter()
    for video in video_results:
        for candidate in video["qa_candidates"]:
            task_family_counter[candidate["task_family"]] += 1
            qa_type_counter[candidate["qa_type"]] += 1
    return task_family_counter, qa_type_counter


def write_summary(
    output_path: Path,
    *,
    source_name: str,
    video_results: List[Dict[str, Any]],
    openqa_items: List[Dict[str, Any]],
    mcq_items: List[Dict[str, Any]],
) -> None:
    task_family_counter, qa_type_counter = summarize_counts(video_results)
    lines: List[str] = []
    lines.append(f"# {source_name} QA Summary")
    lines.append("")
    lines.append(f"- Videos parsed: `{len(video_results)}`")
    lines.append(f"- QA candidates: `{sum(len(video['qa_candidates']) for video in video_results)}`")
    lines.append(f"- OpenQA items: `{len(openqa_items)}`")
    lines.append(f"- MCQ items: `{len(mcq_items)}`")
    lines.append("- Note: all MCQ options are auto-generated weak distractors and need manual review.")
    lines.append("")
    lines.append("## Task Family Counts")
    lines.append("")
    for task_family, count in task_family_counter.most_common():
        lines.append(f"- `{task_family}`: `{count}`")
    lines.append("")
    lines.append("## QA Type Counts")
    lines.append("")
    for qa_type, count in qa_type_counter.most_common():
        lines.append(f"- `{qa_type}`: `{count}`")
    lines.append("")

    for video in video_results:
        lines.append(f"## {video['video_name']}")
        lines.append("")
        lines.append(f"- Source file: `{video['source_file']}`")
        if video["main_task"]:
            lines.append(f"- Main task: `{video['main_task']}`")
        lines.append(f"- Scenario family: `{video['scenario_family']}`")
        lines.append(f"- Candidate count: `{len(video['qa_candidates'])}`")
        lines.append("")
        for candidate in video["qa_candidates"]:
            lines.append(f"### {candidate['qa_id']} | `{candidate['task_family']}` / `{candidate['qa_type']}`")
            lines.append("")
            lines.append(f"- Question: {candidate['question_zh']}")
            lines.append(f"- Answer: `{candidate['answer']}`")
            lines.append(f"- Source label: `{candidate['source_label']}`")
            lines.append(
                f"- Span: `{candidate['source_span']['start']} -> {candidate['source_span']['end']}`"
            )
            lines.append(f"- Formats: `{', '.join(candidate['supported_formats'])}`")
            if candidate.get("mcq_options_auto"):
                lines.append("- Auto MCQ options:")
                for idx, option in enumerate(candidate["mcq_options_auto"], start=1):
                    lines.append(f"  {idx}. `{option}`")
            if candidate.get("notes"):
                lines.append(f"- Notes: {candidate['notes']}")
            lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def build_payload(source_name: str, video_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    task_family_counter, qa_type_counter = summarize_counts(video_results)
    return {
        "schema_version": "switch-current-qa-v1",
        "source_name": source_name,
        "num_videos": len(video_results),
        "num_qa_candidates": sum(len(video["qa_candidates"]) for video in video_results),
        "task_family_counts": dict(task_family_counter),
        "qa_type_counts": dict(qa_type_counter),
        "videos": video_results,
    }


def write_outputs(base_path: Path, source_name: str, video_results: List[Dict[str, Any]]) -> None:
    openqa_items = flatten_openqa(video_results)
    mcq_items = flatten_mcq(video_results)
    payload = build_payload(source_name, video_results)

    qa_path = base_path.with_suffix(".qa_candidates.json")
    openqa_path = base_path.with_suffix(".openqa.json")
    mcq_path = base_path.with_suffix(".mcq.json")
    summary_path = base_path.with_suffix(".qa_summary.md")

    qa_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    openqa_path.write_text(
        json.dumps(
            {
                "schema_version": "switch-current-openqa-v1",
                "source_name": source_name,
                "num_items": len(openqa_items),
                "data": openqa_items,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    mcq_path.write_text(
        json.dumps(
            {
                "schema_version": "switch-current-mcq-v1",
                "source_name": source_name,
                "num_items": len(mcq_items),
                "data": mcq_items,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    write_summary(
        summary_path,
        source_name=source_name,
        video_results=video_results,
        openqa_items=openqa_items,
        mcq_items=mcq_items,
    )


def iter_source_files(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]

    source_files: List[Path] = []
    for path in sorted(input_path.glob("*.json")):
        if any(path.name.endswith(suffix) for suffix in SKIP_OUTPUT_SUFFIXES):
            continue
        source_files.append(path)
    return source_files


def process_file(source_file: Path) -> List[Dict[str, Any]]:
    with source_file.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    video_index = build_video_index(source_file.parent)
    video_results = [
        build_candidates_for_item(item, source_file=source_file, video_index=video_index)
        for item in data
        if item.get("annotations")
    ]
    attach_auto_mcq_options(video_results)
    return video_results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract OpenQA and MCQ annotation sheets from the current SWITCH timeline exports."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("annotations") / "0421" / "switch",
        help="Path to one annotation JSON file or to a directory containing multiple JSON exports.",
    )
    args = parser.parse_args()

    source_files = iter_source_files(args.input)
    if not source_files:
        raise SystemExit(f"No source annotation JSON files found under: {args.input}")

    combined_results: List[Dict[str, Any]] = []
    for source_file in source_files:
        video_results = process_file(source_file)
        write_outputs(source_file.with_suffix(""), source_file.name, video_results)
        combined_results.extend(video_results)
        print(
            f"Wrote outputs for {source_file.name}: "
            f"{len(video_results)} videos, "
            f"{sum(len(video['qa_candidates']) for video in video_results)} QA candidates."
        )

    if args.input.is_dir() and combined_results:
        combined_base = args.input / f"{args.input.name}_all"
        write_outputs(combined_base, f"{args.input.name}_all", combined_results)
        print(
            f"Wrote combined outputs: {len(combined_results)} videos, "
            f"{sum(len(video['qa_candidates']) for video in combined_results)} QA candidates."
        )


if __name__ == "__main__":
    main()
