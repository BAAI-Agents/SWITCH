#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import cv2
    import numpy as np
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise SystemExit(
        "opencv-python-headless is required for SWITCH v2 multiform asset generation. "
        "Install it with: python -m pip install opencv-python-headless"
    ) from exc


TASK_FAMILIES = [
    "vqa_task",
    "vqa_state",
    "action",
    "final_state",
    "verification_action",
    "verification_state",
    "recovery",
]

CAPABILITY_LEVEL = {
    "vqa_task": "L2",
    "vqa_state": "L2",
    "action": "L2",
    "final_state": "L3",
    "verification_action": "L4",
    "verification_state": "L4",
    "recovery": "L4",
}

FORM_SPECS = {
    "vqa_task": ["video2txt"],
    "vqa_state": ["img2txt"],
    "action": ["video2txt", "img2txt", "img2video", "video2video"],
    "final_state": ["video2txt", "img2txt", "img2img", "video2img"],
    "verification_action": ["video2txt", "img2txt", "img2video", "video2video"],
    "verification_state": ["video2txt", "img2txt", "img2img", "video2img"],
    "recovery": ["video2txt"],
}

TEXT_FORMS = {"video2txt", "img2txt"}
VISUAL_CHOICE_FORMS = {"img2video", "video2video", "img2img", "video2img"}

ORDINAL_WORDS = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
    "eleventh": 11,
    "twelfth": 12,
    "thirteenth": 13,
}

CARDINAL_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
}

STRICT_FUTURE_POST_RESULT_MIN_OFFSET_FRAMES = 12
STRICT_FUTURE_POST_RESULT_OFFSET_RATIO = 0.4
STRICT_FUTURE_IMG2IMG_MIN_GAP_FRAMES = 15
STRICT_FUTURE_VIDEO2IMG_MIN_GAP_FRAMES = 30
STRICT_FUTURE_QUERY_CLIP_SECONDS = 1.8
STRICT_FUTURE_PERCEPTUAL_MIN_DISTANCE = 0.03
STRICT_FUTURE_NEGATIVE_DISTANCE_MARGIN = 0.06
STRICT_FUTURE_PRE_RESULT_SEARCH_SECONDS = 1.0
STRICT_FUTURE_SAME_VIEW_WINDOW_SECONDS = 0.8
STRICT_FUTURE_FRAME_SAMPLE_STEP = 2
STRICT_FUTURE_SAME_VIDEO_MIN_PRE_RESULT_GAP_FRAMES = 15
STRICT_FUTURE_SAME_VIDEO_MIN_GT_DISTANCE = 0.13
STRICT_FUTURE_MIN_NEGATIVE_DISTANCE_TO_QUERY = 0.05
STRICT_FUTURE_MIN_NEGATIVE_PAIRWISE_DISTANCE = 0.10
STRICT_FUTURE_SAME_SEMANTIC_MIN_GT_DISTANCE = 0.10
STRICT_FUTURE_CROSS_SCENE_MIN_DISTANCE_TO_QUERY = 0.08
STRICT_FUTURE_CROSS_SCENE_MIN_DISTANCE_TO_GT = 0.12

TASK_TEMPLATES = [
    ("task_goal_full_video", "What overall task is being carried out from start to finish in this video?"),
    ("task_main_goal", "What main goal is the operator trying to complete in the full video?"),
    ("task_end_to_end", "Which task best summarizes what the person is trying to finish in this video?"),
    ("task_summary", "What end-to-end task is pursued throughout the video?"),
]

TASK_SUFFIXES = [
    ("task_suffix_short_phrase", "Answer with a short task phrase."),
    ("task_suffix_brief_phrase", "Reply with the task in a short phrase."),
    ("task_suffix_goal_phrase", "Respond with a brief goal phrase."),
]

VQA_STATE_TEMPLATES = [
    ("vqa_state_current", "Which statement best describes the current state shown in this frame?"),
    ("vqa_state_visible", "What visible state is shown in this image right now?"),
    ("vqa_state_interface", "Which description best matches the current interface or environment state in this frame?"),
    ("vqa_state_observed", "What state can be directly observed from this frame?"),
]

VQA_STATE_SUFFIXES = [
    ("vqa_state_suffix_short", "Answer with a short phrase."),
    ("vqa_state_suffix_brief", "Reply with a brief state phrase."),
    ("vqa_state_suffix_concise", "Respond with a concise description."),
]

ACTION_TEMPLATES = {
    "first_required_action": [
        ("action_first_needed", 'What is the first key action needed to start "{main_task}"?'),
        ("action_first_begin", 'To begin "{main_task}", what does the operator do first?'),
    ],
    "next_action_after_previous": [
        (
            "action_after_previous",
            'After "{previous_action}", what does the operator do next in "{main_task}"?',
        ),
        (
            "action_next_move",
            'Once "{previous_action}" is done, what action keeps "{main_task}" moving forward?',
        ),
    ],
    "last_required_action": [
        (
            "action_last_required",
            'What is the last required action before "{main_task}" can succeed?',
        ),
        (
            "action_final_move",
            'Right before "{main_task}" can succeed, what final action does the operator take?',
        ),
    ],
    "progress_action": [
        (
            "action_progress_forward",
            'At this point in the workflow, what action moves "{main_task}" forward?',
        ),
        (
            "action_progress_next",
            'From the current point in "{main_task}", what action should happen next?',
        ),
    ],
    "selected_floor_number": [
        (
            "action_floor_number",
            'Which floor number does the operator choose to continue "{main_task}"?',
        ),
        (
            "action_floor_selection",
            'What floor number is selected at this point in "{main_task}"?',
        ),
    ],
    "ticket_price_yuan": [
        (
            "action_ticket_price",
            'What fare amount is selected on the ticket machine for "{main_task}"?',
        ),
        (
            "action_ticket_yuan",
            'Which ticket price is chosen on screen during "{main_task}"?',
        ),
    ],
}

ACTION_SUFFIXES = {
    "keyword": [
        ("action_keyword_short", "Answer with a short action phrase."),
        ("action_keyword_brief", "Reply with a brief action phrase."),
        ("action_keyword_concise", "Respond with a concise action phrase."),
    ],
    "numeric": [
        ("action_numeric_number", "Answer with the number only."),
        ("action_numeric_digits", "Reply using only the number."),
        ("action_numeric_plain", "Respond with the number."),
    ],
}

FINAL_STATE_TEMPLATES = {
    "visible_outcome": [
        ("final_change_took_effect", 'What change shows that "{main_task}" has taken effect?'),
        ("final_visible_outcome", 'What outcome is visible once "{main_task}" finishes?'),
        ("final_result_after_success", 'Which result appears after "{main_task}" succeeds?'),
        ("final_confirm_result", 'After the task is completed, what visible result confirms "{main_task}"?'),
    ],
    "arrival_floor_number": [
        ("final_floor_reached", 'Which floor is reached once "{main_task}" succeeds?'),
        ("final_floor_outcome", 'What floor number appears as the final result of "{main_task}"?'),
    ],
}

FINAL_STATE_SUFFIXES = {
    "keyword": [
        ("final_keyword_short", "Answer with a short phrase."),
        ("final_keyword_brief", "Reply with a brief phrase."),
        ("final_keyword_result", "Respond with the result in a short phrase."),
    ],
    "numeric": [
        ("final_numeric_number", "Answer with the number only."),
        ("final_numeric_digits", "Reply using only the number."),
        ("final_numeric_plain", "Respond with the number."),
    ],
}

VERIFICATION_ACTION_TEMPLATES = [
    ("verify_action_person_check", 'How would a person verify that "{main_task}" succeeded?'),
    ("verify_action_operator_check", 'What should the operator check to confirm success in "{main_task}"?'),
    ("verify_action_confirm_worked", 'Which check would confirm that "{main_task}" worked?'),
    ("verify_action_observation_step", 'What observation step verifies success for "{main_task}"?'),
]

VERIFICATION_ACTION_SUFFIXES = [
    ("verify_action_suffix_short", "Answer with a short phrase."),
    ("verify_action_suffix_brief", "Reply with a brief phrase."),
    ("verify_action_suffix_concise", "Respond with a concise phrase."),
]

VERIFICATION_STATE_TEMPLATES = {
    "visible_success_signal": [
        ("verify_state_signal", 'What visible signal confirms success here for "{main_task}"?'),
        ("verify_state_seen", 'What should be seen if "{main_task}" worked?'),
        ("verify_state_observable", 'Which observable result would confirm success in "{main_task}"?'),
        ("verify_state_visible_cue", 'What visible cue shows that "{main_task}" has succeeded?'),
    ],
    "observed_floor_number": [
        ("verify_state_floor_signal", 'Which floor number appears as the success signal for "{main_task}"?'),
        ("verify_state_floor_seen", 'What floor number should be seen when "{main_task}" succeeds?'),
    ],
}

VERIFICATION_STATE_SUFFIXES = {
    "keyword": [
        ("verify_state_keyword_short", "Answer with a short phrase."),
        ("verify_state_keyword_brief", "Reply with a brief phrase."),
        ("verify_state_keyword_concise", "Respond with a concise phrase."),
    ],
    "numeric": [
        ("verify_state_numeric_number", "Answer with the number only."),
        ("verify_state_numeric_digits", "Reply using only the number."),
        ("verify_state_numeric_plain", "Respond with the number."),
    ],
}

RECOVERY_TEMPLATES = {
    "wrong_action_first": [
        ("recovery_wrong_first", "What went wrong first in this interaction?"),
        ("recovery_wrong_action", "Which incorrect action happens first in the recovery sequence?"),
    ],
    "correction_action": [
        ("recovery_fix_action", "How is the mistake corrected in this interaction?"),
        ("recovery_fix_step", "What action is used to recover from the earlier mistake?"),
    ],
    "full_recovery_chain": [
        ("recovery_full_chain", "What is the full wrong-action / signal / fix chain in this video?"),
        ("recovery_error_path", "Summarize the error, intermediate signal, and recovery path in this interaction."),
    ],
}

RECOVERY_SUFFIXES = {
    "wrong_action_first": [
        ("recovery_wrong_structured", "Answer with the field wrong_action."),
        ("recovery_wrong_object", "Reply with a short object containing wrong_action."),
    ],
    "correction_action": [
        ("recovery_fix_structured", "Answer with the field fix_step."),
        ("recovery_fix_object", "Reply with a short object containing fix_step."),
    ],
    "full_recovery_chain": [
        (
            "recovery_chain_structured",
            "Answer with the fields wrong_action, post_wrong_signal, fix_steps, and post_fix_signal.",
        ),
        (
            "recovery_chain_object",
            "Reply with a short object containing wrong_action, post_wrong_signal, fix_steps, and post_fix_signal.",
        ),
    ],
}


@dataclass
class QARecord:
    qa_id: str
    source_file: str
    data_id: str
    video_name: str
    source_video_relpath: str
    source_video_abspath: Path
    scenario_family: str
    task_family: str
    qa_type: str
    main_task: str
    main_verification: str
    answer: str
    source_span: Dict[str, Optional[int]]
    question_zh: str
    answer_explanation_zh: str
    notes: Optional[str]
    slice_tags: List[str]
    fps: float
    total_frames: int
    duration: float
    action_index: Optional[int] = None
    action_total: Optional[int] = None
    previous_action: Optional[str] = None
    last_action_end: Optional[int] = None
    semantic_group: str = "generic"
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VisualFrameOption:
    option_key: str
    source_type: str
    origin_qa_id: str
    source_video_abspath: Path
    frame_index: int
    answer: str
    scenario_family: str
    semantic_group: str


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def remove_tree(path: Path) -> None:
    if not path.exists():
        return

    def _onerror(func: Any, target: str, _exc_info: Any) -> None:
        try:
            os.chmod(target, 0o777)
        except OSError:
            pass
        try:
            func(target)
        except OSError:
            pass

    last_error: Optional[Exception] = None
    for _ in range(5):
        try:
            shutil.rmtree(path, onerror=_onerror)
            if not path.exists():
                return
        except OSError as exc:
            last_error = exc
        time.sleep(0.5)
    if path.exists():
        raise RuntimeError(f"Unable to remove existing output directory: {path}") from last_error


def stable_int(key: str) -> int:
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(digest, 16)


def choose_variant(options: List[Tuple[str, str]], key: str) -> Tuple[str, str]:
    return options[stable_int(key) % len(options)]


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def sentence_case(text: str) -> str:
    text = normalize_spaces(text)
    if not text:
        return ""
    if text[0].isalpha():
        text = text[0].upper() + text[1:]
    return text


def ensure_punctuated(text: str) -> str:
    text = sentence_case(text)
    if text and text[-1] not in ".?!":
        text += "."
    return text


def lower_first(text: str) -> str:
    if not text:
        return text
    if text[0].isalpha():
        return text[0].lower() + text[1:]
    return text


def format_option_block(options: List[str]) -> str:
    letters = ["A", "B", "C", "D"]
    return "".join(
        f"{letters[index]}. {ensure_punctuated(option)}\n" for index, option in enumerate(options)
    )


def make_hardlink_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


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
    direct_candidates = sorted(
        path
        for path in annotation_root.glob("*.qa_candidates.json")
        if not path.name.endswith(".qa_candidates.qa_candidates.json")
    )
    if len(direct_candidates) == 1:
        return direct_candidates[0]
    raise FileNotFoundError(f"Unable to resolve combined qa_candidates file under {annotation_root}")


def build_video_meta(annotation_root: Path) -> Dict[str, Dict[str, Any]]:
    video_meta: Dict[str, Dict[str, Any]] = {}
    for json_path in iter_raw_annotation_jsons(annotation_root):
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
            media_meta = None
            for value in (item.get("data", {}).get("meta") or {}).values():
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


def build_profiles(
    annotation_root: Path,
    candidate_payload: Dict[str, Any],
    video_meta: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    profiles: Dict[str, Dict[str, Any]] = {}
    for video in candidate_payload["videos"]:
        data_id = video["data_id"]
        qas = video["qa_candidates"]
        action_ends = [
            qa["source_span"]["end"]
            for qa in qas
            if qa["task_family"] == "action" and qa["source_span"]["end"] is not None
        ]
        final_starts = [
            qa["source_span"]["start"]
            for qa in qas
            if qa["task_family"] == "final_state" and qa["source_span"]["start"] is not None
        ]
        meta = video_meta.get(data_id, {})
        duration = meta.get("duration") or 0.0
        total_frames = int(meta.get("total_frames") or 0)
        fps = float(meta.get("fps") or 30.0)
        last_action_end = max(action_ends) if action_ends else None
        first_final_start = min(final_starts) if final_starts else None
        delayed_frames = None
        if last_action_end is not None and first_final_start is not None:
            delayed_frames = max(0, int(first_final_start) - int(last_action_end))
        slice_tags: List[str] = []
        if sum(1 for qa in qas if qa["task_family"] == "action") >= 4:
            slice_tags.append("multi_step")
        if sum(1 for qa in qas if qa["task_family"] in {"verification_action", "verification_state"}) >= 3:
            slice_tags.append("verification_heavy")
        if any(qa["task_family"] == "recovery" for qa in qas):
            slice_tags.append("recovery")
        if duration >= 45:
            slice_tags.append("long_horizon")
        if delayed_frames is not None and delayed_frames >= 90:
            slice_tags.append("delayed_effect")
        if not slice_tags:
            slice_tags.append("clean_success")
        relpath = video.get("video_local_path") or video["video_name"]
        abs_path = annotation_root / relpath
        profiles[data_id] = {
            "data_id": data_id,
            "video_name": video["video_name"],
            "source_video_relpath": relpath.replace("\\", "/"),
            "source_video_abspath": abs_path,
            "source_video_exists": abs_path.exists(),
            "scenario_family": video["scenario_family"],
            "main_task": sentence_case(video["main_task"]),
            "main_verification": sentence_case(video.get("main_verification") or ""),
            "duration": duration,
            "total_frames": total_frames,
            "fps": fps,
            "last_action_end": last_action_end,
            "slice_tags": slice_tags,
        }
    return profiles


def source_span_key(record: QARecord) -> Tuple[int, int, str]:
    start = int(record.source_span.get("start") or 0)
    end = int(record.source_span.get("end") or start)
    return (start, end, record.qa_id)


def span_start(span: Optional[Dict[str, Optional[int]]]) -> int:
    if not span:
        return 0
    return int(span.get("start") or 0)


def span_end(span: Optional[Dict[str, Optional[int]]]) -> int:
    if not span:
        return 0
    return int(span.get("end") or span_start(span))


def record_span_start(record: QARecord) -> int:
    return span_start(record.source_span)


def record_span_end(record: QARecord) -> int:
    return span_end(record.source_span)


def find_recovery_state_signal(
    video_records: List[QARecord],
    after_frame: int,
    before_frame: int,
) -> Optional[QARecord]:
    for record in sorted(video_records, key=source_span_key):
        if record.task_family not in {"verification_state", "final_state"}:
            continue
        start = record_span_start(record)
        if after_frame < start < before_frame:
            return record
    return None


def recovery_chain_audit_note(
    last_wrong_end: int,
    first_fix_start: int,
    post_wrong_signal_record: Optional[QARecord],
) -> Optional[str]:
    gap = first_fix_start - last_wrong_end
    if post_wrong_signal_record is None:
        return "No annotated post-wrong state was found between the wrong action and the fix step."
    if gap >= 180:
        signal_delay = record_span_start(post_wrong_signal_record) - last_wrong_end
        if signal_delay <= max(45, int(gap * 0.25)):
            return (
                "The annotated post-wrong signal appears early in a long wrong-to-fix interval; "
                "inspect the clip for a possible unlabeled intermediate system response."
            )
    return None


def build_records(annotation_root: Path) -> Tuple[List[QARecord], Dict[str, Dict[str, Any]], Dict[str, List[QARecord]]]:
    candidate_payload = load_json(resolve_candidate_payload_path(annotation_root))
    video_meta = build_video_meta(annotation_root)
    profiles = build_profiles(annotation_root, candidate_payload, video_meta)
    records: List[QARecord] = []
    by_video: Dict[str, List[QARecord]] = defaultdict(list)

    for video in candidate_payload["videos"]:
        profile = profiles[video["data_id"]]
        if not profile.get("source_video_exists", False):
            continue
        for candidate in video["qa_candidates"]:
            start = candidate["source_span"].get("start")
            end = candidate["source_span"].get("end")
            start_int = int(start) if start is not None else None
            end_int = int(end) if end is not None else None
            record = QARecord(
                qa_id=candidate["qa_id"],
                source_file=candidate["source_file"],
                data_id=video["data_id"],
                video_name=video["video_name"],
                source_video_relpath=profile["source_video_relpath"],
                source_video_abspath=profile["source_video_abspath"],
                scenario_family=video["scenario_family"],
                task_family=candidate["task_family"],
                qa_type=candidate["qa_type"],
                main_task=profile["main_task"],
                main_verification=profile["main_verification"],
                answer=sentence_case(candidate["answer"]),
                source_span={"start": start_int, "end": end_int},
                question_zh=candidate["question_zh"],
                answer_explanation_zh=candidate["answer_explanation_zh"],
                notes=candidate.get("notes"),
                slice_tags=list(profile["slice_tags"]),
                fps=profile["fps"],
                total_frames=profile["total_frames"],
                duration=profile["duration"],
                last_action_end=profile["last_action_end"],
            )
            records.append(record)
            by_video[record.data_id].append(record)

    for data_id, video_records in by_video.items():
        action_records = sorted(
            [record for record in video_records if record.task_family == "action"],
            key=source_span_key,
        )
        for index, record in enumerate(action_records, start=1):
            record.action_index = index
            record.action_total = len(action_records)
            if index > 1:
                record.previous_action = action_records[index - 2].answer
        last_action_hint = action_records[-1].answer if action_records else None
        action_history_hint = None
        if action_records:
            history_slice = action_records[-2:] if len(action_records) >= 2 else action_records
            action_history_hint = " -> ".join(record.answer for record in history_slice)
        for record in video_records:
            if record.task_family == "final_state":
                if last_action_hint:
                    record.extra["last_action_hint"] = last_action_hint
                if action_history_hint:
                    record.extra["action_history_hint"] = action_history_hint

        recovery_records = sorted(
            [record for record in video_records if record.task_family == "recovery"],
            key=source_span_key,
        )
        wrong_actions = [record for record in recovery_records if record.qa_type == "error_action"]
        fix_actions = [record for record in recovery_records if record.qa_type == "correction_action"]
        if wrong_actions and fix_actions:
            last_wrong_end = max(record_span_end(record) for record in wrong_actions)
            first_fix_start = min(record_span_start(record) for record in fix_actions)
            last_fix_end = max(int(record.source_span.get("end") or 0) for record in fix_actions)
            post_wrong_signal_record = find_recovery_state_signal(video_records, last_wrong_end, first_fix_start)
            post_fix_signal_record = None
            for record in sorted(video_records, key=source_span_key):
                if record.task_family not in {"verification_state", "final_state"}:
                    continue
                start = int(record.source_span.get("start") or 0)
                if start >= last_fix_end:
                    post_fix_signal_record = record
                    break
            audit_note = recovery_chain_audit_note(last_wrong_end, first_fix_start, post_wrong_signal_record)
            chain_record = QARecord(
                qa_id=f"{data_id}_recovery_chain",
                source_file=wrong_actions[0].source_file,
                data_id=data_id,
                video_name=wrong_actions[0].video_name,
                source_video_relpath=wrong_actions[0].source_video_relpath,
                source_video_abspath=wrong_actions[0].source_video_abspath,
                scenario_family=wrong_actions[0].scenario_family,
                task_family="recovery",
                qa_type="recovery_chain",
                main_task=wrong_actions[0].main_task,
                main_verification=wrong_actions[0].main_verification,
                answer="Recovery chain",
                source_span={
                    "start": min(int(record.source_span.get("start") or 0) for record in recovery_records),
                    "end": max(int(record.source_span.get("end") or 0) for record in recovery_records),
                },
                question_zh="这段视频里先出现了什么错误动作，之后又如何修正？",
                answer_explanation_zh="该答案同时覆盖错误动作识别和修复动作，贴合 SWITCH v2 的 recovery 闭环评测。",
                notes="Synthetic recovery_chain item derived from the same recovery trajectory.",
                slice_tags=list(wrong_actions[0].slice_tags),
                fps=wrong_actions[0].fps,
                total_frames=wrong_actions[0].total_frames,
                duration=wrong_actions[0].duration,
                last_action_end=wrong_actions[0].last_action_end,
                extra={
                    "wrong_action": wrong_actions[0].answer,
                    "wrong_action_origin_qa_id": wrong_actions[0].qa_id,
                    "wrong_action_span": dict(wrong_actions[0].source_span),
                    "post_wrong_signal": (
                        post_wrong_signal_record.answer if post_wrong_signal_record is not None else None
                    ),
                    "post_wrong_signal_origin_qa_id": (
                        post_wrong_signal_record.qa_id if post_wrong_signal_record is not None else None
                    ),
                    "post_wrong_signal_span": (
                        dict(post_wrong_signal_record.source_span)
                        if post_wrong_signal_record is not None
                        else None
                    ),
                    "fix_steps": [record.answer for record in fix_actions],
                    "fix_action_origin_qa_ids": [record.qa_id for record in fix_actions],
                    "fix_action_spans": [dict(record.source_span) for record in fix_actions],
                    "post_fix_signal": (
                        post_fix_signal_record.answer if post_fix_signal_record is not None else None
                    ),
                    "post_fix_signal_origin_qa_id": (
                        post_fix_signal_record.qa_id if post_fix_signal_record is not None else None
                    ),
                    "post_fix_signal_span": (
                        dict(post_fix_signal_record.source_span)
                        if post_fix_signal_record is not None
                        else None
                    ),
                    "origin_qa_ids": [record.qa_id for record in recovery_records],
                    "recovery_chain_audit_note": audit_note,
                },
            )
            records.append(chain_record)
            video_records.append(chain_record)

        vqa_state_seen: set[Tuple[int, int, str]] = set()
        state_source_records = sorted(
            [
                record
                for record in video_records
                if record.task_family in {"verification_state", "final_state"}
            ],
            key=source_span_key,
        )
        for source_record in state_source_records:
            vqa_state_key = (
                record_span_start(source_record),
                record_span_end(source_record),
                normalize_spaces(source_record.answer).lower(),
            )
            if vqa_state_key in vqa_state_seen:
                continue
            vqa_state_seen.add(vqa_state_key)
            vqa_state_record = QARecord(
                qa_id=f"{source_record.qa_id}_vqa_state",
                source_file=source_record.source_file,
                data_id=source_record.data_id,
                video_name=source_record.video_name,
                source_video_relpath=source_record.source_video_relpath,
                source_video_abspath=source_record.source_video_abspath,
                scenario_family=source_record.scenario_family,
                task_family="vqa_state",
                qa_type="state_description",
                main_task=source_record.main_task,
                main_verification=source_record.main_verification,
                answer=source_record.answer,
                source_span=dict(source_record.source_span),
                question_zh="这张图像中当前显示的界面或环境状态是什么？",
                answer_explanation_zh="该样本来自已标注的状态片段，用于评估模型对当前可见状态的识别能力。",
                notes=f"Synthetic vqa_state item derived from {source_record.qa_id}.",
                slice_tags=list(source_record.slice_tags),
                fps=source_record.fps,
                total_frames=source_record.total_frames,
                duration=source_record.duration,
                action_index=source_record.action_index,
                action_total=source_record.action_total,
                previous_action=source_record.previous_action,
                last_action_end=source_record.last_action_end,
                extra={
                    "source_task_family": source_record.task_family,
                    "source_origin_qa_id": source_record.qa_id,
                },
            )
            records.append(vqa_state_record)
            video_records.append(vqa_state_record)

    for record in records:
        record.semantic_group = derive_semantic_group(record)

    records.sort(key=lambda record: (record.data_id, source_span_key(record)))
    grouped_records: Dict[str, List[QARecord]] = defaultdict(list)
    for record in records:
        grouped_records[record.task_family].append(record)
    return records, profiles, grouped_records


def derive_semantic_group(record: QARecord) -> str:
    answer = normalize_spaces(record.answer).lower()
    task_family = record.task_family

    if task_family == "vqa_task":
        return f"task::{record.scenario_family}"

    if task_family == "action":
        if "elevator up button" in answer or "elevator down button" in answer:
            return "action::direction_button"
        if "button for the" in answer and "floor" in answer:
            return "action::floor_button"
        if "enter the elevator" in answer:
            return "action::enter_elevator"
        if "yuan" in answer:
            return "action::ticket_price"
        if "ticket counter" in answer or "paper reservation form" in answer:
            return "action::collect_ticket_form"
        if "doctor" in answer or "appointment" in answer or "department" in answer:
            return "action::appointment_flow"
        if "screen" in answer or "page" in answer or "machine" in answer:
            return "action::screen_navigation"
        return f"action::{record.scenario_family}"

    if task_family in {"final_state", "verification_state", "vqa_state"}:
        if "arrives at the" in answer and "floor" in answer:
            return "state::elevator_arrival"
        if "door" in answer:
            return "state::door_state"
        if "ticket" in answer or "reservation form" in answer:
            return "state::ticket_or_form"
        if "home screen" in answer or "homepage" in answer:
            return "state::home_screen"
        if "page" in answer or "interface" in answer or "screen" in answer:
            return "state::page_or_interface"
        if "prompt" in answer or "information" in answer:
            return "state::info_signal"
        return f"state::{record.scenario_family}"

    if task_family == "verification_action":
        if "surrounding environment and floor information" in answer:
            return "verify_action::floor_environment"
        if "ticket counter" in answer:
            return "verify_action::ticket_counter"
        if "machine screen" in answer or "homepage information" in answer:
            return "verify_action::machine_screen"
        return f"verify_action::{record.scenario_family}"

    if task_family == "recovery":
        if record.qa_type == "error_action":
            return "recovery::wrong_action"
        if record.qa_type == "correction_action":
            return "recovery::fix_action"
        return "recovery::chain"

    return f"generic::{record.scenario_family}"


def derive_device_family(record: QARecord) -> str:
    if record.scenario_family in {"medical_kiosk", "subway_ticket"}:
        return "public_terminal"
    if record.scenario_family == "elevator":
        return "elevator"
    return record.scenario_family


def extract_floor_number(text: str) -> Optional[int]:
    lowered = normalize_spaces(text).lower()
    if "basement level" in lowered:
        return None
    match = re.search(r"\b(\d+)(?:st|nd|rd|th)? floor\b", lowered)
    if match:
        return int(match.group(1))
    for word, value in ORDINAL_WORDS.items():
        if f"{word} floor" in lowered:
            return value
    return None


def extract_price_yuan(text: str) -> Optional[int]:
    lowered = normalize_spaces(text).lower()
    match = re.search(r"\b(\d+)\s*yuan\b", lowered)
    if match:
        return int(match.group(1))
    for word, value in CARDINAL_WORDS.items():
        if f"{word} yuan" in lowered:
            return value
    return None


def ordinal_aliases(value: int) -> List[str]:
    inverse = {number: word for word, number in ORDINAL_WORDS.items()}
    word = inverse.get(value)
    suffix = "th"
    if value % 10 == 1 and value % 100 != 11:
        suffix = "st"
    elif value % 10 == 2 and value % 100 != 12:
        suffix = "nd"
    elif value % 10 == 3 and value % 100 != 13:
        suffix = "rd"
    aliases = [str(value), f"{value}{suffix}"]
    if word:
        aliases.extend([word, f"{word} floor", f"{value} floor"])
    return aliases


def price_aliases(value: int) -> List[str]:
    inverse = {number: word for word, number in CARDINAL_WORDS.items()}
    word = inverse.get(value)
    aliases = [str(value), f"{value} yuan"]
    if word:
        aliases.extend([word, f"{word} yuan"])
    return aliases


def unique_strings(values: Iterable[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for value in values:
        candidate = normalize_spaces(str(value))
        if not candidate:
            continue
        key = candidate.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(candidate)
    return ordered


def alias_list_for_answer(answer: str) -> List[str]:
    base = sentence_case(answer)
    variants = [base, normalize_spaces(base).lower(), re.sub(r"[.?!]+$", "", base)]
    return unique_strings(variants)


def detect_numeric_info(record: QARecord) -> Optional[Dict[str, Any]]:
    if record.task_family not in {"action", "final_state", "verification_state"}:
        return None

    if record.task_family == "action":
        floor = extract_floor_number(record.answer)
        if floor is not None and "button for the" in record.answer.lower():
            return {
                "kind": "floor_number",
                "value": str(floor),
                "semantic_anchor": "selected_floor_number",
                "numeric_slots": {"selected_floor": floor, "unit": "floor"},
                "alias_list": ordinal_aliases(floor),
            }
        price = extract_price_yuan(record.answer)
        if price is not None:
            return {
                "kind": "ticket_price_yuan",
                "value": str(price),
                "semantic_anchor": "ticket_price_yuan",
                "numeric_slots": {"ticket_price_yuan": price, "unit": "yuan"},
                "alias_list": price_aliases(price),
            }

    if record.task_family == "final_state":
        floor = extract_floor_number(record.answer)
        if floor is not None and "arrives at the" in record.answer.lower():
            return {
                "kind": "arrival_floor_number",
                "value": str(floor),
                "semantic_anchor": "arrival_floor_number",
                "numeric_slots": {"arrival_floor": floor, "unit": "floor"},
                "alias_list": ordinal_aliases(floor),
            }

    if record.task_family in {"verification_state", "vqa_state"}:
        floor = extract_floor_number(record.answer)
        if floor is not None and "arrives at the" in record.answer.lower():
            return {
                "kind": "observed_floor_number",
                "value": str(floor),
                "semantic_anchor": "observed_floor_number" if record.task_family == "verification_state" else "current_floor_number",
                "numeric_slots": {"observed_floor": floor, "unit": "floor"},
                "alias_list": ordinal_aliases(floor),
            }

    return None


def action_anchor(record: QARecord) -> str:
    numeric_info = detect_numeric_info(record)
    if numeric_info is not None:
        return str(numeric_info["semantic_anchor"])
    if record.action_index == 1:
        return "first_required_action"
    if record.action_total and record.action_index == record.action_total:
        return "last_required_action"
    if record.previous_action:
        return "next_action_after_previous"
    return "progress_action"


def render_template(template: str, record: QARecord) -> str:
    values = {
        "main_task": record.main_task,
        "previous_action": record.previous_action or "the previous action",
    }
    return template.format(**values)


def wrap_prompt_for_form(form: str, stem: str, output_modality: str) -> str:
    if form == "img2txt":
        return f"Based on this frame, {lower_first(stem)}"
    if form == "img2video":
        return f"Based on this frame, {lower_first(stem)} Choose the best matching video option."
    if form == "img2img":
        return f"Based on this frame, {lower_first(stem)} Choose the best matching image option."
    if form == "video2img":
        return f"{stem} Choose the image that best matches the expected result."
    if form == "video2video":
        return f"{stem} Choose the best matching video option."
    return stem


def choose_openqa_prompt(record: QARecord, form: str) -> Dict[str, Any]:
    numeric_info = detect_numeric_info(record)

    if record.task_family == "vqa_task":
        template_id, template_text = choose_variant(TASK_TEMPLATES, f"{record.qa_id}:{form}:task")
        suffix_id, suffix_text = choose_variant(TASK_SUFFIXES, f"{record.qa_id}:{form}:task:suffix")
        stem = wrap_prompt_for_form(form, template_text, "text")
        return {
            "query": f"{stem} {suffix_text}",
            "prompt_variant": f"{template_id}__{suffix_id}",
            "semantic_anchor": "task_summary",
            "rewrite_type": "structured_short_answer",
            "output_schema": {"type": "structured_short_answer", "value": "task_summary"},
            "canonical_answer": record.answer,
            "alias_list": alias_list_for_answer(record.answer),
            "numeric_slots": None,
            "GT": record.answer,
            "answer_type": "structured_short_answer",
        }

    if record.task_family == "vqa_state":
        template_id, template_text = choose_variant(VQA_STATE_TEMPLATES, f"{record.qa_id}:{form}:vqa_state")
        suffix_id, suffix_text = choose_variant(VQA_STATE_SUFFIXES, f"{record.qa_id}:{form}:vqa_state:suffix")
        stem = wrap_prompt_for_form(form, template_text, "text")
        if numeric_info is not None:
            return {
                "query": f"{stem} {suffix_text}",
                "prompt_variant": f"{template_id}__{suffix_id}",
                "semantic_anchor": str(numeric_info["semantic_anchor"]),
                "rewrite_type": "numeric",
                "output_schema": {"type": "numeric", "slots": numeric_info["numeric_slots"]},
                "canonical_answer": numeric_info["value"],
                "alias_list": numeric_info["alias_list"],
                "numeric_slots": numeric_info["numeric_slots"],
                "GT": numeric_info["value"],
                "answer_type": "numeric",
            }
        return {
            "query": f"{stem} {suffix_text}",
            "prompt_variant": f"{template_id}__{suffix_id}",
            "semantic_anchor": "current_visible_state",
            "rewrite_type": "keyword",
            "output_schema": {"type": "keyword", "value": "current_state_phrase"},
            "canonical_answer": record.answer,
            "alias_list": alias_list_for_answer(record.answer),
            "numeric_slots": None,
            "GT": record.answer,
            "answer_type": "keyword",
        }

    if record.task_family == "action":
        anchor = action_anchor(record)
        rewrite_type = "numeric" if numeric_info is not None else "keyword"
        template_id, template_text = choose_variant(
            ACTION_TEMPLATES[anchor],
            f"{record.qa_id}:{form}:{anchor}:{rewrite_type}",
        )
        suffix_id, suffix_text = choose_variant(
            ACTION_SUFFIXES[rewrite_type],
            f"{record.qa_id}:{form}:{anchor}:{rewrite_type}:suffix",
        )
        stem = wrap_prompt_for_form(form, render_template(template_text, record), "text")
        if numeric_info is not None:
            return {
                "query": f"{stem} {suffix_text}",
                "prompt_variant": f"{template_id}__{suffix_id}",
                "semantic_anchor": anchor,
                "rewrite_type": "numeric",
                "output_schema": {"type": "numeric", "slots": numeric_info["numeric_slots"]},
                "canonical_answer": numeric_info["value"],
                "alias_list": numeric_info["alias_list"],
                "numeric_slots": numeric_info["numeric_slots"],
                "GT": numeric_info["value"],
                "answer_type": "numeric",
            }
        return {
            "query": f"{stem} {suffix_text}",
            "prompt_variant": f"{template_id}__{suffix_id}",
            "semantic_anchor": anchor,
            "rewrite_type": "keyword",
            "output_schema": {"type": "keyword", "value": "short_action_phrase"},
            "canonical_answer": record.answer,
            "alias_list": alias_list_for_answer(record.answer),
            "numeric_slots": None,
            "GT": record.answer,
            "answer_type": "keyword",
        }

    if record.task_family == "final_state":
        anchor = str(numeric_info["semantic_anchor"]) if numeric_info is not None else "visible_outcome"
        rewrite_type = "numeric" if numeric_info is not None else "keyword"
        template_id, template_text = choose_variant(
            FINAL_STATE_TEMPLATES[anchor],
            f"{record.qa_id}:{form}:{anchor}:{rewrite_type}",
        )
        suffix_id, suffix_text = choose_variant(
            FINAL_STATE_SUFFIXES[rewrite_type],
            f"{record.qa_id}:{form}:{anchor}:{rewrite_type}:suffix",
        )
        stem = wrap_prompt_for_form(form, render_template(template_text, record), "text")
        if numeric_info is not None:
            return {
                "query": f"{stem} {suffix_text}",
                "prompt_variant": f"{template_id}__{suffix_id}",
                "semantic_anchor": anchor,
                "rewrite_type": "numeric",
                "output_schema": {"type": "numeric", "slots": numeric_info["numeric_slots"]},
                "canonical_answer": numeric_info["value"],
                "alias_list": numeric_info["alias_list"],
                "numeric_slots": numeric_info["numeric_slots"],
                "GT": numeric_info["value"],
                "answer_type": "numeric",
            }
        return {
            "query": f"{stem} {suffix_text}",
            "prompt_variant": f"{template_id}__{suffix_id}",
            "semantic_anchor": anchor,
            "rewrite_type": "keyword",
            "output_schema": {"type": "keyword", "value": "outcome_phrase"},
            "canonical_answer": record.answer,
            "alias_list": alias_list_for_answer(record.answer),
            "numeric_slots": None,
            "GT": record.answer,
            "answer_type": "keyword",
        }

    if record.task_family == "verification_action":
        template_id, template_text = choose_variant(
            VERIFICATION_ACTION_TEMPLATES,
            f"{record.qa_id}:{form}:verify_action",
        )
        suffix_id, suffix_text = choose_variant(
            VERIFICATION_ACTION_SUFFIXES,
            f"{record.qa_id}:{form}:verify_action:suffix",
        )
        stem = wrap_prompt_for_form(form, render_template(template_text, record), "text")
        return {
            "query": f"{stem} {suffix_text}",
            "prompt_variant": f"{template_id}__{suffix_id}",
            "semantic_anchor": "verification_check",
            "rewrite_type": "keyword",
            "output_schema": {"type": "keyword", "value": "verification_action_phrase"},
            "canonical_answer": record.answer,
            "alias_list": alias_list_for_answer(record.answer),
            "numeric_slots": None,
            "GT": record.answer,
            "answer_type": "keyword",
        }

    if record.task_family == "verification_state":
        anchor = str(numeric_info["semantic_anchor"]) if numeric_info is not None else "visible_success_signal"
        rewrite_type = "numeric" if numeric_info is not None else "keyword"
        template_id, template_text = choose_variant(
            VERIFICATION_STATE_TEMPLATES[anchor],
            f"{record.qa_id}:{form}:{anchor}:{rewrite_type}",
        )
        suffix_id, suffix_text = choose_variant(
            VERIFICATION_STATE_SUFFIXES[rewrite_type],
            f"{record.qa_id}:{form}:{anchor}:{rewrite_type}:suffix",
        )
        stem = wrap_prompt_for_form(form, render_template(template_text, record), "text")
        if numeric_info is not None:
            return {
                "query": f"{stem} {suffix_text}",
                "prompt_variant": f"{template_id}__{suffix_id}",
                "semantic_anchor": anchor,
                "rewrite_type": "numeric",
                "output_schema": {"type": "numeric", "slots": numeric_info["numeric_slots"]},
                "canonical_answer": numeric_info["value"],
                "alias_list": numeric_info["alias_list"],
                "numeric_slots": numeric_info["numeric_slots"],
                "GT": numeric_info["value"],
                "answer_type": "numeric",
            }
        return {
            "query": f"{stem} {suffix_text}",
            "prompt_variant": f"{template_id}__{suffix_id}",
            "semantic_anchor": anchor,
            "rewrite_type": "keyword",
            "output_schema": {"type": "keyword", "value": "verification_signal_phrase"},
            "canonical_answer": record.answer,
            "alias_list": alias_list_for_answer(record.answer),
            "numeric_slots": None,
            "GT": record.answer,
            "answer_type": "keyword",
        }

    if record.task_family == "recovery":
        if record.qa_type == "error_action":
            anchor = "wrong_action_first"
            gt = {"wrong_action": record.answer}
            canonical_answer = record.answer
        elif record.qa_type == "correction_action":
            anchor = "correction_action"
            gt = {"fix_step": record.answer}
            canonical_answer = record.answer
        else:
            anchor = "full_recovery_chain"
            gt = {
                "wrong_action": record.extra.get("wrong_action"),
                "post_wrong_signal": record.extra.get("post_wrong_signal"),
                "fix_steps": record.extra.get("fix_steps") or [],
                "post_fix_signal": record.extra.get("post_fix_signal"),
            }
            canonical_answer = (
                f"Wrong action: {gt['wrong_action']}; "
                f"Post-wrong signal: {gt['post_wrong_signal'] or 'N/A'}; "
                f"Fix: {' -> '.join(gt['fix_steps'])}; "
                f"Post-fix signal: {gt['post_fix_signal'] or 'N/A'}"
            )
        template_id, template_text = choose_variant(
            RECOVERY_TEMPLATES[anchor],
            f"{record.qa_id}:{form}:{anchor}",
        )
        suffix_id, suffix_text = choose_variant(
            RECOVERY_SUFFIXES[anchor],
            f"{record.qa_id}:{form}:{anchor}:suffix",
        )
        stem = wrap_prompt_for_form(form, template_text, "text")
        return {
            "query": f"{stem} {suffix_text}",
            "prompt_variant": f"{template_id}__{suffix_id}",
            "semantic_anchor": anchor,
            "rewrite_type": "structured_short_answer",
            "output_schema": {
                "type": "object",
                "fields": (
                    ["wrong_action", "post_wrong_signal", "fix_steps", "post_fix_signal"]
                    if record.qa_type == "recovery_chain"
                    else (["wrong_action"] if record.qa_type == "error_action" else ["fix_step"])
                ),
            },
            "canonical_answer": canonical_answer,
            "alias_list": None,
            "numeric_slots": None,
            "GT": gt,
            "answer_type": "structured_short_answer",
        }

    raise ValueError(f"Unsupported task family: {record.task_family}")


def choice_text_value(record: QARecord) -> Tuple[str, Optional[str], Optional[Dict[str, Any]], Optional[List[str]]]:
    numeric_info = detect_numeric_info(record)
    if numeric_info is None:
        return record.answer, None, None, None
    if numeric_info["kind"] == "ticket_price_yuan":
        return record.answer, None, None, None
    return (
        str(numeric_info["value"]),
        str(numeric_info["semantic_anchor"]),
        dict(numeric_info["numeric_slots"]),
        list(numeric_info["alias_list"]),
    )


def candidate_score(target: QARecord, other: QARecord) -> int:
    if target.qa_id == other.qa_id:
        return -10**9
    if target.task_family != other.task_family:
        return -10**9
    score = 0
    if target.qa_type == other.qa_type:
        score += 50
    if target.scenario_family == other.scenario_family:
        score += 40
    if target.semantic_group == other.semantic_group:
        score += 25
    if target.data_id != other.data_id:
        score += 5
    return score


def select_distractors(target: QARecord, pool: List[QARecord], count: int = 3) -> List[QARecord]:
    scored: List[Tuple[int, int, QARecord]] = []
    for candidate in pool:
        score = candidate_score(target, candidate)
        if score <= -10**8:
            continue
        tie_breaker = stable_int(f"{target.qa_id}:{candidate.qa_id}")
        scored.append((score, tie_breaker, candidate))
    scored.sort(key=lambda item: (-item[0], item[1]))
    selected: List[QARecord] = []
    used_values = {normalize_spaces(target.answer).lower()}
    for _, _, candidate in scored:
        value = normalize_spaces(candidate.answer).lower()
        if value in used_values:
            continue
        selected.append(candidate)
        used_values.add(value)
        if len(selected) == count:
            break
    if len(selected) < count:
        fallback = sorted(
            [candidate for candidate in pool if candidate.qa_id != target.qa_id],
            key=lambda candidate: stable_int(f"fallback:{target.qa_id}:{candidate.qa_id}"),
        )
        for candidate in fallback:
            value = normalize_spaces(candidate.answer).lower()
            if value in used_values:
                continue
            selected.append(candidate)
            used_values.add(value)
            if len(selected) == count:
                break
    if len(selected) < count:
        raise RuntimeError(f"Unable to find enough distractors for {target.qa_id}")
    return selected


def mcq_prompt(record: QARecord, form: str, option_modality: str) -> Tuple[str, str, str, str, Any, Optional[Dict[str, Any]], Optional[List[str]]]:
    prompt = choose_openqa_prompt(record, form if form in TEXT_FORMS else "video2txt")
    base_query = prompt["query"]
    if form in TEXT_FORMS:
        return (
            base_query,
            prompt["prompt_variant"],
            prompt["semantic_anchor"],
            prompt["rewrite_type"],
            {"type": "choice", "labels": ["A", "B", "C", "D"], "choice_modality": "text"},
            prompt["numeric_slots"],
            prompt["alias_list"],
        )
    choice_prompt = base_query.rsplit(" ", 1)[0]
    if option_modality == "image":
        query = wrap_prompt_for_form(form, re.sub(r"\s+(Answer|Reply|Respond).*$", "", base_query), "image")
    else:
        query = wrap_prompt_for_form(form, re.sub(r"\s+(Answer|Reply|Respond).*$", "", base_query), "video")
    return (
        query,
        prompt["prompt_variant"],
        prompt["semantic_anchor"],
        "choice",
        {"type": "choice", "labels": ["A", "B", "C", "D"], "choice_modality": option_modality},
        prompt["numeric_slots"],
        prompt["alias_list"],
    )


class AssetWriter:
    def __init__(self) -> None:
        self.frame_cache: Dict[Tuple[str, int], Any] = {}
        self.signature_cache: Dict[Tuple[str, int], Any] = {}

    def read_frame(self, video_path: Path, frame_index: int) -> Any:
        key = (str(video_path), frame_index)
        if key in self.frame_cache:
            return self.frame_cache[key]
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open video: {video_path}")
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok and frame_index > 0:
            capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_index - 1))
            ok, frame = capture.read()
        capture.release()
        if not ok:
            raise RuntimeError(f"Unable to read frame {frame_index} from {video_path}")
        self.frame_cache[key] = frame
        return frame

    def extract_frame(self, video_path: Path, frame_index: int, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            return
        frame = self.read_frame(video_path, frame_index)
        if not cv2.imwrite(str(output_path), frame):
            raise RuntimeError(f"Unable to write image: {output_path}")

    def frame_signature(self, video_path: Path, frame_index: int) -> Any:
        key = (str(video_path), frame_index)
        if key in self.signature_cache:
            return self.signature_cache[key]
        frame = self.read_frame(video_path, frame_index)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
        signature = resized.astype("float32") / 255.0
        self.signature_cache[key] = signature
        return signature

    def frame_distance(
        self,
        video_path_a: Path,
        frame_index_a: int,
        video_path_b: Path,
        frame_index_b: int,
    ) -> float:
        signature_a = self.frame_signature(video_path_a, frame_index_a)
        signature_b = self.frame_signature(video_path_b, frame_index_b)
        return float(np.mean(np.abs(signature_a - signature_b)))

    def extract_clip(
        self,
        video_path: Path,
        start_frame: int,
        end_frame: int,
        fps: float,
        output_path: Path,
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            return
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open video: {video_path}")
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            capture.release()
            raise RuntimeError(f"Unable to write clip: {output_path}")
        current = start_frame
        while current <= end_frame:
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(frame)
            current += 1
        writer.release()
        capture.release()
        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError(f"Empty clip generated: {output_path}")


def clamp_frame(frame: Optional[int], total_frames: int) -> int:
    if total_frames <= 0:
        return 0
    if frame is None:
        return 0
    return max(0, min(int(frame), total_frames - 1))


def midpoint_frame(record: QARecord) -> int:
    start = clamp_frame(record.source_span.get("start"), record.total_frames)
    end = clamp_frame(record.source_span.get("end"), record.total_frames)
    return max(start, (start + end) // 2)


def pre_key_frame(record: QARecord, offset_fraction: float = 0.4) -> int:
    start = clamp_frame(record.source_span.get("start"), record.total_frames)
    offset = max(1, int(record.fps * offset_fraction))
    return clamp_frame(start - offset, record.total_frames)


def final_query_frame(record: QARecord) -> int:
    if record.last_action_end is not None:
        candidate = max(record.last_action_end, int(record.source_span.get("start") or 0) - int(record.fps * 0.4))
        return clamp_frame(candidate, record.total_frames)
    return pre_key_frame(record)


def context_clip_range(record: QARecord, seconds: float = 2.0) -> Tuple[int, int]:
    start = clamp_frame(record.source_span.get("start"), record.total_frames)
    context = max(int(record.fps * 1.5), int(record.fps * seconds))
    min_length = max(12, int(record.fps * 1.0))
    clip_start = max(0, start - context)
    clip_end = max(clip_start + min_length - 1, start - 1)
    clip_end = clamp_frame(clip_end, record.total_frames)
    if clip_end <= clip_start:
        clip_end = clamp_frame(clip_start + min_length - 1, record.total_frames)
    return clip_start, clip_end


def action_clip_range(record: QARecord) -> Tuple[int, int]:
    start = clamp_frame(record.source_span.get("start"), record.total_frames)
    end = clamp_frame(record.source_span.get("end"), record.total_frames)
    pre = max(4, int(record.fps * 0.25))
    post = max(6, int(record.fps * 0.35))
    clip_start = max(0, start - pre)
    clip_end = clamp_frame(max(end, start) + post, record.total_frames)
    if clip_end <= clip_start:
        clip_end = clamp_frame(clip_start + max(10, int(record.fps * 0.8)), record.total_frames)
    return clip_start, clip_end


def final_state_start_frame(record: QARecord) -> int:
    return clamp_frame(record.source_span.get("start"), record.total_frames)


def final_state_end_frame(record: QARecord) -> int:
    end = record.source_span.get("end")
    if end is None:
        end = record.source_span.get("start")
    return clamp_frame(end, record.total_frames)


def stable_post_result_frame(record: QARecord) -> int:
    start = final_state_start_frame(record)
    end = final_state_end_frame(record)
    offset = max(
        STRICT_FUTURE_POST_RESULT_MIN_OFFSET_FRAMES,
        int(record.fps * STRICT_FUTURE_POST_RESULT_OFFSET_RATIO),
    )
    candidate = max(end, start + offset)
    return clamp_frame(candidate, record.total_frames)


def strict_future_query_frame(record: QARecord) -> Tuple[Optional[int], Optional[str]]:
    if record.last_action_end is None:
        return None, "missing_last_action_end"
    query_frame = clamp_frame(record.last_action_end, record.total_frames)
    if query_frame >= final_state_start_frame(record):
        return None, "query_not_before_final_state"
    return query_frame, None


def strict_future_clip_range(record: QARecord) -> Tuple[Optional[Tuple[int, int]], Optional[str]]:
    query_frame, reason = strict_future_query_frame(record)
    if reason is not None or query_frame is None:
        return None, reason
    context = max(int(record.fps * 1.5), int(record.fps * STRICT_FUTURE_QUERY_CLIP_SECONDS))
    clip_start = max(0, query_frame - context + 1)
    clip_end = query_frame
    if clip_end < clip_start:
        return None, "invalid_query_clip"
    if clip_end >= final_state_start_frame(record):
        return None, "query_clip_reaches_final_state"
    return (clip_start, clip_end), None


def state_reference_frame(record: QARecord) -> int:
    if record.task_family == "final_state":
        return stable_post_result_frame(record)
    return midpoint_frame(record)


def canonical_visual_choice_query(record: QARecord, form: str) -> Tuple[str, str, str, Optional[Dict[str, Any]], Optional[List[str]]]:
    numeric_info = detect_numeric_info(record)
    semantic_anchor = str(numeric_info["semantic_anchor"]) if numeric_info is not None else "visible_outcome"
    numeric_slots = numeric_info["numeric_slots"] if numeric_info is not None else None
    alias_list = numeric_info["alias_list"] if numeric_info is not None else alias_list_for_answer(record.answer)
    if form == "img2img":
        action_hint = normalize_spaces(
            str(
                record.extra.get("last_action_hint")
                or record.extra.get("action_history_hint")
                or "the final action"
            )
        ).rstrip(".?!")
        return (
            (
                f'This frame is captured before the result appears. After the operator completes "{action_hint}", '
                f'which image best shows the final state of "{record.main_task}"?'
            ),
            "strict_future_img2img__last_action_hint",
            semantic_anchor,
            numeric_slots,
            alias_list,
        )
    return (
        (
            f'Watch the clip up to just before the outcome is revealed. Which image best represents '
            f'the future result of "{record.main_task}"?'
        ),
        "strict_future_video2img__future_result_prediction",
        semantic_anchor,
        numeric_slots,
        alias_list,
    )


def make_visual_frame_option(
    option_key: str,
    source_type: str,
    origin_qa_id: str,
    source_video_abspath: Path,
    frame_index: int,
    answer: str,
    scenario_family: str,
    semantic_group: str,
) -> VisualFrameOption:
    return VisualFrameOption(
        option_key=option_key,
        source_type=source_type,
        origin_qa_id=origin_qa_id,
        source_video_abspath=source_video_abspath,
        frame_index=frame_index,
        answer=answer,
        scenario_family=scenario_family,
        semantic_group=semantic_group,
    )


def sample_frame_indices(start_frame: int, end_frame: int, step: int = STRICT_FUTURE_FRAME_SAMPLE_STEP) -> List[int]:
    if end_frame < start_frame:
        return []
    indices = list(range(start_frame, end_frame + 1, max(1, step)))
    if indices and indices[-1] != end_frame:
        indices.append(end_frame)
    if not indices:
        indices.append(start_frame)
    return sorted(set(indices))


def is_diverse_against_options(
    asset_writer: AssetWriter,
    candidate_video_path: Path,
    candidate_frame_index: int,
    existing_options: Iterable[VisualFrameOption],
    min_distance: float = STRICT_FUTURE_MIN_NEGATIVE_PAIRWISE_DISTANCE,
) -> bool:
    for option in existing_options:
        if option.source_type == "gold_stable_post_result":
            continue
        distance = asset_writer.frame_distance(
            candidate_video_path,
            candidate_frame_index,
            option.source_video_abspath,
            option.frame_index,
        )
        if distance < min_distance:
            return False
    return True


def same_video_pre_result_option(
    record: QARecord,
    asset_writer: AssetWriter,
    query_frame: int,
    gt_frame: int,
    used_options: Iterable[VisualFrameOption] = (),
) -> Optional[VisualFrameOption]:
    start = final_state_start_frame(record)
    search_start = max(query_frame + 1, start - max(6, int(record.fps * STRICT_FUTURE_PRE_RESULT_SEARCH_SECONDS)))
    search_end = start - max(
        STRICT_FUTURE_SAME_VIDEO_MIN_PRE_RESULT_GAP_FRAMES,
        int(record.fps * 0.5),
    )
    candidates = sample_frame_indices(search_start, search_end)
    if not candidates:
        return None
    eligible: List[Tuple[float, int]] = []
    for frame_index in candidates:
        distance_to_query = asset_writer.frame_distance(
            record.source_video_abspath,
            query_frame,
            record.source_video_abspath,
            frame_index,
        )
        if distance_to_query < STRICT_FUTURE_MIN_NEGATIVE_DISTANCE_TO_QUERY:
            continue
        distance_to_gt = asset_writer.frame_distance(
            record.source_video_abspath,
            frame_index,
            record.source_video_abspath,
            gt_frame,
        )
        if distance_to_gt < STRICT_FUTURE_SAME_VIDEO_MIN_GT_DISTANCE:
            continue
        if not is_diverse_against_options(
            asset_writer,
            record.source_video_abspath,
            frame_index,
            used_options,
        ):
            continue
        eligible.append((distance_to_gt, frame_index))
    if not eligible:
        return None
    _, best_frame = min(eligible, key=lambda item: (item[0], item[1]))
    return make_visual_frame_option(
        option_key=f"frame::{record.data_id}:{best_frame}:same_video_pre_result",
        source_type="same_video_pre_result_hard_negative",
        origin_qa_id=f"frame::{record.data_id}:{best_frame}",
        source_video_abspath=record.source_video_abspath,
        frame_index=best_frame,
        answer="Same-video frame before result reveal",
        scenario_family=record.scenario_family,
        semantic_group="state::same_video_pre_result",
    )


def same_viewpoint_wrong_state_option(
    record: QARecord,
    asset_writer: AssetWriter,
    query_frame: int,
    used_options: Iterable[VisualFrameOption],
) -> Optional[VisualFrameOption]:
    used = {option.option_key for option in used_options}
    start = final_state_start_frame(record)
    window = max(4, int(record.fps * STRICT_FUTURE_SAME_VIEW_WINDOW_SECONDS))
    candidate_frames = sample_frame_indices(
        max(0, query_frame - window),
        min(start - 1, query_frame + max(2, window // 2)),
    )
    candidate_frames = [frame for frame in candidate_frames if frame != query_frame]
    candidate_frames.sort(
        key=lambda frame_index: asset_writer.frame_distance(
            record.source_video_abspath,
            query_frame,
            record.source_video_abspath,
            frame_index,
        )
    )
    for frame_index in candidate_frames:
        distance_to_query = asset_writer.frame_distance(
            record.source_video_abspath,
            query_frame,
            record.source_video_abspath,
            frame_index,
        )
        if distance_to_query < STRICT_FUTURE_MIN_NEGATIVE_DISTANCE_TO_QUERY:
            continue
        option_key = f"frame::{record.data_id}:{frame_index}:same_viewpoint_wrong_state"
        if option_key in used:
            continue
        if not is_diverse_against_options(
            asset_writer,
            record.source_video_abspath,
            frame_index,
            used_options,
        ):
            continue
        return make_visual_frame_option(
            option_key=option_key,
            source_type="same_viewpoint_wrong_state_negative",
            origin_qa_id=f"frame::{record.data_id}:{frame_index}",
            source_video_abspath=record.source_video_abspath,
            frame_index=frame_index,
            answer="Same-viewpoint wrong state frame",
            scenario_family=record.scenario_family,
            semantic_group="state::same_viewpoint_wrong_state",
        )
    return None


def same_scenario_same_semantic_option(
    record: QARecord,
    state_pool: List[QARecord],
    asset_writer: AssetWriter,
    query_frame: int,
    gt_frame: int,
    used_options: Iterable[VisualFrameOption],
) -> Optional[VisualFrameOption]:
    used = {option.option_key for option in used_options}
    target_answer = normalize_spaces(record.answer).lower()
    target_numeric = detect_numeric_info(record)
    candidates: List[Tuple[float, int, QARecord, int]] = []
    for other in state_pool:
        if other.scenario_family != record.scenario_family:
            continue
        if other.semantic_group != record.semantic_group:
            continue
        if normalize_spaces(other.answer).lower() == target_answer:
            continue
        if target_numeric is not None:
            other_numeric = detect_numeric_info(other)
            if other_numeric is None:
                continue
            if other_numeric["kind"] != target_numeric["kind"]:
                continue
            if other_numeric["value"] == target_numeric["value"]:
                continue
        frame_index = state_reference_frame(other)
        option_key = f"qa::{other.qa_id}:{frame_index}"
        if option_key in used:
            continue
        if not is_diverse_against_options(
            asset_writer,
            other.source_video_abspath,
            frame_index,
            used_options,
        ):
            continue
        distance_to_query = asset_writer.frame_distance(
            record.source_video_abspath,
            query_frame,
            other.source_video_abspath,
            frame_index,
        )
        if distance_to_query < STRICT_FUTURE_MIN_NEGATIVE_DISTANCE_TO_QUERY:
            continue
        distance_to_gt = asset_writer.frame_distance(
            record.source_video_abspath,
            gt_frame,
            other.source_video_abspath,
            frame_index,
        )
        if distance_to_gt < STRICT_FUTURE_SAME_SEMANTIC_MIN_GT_DISTANCE:
            continue
        priority = 0 if other.task_family == "final_state" else 1
        candidates.append((distance_to_gt + distance_to_query, priority, other, frame_index))
    if not candidates:
        return None
    _, _, chosen_record, frame_index = min(candidates, key=lambda item: (item[0], item[1], stable_int(f"{record.qa_id}:{item[2].qa_id}:{item[3]}")))
    return make_visual_frame_option(
        option_key=f"qa::{chosen_record.qa_id}:{frame_index}",
        source_type="same_scenario_same_semantic_wrong_outcome",
        origin_qa_id=chosen_record.qa_id,
        source_video_abspath=chosen_record.source_video_abspath,
        frame_index=frame_index,
        answer=chosen_record.answer,
        scenario_family=chosen_record.scenario_family,
        semantic_group=chosen_record.semantic_group,
    )


def same_scenario_other_state_option(
    record: QARecord,
    state_pool: List[QARecord],
    asset_writer: AssetWriter,
    query_frame: int,
    gt_frame: int,
    used_options: Iterable[VisualFrameOption],
) -> Optional[VisualFrameOption]:
    used = {option.option_key for option in used_options}
    candidates: List[Tuple[float, int, QARecord, int]] = []
    for other in state_pool:
        if other.scenario_family != record.scenario_family:
            continue
        if other.semantic_group == record.semantic_group:
            continue
        frame_index = state_reference_frame(other)
        option_key = f"qa::{other.qa_id}:{frame_index}"
        if option_key in used:
            continue
        if not is_diverse_against_options(
            asset_writer,
            other.source_video_abspath,
            frame_index,
            used_options,
        ):
            continue
        distance_to_query = asset_writer.frame_distance(
            record.source_video_abspath,
            query_frame,
            other.source_video_abspath,
            frame_index,
        )
        if distance_to_query < STRICT_FUTURE_MIN_NEGATIVE_DISTANCE_TO_QUERY:
            continue
        distance_to_gt = asset_writer.frame_distance(
            record.source_video_abspath,
            gt_frame,
            other.source_video_abspath,
            frame_index,
        )
        priority = 0 if other.task_family == "verification_state" else 1
        candidates.append((distance_to_query + distance_to_gt, priority, other, frame_index))
    if not candidates:
        return None
    _, _, chosen_record, frame_index = min(
        candidates,
        key=lambda item: (item[0], item[1], stable_int(f"{record.qa_id}:{item[2].qa_id}:{item[3]}")),
    )
    return make_visual_frame_option(
        option_key=f"qa::{chosen_record.qa_id}:{frame_index}",
        source_type="same_scenario_other_state_negative",
        origin_qa_id=chosen_record.qa_id,
        source_video_abspath=chosen_record.source_video_abspath,
        frame_index=frame_index,
        answer=chosen_record.answer,
        scenario_family=chosen_record.scenario_family,
        semantic_group=chosen_record.semantic_group,
    )


def cross_scene_same_device_option(
    record: QARecord,
    state_pool: List[QARecord],
    asset_writer: AssetWriter,
    query_frame: int,
    gt_frame: int,
    used_options: Iterable[VisualFrameOption],
) -> Optional[VisualFrameOption]:
    used = {option.option_key for option in used_options}
    target_device_family = derive_device_family(record)
    candidates: List[Tuple[int, float, int, QARecord, int]] = []
    for other in state_pool:
        if other.data_id == record.data_id:
            continue
        if derive_device_family(other) != target_device_family:
            continue
        if other.semantic_group == record.semantic_group:
            continue
        frame_index = state_reference_frame(other)
        option_key = f"qa::{other.qa_id}:{frame_index}"
        if option_key in used:
            continue
        if not is_diverse_against_options(
            asset_writer,
            other.source_video_abspath,
            frame_index,
            used_options,
        ):
            continue
        distance_to_query = asset_writer.frame_distance(
            record.source_video_abspath,
            query_frame,
            other.source_video_abspath,
            frame_index,
        )
        if distance_to_query < STRICT_FUTURE_CROSS_SCENE_MIN_DISTANCE_TO_QUERY:
            continue
        distance_to_gt = asset_writer.frame_distance(
            record.source_video_abspath,
            gt_frame,
            other.source_video_abspath,
            frame_index,
        )
        if distance_to_gt < STRICT_FUTURE_CROSS_SCENE_MIN_DISTANCE_TO_GT:
            continue
        cross_scenario_priority = 0 if other.scenario_family != record.scenario_family else 1
        task_priority = 0 if other.task_family == "verification_state" else 1
        candidates.append(
            (
                cross_scenario_priority,
                distance_to_query + distance_to_gt,
                task_priority,
                other,
                frame_index,
            )
        )
    if not candidates:
        return None
    _, _, _, chosen_record, frame_index = min(
        candidates,
        key=lambda item: (
            item[0],
            item[1],
            item[2],
            stable_int(f"{record.qa_id}:{item[3].qa_id}:{item[4]}"),
        ),
    )
    return make_visual_frame_option(
        option_key=f"qa::{chosen_record.qa_id}:{frame_index}",
        source_type="cross_scene_same_device_negative",
        origin_qa_id=chosen_record.qa_id,
        source_video_abspath=chosen_record.source_video_abspath,
        frame_index=frame_index,
        answer=chosen_record.answer,
        scenario_family=chosen_record.scenario_family,
        semantic_group=chosen_record.semantic_group,
    )


def build_strict_future_filter_entry(
    form: str,
    record: QARecord,
    reason: str,
    query_frame: Optional[int] = None,
    gt_frame: Optional[int] = None,
    future_gap_frames: Optional[int] = None,
    perceptual_distance: Optional[float] = None,
) -> Dict[str, Any]:
    entry: Dict[str, Any] = {
        "form": form,
        "origin_qa_id": record.qa_id,
        "data_id": record.data_id,
        "scenario_family": record.scenario_family,
        "main_task": record.main_task,
        "filter_reason": reason,
        "last_action_end": record.last_action_end,
        "final_state_start_frame": final_state_start_frame(record),
        "final_state_end_frame": final_state_end_frame(record),
    }
    if query_frame is not None:
        entry["query_frame"] = query_frame
    if gt_frame is not None:
        entry["gt_frame"] = gt_frame
    if future_gap_frames is not None:
        entry["future_gap_frames"] = future_gap_frames
    if perceptual_distance is not None:
        entry["perceptual_distance"] = round(perceptual_distance, 6)
    return entry


def build_final_state_visual_options(
    record: QARecord,
    state_pool: List[QARecord],
    asset_writer: AssetWriter,
    query_frame: int,
    gt_frame: int,
) -> Tuple[Optional[List[VisualFrameOption]], Optional[str]]:
    correct_option = make_visual_frame_option(
        option_key=f"qa::{record.qa_id}:{gt_frame}",
        source_type="gold_stable_post_result",
        origin_qa_id=record.qa_id,
        source_video_abspath=record.source_video_abspath,
        frame_index=gt_frame,
        answer=record.answer,
        scenario_family=record.scenario_family,
        semantic_group=record.semantic_group,
    )
    selected_options: List[VisualFrameOption] = [correct_option]
    same_viewpoint_option = same_viewpoint_wrong_state_option(
        record,
        asset_writer,
        query_frame,
        selected_options,
    )
    if same_viewpoint_option is None:
        return None, "missing_same_viewpoint_wrong_state_negative"

    selected_options.append(same_viewpoint_option)
    same_semantic_option = same_scenario_same_semantic_option(
        record,
        state_pool,
        asset_writer,
        query_frame,
        gt_frame,
        selected_options,
    )
    cross_scene_option = cross_scene_same_device_option(
        record,
        state_pool,
        asset_writer,
        query_frame,
        gt_frame,
        selected_options,
    )
    if same_semantic_option is not None:
        selected_options.append(same_semantic_option)
    elif cross_scene_option is not None:
        same_semantic_option = cross_scene_option
        cross_scene_option = None
        selected_options.append(same_semantic_option)
    else:
        return None, "missing_same_semantic_or_cross_scene_device_negative"

    optional_negative = cross_scene_option
    if optional_negative is None:
        optional_negative = same_video_pre_result_option(
            record,
            asset_writer,
            query_frame,
            gt_frame,
            selected_options,
        )
    if optional_negative is None:
        optional_negative = same_scenario_other_state_option(
            record,
            state_pool,
            asset_writer,
            query_frame,
            gt_frame,
            selected_options,
        )
    if optional_negative is None:
        return None, "missing_optional_diverse_negative"

    return [
        correct_option,
        optional_negative,
        same_semantic_option,
        same_viewpoint_option,
    ], None


def sorted_records(records: Iterable[QARecord]) -> List[QARecord]:
    return sorted(records, key=lambda record: (record.data_id, source_span_key(record)))


def build_video2txt_form(
    output_root: Path,
    family: str,
    records: List[QARecord],
    source_videos: Dict[str, Path],
) -> Dict[str, int]:
    form = "video2txt"
    form_dir = output_root / family / form
    videos_dir = form_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    unique_videos = sorted({record.video_name for record in records})
    for video_name in unique_videos:
        source = source_videos[video_name]
        make_hardlink_or_copy(source, videos_dir / video_name)

    mcq_rows: List[Dict[str, Any]] = []
    openqa_rows: List[Dict[str, Any]] = []
    pool = sorted_records(records)

    for record in pool:
        open_prompt = choose_openqa_prompt(record, form)
        openqa_row = {
            "id": len(openqa_rows),
            "origin_qa_id": record.qa_id,
            "query_video_path": f"videos/{record.video_name}",
            "query": open_prompt["query"],
            "GT": open_prompt["GT"],
            "answer_type": open_prompt["answer_type"],
            "task_family": record.task_family,
            "qa_type": record.qa_type,
            "capability_level": CAPABILITY_LEVEL[record.task_family],
            "scenario_family": record.scenario_family,
            "slice_tags": record.slice_tags,
            "question_zh": record.question_zh,
            "answer_explanation_zh": record.answer_explanation_zh,
            "prompt_variant": open_prompt["prompt_variant"],
            "rewrite_type": open_prompt["rewrite_type"],
            "semantic_anchor": open_prompt["semantic_anchor"],
            "output_schema": open_prompt["output_schema"],
            "canonical_answer": open_prompt["canonical_answer"],
            "alias_list": open_prompt["alias_list"],
            "numeric_slots": open_prompt["numeric_slots"],
            "source_file": record.source_file,
            "source_span": record.source_span,
            "notes": record.notes,
            "candidate_recovery_future": None if family == "recovery" else None,
        }
        if family == "recovery" and record.qa_type == "recovery_chain":
            openqa_row.update(
                {
                    "wrong_action_origin_qa_id": record.extra.get("wrong_action_origin_qa_id"),
                    "wrong_action_span": record.extra.get("wrong_action_span"),
                    "post_wrong_signal_origin_qa_id": record.extra.get("post_wrong_signal_origin_qa_id"),
                    "post_wrong_signal_span": record.extra.get("post_wrong_signal_span"),
                    "fix_action_origin_qa_ids": record.extra.get("fix_action_origin_qa_ids"),
                    "fix_action_spans": record.extra.get("fix_action_spans"),
                    "post_fix_signal_origin_qa_id": record.extra.get("post_fix_signal_origin_qa_id"),
                    "post_fix_signal_span": record.extra.get("post_fix_signal_span"),
                    "recovery_chain_audit_note": record.extra.get("recovery_chain_audit_note"),
                }
            )
        openqa_rows.append(openqa_row)

        if family == "recovery":
            continue

        distractors = select_distractors(record, pool)
        option_values: List[str] = []
        option_origins: List[str] = []
        option_semantic_anchor = None
        option_numeric_slots = None
        option_alias_list = None
        for option_record in [record, *distractors]:
            value, numeric_anchor, numeric_slots, alias_list = choice_text_value(option_record)
            option_values.append(value)
            option_origins.append(option_record.qa_id)
            if option_record.qa_id == record.qa_id:
                option_semantic_anchor = numeric_anchor
                option_numeric_slots = numeric_slots
                option_alias_list = alias_list
        ordering = list(range(4))
        shift = stable_int(f"{record.qa_id}:{form}:choice_order") % 4
        ordering = ordering[shift:] + ordering[:shift]
        ordered_values = [option_values[index] for index in ordering]
        ordered_origins = [option_origins[index] for index in ordering]
        correct_option = "ABCD"[ordering.index(0)]
        prompt_query, prompt_variant, semantic_anchor, rewrite_type, output_schema, numeric_slots, alias_list = mcq_prompt(
            record,
            form,
            "text",
        )
        mcq_rows.append(
            {
                "id": len(mcq_rows),
                "origin_qa_id": record.qa_id,
                "query_video_path": f"videos/{record.video_name}",
                "query": prompt_query + "\n" + format_option_block(ordered_values),
                "GT": correct_option,
                "correct_answer": ordered_values[ordering.index(0)],
                "task_family": record.task_family,
                "qa_type": record.qa_type,
                "capability_level": CAPABILITY_LEVEL[record.task_family],
                "scenario_family": record.scenario_family,
                "slice_tags": record.slice_tags,
                "question_zh": record.question_zh,
                "answer_explanation_zh": record.answer_explanation_zh,
                "prompt_variant": prompt_variant,
                "rewrite_type": rewrite_type,
                "semantic_anchor": semantic_anchor,
                "output_schema": output_schema,
                "canonical_answer": ordered_values[ordering.index(0)],
                "alias_list": option_alias_list if rewrite_type == "choice" else alias_list,
                "numeric_slots": option_numeric_slots or numeric_slots,
                "source_file": record.source_file,
                "source_span": record.source_span,
                "option_a": ensure_punctuated(ordered_values[0]),
                "option_b": ensure_punctuated(ordered_values[1]),
                "option_c": ensure_punctuated(ordered_values[2]),
                "option_d": ensure_punctuated(ordered_values[3]),
                "option_origin_qa_ids": ordered_origins,
                "notes": record.notes,
            }
        )

    write_json(form_dir / "vqa.json", {"data": mcq_rows})
    write_json(form_dir / "openqa.json", {"data": openqa_rows})
    return {"mcq": len(mcq_rows), "openqa": len(openqa_rows)}


def build_img2txt_form(
    output_root: Path,
    family: str,
    records: List[QARecord],
    asset_writer: AssetWriter,
) -> Dict[str, int]:
    form = "img2txt"
    form_dir = output_root / family / form
    imgs_dir = form_dir / "imgs"
    imgs_dir.mkdir(parents=True, exist_ok=True)

    mcq_rows: List[Dict[str, Any]] = []
    openqa_rows: List[Dict[str, Any]] = []
    pool = sorted_records(records)

    for record in pool:
        if family == "vqa_state":
            query_frame = midpoint_frame(record)
        elif family in {"action", "verification_action", "verification_state"}:
            query_frame = pre_key_frame(record)
        else:
            query_frame = final_query_frame(record)
        query_name = f"{record.qa_id}_query_img.jpg"
        asset_writer.extract_frame(record.source_video_abspath, query_frame, imgs_dir / query_name)

        open_prompt = choose_openqa_prompt(record, form)
        openqa_rows.append(
            {
                "id": len(openqa_rows),
                "origin_qa_id": record.qa_id,
                "query_img_path": f"imgs/{query_name}",
                "query": open_prompt["query"],
                "GT": open_prompt["GT"],
                "answer_type": open_prompt["answer_type"],
                "task_family": record.task_family,
                "qa_type": record.qa_type,
                "capability_level": CAPABILITY_LEVEL[record.task_family],
                "scenario_family": record.scenario_family,
                "slice_tags": record.slice_tags,
                "question_zh": record.question_zh,
                "answer_explanation_zh": record.answer_explanation_zh,
                "prompt_variant": open_prompt["prompt_variant"],
                "rewrite_type": open_prompt["rewrite_type"],
                "semantic_anchor": open_prompt["semantic_anchor"],
                "output_schema": open_prompt["output_schema"],
                "canonical_answer": open_prompt["canonical_answer"],
                "alias_list": open_prompt["alias_list"],
                "numeric_slots": open_prompt["numeric_slots"],
                "query_source_frame": query_frame,
                "source_file": record.source_file,
                "source_span": record.source_span,
                "notes": record.notes,
            }
        )

        distractors = select_distractors(record, pool)
        option_values: List[str] = []
        option_origins: List[str] = []
        option_numeric_slots = None
        option_alias_list = None
        for option_record in [record, *distractors]:
            value, _, numeric_slots, alias_list = choice_text_value(option_record)
            option_values.append(value)
            option_origins.append(option_record.qa_id)
            if option_record.qa_id == record.qa_id:
                option_numeric_slots = numeric_slots
                option_alias_list = alias_list
        shift = stable_int(f"{record.qa_id}:{form}:choice_order") % 4
        ordering = list(range(4))
        ordering = ordering[shift:] + ordering[:shift]
        ordered_values = [option_values[index] for index in ordering]
        ordered_origins = [option_origins[index] for index in ordering]
        correct_option = "ABCD"[ordering.index(0)]
        prompt_query, prompt_variant, semantic_anchor, rewrite_type, output_schema, numeric_slots, alias_list = mcq_prompt(
            record,
            form,
            "text",
        )
        mcq_rows.append(
            {
                "id": len(mcq_rows),
                "origin_qa_id": record.qa_id,
                "query_img_path": f"imgs/{query_name}",
                "query": prompt_query + "\n" + format_option_block(ordered_values),
                "GT": correct_option,
                "correct_answer": ordered_values[ordering.index(0)],
                "task_family": record.task_family,
                "qa_type": record.qa_type,
                "capability_level": CAPABILITY_LEVEL[record.task_family],
                "scenario_family": record.scenario_family,
                "slice_tags": record.slice_tags,
                "question_zh": record.question_zh,
                "answer_explanation_zh": record.answer_explanation_zh,
                "prompt_variant": prompt_variant,
                "rewrite_type": rewrite_type,
                "semantic_anchor": semantic_anchor,
                "output_schema": output_schema,
                "canonical_answer": ordered_values[ordering.index(0)],
                "alias_list": option_alias_list or alias_list,
                "numeric_slots": option_numeric_slots or numeric_slots,
                "query_source_frame": query_frame,
                "source_file": record.source_file,
                "source_span": record.source_span,
                "option_a": ensure_punctuated(ordered_values[0]),
                "option_b": ensure_punctuated(ordered_values[1]),
                "option_c": ensure_punctuated(ordered_values[2]),
                "option_d": ensure_punctuated(ordered_values[3]),
                "option_origin_qa_ids": ordered_origins,
                "notes": record.notes,
            }
        )

    write_json(form_dir / "vqa.json", {"data": mcq_rows})
    write_json(form_dir / "openqa.json", {"data": openqa_rows})
    return {"mcq": len(mcq_rows), "openqa": len(openqa_rows)}


def build_img2video_form(
    output_root: Path,
    family: str,
    records: List[QARecord],
    pool: List[QARecord],
    asset_writer: AssetWriter,
) -> Dict[str, int]:
    form = "img2video"
    form_dir = output_root / family / form
    imgs_dir = form_dir / "imgs"
    videos_dir = form_dir / "videos"
    imgs_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)
    mcq_rows: List[Dict[str, Any]] = []

    for record in sorted_records(records):
        query_frame = pre_key_frame(record)
        query_name = f"{record.qa_id}_query_img.jpg"
        asset_writer.extract_frame(record.source_video_abspath, query_frame, imgs_dir / query_name)

        distractors = select_distractors(record, pool)
        options = [record, *distractors]
        shift = stable_int(f"{record.qa_id}:{form}:choice_order") % 4
        ordering = list(range(4))
        ordering = ordering[shift:] + ordering[:shift]
        ordered_records = [options[index] for index in ordering]
        correct_option = "ABCD"[ordering.index(0)]
        option_paths: List[str] = []
        option_ranges: List[Dict[str, int]] = []
        option_origin_ids: List[str] = []
        for option_record in ordered_records:
            clip_start, clip_end = action_clip_range(option_record)
            option_name = f"option_clip_{option_record.qa_id}.mp4"
            asset_writer.extract_clip(
                option_record.source_video_abspath,
                clip_start,
                clip_end,
                option_record.fps,
                videos_dir / option_name,
            )
            option_paths.append(f"videos/{option_name}")
            option_ranges.append({"start_frame": clip_start, "end_frame": clip_end})
            option_origin_ids.append(option_record.qa_id)

        prompt_query, prompt_variant, semantic_anchor, rewrite_type, output_schema, numeric_slots, alias_list = mcq_prompt(
            record,
            form,
            "video",
        )
        mcq_rows.append(
            {
                "id": len(mcq_rows),
                "origin_qa_id": record.qa_id,
                "query_img_path": f"imgs/{query_name}",
                "query": prompt_query,
                "GT": correct_option,
                "task_family": record.task_family,
                "qa_type": record.qa_type,
                "capability_level": CAPABILITY_LEVEL[record.task_family],
                "scenario_family": record.scenario_family,
                "slice_tags": record.slice_tags,
                "question_zh": record.question_zh,
                "answer_explanation_zh": record.answer_explanation_zh,
                "prompt_variant": prompt_variant,
                "rewrite_type": rewrite_type,
                "semantic_anchor": semantic_anchor,
                "output_schema": output_schema,
                "canonical_answer": record.answer,
                "alias_list": alias_list,
                "numeric_slots": numeric_slots,
                "query_source_frame": query_frame,
                "option_videos_path": option_paths,
                "option_source_ranges": option_ranges,
                "option_origin_qa_ids": option_origin_ids,
                "source_file": record.source_file,
                "source_span": record.source_span,
                "notes": record.notes,
            }
        )

    write_json(form_dir / "vqa.json", {"data": mcq_rows})
    write_json(form_dir / "openqa.json", {"data": []})
    return {"mcq": len(mcq_rows), "openqa": 0}


def build_video2video_form(
    output_root: Path,
    family: str,
    records: List[QARecord],
    pool: List[QARecord],
    asset_writer: AssetWriter,
) -> Dict[str, int]:
    form = "video2video"
    form_dir = output_root / family / form
    videos_dir = form_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    mcq_rows: List[Dict[str, Any]] = []

    for record in sorted_records(records):
        query_start, query_end = context_clip_range(record)
        query_name = f"{record.qa_id}_query_video.mp4"
        asset_writer.extract_clip(
            record.source_video_abspath,
            query_start,
            query_end,
            record.fps,
            videos_dir / query_name,
        )

        distractors = select_distractors(record, pool)
        options = [record, *distractors]
        shift = stable_int(f"{record.qa_id}:{form}:choice_order") % 4
        ordering = list(range(4))
        ordering = ordering[shift:] + ordering[:shift]
        ordered_records = [options[index] for index in ordering]
        correct_option = "ABCD"[ordering.index(0)]
        option_paths: List[str] = []
        option_ranges: List[Dict[str, int]] = []
        option_origin_ids: List[str] = []
        for letter, option_record in zip(["A", "B", "C", "D"], ordered_records):
            clip_start, clip_end = action_clip_range(option_record)
            option_name = f"option_clip_{option_record.qa_id}.mp4"
            asset_writer.extract_clip(
                option_record.source_video_abspath,
                clip_start,
                clip_end,
                option_record.fps,
                videos_dir / option_name,
            )
            option_paths.append(f"videos/{option_name}")
            option_ranges.append({"start_frame": clip_start, "end_frame": clip_end})
            option_origin_ids.append(option_record.qa_id)

        prompt_query, prompt_variant, semantic_anchor, rewrite_type, output_schema, numeric_slots, alias_list = mcq_prompt(
            record,
            form,
            "video",
        )
        mcq_rows.append(
            {
                "id": len(mcq_rows),
                "origin_qa_id": record.qa_id,
                "query_video_path": f"videos/{query_name}",
                "query": prompt_query,
                "GT": correct_option,
                "task_family": record.task_family,
                "qa_type": record.qa_type,
                "capability_level": CAPABILITY_LEVEL[record.task_family],
                "scenario_family": record.scenario_family,
                "slice_tags": record.slice_tags,
                "question_zh": record.question_zh,
                "answer_explanation_zh": record.answer_explanation_zh,
                "prompt_variant": prompt_variant,
                "rewrite_type": rewrite_type,
                "semantic_anchor": semantic_anchor,
                "output_schema": output_schema,
                "canonical_answer": record.answer,
                "alias_list": alias_list,
                "numeric_slots": numeric_slots,
                "query_source_range": {"start_frame": query_start, "end_frame": query_end},
                "option_videos_path": option_paths,
                "option_source_ranges": option_ranges,
                "option_origin_qa_ids": option_origin_ids,
                "source_file": record.source_file,
                "source_span": record.source_span,
                "notes": record.notes,
            }
        )

    write_json(form_dir / "vqa.json", {"data": mcq_rows})
    write_json(form_dir / "openqa.json", {"data": []})
    return {"mcq": len(mcq_rows), "openqa": 0}


def build_final_state_img2img_form(
    output_root: Path,
    records: List[QARecord],
    state_pool: List[QARecord],
    asset_writer: AssetWriter,
) -> Dict[str, Any]:
    form = "img2img"
    family = "final_state"
    form_dir = output_root / family / form
    imgs_dir = form_dir / "imgs"
    imgs_dir.mkdir(parents=True, exist_ok=True)
    mcq_rows: List[Dict[str, Any]] = []
    filter_entries: List[Dict[str, Any]] = []
    filter_counter: Counter[str] = Counter()

    for record in sorted_records(records):
        query_frame, reason = strict_future_query_frame(record)
        if reason is not None or query_frame is None:
            filter_counter[str(reason)] += 1
            filter_entries.append(build_strict_future_filter_entry(form, record, str(reason)))
            continue
        gt_frame = stable_post_result_frame(record)
        future_gap_frames = gt_frame - query_frame
        if future_gap_frames < STRICT_FUTURE_IMG2IMG_MIN_GAP_FRAMES:
            reason = "future_gap_below_img2img_threshold"
            filter_counter[reason] += 1
            filter_entries.append(
                build_strict_future_filter_entry(
                    form,
                    record,
                    reason,
                    query_frame=query_frame,
                    gt_frame=gt_frame,
                    future_gap_frames=future_gap_frames,
                )
            )
            continue
        action_hint_source = None
        if record.extra.get("last_action_hint"):
            action_hint_source = "last_action_hint"
        elif record.extra.get("action_history_hint"):
            action_hint_source = "action_history_hint"
        if action_hint_source is None:
            reason = "missing_action_hint"
            filter_counter[reason] += 1
            filter_entries.append(
                build_strict_future_filter_entry(
                    form,
                    record,
                    reason,
                    query_frame=query_frame,
                    gt_frame=gt_frame,
                    future_gap_frames=future_gap_frames,
                )
            )
            continue
        perceptual_distance = asset_writer.frame_distance(
            record.source_video_abspath,
            query_frame,
            record.source_video_abspath,
            gt_frame,
        )
        if perceptual_distance < STRICT_FUTURE_PERCEPTUAL_MIN_DISTANCE:
            reason = "perceptual_distance_below_threshold"
            filter_counter[reason] += 1
            filter_entries.append(
                build_strict_future_filter_entry(
                    form,
                    record,
                    reason,
                    query_frame=query_frame,
                    gt_frame=gt_frame,
                    future_gap_frames=future_gap_frames,
                    perceptual_distance=perceptual_distance,
                )
            )
            continue
        options, reason = build_final_state_visual_options(
            record,
            state_pool,
            asset_writer,
            query_frame,
            gt_frame,
        )
        if options is None or reason is not None:
            filter_counter[str(reason)] += 1
            filter_entries.append(
                build_strict_future_filter_entry(
                    form,
                    record,
                    str(reason),
                    query_frame=query_frame,
                    gt_frame=gt_frame,
                    future_gap_frames=future_gap_frames,
                    perceptual_distance=perceptual_distance,
                )
            )
            continue

        query_name = f"{record.qa_id}_query_img.jpg"
        asset_writer.extract_frame(record.source_video_abspath, query_frame, imgs_dir / query_name)

        shift = stable_int(f"{record.qa_id}:{form}:choice_order") % 4
        ordering = list(range(4))
        ordering = ordering[shift:] + ordering[:shift]
        ordered_options = [options[index] for index in ordering]
        correct_option = "ABCD"[ordering.index(0)]
        option_paths: List[str] = []
        option_frames: List[int] = []
        option_origin_ids: List[str] = []
        option_source_types: List[str] = []
        option_query_distances: List[float] = []
        option_gt_distances: List[float] = []
        for letter, option in zip(["A", "B", "C", "D"], ordered_options):
            option_name = f"option_img_{record.qa_id}_{letter}.jpg"
            asset_writer.extract_frame(option.source_video_abspath, option.frame_index, imgs_dir / option_name)
            option_paths.append(f"imgs/{option_name}")
            option_frames.append(option.frame_index)
            option_origin_ids.append(option.origin_qa_id)
            option_source_types.append(option.source_type)
            option_query_distances.append(
                round(
                    asset_writer.frame_distance(
                        record.source_video_abspath,
                        query_frame,
                        option.source_video_abspath,
                        option.frame_index,
                    ),
                    6,
                )
            )
            option_gt_distances.append(
                round(
                    asset_writer.frame_distance(
                        record.source_video_abspath,
                        gt_frame,
                        option.source_video_abspath,
                        option.frame_index,
                    ),
                    6,
                )
            )

        prompt_query, prompt_variant, semantic_anchor, numeric_slots, alias_list = canonical_visual_choice_query(
            record,
            form,
        )
        mcq_rows.append(
            {
                "id": len(mcq_rows),
                "origin_qa_id": record.qa_id,
                "query_img_path": f"imgs/{query_name}",
                "query": prompt_query,
                "GT": correct_option,
                "task_family": record.task_family,
                "qa_type": record.qa_type,
                "capability_level": CAPABILITY_LEVEL[record.task_family],
                "scenario_family": record.scenario_family,
                "slice_tags": record.slice_tags,
                "question_zh": record.question_zh,
                "answer_explanation_zh": record.answer_explanation_zh,
                "prompt_variant": prompt_variant,
                "rewrite_type": "choice",
                "semantic_anchor": semantic_anchor,
                "output_schema": {"type": "choice", "labels": ["A", "B", "C", "D"], "choice_modality": "image"},
                "canonical_answer": record.answer,
                "alias_list": alias_list,
                "numeric_slots": numeric_slots,
                "query_source_frame": query_frame,
                "query_reference_frame": query_frame,
                "option_imgs_path": option_paths,
                "option_source_frames": option_frames,
                "option_origin_qa_ids": option_origin_ids,
                "option_source_types": option_source_types,
                "option_perceptual_distance_to_query": option_query_distances,
                "option_perceptual_distance_to_gt": option_gt_distances,
                "source_file": record.source_file,
                "source_span": record.source_span,
                "notes": record.notes,
                "subtask_role": "snapshot_transition_prediction",
                "query_temporal_role": "after_last_action_before_result",
                "gt_temporal_role": "stable_post_result",
                "strict_future_pass": True,
                "future_gap_frames": future_gap_frames,
                "perceptual_distance": round(perceptual_distance, 6),
                "action_hint_source": action_hint_source,
                "action_hint_text": record.extra.get(action_hint_source),
                "filter_reason": None,
            }
        )

    write_json(form_dir / "vqa.json", {"data": mcq_rows})
    write_json(form_dir / "openqa.json", {"data": []})
    return {
        "mcq": len(mcq_rows),
        "openqa": 0,
        "strict_future_kept_count": len(mcq_rows),
        "strict_future_filtered_count": len(filter_entries),
        "filter_reason_breakdown": dict(sorted(filter_counter.items())),
        "filter_report": filter_entries,
    }


def build_img2img_form(
    output_root: Path,
    family: str,
    records: List[QARecord],
    asset_writer: AssetWriter,
) -> Dict[str, int]:
    form = "img2img"
    form_dir = output_root / family / form
    imgs_dir = form_dir / "imgs"
    imgs_dir.mkdir(parents=True, exist_ok=True)
    mcq_rows: List[Dict[str, Any]] = []
    pool = sorted_records(records)

    for record in pool:
        query_frame = final_query_frame(record) if family == "final_state" else pre_key_frame(record)
        query_name = f"{record.qa_id}_query_img.jpg"
        asset_writer.extract_frame(record.source_video_abspath, query_frame, imgs_dir / query_name)
        distractors = select_distractors(record, pool)
        options = [record, *distractors]
        shift = stable_int(f"{record.qa_id}:{form}:choice_order") % 4
        ordering = list(range(4))
        ordering = ordering[shift:] + ordering[:shift]
        ordered_records = [options[index] for index in ordering]
        correct_option = "ABCD"[ordering.index(0)]
        option_paths: List[str] = []
        option_frames: List[int] = []
        option_origin_ids: List[str] = []
        for letter, option_record in zip(["A", "B", "C", "D"], ordered_records):
            option_frame = midpoint_frame(option_record)
            option_name = f"option_img_{option_record.qa_id}.jpg"
            asset_writer.extract_frame(option_record.source_video_abspath, option_frame, imgs_dir / option_name)
            option_paths.append(f"imgs/{option_name}")
            option_frames.append(option_frame)
            option_origin_ids.append(option_record.qa_id)
        prompt_query, prompt_variant, semantic_anchor, rewrite_type, output_schema, numeric_slots, alias_list = mcq_prompt(
            record,
            form,
            "image",
        )
        mcq_rows.append(
            {
                "id": len(mcq_rows),
                "origin_qa_id": record.qa_id,
                "query_img_path": f"imgs/{query_name}",
                "query": prompt_query,
                "GT": correct_option,
                "task_family": record.task_family,
                "qa_type": record.qa_type,
                "capability_level": CAPABILITY_LEVEL[record.task_family],
                "scenario_family": record.scenario_family,
                "slice_tags": record.slice_tags,
                "question_zh": record.question_zh,
                "answer_explanation_zh": record.answer_explanation_zh,
                "prompt_variant": prompt_variant,
                "rewrite_type": rewrite_type,
                "semantic_anchor": semantic_anchor,
                "output_schema": output_schema,
                "canonical_answer": record.answer,
                "alias_list": alias_list,
                "numeric_slots": numeric_slots,
                "query_source_frame": query_frame,
                "option_imgs_path": option_paths,
                "option_source_frames": option_frames,
                "option_origin_qa_ids": option_origin_ids,
                "source_file": record.source_file,
                "source_span": record.source_span,
                "notes": record.notes,
            }
        )

    write_json(form_dir / "vqa.json", {"data": mcq_rows})
    write_json(form_dir / "openqa.json", {"data": []})
    return {"mcq": len(mcq_rows), "openqa": 0}


def build_state_video2img_form(
    output_root: Path,
    family: str,
    records: List[QARecord],
    asset_writer: AssetWriter,
) -> Dict[str, int]:
    form = "video2img"
    form_dir = output_root / family / form
    videos_dir = form_dir / "videos"
    imgs_dir = form_dir / "imgs"
    videos_dir.mkdir(parents=True, exist_ok=True)
    imgs_dir.mkdir(parents=True, exist_ok=True)
    mcq_rows: List[Dict[str, Any]] = []
    pool = sorted_records(records)

    for record in pool:
        query_start, query_end = context_clip_range(record)
        query_name = f"{record.qa_id}_query_video.mp4"
        asset_writer.extract_clip(
            record.source_video_abspath,
            query_start,
            query_end,
            record.fps,
            videos_dir / query_name,
        )
        distractors = select_distractors(record, pool)
        options = [record, *distractors]
        shift = stable_int(f"{record.qa_id}:{form}:choice_order") % 4
        ordering = list(range(4))
        ordering = ordering[shift:] + ordering[:shift]
        ordered_records = [options[index] for index in ordering]
        correct_option = "ABCD"[ordering.index(0)]
        option_paths: List[str] = []
        option_frames: List[int] = []
        option_origin_ids: List[str] = []
        for letter, option_record in zip(["A", "B", "C", "D"], ordered_records):
            option_frame = state_reference_frame(option_record)
            option_name = f"option_img_{record.qa_id}_{letter}.jpg"
            asset_writer.extract_frame(option_record.source_video_abspath, option_frame, imgs_dir / option_name)
            option_paths.append(f"imgs/{option_name}")
            option_frames.append(option_frame)
            option_origin_ids.append(option_record.qa_id)
        prompt_query, prompt_variant, semantic_anchor, rewrite_type, output_schema, numeric_slots, alias_list = mcq_prompt(
            record,
            form,
            "image",
        )
        mcq_rows.append(
            {
                "id": len(mcq_rows),
                "origin_qa_id": record.qa_id,
                "query_video_path": f"videos/{query_name}",
                "query": prompt_query,
                "GT": correct_option,
                "task_family": record.task_family,
                "qa_type": record.qa_type,
                "capability_level": CAPABILITY_LEVEL[record.task_family],
                "scenario_family": record.scenario_family,
                "slice_tags": record.slice_tags,
                "question_zh": record.question_zh,
                "answer_explanation_zh": record.answer_explanation_zh,
                "prompt_variant": prompt_variant,
                "rewrite_type": rewrite_type,
                "semantic_anchor": semantic_anchor,
                "output_schema": output_schema,
                "canonical_answer": record.answer,
                "alias_list": alias_list,
                "numeric_slots": numeric_slots,
                "query_source_range": {"start_frame": query_start, "end_frame": query_end},
                "option_imgs_path": option_paths,
                "option_source_frames": option_frames,
                "option_origin_qa_ids": option_origin_ids,
                "source_file": record.source_file,
                "source_span": record.source_span,
                "notes": record.notes,
            }
        )

    write_json(form_dir / "vqa.json", {"data": mcq_rows})
    write_json(form_dir / "openqa.json", {"data": []})
    return {"mcq": len(mcq_rows), "openqa": 0}


def build_video2img_form(
    output_root: Path,
    records: List[QARecord],
    state_pool: List[QARecord],
    asset_writer: AssetWriter,
) -> Dict[str, Any]:
    form = "video2img"
    family = "final_state"
    form_dir = output_root / family / form
    videos_dir = form_dir / "videos"
    imgs_dir = form_dir / "imgs"
    videos_dir.mkdir(parents=True, exist_ok=True)
    imgs_dir.mkdir(parents=True, exist_ok=True)
    mcq_rows: List[Dict[str, Any]] = []
    filter_entries: List[Dict[str, Any]] = []
    filter_counter: Counter[str] = Counter()

    for record in sorted_records(records):
        clip_range, reason = strict_future_clip_range(record)
        if reason is not None or clip_range is None:
            filter_counter[str(reason)] += 1
            filter_entries.append(build_strict_future_filter_entry(form, record, str(reason)))
            continue
        query_start, query_end = clip_range
        gt_frame = stable_post_result_frame(record)
        future_gap_frames = gt_frame - query_end
        if future_gap_frames < STRICT_FUTURE_VIDEO2IMG_MIN_GAP_FRAMES:
            reason = "future_gap_below_video2img_threshold"
            filter_counter[reason] += 1
            filter_entries.append(
                build_strict_future_filter_entry(
                    form,
                    record,
                    reason,
                    query_frame=query_end,
                    gt_frame=gt_frame,
                    future_gap_frames=future_gap_frames,
                )
            )
            continue
        perceptual_distance = asset_writer.frame_distance(
            record.source_video_abspath,
            query_end,
            record.source_video_abspath,
            gt_frame,
        )
        if perceptual_distance < STRICT_FUTURE_PERCEPTUAL_MIN_DISTANCE:
            reason = "perceptual_distance_below_threshold"
            filter_counter[reason] += 1
            filter_entries.append(
                build_strict_future_filter_entry(
                    form,
                    record,
                    reason,
                    query_frame=query_end,
                    gt_frame=gt_frame,
                    future_gap_frames=future_gap_frames,
                    perceptual_distance=perceptual_distance,
                )
            )
            continue
        options, reason = build_final_state_visual_options(
            record,
            state_pool,
            asset_writer,
            query_end,
            gt_frame,
        )
        if options is None or reason is not None:
            filter_counter[str(reason)] += 1
            filter_entries.append(
                build_strict_future_filter_entry(
                    form,
                    record,
                    str(reason),
                    query_frame=query_end,
                    gt_frame=gt_frame,
                    future_gap_frames=future_gap_frames,
                    perceptual_distance=perceptual_distance,
                )
            )
            continue
        query_name = f"{record.qa_id}_query_video.mp4"
        asset_writer.extract_clip(
            record.source_video_abspath,
            query_start,
            query_end,
            record.fps,
            videos_dir / query_name,
        )
        shift = stable_int(f"{record.qa_id}:{form}:choice_order") % 4
        ordering = list(range(4))
        ordering = ordering[shift:] + ordering[:shift]
        ordered_records = [options[index] for index in ordering]
        correct_option = "ABCD"[ordering.index(0)]
        option_paths: List[str] = []
        option_frames: List[int] = []
        option_origin_ids: List[str] = []
        option_source_types: List[str] = []
        option_query_distances: List[float] = []
        option_gt_distances: List[float] = []
        for letter, option_record in zip(["A", "B", "C", "D"], ordered_records):
            option_name = f"option_img_{record.qa_id}_{letter}.jpg"
            asset_writer.extract_frame(option_record.source_video_abspath, option_record.frame_index, imgs_dir / option_name)
            option_paths.append(f"imgs/{option_name}")
            option_frames.append(option_record.frame_index)
            option_origin_ids.append(option_record.origin_qa_id)
            option_source_types.append(option_record.source_type)
            option_query_distances.append(
                round(
                    asset_writer.frame_distance(
                        record.source_video_abspath,
                        query_end,
                        option_record.source_video_abspath,
                        option_record.frame_index,
                    ),
                    6,
                )
            )
            option_gt_distances.append(
                round(
                    asset_writer.frame_distance(
                        record.source_video_abspath,
                        gt_frame,
                        option_record.source_video_abspath,
                        option_record.frame_index,
                    ),
                    6,
                )
            )
        prompt_query, prompt_variant, semantic_anchor, numeric_slots, alias_list = canonical_visual_choice_query(
            record,
            form,
        )
        mcq_rows.append(
            {
                "id": len(mcq_rows),
                "origin_qa_id": record.qa_id,
                "query_video_path": f"videos/{query_name}",
                "query": prompt_query,
                "GT": correct_option,
                "task_family": record.task_family,
                "qa_type": record.qa_type,
                "capability_level": CAPABILITY_LEVEL[record.task_family],
                "scenario_family": record.scenario_family,
                "slice_tags": record.slice_tags,
                "question_zh": record.question_zh,
                "answer_explanation_zh": record.answer_explanation_zh,
                "prompt_variant": prompt_variant,
                "rewrite_type": "choice",
                "semantic_anchor": semantic_anchor,
                "output_schema": {"type": "choice", "labels": ["A", "B", "C", "D"], "choice_modality": "image"},
                "canonical_answer": record.answer,
                "alias_list": alias_list,
                "numeric_slots": numeric_slots,
                "query_source_range": {"start_frame": query_start, "end_frame": query_end},
                "query_reference_frame": query_end,
                "option_imgs_path": option_paths,
                "option_source_frames": option_frames,
                "option_origin_qa_ids": option_origin_ids,
                "option_source_types": option_source_types,
                "option_perceptual_distance_to_query": option_query_distances,
                "option_perceptual_distance_to_gt": option_gt_distances,
                "source_file": record.source_file,
                "source_span": record.source_span,
                "notes": record.notes,
                "subtask_role": "future_result_prediction",
                "query_temporal_role": "context_before_result_reveal",
                "gt_temporal_role": "stable_post_result",
                "strict_future_pass": True,
                "future_gap_frames": future_gap_frames,
                "perceptual_distance": round(perceptual_distance, 6),
                "action_hint_source": None,
                "action_hint_text": None,
                "filter_reason": None,
            }
        )

    write_json(form_dir / "vqa.json", {"data": mcq_rows})
    write_json(form_dir / "openqa.json", {"data": []})
    return {
        "mcq": len(mcq_rows),
        "openqa": 0,
        "strict_future_kept_count": len(mcq_rows),
        "strict_future_filtered_count": len(filter_entries),
        "filter_reason_breakdown": dict(sorted(filter_counter.items())),
        "filter_report": filter_entries,
    }


def form_asset_counts(form_dir: Path) -> Dict[str, int]:
    counts = {
        "query_videos": 0,
        "option_videos": 0,
        "query_imgs": 0,
        "option_imgs": 0,
    }
    for path in form_dir.rglob("*"):
        if not path.is_file():
            continue
        name = path.name
        if name.endswith("_query_video.mp4"):
            counts["query_videos"] += 1
        elif name.startswith("option_clip_") and name.endswith(".mp4"):
            counts["option_videos"] += 1
        elif name.endswith("_query_img.jpg"):
            counts["query_imgs"] += 1
        elif name.startswith("option_img_") and name.endswith(".jpg"):
            counts["option_imgs"] += 1
    return counts


def write_readme(
    output_root: Path,
    manifest: Dict[str, Any],
    form_matrix: Dict[str, Any],
) -> None:
    lines: List[str] = []
    lines.append("# SWITCH HF-like Multiform QA v2")
    lines.append("")
    lines.append(f"- Videos: `{manifest['num_videos']}`")
    lines.append("- Dependency used for asset generation: `opencv-python-headless`")
    lines.append("- This package preserves `hf_innovative_qa_v1` and writes a new versioned directory.")
    lines.append("")
    strict_summary = manifest.get("strict_future_summary") or {}
    if strict_summary:
        lines.append("## Strict Future Final State Forms")
        lines.append("")
        for form_name, stats in strict_summary.items():
            lines.append(
                f"- `final_state/{form_name}`: kept=`{stats['kept_count']}`, "
                f"filtered=`{stats['filtered_count']}`, role=`{stats['subtask_role']}`"
            )
        lines.append("")
    lines.append("## Task / Form Counts")
    lines.append("")
    for family in TASK_FAMILIES:
        lines.append(f"### {family}")
        lines.append("")
        for form_name, stats in form_matrix["task_families"][family].items():
            lines.append(
                f"- `{form_name}`: mcq=`{stats['mcq']}`, openqa=`{stats['openqa']}`, "
                f"input=`{stats['input_modality']}`, output=`{stats['output_modality']}`"
            )
        lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Queries are rewritten with stable prompt variants rather than step-number templates.")
    lines.append("- OpenQA uses `keyword`, `numeric`, or `structured_short_answer` depending on the task.")
    lines.append("- `final_state/img2img` now uses pre-result snapshot plus explicit action hint to predict the future final image.")
    lines.append("- `final_state/video2img` now uses pre-result context video only and filters samples that leak the answer visually.")
    lines.append("- Final-state visual choices now prefer one cross-scene same-device distractor before falling back to same-video negatives.")
    lines.append("- `recovery` keeps `video2txt` as the primary form and exposes structured recovery-chain supervision.")
    lines.append("- `vqa_task` intentionally stays on `video2txt` in this version.")
    output_root.joinpath("README.md").write_text("\n".join(lines), encoding="utf-8")


def validate_outputs(output_root: Path, form_matrix: Dict[str, Any]) -> Dict[str, Any]:
    report: Dict[str, Any] = {"action_query_check": True, "task_openqa_prompt_variants": 0}
    for family in TASK_FAMILIES:
        for form_name, stats in form_matrix["task_families"][family].items():
            form_dir = output_root / family / form_name
            vqa = load_json(form_dir / "vqa.json")
            openqa = load_json(form_dir / "openqa.json")
            assert len(vqa["data"]) == stats["mcq"], f"MCQ count mismatch for {family}/{form_name}"
            assert len(openqa["data"]) == stats["openqa"], f"OpenQA count mismatch for {family}/{form_name}"
            for row in vqa["data"]:
                if "query_video_path" in row:
                    assert (form_dir / row["query_video_path"]).exists()
                if "query_img_path" in row:
                    assert (form_dir / row["query_img_path"]).exists()
                for field in ("option_videos_path", "option_imgs_path"):
                    for asset in row.get(field, []) or []:
                        assert (form_dir / asset).exists()
                assert "rewrite_type" in row
                assert "prompt_variant" in row
                assert "semantic_anchor" in row
                assert "output_schema" in row
                if family == "final_state" and form_name == "img2img":
                    assert row["subtask_role"] == "snapshot_transition_prediction"
                    assert row["query_temporal_role"] == "after_last_action_before_result"
                    assert row["gt_temporal_role"] == "stable_post_result"
                    assert row["strict_future_pass"] is True
                    assert row["future_gap_frames"] >= STRICT_FUTURE_IMG2IMG_MIN_GAP_FRAMES
                    assert row["query_source_frame"] < int(row["source_span"]["start"] or 0)
                    assert row["perceptual_distance"] >= STRICT_FUTURE_PERCEPTUAL_MIN_DISTANCE
                    assert "before the result appears" in row["query"].lower()
                    assert row["action_hint_source"] in {"last_action_hint", "action_history_hint"}
                    assert row["action_hint_text"]
                    gt_index = "ABCD".index(row["GT"])
                    gt_frame = row["option_source_frames"][gt_index]
                    assert gt_frame != row["query_source_frame"]
                    assert row["option_source_types"][gt_index] == "gold_stable_post_result"
                    negative_distances = [
                        distance
                        for index, distance in enumerate(row["option_perceptual_distance_to_query"])
                        if index != gt_index
                    ]
                    assert negative_distances
                    assert min(negative_distances) >= STRICT_FUTURE_MIN_NEGATIVE_DISTANCE_TO_QUERY
                    assert (
                        min(negative_distances)
                        <= row["option_perceptual_distance_to_query"][gt_index] + STRICT_FUTURE_NEGATIVE_DISTANCE_MARGIN
                    )
                    assert any(
                        source_type in {
                            "same_video_pre_result_hard_negative",
                            "same_scenario_same_semantic_wrong_outcome",
                            "same_scenario_other_state_negative",
                            "cross_scene_same_device_negative",
                        }
                        for source_type in row["option_source_types"]
                    )
                    for source_type, frame_index, distance_to_gt in zip(
                        row["option_source_types"],
                        row["option_source_frames"],
                        row["option_perceptual_distance_to_gt"],
                    ):
                        if source_type != "same_video_pre_result_hard_negative":
                            continue
                        assert int(row["source_span"]["start"] or 0) - frame_index >= max(
                            STRICT_FUTURE_SAME_VIDEO_MIN_PRE_RESULT_GAP_FRAMES,
                            15,
                        )
                        assert distance_to_gt >= STRICT_FUTURE_SAME_VIDEO_MIN_GT_DISTANCE
                    for left in range(4):
                        for right in range(left + 1, 4):
                            if (
                                row["option_source_types"][left] == "gold_stable_post_result"
                                or row["option_source_types"][right] == "gold_stable_post_result"
                            ):
                                continue
                            form_image_root = form_dir
                            left_frame = cv2.imread(str(form_image_root / row["option_imgs_path"][left]), cv2.IMREAD_GRAYSCALE)
                            right_frame = cv2.imread(str(form_image_root / row["option_imgs_path"][right]), cv2.IMREAD_GRAYSCALE)
                            left_frame = cv2.resize(left_frame, (32, 32), interpolation=cv2.INTER_AREA).astype("float32") / 255.0
                            right_frame = cv2.resize(right_frame, (32, 32), interpolation=cv2.INTER_AREA).astype("float32") / 255.0
                            pairwise_distance = float(np.mean(np.abs(left_frame - right_frame)))
                            assert pairwise_distance >= STRICT_FUTURE_MIN_NEGATIVE_PAIRWISE_DISTANCE
                if family == "final_state" and form_name == "video2img":
                    assert row["subtask_role"] == "future_result_prediction"
                    assert row["query_temporal_role"] == "context_before_result_reveal"
                    assert row["gt_temporal_role"] == "stable_post_result"
                    assert row["strict_future_pass"] is True
                    assert row["future_gap_frames"] >= STRICT_FUTURE_VIDEO2IMG_MIN_GAP_FRAMES
                    assert row["query_source_range"]["end_frame"] <= int(row["source_span"]["start"] or 0) - 1
                    assert row["perceptual_distance"] >= STRICT_FUTURE_PERCEPTUAL_MIN_DISTANCE
                    assert "before the outcome is revealed" in row["query"].lower()
                    assert row["action_hint_source"] is None
                    gt_index = "ABCD".index(row["GT"])
                    gt_frame = row["option_source_frames"][gt_index]
                    assert not (
                        row["query_source_range"]["start_frame"] <= gt_frame <= row["query_source_range"]["end_frame"]
                    )
                    assert row["option_source_types"][gt_index] == "gold_stable_post_result"
                    negative_distances = [
                        distance
                        for index, distance in enumerate(row["option_perceptual_distance_to_query"])
                        if index != gt_index
                    ]
                    assert negative_distances
                    assert min(negative_distances) >= STRICT_FUTURE_MIN_NEGATIVE_DISTANCE_TO_QUERY
                    assert (
                        min(negative_distances)
                        <= row["option_perceptual_distance_to_query"][gt_index] + STRICT_FUTURE_NEGATIVE_DISTANCE_MARGIN
                    )
                    assert any(
                        source_type in {
                            "same_video_pre_result_hard_negative",
                            "same_scenario_same_semantic_wrong_outcome",
                            "same_scenario_other_state_negative",
                            "cross_scene_same_device_negative",
                        }
                        for source_type in row["option_source_types"]
                    )
                    for source_type, frame_index, distance_to_gt in zip(
                        row["option_source_types"],
                        row["option_source_frames"],
                        row["option_perceptual_distance_to_gt"],
                    ):
                        if source_type != "same_video_pre_result_hard_negative":
                            continue
                        assert int(row["source_span"]["start"] or 0) - frame_index >= max(
                            STRICT_FUTURE_SAME_VIDEO_MIN_PRE_RESULT_GAP_FRAMES,
                            15,
                        )
                        assert distance_to_gt >= STRICT_FUTURE_SAME_VIDEO_MIN_GT_DISTANCE
                    for left in range(4):
                        for right in range(left + 1, 4):
                            if (
                                row["option_source_types"][left] == "gold_stable_post_result"
                                or row["option_source_types"][right] == "gold_stable_post_result"
                            ):
                                continue
                            form_image_root = form_dir
                            left_frame = cv2.imread(str(form_image_root / row["option_imgs_path"][left]), cv2.IMREAD_GRAYSCALE)
                            right_frame = cv2.imread(str(form_image_root / row["option_imgs_path"][right]), cv2.IMREAD_GRAYSCALE)
                            left_frame = cv2.resize(left_frame, (32, 32), interpolation=cv2.INTER_AREA).astype("float32") / 255.0
                            right_frame = cv2.resize(right_frame, (32, 32), interpolation=cv2.INTER_AREA).astype("float32") / 255.0
                            pairwise_distance = float(np.mean(np.abs(left_frame - right_frame)))
                            assert pairwise_distance >= STRICT_FUTURE_MIN_NEGATIVE_PAIRWISE_DISTANCE
            for row in openqa["data"]:
                if "query_video_path" in row:
                    assert (form_dir / row["query_video_path"]).exists()
                if "query_img_path" in row:
                    assert (form_dir / row["query_img_path"]).exists()
                assert "rewrite_type" in row
                assert "prompt_variant" in row
                assert "semantic_anchor" in row
                assert "output_schema" in row
                if row["rewrite_type"] == "numeric":
                    assert row.get("numeric_slots"), f"Missing numeric slots for {row['origin_qa_id']}"
                if row["task_family"] == "recovery" and row["qa_type"] == "recovery_chain":
                    assert isinstance(row["GT"], dict)
                    assert {"wrong_action", "post_wrong_signal", "fix_steps", "post_fix_signal"} <= set(row["GT"].keys())

    action_openqa = load_json(output_root / "action" / "video2txt" / "openqa.json")["data"]
    assert all("key action step" not in row["query"].lower() for row in action_openqa)
    report["action_query_check"] = True

    vqa_task_openqa = load_json(output_root / "vqa_task" / "video2txt" / "openqa.json")["data"]
    prompt_variants = {row["prompt_variant"] for row in vqa_task_openqa}
    assert len(prompt_variants) > 1
    report["task_openqa_prompt_variants"] = len(prompt_variants)
    strict_future_report = load_json(output_root / "strict_future_filter_report.json")
    report["strict_future_filter_report_present"] = True
    report["strict_future_filter_forms"] = sorted(strict_future_report["forms"].keys())
    return report


def build_form_matrix(output_root: Path, family_counts: Dict[str, Dict[str, Dict[str, int]]]) -> Dict[str, Any]:
    matrix: Dict[str, Any] = {
        "schema_version": "switch-hf-innovative-qa-v2-multiform",
        "dependency": "opencv-python-headless",
        "task_families": {},
    }
    for family in TASK_FAMILIES:
        matrix["task_families"][family] = {}
        for form_name in FORM_SPECS[family]:
            form_dir = output_root / family / form_name
            input_modality = "video" if form_name.startswith("video2") else "image"
            output_modality = form_name.split("2", 1)[1].replace("txt", "text")
            counts = family_counts[family][form_name]
            asset_counts = form_asset_counts(form_dir)
            matrix["task_families"][family][form_name] = {
                "input_modality": input_modality,
                "output_modality": output_modality,
                "mcq": counts["mcq"],
                "openqa": counts["openqa"],
                "asset_counts": asset_counts,
                "candidate_recovery_future_metadata_only": family == "recovery",
            }
            if "strict_future_kept_count" in counts:
                matrix["task_families"][family][form_name]["strict_future_kept_count"] = counts["strict_future_kept_count"]
                matrix["task_families"][family][form_name]["strict_future_filtered_count"] = counts["strict_future_filtered_count"]
                matrix["task_families"][family][form_name]["filter_reason_breakdown"] = counts["filter_reason_breakdown"]
    return matrix


def build_manifest(
    profiles: Dict[str, Dict[str, Any]],
    form_matrix: Dict[str, Any],
    validation_report: Dict[str, Any],
) -> Dict[str, Any]:
    strict_future_summary: Dict[str, Any] = {}
    for form_name in ("img2img", "video2img"):
        stats = form_matrix["task_families"]["final_state"].get(form_name) or {}
        if "strict_future_kept_count" not in stats:
            continue
        strict_future_summary[form_name] = {
            "kept_count": stats["strict_future_kept_count"],
            "filtered_count": stats["strict_future_filtered_count"],
            "filter_reason_breakdown": stats["filter_reason_breakdown"],
            "subtask_role": (
                "snapshot_transition_prediction"
                if form_name == "img2img"
                else "future_result_prediction"
            ),
        }
    return {
        "schema_version": "switch-hf-innovative-qa-v2-multiform",
        "num_videos": len(profiles),
        "task_families": TASK_FAMILIES,
        "dependency": "opencv-python-headless",
        "form_matrix": form_matrix["task_families"],
        "validation_report": validation_report,
        "strict_future_summary": strict_future_summary,
        "notes": [
            "This package keeps hf_innovative_qa_v1 untouched and writes a separate v2 multiform directory.",
            "Query prompts are rewritten with stable semantic prompt variants.",
            "OpenQA uses keyword, numeric, or structured_short_answer instead of open_judge-heavy answers.",
            "vqa_task remains video2txt in this version by design.",
            "final_state visual forms now use strict-future anti-leakage rules and may filter low-quality samples.",
            "recovery keeps video2txt as the primary form and exposes metadata for candidate recovery future extension.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build SWITCH multiform QA dataset from extracted qa_candidates."
    )
    parser.add_argument(
        "--annotation-root",
        type=Path,
        default=Path("annotations") / "0421" / "switch",
        help="Directory containing raw annotation JSON, videos, and combined qa_candidates outputs.",
    )
    parser.add_argument(
        "--output-dirname",
        type=str,
        default="hf_innovative_qa_v2_multiform",
        help="Name of the output dataset directory created under annotation-root.",
    )
    args = parser.parse_args()

    annotation_root = args.annotation_root
    output_root = annotation_root / args.output_dirname
    if output_root.exists():
        remove_tree(output_root)

    records, profiles, grouped_records = build_records(annotation_root)
    source_videos = {
        profile["video_name"]: profile["source_video_abspath"]
        for profile in profiles.values()
    }
    asset_writer = AssetWriter()

    family_counts: Dict[str, Dict[str, Dict[str, int]]] = {
        family: {} for family in TASK_FAMILIES
    }
    strict_future_filter_report: Dict[str, Any] = {
        "schema_version": "switch-hf-innovative-qa-v2-strict-future-filter-report",
        "forms": {},
    }

    for family in TASK_FAMILIES:
        family_records = sorted_records(grouped_records[family])
        if "video2txt" not in FORM_SPECS[family]:
            continue
        print(f"Building {family}/video2txt with {len(family_records)} records")
        family_counts[family]["video2txt"] = build_video2txt_form(
            output_root,
            family,
            family_records,
            source_videos,
        )

    print(f"Building vqa_state/img2txt with {len(grouped_records['vqa_state'])} records")
    family_counts["vqa_state"]["img2txt"] = build_img2txt_form(
        output_root,
        "vqa_state",
        grouped_records["vqa_state"],
        asset_writer,
    )

    print(f"Building action/img2txt with {len(grouped_records['action'])} records")
    family_counts["action"]["img2txt"] = build_img2txt_form(
        output_root,
        "action",
        grouped_records["action"],
        asset_writer,
    )
    print(f"Building action/img2video with {len(grouped_records['action'])} records")
    family_counts["action"]["img2video"] = build_img2video_form(
        output_root,
        "action",
        grouped_records["action"],
        grouped_records["action"],
        asset_writer,
    )
    print(f"Building action/video2video with {len(grouped_records['action'])} records")
    family_counts["action"]["video2video"] = build_video2video_form(
        output_root,
        "action",
        grouped_records["action"],
        grouped_records["action"],
        asset_writer,
    )

    print(f"Building final_state/img2txt with {len(grouped_records['final_state'])} records")
    family_counts["final_state"]["img2txt"] = build_img2txt_form(
        output_root,
        "final_state",
        grouped_records["final_state"],
        asset_writer,
    )
    print(f"Building final_state/img2img with {len(grouped_records['final_state'])} records")
    state_pool = sorted_records(grouped_records["final_state"] + grouped_records["verification_state"])
    family_counts["final_state"]["img2img"] = build_final_state_img2img_form(
        output_root,
        grouped_records["final_state"],
        state_pool,
        asset_writer,
    )
    strict_future_filter_report["forms"]["final_state/img2img"] = family_counts["final_state"]["img2img"]["filter_report"]
    print(f"Building final_state/video2img with {len(grouped_records['final_state'])} records")
    family_counts["final_state"]["video2img"] = build_video2img_form(
        output_root,
        grouped_records["final_state"],
        state_pool,
        asset_writer,
    )
    strict_future_filter_report["forms"]["final_state/video2img"] = family_counts["final_state"]["video2img"]["filter_report"]

    print(
        f"Building verification_action/img2txt with {len(grouped_records['verification_action'])} records"
    )
    family_counts["verification_action"]["img2txt"] = build_img2txt_form(
        output_root,
        "verification_action",
        grouped_records["verification_action"],
        asset_writer,
    )
    print(
        f"Building verification_action/img2video with {len(grouped_records['verification_action'])} records"
    )
    family_counts["verification_action"]["img2video"] = build_img2video_form(
        output_root,
        "verification_action",
        grouped_records["verification_action"],
        grouped_records["verification_action"],
        asset_writer,
    )
    print(
        f"Building verification_action/video2video with {len(grouped_records['verification_action'])} records"
    )
    family_counts["verification_action"]["video2video"] = build_video2video_form(
        output_root,
        "verification_action",
        grouped_records["verification_action"],
        grouped_records["verification_action"],
        asset_writer,
    )

    print(
        f"Building verification_state/img2txt with {len(grouped_records['verification_state'])} records"
    )
    family_counts["verification_state"]["img2txt"] = build_img2txt_form(
        output_root,
        "verification_state",
        grouped_records["verification_state"],
        asset_writer,
    )
    print(
        f"Building verification_state/img2img with {len(grouped_records['verification_state'])} records"
    )
    family_counts["verification_state"]["img2img"] = build_img2img_form(
        output_root,
        "verification_state",
        grouped_records["verification_state"],
        asset_writer,
    )
    print(
        f"Building verification_state/video2img with {len(grouped_records['verification_state'])} records"
    )
    family_counts["verification_state"]["video2img"] = build_state_video2img_form(
        output_root,
        "verification_state",
        grouped_records["verification_state"],
        asset_writer,
    )

    form_matrix = build_form_matrix(output_root, family_counts)
    write_json(output_root / "strict_future_filter_report.json", strict_future_filter_report)
    validation_report = validate_outputs(output_root, form_matrix)
    manifest = build_manifest(profiles, form_matrix, validation_report)

    write_json(output_root / "form_matrix.json", form_matrix)
    write_json(output_root / "dataset_manifest.json", manifest)
    write_readme(output_root, manifest, form_matrix)

    print(f"Wrote SWITCH multiform QA v2 dataset to: {output_root}")
    for family in TASK_FAMILIES:
        for form_name in FORM_SPECS[family]:
            counts = family_counts[family][form_name]
            print(
                f"  - {family}/{form_name}: mcq={counts['mcq']}, openqa={counts['openqa']}"
            )


if __name__ == "__main__":
    main()
