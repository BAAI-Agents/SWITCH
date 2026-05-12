#!/usr/bin/env python3
"""
Build a single-image + prompt video-generation dataset from the current
SWITCH Action_1 Label Studio export.

Outputs:
- JSONL dataset for generation-model tasks
- Markdown summary for human review
- Extracted anchor frames under a dedicated folder

The script is intentionally conservative:
- it only uses information that can be recovered relatively stably from the
  current timeline annotations
- when semantic details are weak or inferred, it records quality flags
- retry samples are optional and may be zero for the current data
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import imageio_ffmpeg


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


WORD_TO_NUM = {
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
}


GENERIC_VERIFICATION_PATTERNS = (
    "check if the screen has switched",
    "check the screen has switched",
)


GENERIC_ACTION_PATTERNS = (
    "click the button",
    "press the elevator button",
)


@dataclass
class Segment:
    label: str
    text: str
    start: Optional[float]
    end: Optional[float]


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def split_payload(raw_text: str, label: str) -> str:
    raw_text = normalize_spaces(raw_text)
    if not raw_text:
        return ""

    parts = re.split(r"[:：;；]\s*", raw_text, maxsplit=1)
    if len(parts) == 2:
        return parts[1].strip()

    if raw_text.startswith(label):
        return raw_text[len(label) :].strip(" :：;；")
    return raw_text


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
        ranges = value.get("ranges", []) or []
        start = ranges[0].get("start") if ranges else None
        end = ranges[0].get("end") if ranges else None
        segments.append(Segment(label=label, text=payload, start=start, end=end))

    segments.sort(key=lambda s: ((s.start or 0), (s.end or 0), s.label))
    return segments


def infer_summary_labels(segments: Iterable[Segment]) -> List[str]:
    return [seg.label for seg in segments if seg.label not in KNOWN_LABELS]


def choose_main_task(demands: List[str], summaries: List[str]) -> str:
    if summaries:
        return summaries[0]
    if demands:
        return max(demands, key=len)
    return ""


def slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "unknown"


def find_local_video_path(video_name: str, input_dir: Path) -> Path:
    candidate = input_dir / video_name
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Local video not found for {video_name}: {candidate}")


def ordinal_or_numeric_floor(text: str) -> Optional[int]:
    text_l = text.lower()
    digit_match = re.search(r"(\d+)(?:st|nd|rd|th)?\s+floor", text_l)
    if digit_match:
        return int(digit_match.group(1))

    for word, value in WORD_TO_NUM.items():
        if f"{word} floor" in text_l:
            return value
    return None


def infer_device_family(target_object: str, main_task: str) -> str:
    raw = f"{target_object} {main_task}".lower()
    if "hospital registration machine" in raw:
        return "hospital_registration_machine"
    if "subway ticket machine" in raw:
        return "subway_ticket_machine"
    if "elevator" in raw:
        return "elevator_system"
    return slugify(target_object or main_task or "unknown_device")


def infer_task_intent(main_task: str, target_object: str, demands: Optional[List[str]] = None) -> str:
    task_text = " ".join([main_task, *(demands or [])]).lower()
    object_text = (target_object or "").lower()

    if "query" in task_text and "project" in task_text:
        return "query_project"
    if "cancel appointment" in task_text:
        return "cancel_registration_appointment"
    if "appointment" in task_text or "registration" in task_text:
        return "make_registration_appointment"
    if "subway ticket" in task_text or ("ticket" in task_text and "station" in task_text):
        return "buy_subway_ticket"
    if "elevator" in task_text and "floor" in task_text:
        return "go_to_target_floor"
    if "subway ticket machine" in object_text:
        return "buy_subway_ticket"
    if "hospital registration machine" in object_text:
        return "public_service_terminal_task"
    if "elevator" in object_text:
        return "go_to_target_floor"
    return slugify(main_task or target_object or "unknown_task")


def humanize_identifier(text: str) -> str:
    return normalize_spaces((text or "").replace("_", " "))


def infer_goal_slots(
    main_task: str,
    demands: List[str],
    initial_positions: List[str],
    execute_actions: List[Segment],
) -> Dict[str, Any]:
    slots: Dict[str, Any] = {}

    if initial_positions:
        init = initial_positions[0]
        floor = ordinal_or_numeric_floor(init)
        if floor is not None:
            slots["current_floor"] = floor
        if "station" in init.lower():
            slots["origin_station"] = init

    floor = ordinal_or_numeric_floor(main_task)
    if floor is not None:
        slots["target_floor"] = floor

    if len(demands) > 1:
        extra_goal = demands[1]
        floor = ordinal_or_numeric_floor(extra_goal)
        if floor is not None:
            slots["target_floor"] = floor
        elif "station" in extra_goal.lower():
            slots["destination_station"] = extra_goal
        elif "want to go to" in extra_goal.lower():
            slots["destination"] = re.sub(
                r"(?i)^want to go to\s+", "", extra_goal
            ).strip()

    lowered_task = main_task.lower()
    price_match = re.search(r"(\d+)\s*-\s*yuan|(\d+)\s*yuan", lowered_task)
    if price_match:
        price = price_match.group(1) or price_match.group(2)
        slots["ticket_price"] = f"{price}_yuan"

    if "buy one" in lowered_task or "1 ticket" in lowered_task:
        slots["ticket_count"] = 1

    from_to = re.search(r"from (.+?) to (.+?)(?:$|\.|,)", main_task, re.IGNORECASE)
    if from_to:
        slots["origin_station"] = from_to.group(1).strip()
        slots["destination_station"] = from_to.group(2).strip()

    for seg in execute_actions:
        text_l = seg.text.lower()
        if "2 yuan" in text_l:
            slots.setdefault("ticket_price", "2_yuan")
        if "1 sheet" in text_l or "1 ticket" in text_l:
            slots.setdefault("ticket_count", 1)
        action_floor = ordinal_or_numeric_floor(seg.text)
        if action_floor is not None:
            slots.setdefault("target_floor", action_floor)

    return slots


def infer_success_condition(task_intent: str, final_state: Optional[str], goal_slots: Dict[str, Any]) -> str:
    if final_state:
        return final_state
    if task_intent == "go_to_target_floor" and goal_slots.get("target_floor") is not None:
        return f"arrive_at_floor_{goal_slots['target_floor']}"
    if task_intent == "buy_subway_ticket":
        return "subway_ticket_obtained"
    if task_intent == "query_project":
        return "project_query_completed"
    if task_intent == "make_registration_appointment":
        return "appointment_registration_successful"
    if task_intent == "cancel_registration_appointment":
        return "appointment_cancellation_successful"
    return "task_completed"


def infer_action_target(action_text: str) -> Optional[str]:
    text_l = action_text.lower()
    quoted = re.findall(r"'([^']+)'", action_text)
    if quoted:
        return quoted[0]
    if "project search button" in text_l:
        return "project_search_button"
    if "next page button" in text_l:
        return "next_page_button"
    if "back button" in text_l:
        return "back_button"
    if "home button" in text_l:
        return "home_button"
    if "appointment registration button" in text_l:
        return "appointment_registration_button"
    if "e-voucher button" in text_l:
        return "e_voucher_button"
    if "select doctor button" in text_l:
        return "select_doctor_button"
    if "stomatology button" in text_l:
        return "stomatology_button"
    if "general dentistry button" in text_l:
        return "general_dentistry_button"
    if "identity card button" in text_l:
        return "identity_card_button"
    if "cancel appointment button" in text_l:
        return "cancel_appointment_button"
    if "confirm" in text_l:
        return "confirm_button"
    if "coin" in text_l:
        return "coin_slot"
    if "ticket pickup" in text_l:
        return "ticket_output_slot"
    if "sixth floor" in text_l:
        return "floor_6_button"
    if "tenth floor" in text_l:
        return "floor_10_button"
    if "11th floor" in text_l or "eleventh floor" in text_l:
        return "floor_11_button"
    if "elevator button" in text_l or "click the button" in text_l:
        return "elevator_button"
    if "put down id card" in text_l:
        return "id_card_reader"
    if "select time" in text_l:
        return "time_slot"
    return None


def infer_action_parameter(action_text: str) -> Optional[Dict[str, Any]]:
    params: Dict[str, Any] = {}
    text_l = action_text.lower()
    floor = ordinal_or_numeric_floor(action_text)
    if floor is not None:
        params["target_floor"] = floor
    if "2 yuan" in text_l:
        params["ticket_price"] = "2_yuan"
    if "1 sheet" in text_l:
        params["ticket_count"] = 1
    if "stomatology" in text_l:
        params["department"] = "stomatology"
    if "general dentistry" in text_l:
        params["service_type"] = "general_dentistry"
    if "select time" in text_l:
        params["time_slot"] = "selected_time"
    if "coin" in text_l:
        params["payment_method"] = "coin"
    return params or None


def infer_ui_changes_from_action(action_text: str, verification_text: Optional[str]) -> List[str]:
    action_l = action_text.lower()
    verification_l = (verification_text or "").lower()
    changes: List[str] = []

    if "screen has switched" in verification_l:
        changes.append("screen switches to the next relevant page or state")
    if "project search button" in action_l:
        changes.append("project query page becomes visible")
    if "other function button" in action_l:
        changes.append("other functions page becomes visible")
    if "next page button" in action_l:
        changes.append("the next page becomes visible")
    if "back button" in action_l:
        changes.append("the previous page becomes visible")
    if "home button" in action_l:
        changes.append("the home page becomes visible")
    if "appointment registration button" in action_l:
        changes.append("appointment registration page becomes visible")
    if "e-voucher button" in action_l:
        changes.append("the e-voucher page becomes visible")
    if "select doctor button" in action_l:
        changes.append("doctor selection page becomes visible")
    if "stomatology button" in action_l:
        changes.append("the stomatology option is selected")
    if "general dentistry button" in action_l:
        changes.append("the general dentistry option is selected")
    if "2 yuan" in action_l:
        changes.append("the 2-yuan ticket option is selected")
    if "1 sheet" in action_l:
        changes.append("the one-ticket option is selected")
    if "identity card button" in action_l:
        changes.append("identity card verification page becomes visible")
    if "confirm" in action_l:
        changes.append("the confirmation state or next page becomes visible")
    if "elevator button" in action_l or "click the button" in action_l:
        changes.append("the elevator call button lights up")
    if "sixth floor" in action_l or "tenth floor" in action_l or "11th floor" in action_l or "eleventh floor" in action_l:
        changes.append("the selected floor button lights up")

    return list(dict.fromkeys(changes))


def infer_expected_verification(
    verification_text: Optional[str],
    env_text: Optional[str],
    final_text: Optional[str],
) -> Tuple[List[str], List[str]]:
    ver_ui: List[str] = []
    ver_physical: List[str] = []
    verification_l = (verification_text or "").lower()

    if "screen has switched" in verification_l:
        ver_ui.append("the screen shows the expected next page or state")
    if "door is open" in verification_l or "door open" in verification_l:
        ver_physical.append("the elevator door is open")
    if "door is closed" in verification_l or "door closed" in verification_l:
        ver_physical.append("the elevator door is closed")
    if "ticket counter" in verification_l:
        if env_text and "receipt" in env_text.lower():
            ver_physical.append("the receipt is visible at the ticket or receipt counter")
        else:
            ver_physical.append("the ticket is visible at the ticket counter")
    if "did you get the subway ticket" in verification_l:
        ver_physical.append("the subway ticket is in hand")
    if "check the ticket counter" in verification_l:
        ver_physical.append("the receipt or output item appears at the counter")
    if "feel the wind" in verification_l:
        ver_physical.append("airflow is perceptible")

    if final_text:
        final_l = final_text.lower()
        if "floor" in final_l:
            ver_physical.append(final_text)
        if "ticket" in final_l or "receipt" in final_l:
            ver_physical.append(final_text)
        if "successful" in final_l or "completed" in final_l:
            ver_ui.append(final_text)

    return list(dict.fromkeys(ver_ui)), list(dict.fromkeys(ver_physical))


def infer_expected_physical_changes(env_text: Optional[str], final_text: Optional[str]) -> List[str]:
    changes: List[str] = []
    if env_text:
        changes.append(env_text)
    if final_text and final_text not in changes:
        changes.append(final_text)
    return changes


def infer_retry_samples(*_: Any) -> List[Dict[str, Any]]:
    # No explicit retry semantics appear in the current Action_1 annotations.
    return []


def build_source_span(seg: Optional[Segment]) -> Optional[Dict[str, Any]]:
    if not seg:
        return None
    return {"label": seg.label, "start": seg.start, "end": seg.end, "text": seg.text}


def get_next_segment(segments: List[Segment], start_index: int, allowed_labels: Tuple[str, ...]) -> Optional[Segment]:
    for seg in segments[start_index + 1 :]:
        if seg.label in allowed_labels:
            return seg
    return None


def find_matching_verification(action_seg: Segment, verifications: List[Segment]) -> Optional[Segment]:
    for seg in verifications:
        if seg.start == action_seg.start and seg.end == action_seg.end:
            return seg
    for seg in verifications:
        if seg.start == action_seg.start:
            return seg
    return None


def extract_frame(video_path: Path, timestamp_sec: float, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe,
        "-y",
        "-ss",
        f"{max(timestamp_sec, 0.0):.3f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to extract frame from {video_path} at {timestamp_sec:.3f}s: {result.stderr}"
        )


def make_prompt_en_state_transition(device_family: str, goal_text: str, action_step: str) -> str:
    device_label = humanize_identifier(device_family)
    return (
        f"Given this first-person image of a {device_label}, generate a short video "
        f"showing the next state transition for the task: {goal_text}. "
        f"The next action is {action_step}. Show the expected UI changes and physical changes naturally."
    )


def make_prompt_zh_state_transition(device_family: str, goal_text: str, action_step: str) -> str:
    device_label = humanize_identifier(device_family)
    return (
        f"给定这张来自{device_label}的第一人称图像，请生成一段短视频，展示任务“{goal_text}”在下一步会发生的状态转移。"
        f"下一步动作是“{action_step}”。请自然体现预期的界面变化和物理环境变化。"
    )


def make_prompt_en_final(goal_text: str, success_condition: str) -> str:
    return (
        f"Given this first-person image, generate a short video showing the successful completion of the task: "
        f"{goal_text}. The video should end when the success condition is satisfied: {success_condition}."
    )


def make_prompt_zh_final(goal_text: str, success_condition: str) -> str:
    return (
        f"给定这张第一人称图像，请生成一段短视频，展示任务“{goal_text}”成功完成的过程。"
        f"视频应在满足成功条件“{success_condition}”时结束。"
    )


def make_prompt_en_verification(goal_text: str) -> str:
    return (
        f"Given this first-person image, generate a short video showing what should be observed when verifying "
        f"whether the task {goal_text} has succeeded. Emphasize the relevant UI and physical evidence."
    )


def make_prompt_zh_verification(goal_text: str) -> str:
    return (
        f"给定这张第一人称图像，请生成一段短视频，展示在验证任务“{goal_text}”是否成功时，应该观察到的结果。"
        f"请突出关键的界面证据和物理证据。"
    )


def make_prompt_en_recovery(goal_text: str, correction_action: str, post_fix_state: str) -> str:
    return (
        f"Given this first-person image showing an incorrect state during the task {goal_text}, generate a short "
        f"recovery video where the mistake is corrected by {correction_action}, the system returns to "
        f"{post_fix_state}, and then proceeds toward successful completion."
    )


def make_prompt_zh_recovery(goal_text: str, correction_action: str, post_fix_state: str) -> str:
    return (
        f"给定这张展示任务“{goal_text}”中错误状态的第一人称图像，请生成一段短视频，展示如何通过“{correction_action}”修正错误，"
        f"使系统回到“{post_fix_state}”，并继续走向正确完成。"
    )


def make_prompt_en_retry(goal_text: str, retry_trigger_type: str, retry_action: str, post_retry_state: str) -> str:
    return (
        f"Given this first-person image, generate a short video where the same task {goal_text} is retried because "
        f"{retry_trigger_type}. Show the retry action {retry_action} and the resulting state {post_retry_state}."
    )


def make_prompt_zh_retry(goal_text: str, retry_trigger_type: str, retry_action: str, post_retry_state: str) -> str:
    return (
        f"给定这张第一人称图像，请生成一段短视频，展示任务“{goal_text}”因为“{retry_trigger_type}”而进行再次尝试。"
        f"请表现重试动作“{retry_action}”以及产生的状态“{post_retry_state}”。"
    )


def base_sample_dict(
    sample_id: str,
    source_video: str,
    task_type: str,
    device_family: str,
    task_intent: str,
    goal_text: str,
    goal_slots: Dict[str, Any],
    anchor_frame_time: float,
    anchor_frame_path: str,
    prompt_en: str,
    prompt_zh: str,
    expected_ui_changes: List[str],
    expected_physical_changes: List[str],
    expected_verification_ui: List[str],
    expected_verification_physical: List[str],
    expected_final_state: Optional[str],
    source_spans: List[Dict[str, Any]],
    quality_flags: List[str],
    error_action: Optional[str] = None,
    error_state: Optional[str] = None,
    correction_action: Optional[str] = None,
    post_fix_state: Optional[str] = None,
    retry_trigger_type: Optional[str] = None,
    retry_outcome: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "sample_id": sample_id,
        "source_video": source_video,
        "task_type": task_type,
        "device_family": device_family,
        "task_intent": task_intent,
        "goal_text": goal_text,
        "goal_slots": goal_slots,
        "anchor_frame_time": round(anchor_frame_time, 3),
        "anchor_frame_path": anchor_frame_path,
        "prompt_en": prompt_en,
        "prompt_zh": prompt_zh,
        "expected_ui_changes": expected_ui_changes,
        "expected_physical_changes": expected_physical_changes,
        "expected_verification_ui": expected_verification_ui,
        "expected_verification_physical": expected_verification_physical,
        "expected_final_state": expected_final_state,
        "error_action": error_action,
        "error_state": error_state,
        "correction_action": correction_action,
        "post_fix_state": post_fix_state,
        "retry_trigger_type": retry_trigger_type,
        "retry_outcome": retry_outcome,
        "source_spans": source_spans,
        "quality_flags": sorted(set(quality_flags)),
    }


def build_samples_for_item(
    item: Dict[str, Any],
    input_dir: Path,
    frame_dir: Path,
) -> List[Dict[str, Any]]:
    video_name = Path(item.get("data", {}).get("Action_1", "")).name or f"{item.get('id')}.mp4"
    local_video = find_local_video_path(video_name, input_dir)
    meta = item.get("data", {}).get("meta", {}).get("Action_1", {})
    fps = float(meta.get("fps") or 30.0)
    segments = parse_segments(item)

    by_label: Dict[str, List[Segment]] = defaultdict(list)
    for seg in segments:
        by_label[seg.label].append(seg)

    demands = [seg.text for seg in by_label["Demand"] if seg.text]
    summaries = infer_summary_labels(segments)
    main_task = choose_main_task(demands, summaries)
    target_object = by_label["Target object"][0].text if by_label["Target object"] else ""
    initial_positions = [seg.text for seg in by_label["Initial position"] if seg.text]
    execute_actions = by_label["Execute action"]
    verification_actions = by_label["Verification action"]
    env_statuses = by_label["Environmental status"]
    final_conditions = by_label["Physical environmental condition"]
    incorrect_actions = by_label["Incorrect action"]
    correction_actions = by_label["Correction action"]

    device_family = infer_device_family(target_object, main_task)
    task_intent = infer_task_intent(main_task, target_object, demands)
    goal_slots = infer_goal_slots(main_task, demands, initial_positions, execute_actions)
    final_state_text = final_conditions[-1].text if final_conditions else None
    success_condition = infer_success_condition(task_intent, final_state_text, goal_slots)

    base_quality_flags: List[str] = []
    if not goal_slots:
        base_quality_flags.append("missing_goal_slot")

    samples: List[Dict[str, Any]] = []
    sample_counter = 1

    def next_sample_id(task_type: str) -> str:
        nonlocal sample_counter
        sid = f"{Path(video_name).stem}_{task_type}_{sample_counter:03d}"
        sample_counter += 1
        return sid

    def create_anchor_frame(sample_id: str, timestamp_sec: float) -> str:
        anchor_path = frame_dir / f"{sample_id}.jpg"
        extract_frame(local_video, timestamp_sec, anchor_path)
        return str(anchor_path.relative_to(input_dir.parent)).replace("\\", "/")

    # State transition samples from Execute action segments.
    for idx, action_seg in enumerate(execute_actions):
        action_text = action_seg.text
        verification_seg = find_matching_verification(action_seg, verification_actions)
        next_state_seg = get_next_segment(
            segments,
            segments.index(action_seg),
            ("Environmental status", "Physical environmental condition"),
        )
        env_text = next_state_seg.text if next_state_seg and next_state_seg.label == "Environmental status" else None
        final_text = next_state_seg.text if next_state_seg and next_state_seg.label == "Physical environmental condition" else None
        ver_ui, ver_physical = infer_expected_verification(
            verification_seg.text if verification_seg else None,
            env_text,
            final_text,
        )
        expected_ui_changes = infer_ui_changes_from_action(
            action_text,
            verification_seg.text if verification_seg else None,
        )
        expected_physical_changes = infer_expected_physical_changes(env_text, final_text)

        quality_flags = list(base_quality_flags)
        if any(pattern in action_text.lower() for pattern in GENERIC_ACTION_PATTERNS):
            quality_flags.append("generic_action_text")
        if verification_seg and any(
            pattern in verification_seg.text.lower() for pattern in GENERIC_VERIFICATION_PATTERNS
        ):
            quality_flags.append("generic_verification_text")

        anchor_time = (action_seg.start or 0) / fps
        sample_id = next_sample_id("state_transition_video")
        anchor_frame_path = create_anchor_frame(sample_id, anchor_time)
        prompt_en = make_prompt_en_state_transition(device_family, main_task, action_text)
        prompt_zh = make_prompt_zh_state_transition(device_family, main_task, action_text)
        source_spans = [
            span
            for span in (
                build_source_span(action_seg),
                build_source_span(verification_seg),
                build_source_span(next_state_seg),
            )
            if span
        ]

        samples.append(
            base_sample_dict(
                sample_id=sample_id,
                source_video=str(local_video.relative_to(input_dir.parent)).replace("\\", "/"),
                task_type="state_transition_video",
                device_family=device_family,
                task_intent=task_intent,
                goal_text=main_task,
                goal_slots=goal_slots,
                anchor_frame_time=anchor_time,
                anchor_frame_path=anchor_frame_path,
                prompt_en=prompt_en,
                prompt_zh=prompt_zh,
                expected_ui_changes=expected_ui_changes,
                expected_physical_changes=expected_physical_changes,
                expected_verification_ui=ver_ui,
                expected_verification_physical=ver_physical,
                expected_final_state=final_state_text,
                source_spans=source_spans,
                quality_flags=quality_flags,
            )
        )

    # Final-state sample: one per video when possible.
    if final_conditions:
        anchor_seg = by_label["Initial position"][0] if by_label["Initial position"] else segments[0]
        last_ver = verification_actions[-1] if verification_actions else None
        env_text = env_statuses[-1].text if env_statuses else None
        ver_ui, ver_physical = infer_expected_verification(
            last_ver.text if last_ver else None,
            env_text,
            final_conditions[-1].text,
        )
        quality_flags = list(base_quality_flags)
        if last_ver and any(pattern in last_ver.text.lower() for pattern in GENERIC_VERIFICATION_PATTERNS):
            quality_flags.append("generic_verification_text")

        anchor_time = (anchor_seg.start or 0) / fps
        sample_id = next_sample_id("final_state_video")
        anchor_frame_path = create_anchor_frame(sample_id, anchor_time)
        source_spans = [
            span
            for span in (
                build_source_span(anchor_seg),
                build_source_span(last_ver),
                build_source_span(final_conditions[-1]),
            )
            if span
        ]
        samples.append(
            base_sample_dict(
                sample_id=sample_id,
                source_video=str(local_video.relative_to(input_dir.parent)).replace("\\", "/"),
                task_type="final_state_video",
                device_family=device_family,
                task_intent=task_intent,
                goal_text=main_task,
                goal_slots=goal_slots,
                anchor_frame_time=anchor_time,
                anchor_frame_path=anchor_frame_path,
                prompt_en=make_prompt_en_final(main_task, success_condition),
                prompt_zh=make_prompt_zh_final(main_task, success_condition),
                expected_ui_changes=[],
                expected_physical_changes=infer_expected_physical_changes(env_text, final_conditions[-1].text),
                expected_verification_ui=ver_ui,
                expected_verification_physical=ver_physical,
                expected_final_state=final_conditions[-1].text,
                source_spans=source_spans,
                quality_flags=quality_flags,
            )
        )

    # Verification-state samples: one per verification segment.
    for verification_seg in verification_actions:
        next_state_seg = get_next_segment(
            segments,
            segments.index(verification_seg),
            ("Environmental status", "Physical environmental condition"),
        )
        env_text = next_state_seg.text if next_state_seg and next_state_seg.label == "Environmental status" else None
        final_text = next_state_seg.text if next_state_seg and next_state_seg.label == "Physical environmental condition" else final_state_text
        ver_ui, ver_physical = infer_expected_verification(verification_seg.text, env_text, final_text)
        quality_flags = list(base_quality_flags)
        if any(pattern in verification_seg.text.lower() for pattern in GENERIC_VERIFICATION_PATTERNS):
            quality_flags.append("generic_verification_text")

        anchor_time = (verification_seg.start or 0) / fps
        sample_id = next_sample_id("verification_state_video")
        anchor_frame_path = create_anchor_frame(sample_id, anchor_time)
        source_spans = [
            span
            for span in (
                build_source_span(verification_seg),
                build_source_span(next_state_seg),
            )
            if span
        ]
        samples.append(
            base_sample_dict(
                sample_id=sample_id,
                source_video=str(local_video.relative_to(input_dir.parent)).replace("\\", "/"),
                task_type="verification_state_video",
                device_family=device_family,
                task_intent=task_intent,
                goal_text=main_task,
                goal_slots=goal_slots,
                anchor_frame_time=anchor_time,
                anchor_frame_path=anchor_frame_path,
                prompt_en=make_prompt_en_verification(main_task),
                prompt_zh=make_prompt_zh_verification(main_task),
                expected_ui_changes=[],
                expected_physical_changes=infer_expected_physical_changes(env_text, final_text),
                expected_verification_ui=ver_ui,
                expected_verification_physical=ver_physical,
                expected_final_state=final_text,
                source_spans=source_spans,
                quality_flags=quality_flags,
            )
        )

    # Recovery sample: one per video when both error and correction exist.
    if incorrect_actions and correction_actions:
        error_seg = incorrect_actions[0]
        correction_seg = correction_actions[0]
        next_final = final_conditions[-1] if final_conditions else None
        quality_flags = list(base_quality_flags)
        quality_flags.append("inferred_error_state")
        quality_flags.append("inferred_post_fix_state")
        if verification_actions and any(
            pattern in verification_actions[-1].text.lower() for pattern in GENERIC_VERIFICATION_PATTERNS
        ):
            quality_flags.append("generic_verification_text")

        error_state = f"state after error: {error_seg.text}"
        post_fix_state = f"state after correction: {correction_seg.text}"
        anchor_time = (error_seg.end or error_seg.start or 0) / fps
        sample_id = next_sample_id("recovery_video")
        anchor_frame_path = create_anchor_frame(sample_id, anchor_time)
        source_spans = [
            span
            for span in (
                build_source_span(error_seg),
                build_source_span(correction_seg),
                build_source_span(next_final),
            )
            if span
        ]
        samples.append(
            base_sample_dict(
                sample_id=sample_id,
                source_video=str(local_video.relative_to(input_dir.parent)).replace("\\", "/"),
                task_type="recovery_video",
                device_family=device_family,
                task_intent=task_intent,
                goal_text=main_task,
                goal_slots=goal_slots,
                anchor_frame_time=anchor_time,
                anchor_frame_path=anchor_frame_path,
                prompt_en=make_prompt_en_recovery(main_task, correction_seg.text, post_fix_state),
                prompt_zh=make_prompt_zh_recovery(main_task, correction_seg.text, post_fix_state),
                expected_ui_changes=[],
                expected_physical_changes=infer_expected_physical_changes(
                    env_statuses[-1].text if env_statuses else None,
                    next_final.text if next_final else None,
                ),
                expected_verification_ui=[],
                expected_verification_physical=[],
                expected_final_state=next_final.text if next_final else final_state_text,
                source_spans=source_spans,
                quality_flags=quality_flags,
                error_action=error_seg.text,
                error_state=error_state,
                correction_action=correction_seg.text,
                post_fix_state=post_fix_state,
            )
        )

    # Retry samples: current annotations do not explicitly support them.
    samples.extend(infer_retry_samples())
    return samples


def write_jsonl(output_path: Path, samples: List[Dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def write_summary(output_path: Path, samples: List[Dict[str, Any]]) -> None:
    task_counter = Counter(sample["task_type"] for sample in samples)
    video_counter = Counter(sample["source_video"] for sample in samples)
    flag_counter = Counter()
    for sample in samples:
        flag_counter.update(sample["quality_flags"])

    lines: List[str] = []
    lines.append("# Single-Image Generation Dataset Summary")
    lines.append("")
    lines.append(f"- Samples generated: `{len(samples)}`")
    lines.append(f"- Task types generated: `{len(task_counter)}`")
    lines.append("")
    lines.append("## Task Type Counts")
    lines.append("")
    for task_type in (
        "state_transition_video",
        "final_state_video",
        "verification_state_video",
        "recovery_video",
        "retry_video",
    ):
        lines.append(f"- `{task_type}`: `{task_counter.get(task_type, 0)}`")
    lines.append("")
    lines.append("## Video Coverage")
    lines.append("")
    for source_video, count in sorted(video_counter.items()):
        lines.append(f"- `{source_video}`: `{count}`")
    lines.append("")
    lines.append("## Quality Flag Counts")
    lines.append("")
    if flag_counter:
        for flag, count in flag_counter.most_common():
            lines.append(f"- `{flag}`: `{count}`")
    else:
        lines.append("- No quality flags were added.")
    lines.append("")
    lines.append("## Evaluation Dimensions")
    lines.append("")
    lines.append("Each generated video should be scored on a 0-5 scale for the following dimensions:")
    lines.append("")
    lines.append("1. `Task Alignment` (weight 25)")
    lines.append("   - 5: Fully matches `goal_text` and `goal_slots`.")
    lines.append("   - 3: Mostly correct, but one key goal parameter is wrong.")
    lines.append("   - 1: Only the scene or device is preserved; the task drifts.")
    lines.append("   - 0: Completely unrelated.")
    lines.append("")
    lines.append("2. `Action Plausibility` (weight 20)")
    lines.append("   - 5: The action order, object interaction, and physical logic are all plausible.")
    lines.append("   - 3: Mostly plausible, but with minor skips or awkwardness.")
    lines.append("   - 1: Clearly inconsistent with device interaction logic.")
    lines.append("   - 0: The action is unrelated or uninterpretable.")
    lines.append("")
    lines.append("3. `State Transition Correctness` (weight 20)")
    lines.append("   - Score both `UI transition` and `Physical transition` on 0-5, then average them.")
    lines.append("")
    lines.append("4. `Verification Evidence Sufficiency` (weight 20)")
    lines.append("   - 5: Key verification indicators are clearly visible.")
    lines.append("   - 3: Partial indicators are shown, but evidence is weak.")
    lines.append("   - 1: Only vague changes are visible.")
    lines.append("   - 0: The result cannot be verified.")
    lines.append("")
    lines.append("5. `Video Quality & Temporal Coherence` (weight 15)")
    lines.append("   - 5: Stable subject, coherent motion, visually usable.")
    lines.append("   - 3: Minor artifacts, still judgeable.")
    lines.append("   - 1: Severe temporal breakage.")
    lines.append("   - 0: Not usable.")
    lines.append("")
    lines.append("## Weighted Score Formula")
    lines.append("")
    lines.append("- `Task Alignment`: 25")
    lines.append("- `Action Plausibility`: 20")
    lines.append("- `State Transition Correctness`: 20")
    lines.append("- `Verification Evidence Sufficiency`: 20")
    lines.append("- `Video Quality & Temporal Coherence`: 15")
    lines.append("")
    lines.append("Total score is computed on a weighted 100-point scale.")
    lines.append("")
    lines.append("Auxiliary metrics:")
    lines.append("- `Pass@Task`: ratio of samples with Task Alignment >= 4")
    lines.append("- `Pass@Verify`: ratio of samples with Verification Evidence Sufficiency >= 4")
    lines.append("")
    lines.append("## Example Samples")
    lines.append("")
    for sample in samples[:10]:
        lines.append(f"### {sample['sample_id']}")
        lines.append("")
        lines.append(f"- Task type: `{sample['task_type']}`")
        lines.append(f"- Source video: `{sample['source_video']}`")
        lines.append(f"- Anchor frame: `{sample['anchor_frame_path']}` @ `{sample['anchor_frame_time']}`s")
        lines.append(f"- Goal: `{sample['goal_text']}`")
        lines.append(f"- Prompt EN: {sample['prompt_en']}")
        if sample["quality_flags"]:
            lines.append(f"- Quality flags: `{', '.join(sample['quality_flags'])}`")
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def resolve_default_input(input_dir: Path) -> Path:
    candidates = [
        p
        for p in input_dir.glob("*Action_1.json")
        if not p.name.endswith(".qa_candidates.json")
    ]
    if not candidates:
        raise FileNotFoundError(f"Could not find Action_1 JSON under {input_dir}")
    return sorted(candidates)[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a single-image + prompt generation dataset from SWITCH Action_1 annotations."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("30fps"),
        help="Directory containing the Action_1 JSON and local videos.",
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        default=None,
        help="Optional explicit path to the Action_1 Label Studio JSON export.",
    )
    parser.add_argument(
        "--qa-input",
        type=Path,
        default=None,
        help="Optional QA candidate JSON. Reserved for future enrichment; not used as primary truth.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path("30fps") / "single_image_gen_dataset.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("30fps") / "single_image_gen_dataset_summary.md",
        help="Output Markdown summary path.",
    )
    parser.add_argument(
        "--frame-dir",
        type=Path,
        default=Path("30fps") / "single_image_frames",
        help="Directory for extracted anchor frames.",
    )
    args = parser.parse_args()

    input_json = args.input_json or resolve_default_input(args.input_dir)
    with input_json.open("r", encoding="utf-8") as f:
        items = json.load(f)

    all_samples: List[Dict[str, Any]] = []
    for item in items:
        all_samples.extend(build_samples_for_item(item, args.input_dir, args.frame_dir))

    write_jsonl(args.output_jsonl, all_samples)
    write_summary(args.output_md, all_samples)

    print(f"Wrote JSONL: {args.output_jsonl}")
    print(f"Wrote summary: {args.output_md}")
    print(f"Generated samples: {len(all_samples)}")


if __name__ == "__main__":
    main()
