#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2


PROMPT_VERSION = "0424_generation_prompt_review_v2_action_conditioned"

PRE_ACTION_INPUT_CLIP_FRAMES = 45

IMAGE_SAMPLE_IDS = [
    "011_action_004",
    "013_action_006",
    "051_action_004",
    "052_action_008",
    "015_action_007",
    "027_action_005",
    "035_action_004",
    "022_action_004",
    "014_action_003",
    "078_action_003",
]

VIDEO_SAMPLE_IDS = [
    "012_action_004",
    "030_action_007",
    "035_action_003",
    "014_action_005",
    "078_action_005",
]

SELECTED_SAMPLE_IDS = VIDEO_SAMPLE_IDS + IMAGE_SAMPLE_IDS

SCENE_INVARIANTS = [
    "keep the same first-person viewpoint",
    "preserve the same device structure and layout",
    "preserve the same object and hand identity",
]

SCENE_INVARIANTS_ZH = [
    "保持相同的第一人称视角",
    "保持相同的设备结构与布局",
    "保持相同的物体与手部身份",
]

FORBIDDEN_CHANGES = [
    "do not redesign the UI layout",
    "do not invent new buttons, screens, or objects",
    "do not jump to a different camera viewpoint",
]

FORBIDDEN_CHANGES_ZH = [
    "不要重新设计界面布局",
    "不要虚构新的按钮、屏幕或无关物体",
    "不要跳到不同的摄像机视角",
]

TEXT_REPLACEMENTS = {
    "floorr": "floor",
    "The elevator door close": "The elevator door closes",
    "The elevator door close.": "The elevator door closes.",
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

ACTION_LABELS = {
    "action-type",
    "action_requirement",
    "action_description",
    "action_step_id",
}

UI_LABELS = {"ui_change", "ui_state"}
PHYSICAL_LABELS = {"physical_world_change", "physical_world_state"}
EVIDENCE_LABELS = UI_LABELS | PHYSICAL_LABELS

DEVICE_LABELS_EN = {
    "elevator_system": "elevator system",
    "subway_ticket_machine": "subway ticket machine",
    "hospital_registration_machine": "hospital self-service machine",
}

DEVICE_LABELS_ZH = {
    "elevator_system": "电梯系统",
    "subway_ticket_machine": "地铁售票机",
    "hospital_registration_machine": "医院自助机",
}

SELECTION_NOTES = {
    "012_action_004": {
        "selection_reason": "Medical kiosk sample with strong UI and physical evidence after the next action.",
        "watch_for": [
            "Whether the prompt clearly says the output must first show the next action, not skip to the state.",
            "Whether the mixed UI and physical evidence remains easy to follow.",
        ],
    },
    "030_action_007": {
        "selection_reason": "Medical kiosk department-selection action with a short and clean UI outcome.",
        "watch_for": [
            "Whether the stop condition is local enough.",
            "Whether the prompt overstates unsupported physical evidence.",
        ],
    },
    "035_action_003": {
        "selection_reason": "Subway ticket machine screen-selection action with immediate UI feedback.",
        "watch_for": [
            "Whether the screen transition is described specifically enough.",
            "Whether the prompt stays local and does not drift toward coin insertion or ticket pickup.",
        ],
    },
    "014_action_005": {
        "selection_reason": "Elevator in-cabin button press with a longer physical continuation after the action.",
        "watch_for": [
            "Whether the action-result chain remains coherent across door and floor changes.",
            "Whether the stop condition stays at the local floor-arrival result rather than the whole task narrative.",
        ],
    },
    "078_action_005": {
        "selection_reason": "The only scenario_family=other action sample, useful for checking atypical scene grouping.",
        "watch_for": [
            "Whether device-family inference still makes sense under the other label.",
            "Whether the prompt remains concrete despite the unusual scenario grouping.",
        ],
    },
    "011_action_004": {
        "selection_reason": "Simple medical kiosk navigation action with a compact next-page result.",
        "watch_for": [
            "Whether the image-to-video prompt makes the next action explicit.",
            "Whether the output is constrained to the immediate page-entry result.",
        ],
    },
    "013_action_006": {
        "selection_reason": "Medical kiosk cancellation action with a clear page-jump result.",
        "watch_for": [
            "Whether the prompt avoids leaking later cancellation steps.",
            "Whether the result page is treated as the stop condition rather than the full cancellation success.",
        ],
    },
    "051_action_004": {
        "selection_reason": "Medical kiosk action with both device-side UI and user-side physical evidence.",
        "watch_for": [
            "Whether the physical interaction is described as required evidence rather than optional flavor.",
            "Whether the prompt remains visually grounded.",
        ],
    },
    "052_action_008": {
        "selection_reason": "Medical kiosk department-choice action with a concise doctor-list outcome.",
        "watch_for": [
            "Whether the next action is concrete enough for generation.",
            "Whether the evidence list is specific but not redundant.",
        ],
    },
    "015_action_007": {
        "selection_reason": "Subway ticket machine ticket-pickup action emphasizing physical-world completion.",
        "watch_for": [
            "Whether a physical-only result still gives enough supervision.",
            "Whether the prompt avoids hallucinating extra UI evidence.",
        ],
    },
    "027_action_005": {
        "selection_reason": "Subway ticket machine confirmation tap leading to a compact screen state.",
        "watch_for": [
            "Whether the prompt captures the one-step UI result cleanly.",
            "Whether it avoids dragging in later coin-insertion actions.",
        ],
    },
    "035_action_004": {
        "selection_reason": "Single-frame subway ticket machine action that is easy to inspect as an image anchor.",
        "watch_for": [
            "Whether the action is still explicit enough from a static pre-action frame.",
            "Whether the screen result is described in a way a model can actually realize.",
        ],
    },
    "022_action_004": {
        "selection_reason": "Elevator enter-and-go-up action with multiple subsequent cabin cues.",
        "watch_for": [
            "Whether the temporal stages are ordered clearly.",
            "Whether the prompt becomes too long for a single transition sample.",
        ],
    },
    "014_action_003": {
        "selection_reason": "Elevator hall-call action with a short and visually clear door-opening result.",
        "watch_for": [
            "Whether the local stop condition is precise enough.",
            "Whether the prompt avoids implying cabin-entry actions that belong to the next step.",
        ],
    },
    "078_action_003": {
        "selection_reason": "Other-labeled elevator hall-call action for cross-checking scene generalization.",
        "watch_for": [
            "Whether the prompt remains grounded even when the scenario label is atypical.",
            "Whether the direct result is strong enough without UI evidence.",
        ],
    },
}


@dataclass
class Segment:
    label: str
    text: str
    start: Optional[int]
    end: Optional[int]


@dataclass
class ActionEvent:
    action_type: str
    action_requirement: str
    action_description: str
    step_id: str
    start: Optional[int]
    end: Optional[int]


@dataclass
class VideoTimeline:
    data_id: str
    video_name: str
    fps: float
    total_frames: int
    segments: List[Segment]
    action_events: List[ActionEvent]
    local_video_path: Path


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def clean_text(text: Optional[str]) -> str:
    cleaned = normalize_spaces(text or "")
    for src, dst in TEXT_REPLACEMENTS.items():
        cleaned = cleaned.replace(src, dst)
    cleaned = re.sub(r"\s+\.", ".", cleaned)
    if cleaned and cleaned[0].isalpha():
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned


def normalize_action_type(text: str) -> str:
    normalized = normalize_spaces(text).lower()
    return ACTION_TYPE_MAP.get(normalized, normalized)


def get_meta_text(result: Dict[str, Any]) -> str:
    meta = result.get("meta") or {}
    value = meta.get("text")
    if isinstance(value, list) and value:
        return str(value[0])
    if isinstance(value, str):
        return value
    return ""


def get_first_range(value: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    ranges = value.get("ranges") or []
    if not ranges:
        return None, None
    first = ranges[0]
    start = first.get("start")
    end = first.get("end")
    return int(start) if start is not None else None, int(end) if end is not None else None


def infer_video_name(item: Dict[str, Any]) -> str:
    data = item.get("data") or {}
    for _, value in data.items():
        if isinstance(value, str) and value.lower().endswith(".mp4"):
            return Path(value).name
    return f"{item.get('id')}.mp4"


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
    segments.sort(key=lambda seg: ((seg.start or 0), (seg.end or 0), seg.label, seg.text))
    return segments


def build_action_events(segments: Iterable[Segment]) -> List[ActionEvent]:
    grouped: Dict[Tuple[Optional[int], Optional[int]], Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    for segment in segments:
        if segment.label not in ACTION_LABELS:
            continue
        grouped[(segment.start, segment.end)][segment.label].append(segment.text)

    events: List[ActionEvent] = []
    for (start, end), payload in grouped.items():
        action_type = normalize_action_type((payload.get("action-type") or [""])[0])
        action_requirement = clean_text((payload.get("action_requirement") or [""])[0])
        action_description = clean_text((payload.get("action_description") or [""])[0]) or action_requirement
        step_id = normalize_spaces((payload.get("action_step_id") or [""])[0])
        events.append(
            ActionEvent(
                action_type=action_type,
                action_requirement=action_requirement,
                action_description=action_description,
                step_id=step_id,
                start=start,
                end=end,
            )
        )

    def sort_key(event: ActionEvent) -> Tuple[int, int, int]:
        try:
            step_num = int(event.step_id)
        except (TypeError, ValueError):
            step_num = 10**9
        return (event.start or 0, event.end or 0, step_num)

    events.sort(key=sort_key)
    return events


def load_timelines(annotation_root: Path) -> Dict[str, VideoTimeline]:
    raw_path = annotation_root / "SWITCHAction_2.json"
    items = json.loads(raw_path.read_text(encoding="utf-8"))
    timelines: Dict[str, VideoTimeline] = {}
    for item in items:
        segments = parse_segments(item)
        if not segments:
            continue
        by_label: Dict[str, List[Segment]] = defaultdict(list)
        for segment in segments:
            by_label[segment.label].append(segment)
        data_id = (by_label.get("data_id") or [Segment("data_id", Path(infer_video_name(item)).stem, 0, 0)])[0].text
        video_name = infer_video_name(item)
        media_meta = next(
            (
                value
                for value in (item.get("data", {}).get("meta") or {}).values()
                if isinstance(value, dict) and "fps" in value
            ),
            {},
        )
        timelines[data_id] = VideoTimeline(
            data_id=data_id,
            video_name=video_name,
            fps=float(media_meta.get("fps") or 30.0),
            total_frames=int(media_meta.get("total_frames") or 0),
            segments=segments,
            action_events=build_action_events(segments),
            local_video_path=annotation_root / video_name,
        )
    return timelines


def load_inventory(annotation_root: Path) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    payload = json.loads((annotation_root / "SWITCHAction_2.qa_candidates.json").read_text(encoding="utf-8"))
    videos: Dict[str, Dict[str, Any]] = {}
    candidates: Dict[str, Dict[str, Any]] = {}
    by_video: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"candidates": []})
    for video in payload["videos"]:
        video_info = {
            "data_id": video["data_id"],
            "video_name": video["video_name"],
            "scenario_family": video["scenario_family"],
            "main_task": clean_text(video["main_task"]),
            "main_verification": clean_text(video["main_verification"]),
        }
        videos[video["data_id"]] = video_info
        grouped = {"video_info": video_info, "candidates": []}
        for qa in video["qa_candidates"]:
            span = qa.get("source_span") or {}
            candidate = {
                "qa_id": qa["qa_id"],
                "data_id": video["data_id"],
                "task_family": qa["task_family"],
                "qa_type": qa["qa_type"],
                "source_label": qa["source_label"],
                "source_span": {
                    "start": int(span["start"]) if span.get("start") is not None else None,
                    "end": int(span["end"]) if span.get("end") is not None else None,
                },
                "answer": clean_text(qa["answer"]),
            }
            candidates[qa["qa_id"]] = candidate
            grouped["candidates"].append(candidate)
        grouped["candidates"].sort(key=lambda item: ((item["source_span"]["start"] or 0), (item["source_span"]["end"] or 0), item["qa_id"]))
        by_video[video["data_id"]] = grouped
    return videos, candidates, by_video


def modality_for_label(label: Optional[str]) -> Optional[str]:
    if label in UI_LABELS:
        return "ui"
    if label in PHYSICAL_LABELS:
        return "physical"
    return None


def unique_texts(values: Iterable[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for value in values:
        candidate = clean_text(value)
        if not candidate:
            continue
        key = candidate.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(candidate)
    return ordered


def list_or_none(values: Sequence[str]) -> Optional[List[str]]:
    cleaned = unique_texts(values)
    return cleaned or None


def add_evidence(ui_values: List[str], physical_values: List[str], label: Optional[str], text: Optional[str]) -> None:
    cleaned = clean_text(text or "")
    if not cleaned:
        return
    modality = modality_for_label(label)
    if modality == "ui":
        ui_values.append(cleaned)
    elif modality == "physical":
        physical_values.append(cleaned)


def build_required_evidence_from_segments(segments: Iterable[Segment]) -> Tuple[Optional[List[str]], Optional[List[str]]]:
    ui_values: List[str] = []
    physical_values: List[str] = []
    for segment in segments:
        add_evidence(ui_values, physical_values, segment.label, segment.text)
    return list_or_none(ui_values), list_or_none(physical_values)


def next_boundary_event(timeline: VideoTimeline, after_frame: int) -> Optional[ActionEvent]:
    for event in timeline.action_events:
        if event.start is None:
            continue
        if event.action_type not in {"execute action", "verification action"}:
            continue
        if event.start > after_frame:
            return event
    return None


def segments_between(
    timeline: VideoTimeline,
    start_after: int,
    end_before: Optional[int],
    labels: Sequence[str],
) -> List[Segment]:
    selected: List[Segment] = []
    label_set = set(labels)
    for segment in timeline.segments:
        if segment.label not in label_set or segment.start is None:
            continue
        if segment.start <= start_after:
            continue
        if end_before is not None and segment.start >= end_before:
            continue
        selected.append(segment)
    selected.sort(key=lambda segment: ((segment.start or 0), (segment.end or 0), segment.label, segment.text))
    return selected


def segments_near_frame(timeline: VideoTimeline, frame_index: int, window: int, labels: Sequence[str]) -> List[Segment]:
    selected: List[Segment] = []
    label_set = set(labels)
    for segment in timeline.segments:
        if segment.label not in label_set or segment.start is None:
            continue
        if abs(segment.start - frame_index) <= window:
            selected.append(segment)
    selected.sort(key=lambda segment: (abs((segment.start or frame_index) - frame_index), segment.start or 0, segment.end or 0))
    return selected


def nearest_preceding_event(timeline: VideoTimeline, frame_index: int, action_type: str) -> Optional[ActionEvent]:
    chosen: Optional[ActionEvent] = None
    for event in timeline.action_events:
        if event.action_type != action_type or event.start is None:
            continue
        if event.start <= frame_index:
            chosen = event
    return chosen


def nearest_verification_hint(timeline: VideoTimeline, frame_index: int) -> Optional[str]:
    candidates = [event for event in timeline.action_events if event.action_type == "verification action" and event.start is not None]
    if not candidates:
        return None
    candidates.sort(key=lambda event: min(abs(frame_index - (event.start or 0)), abs(frame_index - (event.end or event.start or 0))))
    return clean_text(candidates[0].action_description)


def get_final_state_frame(timeline: VideoTimeline) -> Optional[int]:
    for segment in timeline.segments:
        if segment.label == "is_final_state":
            return segment.start
    return None


def collect_final_support_evidence(
    timeline: VideoTimeline,
    primary_start: int,
    primary_label: str,
    primary_text: str,
) -> Tuple[Optional[List[str]], Optional[List[str]], List[Segment]]:
    final_state_frame = get_final_state_frame(timeline)
    windows = [(max(0, primary_start - 90), primary_start + 45)]
    if final_state_frame is not None:
        windows.append((max(0, final_state_frame - 90), final_state_frame + 45))
    support_segments: List[Segment] = []
    for segment in timeline.segments:
        if segment.label not in EVIDENCE_LABELS or segment.start is None:
            continue
        if (
            segment.label == primary_label
            and segment.start == primary_start
            and clean_text(segment.text) == clean_text(primary_text)
        ):
            continue
        for start, end in windows:
            if start <= segment.start <= end:
                support_segments.append(segment)
                break
    support_segments.sort(key=lambda segment: ((segment.start or 0), (segment.end or 0), segment.label, segment.text))
    ui_values: List[str] = []
    physical_values: List[str] = []
    add_evidence(ui_values, physical_values, primary_label, primary_text)
    for segment in support_segments:
        add_evidence(ui_values, physical_values, segment.label, segment.text)
    return list_or_none(ui_values), list_or_none(physical_values), support_segments


def build_source_span(label: str, text: str, start: Optional[int], end: Optional[int]) -> Dict[str, Any]:
    return {
        "label": label,
        "text": clean_text(text),
        "start": start,
        "end": end,
    }


def infer_device_family(scenario_family: str, main_task: str) -> str:
    if scenario_family == "elevator":
        return "elevator_system"
    if scenario_family == "subway_ticket":
        return "subway_ticket_machine"
    if scenario_family == "medical_kiosk":
        return "hospital_registration_machine"
    lowered = main_task.lower()
    if "elevator" in lowered or "floor" in lowered:
        return "elevator_system"
    if "ticket" in lowered or "subway" in lowered:
        return "subway_ticket_machine"
    return "hospital_registration_machine"


def with_indefinite_article(device_family: str) -> str:
    device_text = DEVICE_LABELS_EN.get(device_family, device_family.replace("_", " "))
    article = "an" if device_text[0].lower() in {"a", "e", "i", "o", "u"} else "a"
    return f"{article} {device_text}"


def render_inline_list(values: Optional[Sequence[str]]) -> str:
    if not values:
        return "null"
    return "; ".join(clean_text(value) for value in values if clean_text(value))


def build_prompt_en_state_transition_video(
    device_family: str,
    goal_text: str,
    next_action: Optional[str],
    required_ui: Optional[List[str]],
    required_physical: Optional[List[str]],
    stop_condition: Optional[str],
    temporal_stages: Optional[List[str]],
    input_reference_text: str,
) -> str:
    lines = [
        f"You are given an egocentric input video clip of {with_indefinite_article(device_family)}.",
        "Input-output mode: video_to_video",
        "Task type: state_transition_video",
        f"Goal: {goal_text}",
        f"The input clip already shows this action segment: {input_reference_text}",
        "Generate the next short output video that starts immediately after the input clip ends.",
        "The output video must:",
        "- preserve the same device layout, viewpoint, and object identities;",
        "- continue naturally from the end of the input clip;",
        "- show the direct visible result of the action already shown in the input clip;",
        f"- stop once this stop condition is satisfied: {stop_condition or 'null'}",
        "Required visible evidence in the output video:",
        f"- UI: {render_inline_list(required_ui)}",
        f"- Physical: {render_inline_list(required_physical)}",
        "Please:",
    ]
    lines.extend(f"- {item}" for item in SCENE_INVARIANTS)
    lines.append("Do not:")
    lines.extend(f"- {item}" for item in FORBIDDEN_CHANGES)
    lines.append("- do not replay the input action from the beginning")
    lines.append("- do not jump directly to the overall success state")
    if temporal_stages:
        lines.append("Expected post-action continuation stages:")
        lines.extend(f"- {item}" for item in temporal_stages)
    if next_action:
        lines.append(f"Next action in sequence: {next_action}")
    return "\n".join(lines)


def build_prompt_zh_state_transition_video(
    device_family: str,
    goal_text: str,
    next_action: Optional[str],
    required_ui: Optional[List[str]],
    required_physical: Optional[List[str]],
    stop_condition: Optional[str],
    temporal_stages: Optional[List[str]],
    input_reference_text: str,
) -> str:
    lines = [
        f"你将看到一段来自{DEVICE_LABELS_ZH.get(device_family, device_family)}的第一人称输入视频片段。",
        "输入输出模式：video_to_video",
        "任务类型：state_transition_video",
        f"任务目标：{goal_text}",
        f"输入片段已经展示了这段动作过程：{input_reference_text}",
        "请生成一段紧接在输入片段之后的短输出视频。",
        "输出视频必须：",
        "- 保持相同的设备布局、观察视角和对象身份；",
        "- 与输入片段结尾自然衔接；",
        "- 展示该动作带来的直接可见结果；",
        f"- 当满足以下停止条件时结束：{stop_condition or 'null'}",
        "输出视频中必须出现的可见证据：",
        f"- 界面证据：{render_inline_list(required_ui)}",
        f"- 物理证据：{render_inline_list(required_physical)}",
        "请：",
    ]
    lines.extend(f"- {item}" for item in SCENE_INVARIANTS_ZH)
    lines.append("不要：")
    lines.extend(f"- {item}" for item in FORBIDDEN_CHANGES_ZH)
    lines.append("- 不要从头重放输入片段中已经出现的动作")
    lines.append("- 不要直接跳到整个任务的全局成功状态")
    if temporal_stages:
        lines.append("建议遵循的动作后续阶段：")
        lines.extend(f"- {item}" for item in temporal_stages)
    if next_action:
        lines.append(f"当前序列中的下一步动作：{next_action}")
    return "\n".join(lines)


def build_prompt_en_action_video_to_video(
    device_family: str,
    goal_text: str,
    next_action: str,
    required_ui: Optional[List[str]],
    required_physical: Optional[List[str]],
    stop_condition: Optional[str],
    temporal_stages: Optional[List[str]],
) -> str:
    lines = [
        f"You are given an egocentric input video clip of {with_indefinite_article(device_family)}.",
        "Input-output mode: video_to_video",
        "Task type: state_transition_video",
        f"Goal: {goal_text}",
        "The input clip ends immediately before the next required action begins.",
        f"The next required action to show is: {next_action}",
        "Generate the next short output video that starts immediately after the input clip ends.",
        "The output video must:",
        "- preserve the same device layout, viewpoint, and object identities;",
        "- continue naturally from the end of the input clip;",
        f"- first show this next action: {next_action};",
        "- then show the direct visible result of that action;",
        f"- stop once this stop condition is satisfied: {stop_condition or 'null'}",
        "Required visible evidence in the output video:",
        f"- UI: {render_inline_list(required_ui)}",
        f"- Physical: {render_inline_list(required_physical)}",
        "Please:",
    ]
    lines.extend(f"- {item}" for item in SCENE_INVARIANTS)
    lines.append("Do not:")
    lines.extend(f"- {item}" for item in FORBIDDEN_CHANGES)
    lines.append("- do not jump directly to the overall success state")
    if temporal_stages:
        lines.append("Expected continuation stages:")
        lines.extend(f"- {item}" for item in temporal_stages)
    return "\n".join(lines)


def build_prompt_zh_action_video_to_video(
    device_family: str,
    goal_text: str,
    next_action: str,
    required_ui: Optional[List[str]],
    required_physical: Optional[List[str]],
    stop_condition: Optional[str],
    temporal_stages: Optional[List[str]],
) -> str:
    lines = [
        f"你将看到一段来自{DEVICE_LABELS_ZH.get(device_family, device_family)}的第一人称输入视频片段。",
        "输入输出模式：video_to_video",
        "任务类型：state_transition_video",
        f"任务目标：{goal_text}",
        "输入片段结束在下一步关键动作发生之前。",
        f"接下来必须发生的动作是：{next_action}",
        "请生成一段紧接在输入片段之后的短输出视频。",
        "输出视频必须：",
        "- 保持相同的设备布局、观察视角和对象身份；",
        "- 与输入片段结尾自然衔接；",
        f"- 先展示这个下一步动作：{next_action}；",
        "- 再展示该动作带来的直接可见结果；",
        f"- 当满足以下停止条件时结束：{stop_condition or 'null'}",
        "输出视频中必须出现的可见证据：",
        f"- 界面证据：{render_inline_list(required_ui)}",
        f"- 物理证据：{render_inline_list(required_physical)}",
        "请：",
    ]
    lines.extend(f"- {item}" for item in SCENE_INVARIANTS_ZH)
    lines.append("不要：")
    lines.extend(f"- {item}" for item in FORBIDDEN_CHANGES_ZH)
    lines.append("- 不要直接跳到整个任务的全局成功状态")
    if temporal_stages:
        lines.append("建议遵循的后续阶段：")
        lines.extend(f"- {item}" for item in temporal_stages)
    return "\n".join(lines)


def build_prompt_en_action_image_to_video(
    device_family: str,
    goal_text: str,
    next_action: str,
    required_ui: Optional[List[str]],
    required_physical: Optional[List[str]],
    stop_condition: Optional[str],
    temporal_stages: Optional[List[str]],
) -> str:
    lines = [
        f"You are given an egocentric input image of {with_indefinite_article(device_family)}.",
        "Input-output mode: image_to_video",
        "Task type: state_transition_video",
        f"Goal: {goal_text}",
        "This image is a pre-action anchor frame taken immediately before the next required action.",
        f"The next required action to show is: {next_action}",
        "Generate a short output video consistent with the same scene.",
        "The output video must:",
        "- preserve the same device layout, viewpoint, and object identities;",
        f"- first show this next action: {next_action};",
        "- then show the direct visible result of that action;",
        f"- stop once this stop condition is satisfied: {stop_condition or 'null'}",
        "Required visible evidence before the video ends:",
        f"- UI: {render_inline_list(required_ui)}",
        f"- Physical: {render_inline_list(required_physical)}",
        "Please:",
    ]
    lines.extend(f"- {item}" for item in SCENE_INVARIANTS)
    lines.append("Do not:")
    lines.extend(f"- {item}" for item in FORBIDDEN_CHANGES)
    lines.append("- do not jump directly to the overall success state")
    if temporal_stages:
        lines.append("Expected continuation stages:")
        lines.extend(f"- {item}" for item in temporal_stages)
    return "\n".join(lines)


def build_prompt_zh_action_image_to_video(
    device_family: str,
    goal_text: str,
    next_action: str,
    required_ui: Optional[List[str]],
    required_physical: Optional[List[str]],
    stop_condition: Optional[str],
    temporal_stages: Optional[List[str]],
) -> str:
    lines = [
        f"你将看到一张来自{DEVICE_LABELS_ZH.get(device_family, device_family)}的第一人称输入图片。",
        "输入输出模式：image_to_video",
        "任务类型：state_transition_video",
        f"任务目标：{goal_text}",
        "这张图片是下一步关键动作发生前的锚点帧。",
        f"接下来必须发生的动作是：{next_action}",
        "请生成一段与同一场景一致的短输出视频。",
        "输出视频必须：",
        "- 保持相同的设备布局、观察视角和对象身份；",
        f"- 先展示这个下一步动作：{next_action}；",
        "- 再展示该动作带来的直接可见结果；",
        f"- 当满足以下停止条件时结束：{stop_condition or 'null'}",
        "视频结束前必须出现的可见证据：",
        f"- 界面证据：{render_inline_list(required_ui)}",
        f"- 物理证据：{render_inline_list(required_physical)}",
        "请：",
    ]
    lines.extend(f"- {item}" for item in SCENE_INVARIANTS_ZH)
    lines.append("不要：")
    lines.extend(f"- {item}" for item in FORBIDDEN_CHANGES_ZH)
    lines.append("- 不要直接跳到整个任务的全局成功状态")
    if temporal_stages:
        lines.append("建议遵循的后续阶段：")
        lines.extend(f"- {item}" for item in temporal_stages)
    return "\n".join(lines)


def build_prompt_en_verification_image(
    device_family: str,
    goal_text: str,
    required_ui: Optional[List[str]],
    required_physical: Optional[List[str]],
    stop_condition: str,
    verification_hint: Optional[str],
) -> str:
    lines = [
        f"You are given an egocentric input image of {with_indefinite_article(device_family)}.",
        "Input-output mode: image_to_video",
        "Task type: verification_state_video",
        "This image is an annotated single-frame verification-state cue for the task.",
        f"Goal being verified: {goal_text}",
        "Generate a short video consistent with the same scene, using this image as the target-state reference.",
        "The video must:",
        "- preserve the same device layout, viewpoint, and object identities;",
        "- focus on visible verification evidence only;",
        f"- stop once this verification state is clearly observable: {stop_condition}",
        "Required visible evidence before the video ends:",
        f"- UI: {render_inline_list(required_ui)}",
        f"- Physical: {render_inline_list(required_physical)}",
    ]
    if verification_hint:
        lines.append(f"Optional verification hint: {verification_hint}")
    lines.append("Please:")
    lines.extend(f"- {item}" for item in SCENE_INVARIANTS)
    lines.append("Do not:")
    lines.extend(f"- {item}" for item in FORBIDDEN_CHANGES)
    lines.append("- do not invent unrelated actions that are not implied by this state cue")
    return "\n".join(lines)


def build_prompt_zh_verification_image(
    device_family: str,
    goal_text: str,
    required_ui: Optional[List[str]],
    required_physical: Optional[List[str]],
    stop_condition: str,
    verification_hint: Optional[str],
) -> str:
    lines = [
        f"你将看到一张来自{DEVICE_LABELS_ZH.get(device_family, device_family)}的第一人称输入图像。",
        "输入输出模式：image_to_video",
        "任务类型：verification_state_video",
        "这张图像来自任务中的单帧验证状态标注，请将其作为目标状态参考。",
        f"正在验证的目标：{goal_text}",
        "请生成一段与同一场景一致的短视频，并让所需的可见验证证据清晰出现。",
        "生成视频必须：",
        "- 保持相同的设备布局、观察视角和对象身份；",
        "- 只关注可见验证证据；",
        f"- 当以下验证状态清晰可见时结束：{stop_condition}",
        "视频结束前必须出现的可见证据：",
        f"- 界面证据：{render_inline_list(required_ui)}",
        f"- 物理证据：{render_inline_list(required_physical)}",
    ]
    if verification_hint:
        lines.append(f"可选验证提示：{verification_hint}")
    lines.append("请：")
    lines.extend(f"- {item}" for item in SCENE_INVARIANTS_ZH)
    lines.append("不要：")
    lines.extend(f"- {item}" for item in FORBIDDEN_CHANGES_ZH)
    lines.append("- 不要编造与该状态无关的新动作")
    return "\n".join(lines)


def build_prompt_en_final_image(
    device_family: str,
    goal_text: str,
    required_ui: Optional[List[str]],
    required_physical: Optional[List[str]],
    stop_condition: str,
    verification_hint: Optional[str],
) -> str:
    lines = [
        f"You are given an egocentric input image of {with_indefinite_article(device_family)}.",
        "Input-output mode: image_to_video",
        "Task type: final_state_video",
        "This image is an annotated single-frame final-state cue for the task.",
        f"Goal: {goal_text}",
        "Generate a short video consistent with the same scene, using this image as the target final-state reference.",
        "The video must:",
        "- preserve the same device layout, viewpoint, and object identities;",
        "- progress toward successful completion in a visually coherent way;",
        f"- stop once this stop condition is satisfied: {stop_condition}",
        "Required visible success evidence before the video ends:",
        f"- UI: {render_inline_list(required_ui)}",
        f"- Physical: {render_inline_list(required_physical)}",
    ]
    if verification_hint:
        lines.append(f"Optional verification hint: {verification_hint}")
    lines.append("Please:")
    lines.extend(f"- {item}" for item in SCENE_INVARIANTS)
    lines.append("Do not:")
    lines.extend(f"- {item}" for item in FORBIDDEN_CHANGES)
    lines.append("- do not add unrelated intermediate outcomes that contradict the task")
    return "\n".join(lines)


def build_prompt_zh_final_image(
    device_family: str,
    goal_text: str,
    required_ui: Optional[List[str]],
    required_physical: Optional[List[str]],
    stop_condition: str,
    verification_hint: Optional[str],
) -> str:
    lines = [
        f"你将看到一张来自{DEVICE_LABELS_ZH.get(device_family, device_family)}的第一人称输入图像。",
        "输入输出模式：image_to_video",
        "任务类型：final_state_video",
        "这张图像来自任务中的单帧最终状态标注，请将其作为目标终态参考。",
        f"任务目标：{goal_text}",
        "请生成一段与同一场景一致的短视频，并自然推进到成功完成状态。",
        "生成视频必须：",
        "- 保持相同的设备布局、观察视角和对象身份；",
        "- 以视觉上连贯的方式推进到成功终态；",
        f"- 当满足以下停止条件时结束：{stop_condition}",
        "视频结束前必须出现的成功证据：",
        f"- 界面证据：{render_inline_list(required_ui)}",
        f"- 物理证据：{render_inline_list(required_physical)}",
    ]
    if verification_hint:
        lines.append(f"可选验证提示：{verification_hint}")
    lines.append("请：")
    lines.extend(f"- {item}" for item in SCENE_INVARIANTS_ZH)
    lines.append("不要：")
    lines.extend(f"- {item}" for item in FORBIDDEN_CHANGES_ZH)
    lines.append("- 不要加入与任务结果矛盾的中间状态")
    return "\n".join(lines)


def build_prompt_en_verification_video(
    device_family: str,
    goal_text: str,
    required_ui: Optional[List[str]],
    required_physical: Optional[List[str]],
    stop_condition: str,
    verification_hint: Optional[str],
    input_reference_text: str,
) -> str:
    lines = [
        f"You are given an egocentric input video clip of {with_indefinite_article(device_family)}.",
        "Input-output mode: video_to_video",
        "Task type: verification_state_video",
        f"Goal being verified: {goal_text}",
        f"The input clip already shows this verification/check action: {input_reference_text}",
        "Generate the next short output video that continues from the input clip and makes the verification result visible.",
        "The video must:",
        "- preserve the same device layout, viewpoint, and object identities;",
        "- focus on visible verification evidence only;",
        f"- stop once this verification result is clearly observable: {stop_condition}",
        "Required visible evidence before the video ends:",
        f"- UI: {render_inline_list(required_ui)}",
        f"- Physical: {render_inline_list(required_physical)}",
    ]
    if verification_hint:
        lines.append(f"Optional verification hint: {verification_hint}")
    lines.append("Please:")
    lines.extend(f"- {item}" for item in SCENE_INVARIANTS)
    lines.append("Do not:")
    lines.extend(f"- {item}" for item in FORBIDDEN_CHANGES)
    lines.append("- do not turn the check action into a new task-execution action")
    return "\n".join(lines)


def build_prompt_zh_verification_video(
    device_family: str,
    goal_text: str,
    required_ui: Optional[List[str]],
    required_physical: Optional[List[str]],
    stop_condition: str,
    verification_hint: Optional[str],
    input_reference_text: str,
) -> str:
    lines = [
        f"你将看到一段来自{DEVICE_LABELS_ZH.get(device_family, device_family)}的第一人称输入视频片段。",
        "输入输出模式：video_to_video",
        "任务类型：verification_state_video",
        f"正在验证的目标：{goal_text}",
        f"输入片段已经展示了这段验证/检查动作：{input_reference_text}",
        "请生成一段紧接在输入片段之后的短输出视频，让验证结果在视频中清晰可见。",
        "生成视频必须：",
        "- 保持相同的设备布局、观察视角和对象身份；",
        "- 只关注可见验证证据；",
        f"- 当以下验证结果清晰可见时结束：{stop_condition}",
        "视频结束前必须出现的可见证据：",
        f"- 界面证据：{render_inline_list(required_ui)}",
        f"- 物理证据：{render_inline_list(required_physical)}",
    ]
    if verification_hint:
        lines.append(f"可选验证提示：{verification_hint}")
    lines.append("请：")
    lines.extend(f"- {item}" for item in SCENE_INVARIANTS_ZH)
    lines.append("不要：")
    lines.extend(f"- {item}" for item in FORBIDDEN_CHANGES_ZH)
    lines.append("- 不要把检查动作误写成新的执行任务动作")
    return "\n".join(lines)


def build_prompt_en_recovery_video(
    device_family: str,
    goal_text: str,
    error_action: str,
    error_state: Optional[str],
    correction_actions: Sequence[str],
    post_fix_state: Optional[str],
    required_ui: Optional[List[str]],
    required_physical: Optional[List[str]],
    input_reference_text: str,
) -> str:
    correction_text = render_inline_list(correction_actions)
    lines = [
        f"You are given an egocentric input video clip of {with_indefinite_article(device_family)}.",
        "Input-output mode: video_to_video",
        "Task type: recovery_video",
        f"Goal: {goal_text}",
        f"The input clip shows this wrong-action segment: {input_reference_text}",
        f"Observed error after the input clip: {error_state or 'null'}",
        f"Correction actions in sequence: {correction_text}",
        f"Post-fix state to reach first: {post_fix_state or 'null'}",
        "Generate the next short output video that starts immediately after the input clip, corrects the mistake, reaches the post-fix state, and then continues toward task success.",
        "Required visible success evidence before the video ends:",
        f"- UI: {render_inline_list(required_ui)}",
        f"- Physical: {render_inline_list(required_physical)}",
        "Please:",
    ]
    lines.extend(f"- {item}" for item in SCENE_INVARIANTS)
    lines.append("Do not:")
    lines.extend(f"- {item}" for item in FORBIDDEN_CHANGES)
    lines.append("- do not skip the correction stage")
    lines.append("- do not jump directly to the final success state")
    lines.append(f"- do not repeat the wrong action as if it were correct: {error_action}")
    return "\n".join(lines)


def build_prompt_zh_recovery_video(
    device_family: str,
    goal_text: str,
    error_action: str,
    error_state: Optional[str],
    correction_actions: Sequence[str],
    post_fix_state: Optional[str],
    required_ui: Optional[List[str]],
    required_physical: Optional[List[str]],
    input_reference_text: str,
) -> str:
    correction_text = render_inline_list(correction_actions)
    lines = [
        f"你将看到一段来自{DEVICE_LABELS_ZH.get(device_family, device_family)}的第一人称输入视频片段。",
        "输入输出模式：video_to_video",
        "任务类型：recovery_video",
        f"任务目标：{goal_text}",
        f"输入片段展示了这段错误动作过程：{input_reference_text}",
        f"输入片段之后的错误状态：{error_state or 'null'}",
        f"需要按顺序执行的修正动作：{correction_text}",
        f"首先需要回到的修正后状态：{post_fix_state or 'null'}",
        "请生成一段紧接在输入片段之后的短输出视频，先纠正错误，再回到修正后状态，最后继续推进到任务成功。",
        "视频结束前必须出现的成功证据：",
        f"- 界面证据：{render_inline_list(required_ui)}",
        f"- 物理证据：{render_inline_list(required_physical)}",
        "请：",
    ]
    lines.extend(f"- {item}" for item in SCENE_INVARIANTS_ZH)
    lines.append("不要：")
    lines.extend(f"- {item}" for item in FORBIDDEN_CHANGES_ZH)
    lines.append("- 不要跳过修正阶段")
    lines.append("- 不要直接跳到最终成功状态")
    lines.append(f"- 不要把错误动作再次当成正确动作重复：{error_action}")
    return "\n".join(lines)


def extract_frame(video_path: Path, frame_index: int, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
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


def extract_clip(video_path: Path, start_frame: int, end_frame: int, output_path: Path) -> None:
    if end_frame < start_frame:
        raise ValueError(f"Invalid clip span: {start_frame}-{end_frame}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames > 0:
        start_frame = max(0, min(start_frame, total_frames - 1))
        end_frame = max(start_frame, min(end_frame, total_frames - 1))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Unable to create video writer: {output_path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current = start_frame
    try:
        while current <= end_frame:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Unable to read frame {current} from {video_path}")
            writer.write(frame)
            current += 1
    finally:
        writer.release()
        capture.release()


def build_review_sample(
    sample_id: str,
    annotation_root: Path,
    output_root: Path,
    timelines: Dict[str, VideoTimeline],
    videos: Dict[str, Dict[str, Any]],
    candidates: Dict[str, Dict[str, Any]],
    by_video: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    candidate = candidates[sample_id]
    if candidate["task_family"] != "action":
        raise RuntimeError(f"Selected sample must be an action candidate: {sample_id}")

    data_id = candidate["data_id"]
    video_info = videos[data_id]
    timeline = timelines[data_id]
    device_family = infer_device_family(video_info["scenario_family"], video_info["main_task"])
    source_span = candidate["source_span"]
    start = int(source_span["start"] or 0)
    end = int(source_span["end"] or start)
    note = SELECTION_NOTES[sample_id]

    input_modality: str
    review_asset_kind: str
    review_asset_path: str
    input_context_span: Dict[str, int]

    if sample_id in VIDEO_SAMPLE_IDS:
        input_end = max(0, start - 1)
        input_start = max(0, input_end - PRE_ACTION_INPUT_CLIP_FRAMES + 1)
        review_video_path = output_root / "videos" / f"{sample_id}.mp4"
        extract_clip(timeline.local_video_path, input_start, input_end, review_video_path)
        input_modality = "video"
        review_asset_kind = "video"
        review_asset_path = f"videos/{sample_id}.mp4"
        input_context_span = {"start": input_start, "end": input_end}
    else:
        anchor_frame = max(0, start - 1)
        review_image_path = output_root / "images" / f"{sample_id}.jpg"
        extract_frame(timeline.local_video_path, anchor_frame, review_image_path)
        input_modality = "image"
        review_asset_kind = "image"
        review_asset_path = f"images/{sample_id}.jpg"
        input_context_span = {"frame": anchor_frame}

    task_type = "state_transition_video"
    next_action = candidate["answer"]
    boundary = next_boundary_event(timeline, end)
    window_segments = segments_between(timeline, end, boundary.start if boundary else None, sorted(EVIDENCE_LABELS))
    required_ui, required_physical = build_required_evidence_from_segments(window_segments)
    evidence_texts = [clean_text(segment.text) for segment in window_segments]
    temporal_stages = list_or_none([next_action, *evidence_texts[:6]])
    stop_condition = clean_text(window_segments[-1].text) if window_segments else None
    quality_flags: List[str] = []
    source_spans: List[Dict[str, Any]] = [
        build_source_span(candidate["source_label"], candidate["answer"], start, end)
    ]
    if not required_ui:
        quality_flags.append("missing_required_evidence_ui")
    if not required_physical:
        quality_flags.append("missing_required_evidence_physical")
    if stop_condition is None:
        quality_flags.append("missing_stop_condition")

    for segment in window_segments[:6]:
        source_spans.append(build_source_span(segment.label, segment.text, segment.start, segment.end))

    if input_modality == "video":
        prompt_en = build_prompt_en_action_video_to_video(
            device_family=device_family,
            goal_text=video_info["main_task"],
            next_action=next_action,
            required_ui=required_ui,
            required_physical=required_physical,
            stop_condition=stop_condition,
            temporal_stages=temporal_stages,
        )
        prompt_zh = build_prompt_zh_action_video_to_video(
            device_family=device_family,
            goal_text=video_info["main_task"],
            next_action=next_action,
            required_ui=required_ui,
            required_physical=required_physical,
            stop_condition=stop_condition,
            temporal_stages=temporal_stages,
        )
    else:
        prompt_en = build_prompt_en_action_image_to_video(
            device_family=device_family,
            goal_text=video_info["main_task"],
            next_action=next_action,
            required_ui=required_ui,
            required_physical=required_physical,
            stop_condition=stop_condition,
            temporal_stages=temporal_stages,
        )
        prompt_zh = build_prompt_zh_action_image_to_video(
            device_family=device_family,
            goal_text=video_info["main_task"],
            next_action=next_action,
            required_ui=required_ui,
            required_physical=required_physical,
            stop_condition=stop_condition,
            temporal_stages=temporal_stages,
        )

    return {
        "sample_id": sample_id,
        "prompt_version": PROMPT_VERSION,
        "scenario_family": video_info["scenario_family"],
        "task_type": task_type,
        "source_video": f"annotations/0424/SWITCHAction_2/{timeline.video_name}",
        "input_modality": input_modality,
        "output_modality": "video",
        "review_asset_kind": review_asset_kind,
        "review_asset_path": review_asset_path,
        "input_context_span": input_context_span,
        "goal_text": video_info["main_task"],
        "overall_success_condition": video_info["main_verification"],
        "anchor_source_type": "next_action_target",
        "anchor_source_span": build_source_span("action_description", candidate["answer"], start, end),
        "next_action": next_action,
        "required_evidence_ui": required_ui,
        "required_evidence_physical": required_physical,
        "temporal_stages": temporal_stages,
        "stop_condition": stop_condition,
        "verification_action_hint": None,
        "error_action": None,
        "error_state": None,
        "correction_actions": None,
        "post_fix_state": None,
        "quality_flags": quality_flags,
        "source_spans": source_spans,
        "prompt_en": prompt_en,
        "prompt_zh": prompt_zh,
        "review_focus": note,
    }


def write_json(output_path: Path, payload: List[Dict[str, Any]]) -> None:
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def prune_review_assets(output_root: Path, samples: Sequence[Dict[str, Any]]) -> None:
    valid_relpaths = {sample["review_asset_path"] for sample in samples}
    for folder_name in ("images", "videos"):
        folder = output_root / folder_name
        if not folder.exists():
            continue
        for path in folder.iterdir():
            if not path.is_file():
                continue
            relpath = path.relative_to(output_root).as_posix()
            if relpath not in valid_relpaths:
                path.unlink()


def render_list(values: Sequence[str]) -> List[str]:
    return [f"- {value}" for value in values]


def inline_value(value: Any) -> str:
    if value is None:
        return "`null`"
    if isinstance(value, list):
        if not value:
            return "`[]`"
        return "; ".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def write_markdown(output_path: Path, samples: Sequence[Dict[str, Any]]) -> None:
    lines: List[str] = [
        "# 0424 Action-Conditioned Generation Prompt Review Bundle",
        "",
        "这些样本直接面向生成式 prompt review，而不是 QA review。它们全部限制为 action-conditioned 样本，因此每条样本都必须带有明确的非空 `next_action`。",
        "",
    ]
    for sample in samples:
        focus = sample["review_focus"]
        lines.extend(
            [
                f"## {sample['sample_id']}",
                "",
                f"- 输入资产: `{sample['review_asset_path']}`",
                f"- 输入上下文范围: {inline_value(sample['input_context_span'])}",
                f"- 输入模态: `{sample['input_modality']}`",
                f"- 输出模态: `{sample['output_modality']}`",
                f"- 场景类型: `{sample['scenario_family']}`",
                f"- 任务类型: `{sample['task_type']}`",
                f"- 目标任务: `{sample['goal_text']}`",
                f"- 总体成功条件: {inline_value(sample['overall_success_condition'])}",
                f"- 锚点来源类型: {inline_value(sample['anchor_source_type'])}",
                f"- 锚点来源范围: {inline_value(sample['anchor_source_span'])}",
                f"- 下一步动作: {inline_value(sample['next_action'])}",
                f"- 界面证据: {inline_value(sample['required_evidence_ui'])}",
                f"- 物理证据: {inline_value(sample['required_evidence_physical'])}",
                f"- 停止条件: {inline_value(sample['stop_condition'])}",
                f"- 验证提示: {inline_value(sample['verification_action_hint'])}",
                f"- 错误动作: {inline_value(sample['error_action'])}",
                f"- 错误状态: {inline_value(sample['error_state'])}",
                f"- 修正动作: {inline_value(sample['correction_actions'])}",
                f"- 修正后状态: {inline_value(sample['post_fix_state'])}",
                f"- 质量标记: {inline_value(sample['quality_flags'])}",
                "",
                "### 选择原因",
                "",
                focus["selection_reason"],
                "",
                "### 建议重点检查",
                "",
            ]
        )
        lines.extend(render_list(focus["watch_for"]))
        lines.extend(
            [
                "",
                "### Prompt EN",
                "",
                "```text",
                sample["prompt_en"],
                "```",
                "",
                "### Prompt ZH",
                "",
                "```text",
                sample["prompt_zh"],
                "```",
                "",
                "### Source Spans",
                "",
                "```json",
                json.dumps(sample["source_spans"], ensure_ascii=False, indent=2),
                "```",
                "",
            ]
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    annotation_root = repo_root / "annotations" / "0424" / "SWITCHAction_2"
    output_root = annotation_root / "generation_prompt_review_bundle_5video_10image"
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "images").mkdir(parents=True, exist_ok=True)
    (output_root / "videos").mkdir(parents=True, exist_ok=True)

    timelines = load_timelines(annotation_root)
    videos, candidates, by_video = load_inventory(annotation_root)
    samples = [
        build_review_sample(sample_id, annotation_root, output_root, timelines, videos, candidates, by_video)
        for sample_id in SELECTED_SAMPLE_IDS
    ]
    prune_review_assets(output_root, samples)
    write_json(output_root / "selected_samples.json", samples)
    write_markdown(output_root / "selected_samples.md", samples)

    print(f"Wrote bundle: {output_root}")
    print(f"Selected samples: {len(samples)}")
    print(f"Image samples: {len(IMAGE_SAMPLE_IDS)}")
    print(f"Video samples: {len(VIDEO_SAMPLE_IDS)}")
    print(f"Scenario counts: {dict(Counter(sample['scenario_family'] for sample in samples))}")


if __name__ == "__main__":
    main()
