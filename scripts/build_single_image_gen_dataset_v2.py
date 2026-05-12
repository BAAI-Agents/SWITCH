#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2


PROMPT_VERSION = "single_image_gen_prompt_v2_full"

SCENE_INVARIANTS = [
    "keep the same first-person viewpoint",
    "preserve the same device structure and button layout",
    "preserve the same object and hand identity",
]

SCENE_INVARIANTS_ZH = [
    "保持相同的第一人称视角",
    "保持相同的设备结构和按钮布局",
    "保持相同的物体与手部身份",
]

FORBIDDEN_CHANGES = [
    "do not redesign the UI layout",
    "do not invent new buttons or displays",
    "do not add unrelated objects",
    "do not jump to a different camera viewpoint",
    "do not skip the required next action",
]

FORBIDDEN_CHANGES_ZH = [
    "不要重新设计界面布局",
    "不要虚构新的按钮或显示内容",
    "不要添加无关物体",
    "不要切换到不同的摄像机视角",
    "不要跳过要求的下一步动作",
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
STATE_LABELS = {"ui_state", "physical_world_state"}

EXPECTED_TASK_COUNTS = {
    "state_transition_video": 84,
    "final_state_video": 22,
    "verification_state_video": 86,
    "recovery_video": 3,
}

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
class Candidate:
    qa_id: str
    source_file: str
    data_id: str
    video_name: str
    video_local_path: str
    scenario_family: str
    task_family: str
    qa_type: str
    main_task: str
    main_verification: str
    source_label: str
    source_span: Dict[str, Optional[int]]
    answer: str


@dataclass
class VideoInventory:
    data_id: str
    video_name: str
    video_local_path: str
    scenario_family: str
    main_task: str
    main_verification: str
    candidates: List[Candidate] = field(default_factory=list)


@dataclass
class VideoTimeline:
    data_id: str
    source_file: str
    video_name: str
    fps: float
    total_frames: int
    segments: List[Segment]
    action_events: List[ActionEvent]
    local_video_path: Optional[Path] = None


class AssetWriter:
    def __init__(self) -> None:
        self.frame_cache: Dict[Tuple[str, int], Any] = {}

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


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def sentence_case(text: str) -> str:
    text = normalize_spaces(text)
    if not text:
        return ""
    if text[0].isalpha():
        return text[0].upper() + text[1:]
    return text


def clean_text(text: str) -> str:
    cleaned = normalize_spaces(text)
    for src, dst in TEXT_REPLACEMENTS.items():
        cleaned = cleaned.replace(src, dst)
    cleaned = cleaned.replace("  ", " ")
    cleaned = re.sub(r"\s+\.", ".", cleaned)
    return sentence_case(cleaned)


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
    for key, value in data.items():
        if key == "meta":
            continue
        if isinstance(value, str) and value.lower().endswith(".mp4"):
            return Path(value).name
    return f"{item.get('id')}.mp4"


def span_sort_key(span: Dict[str, Optional[int]]) -> Tuple[int, int]:
    start = int(span.get("start") or 0)
    end = int(span.get("end") or start)
    return start, end


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


def compat_list(values: Optional[Sequence[str]]) -> List[str]:
    if not values:
        return []
    return [clean_text(value) for value in values if clean_text(value)]


def humanize_identifier(text: str) -> str:
    return normalize_spaces((text or "").replace("_", " "))


def render_inline_list(values: Optional[Sequence[str]]) -> str:
    if not values:
        return "null"
    return "; ".join(values)


def safe_name(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return text.strip("._") or "item"


def relative_repo_path(path: Path, repo_root: Path) -> str:
    return path.relative_to(repo_root).as_posix()


def clamp_frame(frame_index: int, total_frames: int) -> int:
    if total_frames > 0:
        return max(0, min(frame_index, total_frames - 1))
    return max(0, frame_index)


def modality_for_label(label: str) -> Optional[str]:
    if label in UI_LABELS:
        return "ui"
    if label in PHYSICAL_LABELS:
        return "physical"
    return None


def build_source_span(label: str, text: str, start: Optional[int], end: Optional[int]) -> Dict[str, Any]:
    return {
        "label": label,
        "text": clean_text(text),
        "start": start,
        "end": end,
    }


def find_raw_annotation_paths(annotation_root: Path) -> List[Path]:
    action1_candidates = []
    for path in annotation_root.glob("SWITCH*Action_1.json"):
        if not path.is_file():
            continue
        name = path.name
        if any(token in name for token in (".mcq.", ".openqa.", ".qa_candidates", ".qa_summary")):
            continue
        action1_candidates.append(path)
    if not action1_candidates:
        raise FileNotFoundError(f"Could not find Action_1 JSON under {annotation_root}")
    action2_path = annotation_root / "SWITCH_Action_2.json"
    if not action2_path.exists():
        raise FileNotFoundError(f"Could not find Action_2 JSON under {annotation_root}")
    return [sorted(action1_candidates)[0], action2_path]


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


def build_video_index(annotation_root: Path, fallback_video_dir: Path) -> Dict[str, Path]:
    video_index: Dict[str, Path] = {}
    for subdir in sorted(annotation_root.glob("SWITCHAction_*")):
        if not subdir.is_dir():
            continue
        for video_path in sorted(subdir.rglob("*.mp4")):
            video_index[video_path.name] = video_path
    if fallback_video_dir.exists():
        for video_path in sorted(fallback_video_dir.glob("*.mp4")):
            video_index.setdefault(video_path.name, video_path)
    return video_index


def load_inventory(annotation_root: Path) -> Dict[str, VideoInventory]:
    payload = json.loads((annotation_root / "switch_all.qa_candidates.json").read_text(encoding="utf-8"))
    inventory: Dict[str, VideoInventory] = {}
    for video_payload in payload["videos"]:
        video = VideoInventory(
            data_id=video_payload["data_id"],
            video_name=video_payload["video_name"],
            video_local_path=video_payload["video_local_path"],
            scenario_family=video_payload["scenario_family"],
            main_task=clean_text(video_payload["main_task"]),
            main_verification=clean_text(video_payload["main_verification"]),
        )
        for qa in video_payload["qa_candidates"]:
            span = qa.get("source_span") or {}
            candidate = Candidate(
                qa_id=qa["qa_id"],
                source_file=qa["source_file"],
                data_id=video_payload["data_id"],
                video_name=video_payload["video_name"],
                video_local_path=video_payload["video_local_path"],
                scenario_family=video_payload["scenario_family"],
                task_family=qa["task_family"],
                qa_type=qa["qa_type"],
                main_task=clean_text(video_payload["main_task"]),
                main_verification=clean_text(video_payload["main_verification"]),
                source_label=qa["source_label"],
                source_span={
                    "start": int(span["start"]) if span.get("start") is not None else None,
                    "end": int(span["end"]) if span.get("end") is not None else None,
                },
                answer=clean_text(qa["answer"]),
            )
            video.candidates.append(candidate)
        video.candidates.sort(key=lambda candidate: (span_sort_key(candidate.source_span), candidate.qa_id))
        inventory[video.data_id] = video
    return inventory


def load_timelines(annotation_root: Path, video_index: Dict[str, Path]) -> Dict[str, VideoTimeline]:
    timelines: Dict[str, VideoTimeline] = {}
    for source_path in find_raw_annotation_paths(annotation_root):
        items = json.loads(source_path.read_text(encoding="utf-8"))
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
            action_events = build_action_events(segments)
            timelines[data_id] = VideoTimeline(
                data_id=data_id,
                source_file=source_path.name,
                video_name=video_name,
                fps=float(media_meta.get("fps") or 30.0),
                total_frames=int(media_meta.get("total_frames") or 0),
                segments=segments,
                action_events=action_events,
                local_video_path=video_index.get(video_name),
            )
    return timelines


def find_segment(
    timeline: VideoTimeline,
    label: str,
    start: Optional[int],
    end: Optional[int],
    text: Optional[str] = None,
) -> Optional[Segment]:
    target_text = clean_text(text or "")
    for segment in timeline.segments:
        if segment.label != label:
            continue
        if segment.start != start or segment.end != end:
            continue
        if target_text and clean_text(segment.text) != target_text:
            continue
        return segment
    return None


def find_action_event(
    timeline: VideoTimeline,
    start: Optional[int],
    end: Optional[int],
    description: Optional[str] = None,
    allowed_types: Optional[Sequence[str]] = None,
) -> Optional[ActionEvent]:
    target_desc = clean_text(description or "")
    for event in timeline.action_events:
        if allowed_types and event.action_type not in allowed_types:
            continue
        if event.start == start and event.end == end:
            if not target_desc or clean_text(event.action_description) == target_desc:
                return event
    for event in timeline.action_events:
        if allowed_types and event.action_type not in allowed_types:
            continue
        if target_desc and clean_text(event.action_description) == target_desc:
            return event
    return None


def get_final_state_frame(timeline: VideoTimeline) -> Optional[int]:
    for segment in timeline.segments:
        if segment.label == "is_final_state":
            return segment.start
    return None


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


def segments_in_windows(
    timeline: VideoTimeline,
    windows: Sequence[Tuple[int, int]],
    labels: Sequence[str],
) -> List[Segment]:
    label_set = set(labels)
    selected: List[Segment] = []
    for segment in timeline.segments:
        if segment.label not in label_set or segment.start is None:
            continue
        for window_start, window_end in windows:
            if window_start <= segment.start <= window_end:
                selected.append(segment)
                break
    selected.sort(key=lambda segment: ((segment.start or 0), (segment.end or 0), segment.label, segment.text))
    return selected


def add_evidence(
    ui_values: List[str],
    physical_values: List[str],
    label: Optional[str],
    text: Optional[str],
) -> None:
    cleaned = clean_text(text or "")
    if not cleaned:
        return
    modality = modality_for_label(label or "")
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


def select_secondary_segments(
    window_segments: Iterable[Segment],
    primary_modality: str,
    candidate_frame: int,
) -> List[Segment]:
    secondary_labels = PHYSICAL_LABELS if primary_modality == "ui" else UI_LABELS
    candidates = [
        segment
        for segment in window_segments
        if segment.label in secondary_labels and segment.start is not None and abs(segment.start - candidate_frame) <= 90
    ]
    candidates.sort(
        key=lambda segment: (
            abs((segment.start or candidate_frame) - candidate_frame),
            segment.start or 0,
            segment.end or 0,
            segment.label,
        )
    )
    chosen: List[Segment] = []
    seen = set()
    for segment in candidates:
        key = (segment.label, clean_text(segment.text).lower())
        if key in seen:
            continue
        seen.add(key)
        chosen.append(segment)
        if len(chosen) == 2:
            break
    return chosen


def nearest_preceding_event(timeline: VideoTimeline, frame_index: int, action_type: str) -> Optional[ActionEvent]:
    chosen: Optional[ActionEvent] = None
    for event in timeline.action_events:
        if event.action_type != action_type or event.start is None:
            continue
        if event.start <= frame_index:
            chosen = event
    return chosen


def select_verification_anchor(timeline: VideoTimeline, candidate_frame: int) -> Tuple[Optional[ActionEvent], str]:
    verification_event = nearest_preceding_event(timeline, candidate_frame, "verification action")
    if verification_event is not None:
        return verification_event, "verification_action"
    action_event = nearest_preceding_event(timeline, candidate_frame, "execute action")
    if action_event is not None:
        return action_event, "action"
    return None, "frame"


def distance_to_event(frame_index: int, event: ActionEvent) -> int:
    start = event.start or 0
    end = event.end or start
    if start <= frame_index <= end:
        return 0
    return min(abs(frame_index - start), abs(frame_index - end))


def nearest_verification_hint(timeline: VideoTimeline, candidate_frame: int) -> Optional[str]:
    candidates = [event for event in timeline.action_events if event.action_type == "verification action"]
    if not candidates:
        return None
    candidates.sort(key=lambda event: (distance_to_event(candidate_frame, event), event.start or 0))
    return clean_text(candidates[0].action_description)


def primary_evidence(required_ui: Optional[List[str]], required_physical: Optional[List[str]]) -> Optional[str]:
    if required_ui:
        return required_ui[0]
    if required_physical:
        return required_physical[0]
    return None


def support_windows_for_final_state(primary_frame: int, final_state_frame: Optional[int]) -> List[Tuple[int, int]]:
    windows = [(max(0, primary_frame - 90), primary_frame + 45)]
    if final_state_frame is not None:
        windows.append((max(0, final_state_frame - 90), final_state_frame + 45))
    deduped: List[Tuple[int, int]] = []
    for window in windows:
        if window not in deduped:
            deduped.append(window)
    return deduped


def collect_final_support_evidence(
    timeline: VideoTimeline,
    primary_segment: Segment,
    final_state_frame: Optional[int],
) -> Tuple[Optional[List[str]], Optional[List[str]], List[Segment]]:
    windows = support_windows_for_final_state(primary_segment.start or 0, final_state_frame)
    support_segments = segments_in_windows(timeline, windows, sorted(EVIDENCE_LABELS))
    support_segments = [
        segment
        for segment in support_segments
        if not (
            segment.label == primary_segment.label
            and segment.start == primary_segment.start
            and segment.end == primary_segment.end
            and clean_text(segment.text) == clean_text(primary_segment.text)
        )
    ]
    ui_values: List[str] = []
    physical_values: List[str] = []
    add_evidence(ui_values, physical_values, primary_segment.label, primary_segment.text)
    for segment in support_segments:
        add_evidence(ui_values, physical_values, segment.label, segment.text)
    return list_or_none(ui_values), list_or_none(physical_values), support_segments


def choose_error_state(
    timeline: VideoTimeline,
    wrong_action: Candidate,
    first_fix: Candidate,
) -> Optional[str]:
    state_segments = segments_between(
        timeline,
        int(wrong_action.source_span.get("end") or wrong_action.source_span.get("start") or 0),
        int(first_fix.source_span.get("start") or 0),
        ["ui_state", "physical_world_state", "ui_change", "physical_world_change"],
    )
    if not state_segments:
        return None
    return clean_text(state_segments[0].text)


def ordinal_or_numeric_floor(text: str) -> Optional[int]:
    lowered = normalize_spaces(text).lower()
    match = re.search(r"(\d+)(?:st|nd|rd|th)?\s+floor", lowered)
    if match:
        return int(match.group(1))
    for word, value in ORDINAL_WORDS.items():
        if f"{word} floor" in lowered:
            return value
    return None


def infer_device_family(scenario_family: str, main_task: str) -> str:
    if scenario_family == "elevator":
        return "elevator_system"
    if scenario_family == "subway_ticket":
        return "subway_ticket_machine"
    if scenario_family == "medical_kiosk":
        return "hospital_registration_machine"
    lowered = main_task.lower()
    if "elevator" in lowered:
        return "elevator_system"
    if "ticket" in lowered or "subway" in lowered:
        return "subway_ticket_machine"
    return "hospital_registration_machine"


def infer_task_intent(main_task: str, scenario_family: str) -> str:
    lowered = main_task.lower()
    if "query" in lowered and "project" in lowered:
        return "query_project"
    if "cancel appointment" in lowered:
        return "cancel_registration_appointment"
    if "appointment" in lowered or "registration" in lowered:
        return "make_registration_appointment"
    if "subway ticket" in lowered or ("ticket" in lowered and "station" in lowered):
        return "buy_subway_ticket"
    if "elevator" in lowered and "floor" in lowered:
        return "go_to_target_floor"
    if scenario_family == "subway_ticket":
        return "buy_subway_ticket"
    if scenario_family == "elevator":
        return "go_to_target_floor"
    return "public_service_terminal_task"


def infer_goal_slots(main_task: str, action_answers: Sequence[str]) -> Dict[str, Any]:
    slots: Dict[str, Any] = {}
    task_lower = main_task.lower()
    target_floor = ordinal_or_numeric_floor(main_task)
    if target_floor is not None:
        slots["target_floor"] = target_floor
    from_to = re.search(r"from (.+?) to (.+?)(?:$|\.|,)", main_task, re.IGNORECASE)
    if from_to:
        slots["origin_station"] = clean_text(from_to.group(1))
        slots["destination_station"] = clean_text(from_to.group(2))
    price_match = re.search(r"(\d+)\s*-\s*yuan|(\d+)\s*yuan", task_lower)
    if price_match:
        price = price_match.group(1) or price_match.group(2)
        slots["ticket_price"] = f"{price}_yuan"
    if "buy one" in task_lower or "1 ticket" in task_lower:
        slots["ticket_count"] = 1
    for answer in action_answers:
        answer_lower = answer.lower()
        if "2 yuan" in answer_lower:
            slots.setdefault("ticket_price", "2_yuan")
        if "3 yuan" in answer_lower:
            slots.setdefault("ticket_price", "3_yuan")
        if "1 ticket" in answer_lower or "1 sheet" in answer_lower:
            slots.setdefault("ticket_count", 1)
        floor = ordinal_or_numeric_floor(answer)
        if floor is not None:
            slots.setdefault("target_floor", floor)
    return slots


def device_label_zh(device_family: str) -> str:
    mapping = {
        "elevator_system": "电梯系统",
        "subway_ticket_machine": "地铁售票机",
        "hospital_registration_machine": "医院自助机",
    }
    return mapping.get(device_family, device_family)


def build_prompt_en_state_transition(
    device_family: str,
    goal_text: str,
    next_action: str,
    required_ui: Optional[List[str]],
    required_physical: Optional[List[str]],
    stop_condition: Optional[str],
    temporal_stages: Optional[List[str]],
) -> str:
    lines = [
        f"You are given an egocentric anchor frame of a {humanize_identifier(device_family)}.",
        "Task type: state_transition_video",
        f"Goal: {goal_text}",
        f"Next action in sequence: {next_action}",
        "The video must:",
        "- preserve the same device layout, viewpoint, and object identities;",
        "- show the required next action before showing its effect;",
        "- show the expected local UI changes and physical changes;",
        f"- stop once this stop condition is satisfied: {stop_condition or 'null'}",
        "Required visible evidence before the video ends:",
        f"- UI: {render_inline_list(required_ui)}",
        f"- Physical: {render_inline_list(required_physical)}",
        "Please:",
    ]
    lines.extend(f"- {item}" for item in SCENE_INVARIANTS)
    lines.append("Do not:")
    lines.extend(f"- {item}" for item in FORBIDDEN_CHANGES)
    lines.append("Temporal stages to follow:")
    if temporal_stages:
        lines.extend(f"- {stage}" for stage in temporal_stages)
    else:
        lines.append("- null")
    return "\n".join(lines)


def build_prompt_zh_state_transition(
    device_family: str,
    goal_text: str,
    next_action: str,
    required_ui: Optional[List[str]],
    required_physical: Optional[List[str]],
    stop_condition: Optional[str],
    temporal_stages: Optional[List[str]],
) -> str:
    lines = [
        f"你将看到一张来自{device_label_zh(device_family)}的第一人称锚点帧。",
        "任务类型：state_transition_video",
        f"任务目标：{goal_text}",
        f"当前序列中的下一步动作：{next_action}",
        "生成视频时必须满足：",
        "- 保持相同的设备布局、观察视角和对象身份；",
        "- 先展示要求的下一步动作，再展示该动作带来的结果；",
        "- 展示局部可见的界面变化和物理变化；",
        f"- 当以下停止条件满足时结束视频：{stop_condition or 'null'}",
        "视频结束前必须能看到的证据：",
        f"- 界面证据：{render_inline_list(required_ui)}",
        f"- 物理证据：{render_inline_list(required_physical)}",
        "请：",
    ]
    lines.extend(f"- {item}" for item in SCENE_INVARIANTS_ZH)
    lines.append("不要：")
    lines.extend(f"- {item}" for item in FORBIDDEN_CHANGES_ZH)
    lines.append("建议遵循的时序阶段：")
    if temporal_stages:
        lines.extend(f"- {stage}" for stage in temporal_stages)
    else:
        lines.append("- null")
    return "\n".join(lines)


def build_prompt_en_verification(
    device_family: str,
    goal_text: str,
    required_ui: Optional[List[str]],
    required_physical: Optional[List[str]],
) -> str:
    lines = [
        f"You are given an egocentric anchor frame of a {humanize_identifier(device_family)}.",
        "Task type: verification_state_video",
        f"Goal being verified: {goal_text}",
        "Generate a short video showing what should be observed when checking whether the task has succeeded.",
        "Do not add unnecessary new actions.",
        "Focus on visible evidence only.",
        "Required visible evidence:",
        f"- UI: {render_inline_list(required_ui)}",
        f"- Physical: {render_inline_list(required_physical)}",
        "Please:",
    ]
    lines.extend(f"- {item}" for item in SCENE_INVARIANTS)
    lines.append("Do not:")
    lines.extend(f"- {item}" for item in FORBIDDEN_CHANGES)
    return "\n".join(lines)


def build_prompt_zh_verification(
    device_family: str,
    goal_text: str,
    required_ui: Optional[List[str]],
    required_physical: Optional[List[str]],
) -> str:
    lines = [
        f"你将看到一张来自{device_label_zh(device_family)}的第一人称锚点帧。",
        "任务类型：verification_state_video",
        f"正在验证的目标：{goal_text}",
        "请生成一段短视频，展示当检查该任务是否已经成功时，应该观察到什么。",
        "不要额外添加不必要的新动作。",
        "只关注可见证据。",
        "必须出现的可见证据：",
        f"- 界面证据：{render_inline_list(required_ui)}",
        f"- 物理证据：{render_inline_list(required_physical)}",
        "请：",
    ]
    lines.extend(f"- {item}" for item in SCENE_INVARIANTS_ZH)
    lines.append("不要：")
    lines.extend(f"- {item}" for item in FORBIDDEN_CHANGES_ZH)
    return "\n".join(lines)


def build_prompt_en_final(
    device_family: str,
    goal_text: str,
    required_ui: Optional[List[str]],
    required_physical: Optional[List[str]],
    stop_condition: str,
) -> str:
    lines = [
        f"You are given an egocentric anchor frame of a {humanize_identifier(device_family)}.",
        "Task type: final_state_video",
        f"Goal: {goal_text}",
        "Generate a short video showing the successful completion of the task.",
        "The video must:",
        "- preserve the same device layout, viewpoint, and object identities;",
        "- progress naturally toward the successful end state;",
        f"- stop once this stop condition is satisfied: {stop_condition}",
        "Required visible success evidence before the video ends:",
        f"- UI: {render_inline_list(required_ui)}",
        f"- Physical: {render_inline_list(required_physical)}",
        "Please:",
    ]
    lines.extend(f"- {item}" for item in SCENE_INVARIANTS)
    lines.append("Do not:")
    lines.extend(f"- {item}" for item in FORBIDDEN_CHANGES)
    return "\n".join(lines)


def build_prompt_zh_final(
    device_family: str,
    goal_text: str,
    required_ui: Optional[List[str]],
    required_physical: Optional[List[str]],
    stop_condition: str,
) -> str:
    lines = [
        f"你将看到一张来自{device_label_zh(device_family)}的第一人称锚点帧。",
        "任务类型：final_state_video",
        f"任务目标：{goal_text}",
        "请生成一段短视频，展示任务成功完成的过程。",
        "生成视频时必须满足：",
        "- 保持相同的设备布局、观察视角和对象身份；",
        "- 自然推进到成功结束状态；",
        f"- 当以下停止条件满足时结束视频：{stop_condition}",
        "视频结束前必须能看到的成功证据：",
        f"- 界面证据：{render_inline_list(required_ui)}",
        f"- 物理证据：{render_inline_list(required_physical)}",
        "请：",
    ]
    lines.extend(f"- {item}" for item in SCENE_INVARIANTS_ZH)
    lines.append("不要：")
    lines.extend(f"- {item}" for item in FORBIDDEN_CHANGES_ZH)
    return "\n".join(lines)


def build_prompt_en_recovery(
    device_family: str,
    goal_text: str,
    error_action: str,
    error_state: Optional[str],
    correction_actions: Sequence[str],
    post_fix_state: Optional[str],
    required_ui: Optional[List[str]],
    required_physical: Optional[List[str]],
) -> str:
    correction_text = "; then ".join(correction_actions) if correction_actions else "null"
    lines = [
        f"You are given an egocentric anchor frame of a {humanize_identifier(device_family)} in an incorrect state.",
        "Task type: recovery_video",
        f"Goal: {goal_text}",
        f"Observed error: {error_action} / {error_state or 'null'}",
        f"Correction actions in sequence: {correction_text}",
        f"State after correction: {post_fix_state or 'null'}",
        "Generate a short recovery video that:",
        "1. corrects the current mistake,",
        "2. visibly returns to the specified post-fix state,",
        "3. then proceeds toward successful completion.",
        "Required success evidence:",
        f"- UI: {render_inline_list(required_ui)}",
        f"- Physical: {render_inline_list(required_physical)}",
        "Please:",
    ]
    lines.extend(f"- {item}" for item in SCENE_INVARIANTS)
    lines.append("Do not:")
    lines.extend(f"- {item}" for item in FORBIDDEN_CHANGES)
    lines.append("Do not skip the correction stage.")
    lines.append("Do not jump directly to the success state.")
    return "\n".join(lines)


def build_prompt_zh_recovery(
    device_family: str,
    goal_text: str,
    error_action: str,
    error_state: Optional[str],
    correction_actions: Sequence[str],
    post_fix_state: Optional[str],
    required_ui: Optional[List[str]],
    required_physical: Optional[List[str]],
) -> str:
    correction_text = "，然后".join(correction_actions) if correction_actions else "null"
    lines = [
        f"你将看到一张来自{device_label_zh(device_family)}、且已经处于错误状态的第一人称锚点帧。",
        "任务类型：recovery_video",
        f"任务目标：{goal_text}",
        f"观测到的错误：{error_action} / {error_state or 'null'}",
        f"按顺序执行的修正动作：{correction_text}",
        f"修正后的状态：{post_fix_state or 'null'}",
        "请生成一段恢复视频，要求：",
        "1. 先纠正当前错误；",
        "2. 明确回到指定的修正后状态；",
        "3. 再继续朝着成功完成推进。",
        "必须出现的成功证据：",
        f"- 界面证据：{render_inline_list(required_ui)}",
        f"- 物理证据：{render_inline_list(required_physical)}",
        "请：",
    ]
    lines.extend(f"- {item}" for item in SCENE_INVARIANTS_ZH)
    lines.append("不要：")
    lines.extend(f"- {item}" for item in FORBIDDEN_CHANGES_ZH)
    lines.append("不要跳过修正阶段。")
    lines.append("不要直接跳到成功状态。")
    return "\n".join(lines)


def sample_id_from_candidate(candidate: Candidate, task_type: str) -> str:
    match = re.search(r"_(\d+)$", candidate.qa_id)
    suffix = match.group(1) if match else "000"
    return f"{candidate.data_id}_{task_type}_{suffix}"


def qa_numeric_suffix(qa_id: str) -> str:
    match = re.search(r"_(\d+)$", qa_id)
    return match.group(1) if match else "000"


def build_base_sample(
    *,
    sample_id: str,
    candidate: Optional[Candidate],
    inventory: VideoInventory,
    timeline: VideoTimeline,
    source_video: str,
    task_type: str,
    device_family: str,
    task_intent: str,
    goal_slots: Dict[str, Any],
    anchor_frame: int,
    anchor_frame_path: str,
    anchor_source_type: str,
    anchor_source_span: Optional[Dict[str, Any]],
    prompt_en: str,
    prompt_zh: str,
    next_action: Optional[str],
    required_ui: Optional[List[str]],
    required_physical: Optional[List[str]],
    temporal_stages: Optional[List[str]],
    stop_condition: Optional[str],
    overall_success_condition: str,
    verification_action_hint: Optional[str],
    final_state_frame: Optional[int],
    correction_actions: Optional[List[str]],
    expected_final_state: Optional[str],
    error_action: Optional[str],
    error_state: Optional[str],
    correction_action: Optional[str],
    post_fix_state: Optional[str],
    source_spans: List[Dict[str, Any]],
    quality_flags: List[str],
) -> Dict[str, Any]:
    return {
        "sample_id": sample_id,
        "source_qa_id": candidate.qa_id if candidate else None,
        "source_video": source_video,
        "task_type": task_type,
        "device_family": device_family,
        "task_intent": task_intent,
        "goal_text": inventory.main_task,
        "goal_slots": goal_slots,
        "anchor_frame": anchor_frame,
        "anchor_frame_time": round(anchor_frame / timeline.fps, 3),
        "anchor_frame_path": anchor_frame_path,
        "anchor_source_type": anchor_source_type,
        "anchor_source_span": anchor_source_span,
        "prompt_version": PROMPT_VERSION,
        "prompt_en": prompt_en,
        "prompt_zh": prompt_zh,
        "next_action": next_action,
        "required_evidence_ui": required_ui,
        "required_evidence_physical": required_physical,
        "temporal_stages": temporal_stages,
        "stop_condition": stop_condition,
        "overall_success_condition": overall_success_condition,
        "scene_invariants": list(SCENE_INVARIANTS),
        "forbidden_changes": list(FORBIDDEN_CHANGES),
        "verification_action_hint": verification_action_hint,
        "final_state_frame": final_state_frame,
        "correction_actions": correction_actions,
        "expected_ui_changes": compat_list(required_ui),
        "expected_physical_changes": compat_list(required_physical),
        "expected_verification_ui": compat_list(required_ui),
        "expected_verification_physical": compat_list(required_physical),
        "expected_final_state": expected_final_state,
        "error_action": error_action,
        "error_state": error_state,
        "correction_action": correction_action,
        "post_fix_state": post_fix_state,
        "retry_trigger_type": None,
        "retry_outcome": None,
        "source_spans": source_spans,
        "quality_flags": sorted(set(flag for flag in quality_flags if flag)),
    }


def extract_anchor_frame(
    asset_writer: AssetWriter,
    timeline: VideoTimeline,
    repo_root: Path,
    frames_dir: Path,
    sample_id: str,
    anchor_frame: int,
) -> str:
    if timeline.local_video_path is None:
        raise FileNotFoundError(f"Missing local video for {timeline.video_name}")
    frame_path = frames_dir / f"{safe_name(sample_id)}.jpg"
    asset_writer.extract_frame(timeline.local_video_path, anchor_frame, frame_path)
    return relative_repo_path(frame_path, repo_root)


def build_state_transition_samples(
    repo_root: Path,
    output_frames_dir: Path,
    asset_writer: AssetWriter,
    inventory: Dict[str, VideoInventory],
    timelines: Dict[str, VideoTimeline],
) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    for data_id in sorted(inventory):
        video = inventory[data_id]
        timeline = timelines[data_id]
        device_family = infer_device_family(video.scenario_family, video.main_task)
        task_intent = infer_task_intent(video.main_task, video.scenario_family)
        action_answers = [candidate.answer for candidate in video.candidates if candidate.task_family == "action"]
        goal_slots = infer_goal_slots(video.main_task, action_answers)
        final_candidate = next((candidate for candidate in video.candidates if candidate.task_family == "final_state"), None)
        for candidate in [item for item in video.candidates if item.task_family == "action"]:
            current_event = find_action_event(
                timeline,
                candidate.source_span.get("start"),
                candidate.source_span.get("end"),
                candidate.answer,
                allowed_types=["execute action"],
            )
            if current_event is None:
                raise RuntimeError(f"Unable to match action event for {candidate.qa_id}")
            anchor_frame = clamp_frame(
                int(candidate.source_span.get("start") or 0) - 1,
                timeline.total_frames,
            )
            next_event = next_boundary_event(timeline, int(candidate.source_span.get("end") or candidate.source_span.get("start") or 0))
            window_segments = segments_between(
                timeline,
                int(candidate.source_span.get("end") or candidate.source_span.get("start") or 0),
                next_event.start if next_event else None,
                sorted(EVIDENCE_LABELS),
            )
            required_ui, required_physical = build_required_evidence_from_segments(window_segments)
            evidence_texts = [clean_text(segment.text) for segment in window_segments]
            temporal_stages = list_or_none([candidate.answer, *evidence_texts[:6]])
            stop_condition = clean_text(window_segments[-1].text) if window_segments else primary_evidence(required_ui, required_physical)
            final_state_frame = get_final_state_frame(timeline)
            sample_id = sample_id_from_candidate(candidate, "state_transition_video")
            anchor_frame_path = extract_anchor_frame(asset_writer, timeline, repo_root, output_frames_dir, sample_id, anchor_frame)
            source_spans = [
                build_source_span("action_description", candidate.answer, candidate.source_span.get("start"), candidate.source_span.get("end"))
            ]
            for segment in window_segments[:6]:
                source_spans.append(build_source_span(segment.label, segment.text, segment.start, segment.end))
            quality_flags: List[str] = []
            if not required_ui:
                quality_flags.append("missing_required_evidence_ui")
            if not required_physical:
                quality_flags.append("missing_required_evidence_physical")
            if stop_condition is None:
                quality_flags.append("missing_stop_condition")
            if timeline.local_video_path and "30fps" in timeline.local_video_path.parts:
                quality_flags.append("fallback_video_source")
            samples.append(
                build_base_sample(
                    sample_id=sample_id,
                    candidate=candidate,
                    inventory=video,
                    timeline=timeline,
                    source_video=relative_repo_path(timeline.local_video_path, repo_root),
                    task_type="state_transition_video",
                    device_family=device_family,
                    task_intent=task_intent,
                    goal_slots=goal_slots,
                    anchor_frame=anchor_frame,
                    anchor_frame_path=anchor_frame_path,
                    anchor_source_type="action",
                    anchor_source_span=build_source_span(
                        "action_description",
                        candidate.answer,
                        candidate.source_span.get("start"),
                        candidate.source_span.get("end"),
                    ),
                    prompt_en=build_prompt_en_state_transition(
                        device_family,
                        video.main_task,
                        candidate.answer,
                        required_ui,
                        required_physical,
                        stop_condition,
                        temporal_stages,
                    ),
                    prompt_zh=build_prompt_zh_state_transition(
                        device_family,
                        video.main_task,
                        candidate.answer,
                        required_ui,
                        required_physical,
                        stop_condition,
                        temporal_stages,
                    ),
                    next_action=candidate.answer,
                    required_ui=required_ui,
                    required_physical=required_physical,
                    temporal_stages=temporal_stages,
                    stop_condition=stop_condition,
                    overall_success_condition=video.main_verification,
                    verification_action_hint=None,
                    final_state_frame=final_state_frame,
                    correction_actions=None,
                    expected_final_state=final_candidate.answer if final_candidate else video.main_verification,
                    error_action=None,
                    error_state=None,
                    correction_action=None,
                    post_fix_state=None,
                    source_spans=source_spans,
                    quality_flags=quality_flags,
                )
            )
    return samples


def build_verification_state_samples(
    repo_root: Path,
    output_frames_dir: Path,
    asset_writer: AssetWriter,
    inventory: Dict[str, VideoInventory],
    timelines: Dict[str, VideoTimeline],
) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    for data_id in sorted(inventory):
        video = inventory[data_id]
        timeline = timelines[data_id]
        device_family = infer_device_family(video.scenario_family, video.main_task)
        task_intent = infer_task_intent(video.main_task, video.scenario_family)
        action_answers = [candidate.answer for candidate in video.candidates if candidate.task_family == "action"]
        goal_slots = infer_goal_slots(video.main_task, action_answers)
        final_candidate = next((candidate for candidate in video.candidates if candidate.task_family == "final_state"), None)
        for candidate in [item for item in video.candidates if item.task_family == "verification_state"]:
            source_segment = find_segment(
                timeline,
                candidate.source_label,
                candidate.source_span.get("start"),
                candidate.source_span.get("end"),
                candidate.answer,
            )
            if source_segment is None:
                raise RuntimeError(f"Unable to match verification state segment for {candidate.qa_id}")
            anchor_event, anchor_source_type = select_verification_anchor(timeline, int(candidate.source_span.get("start") or 0))
            anchor_start = anchor_event.start if anchor_event and anchor_event.start is not None else int(candidate.source_span.get("start") or 0)
            anchor_end = anchor_event.end if anchor_event and anchor_event.end is not None else int(candidate.source_span.get("end") or anchor_start)
            anchor_frame = clamp_frame(anchor_start - 1, timeline.total_frames)
            boundary_event = next_boundary_event(timeline, anchor_end)
            window_segments = segments_between(
                timeline,
                anchor_end,
                boundary_event.start if boundary_event else None,
                sorted(EVIDENCE_LABELS),
            )
            primary_modality = modality_for_label(candidate.source_label) or "ui"
            secondary_segments = select_secondary_segments(window_segments, primary_modality, int(candidate.source_span.get("start") or 0))
            ui_values: List[str] = []
            physical_values: List[str] = []
            add_evidence(ui_values, physical_values, candidate.source_label, candidate.answer)
            for segment in secondary_segments:
                add_evidence(ui_values, physical_values, segment.label, segment.text)
            required_ui = list_or_none(ui_values)
            required_physical = list_or_none(physical_values)
            stop_condition = candidate.answer
            sample_id = sample_id_from_candidate(candidate, "verification_state_video")
            anchor_frame_path = extract_anchor_frame(asset_writer, timeline, repo_root, output_frames_dir, sample_id, anchor_frame)
            source_spans = [
                build_source_span(candidate.source_label, candidate.answer, candidate.source_span.get("start"), candidate.source_span.get("end"))
            ]
            if anchor_event is not None:
                source_spans.insert(
                    0,
                    build_source_span(
                        "action_description",
                        anchor_event.action_description,
                        anchor_event.start,
                        anchor_event.end,
                    ),
                )
            for segment in secondary_segments:
                source_spans.append(build_source_span(segment.label, segment.text, segment.start, segment.end))
            quality_flags: List[str] = []
            if anchor_source_type == "action":
                quality_flags.append("anchor_from_action_fallback")
            if not required_ui:
                quality_flags.append("missing_required_evidence_ui")
            if not required_physical:
                quality_flags.append("missing_required_evidence_physical")
            verification_hint = nearest_verification_hint(timeline, int(candidate.source_span.get("start") or 0))
            if verification_hint is None:
                quality_flags.append("missing_verification_action_hint")
            if timeline.local_video_path and "30fps" in timeline.local_video_path.parts:
                quality_flags.append("fallback_video_source")
            samples.append(
                build_base_sample(
                    sample_id=sample_id,
                    candidate=candidate,
                    inventory=video,
                    timeline=timeline,
                    source_video=relative_repo_path(timeline.local_video_path, repo_root),
                    task_type="verification_state_video",
                    device_family=device_family,
                    task_intent=task_intent,
                    goal_slots=goal_slots,
                    anchor_frame=anchor_frame,
                    anchor_frame_path=anchor_frame_path,
                    anchor_source_type=anchor_source_type,
                    anchor_source_span=build_source_span(
                        "action_description",
                        anchor_event.action_description if anchor_event else "anchor frame",
                        anchor_start,
                        anchor_end,
                    )
                    if anchor_event
                    else None,
                    prompt_en=build_prompt_en_verification(
                        device_family,
                        video.main_task,
                        required_ui,
                        required_physical,
                    ),
                    prompt_zh=build_prompt_zh_verification(
                        device_family,
                        video.main_task,
                        required_ui,
                        required_physical,
                    ),
                    next_action=None,
                    required_ui=required_ui,
                    required_physical=required_physical,
                    temporal_stages=None,
                    stop_condition=stop_condition,
                    overall_success_condition=video.main_verification,
                    verification_action_hint=verification_hint,
                    final_state_frame=get_final_state_frame(timeline),
                    correction_actions=None,
                    expected_final_state=final_candidate.answer if final_candidate else video.main_verification,
                    error_action=None,
                    error_state=None,
                    correction_action=None,
                    post_fix_state=None,
                    source_spans=source_spans,
                    quality_flags=quality_flags,
                )
            )
    return samples


def build_final_state_samples(
    repo_root: Path,
    output_frames_dir: Path,
    asset_writer: AssetWriter,
    inventory: Dict[str, VideoInventory],
    timelines: Dict[str, VideoTimeline],
) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    for data_id in sorted(inventory):
        video = inventory[data_id]
        timeline = timelines[data_id]
        device_family = infer_device_family(video.scenario_family, video.main_task)
        task_intent = infer_task_intent(video.main_task, video.scenario_family)
        action_answers = [candidate.answer for candidate in video.candidates if candidate.task_family == "action"]
        goal_slots = infer_goal_slots(video.main_task, action_answers)
        first_action = next((event for event in timeline.action_events if event.action_type == "execute action"), None)
        final_candidate = next((candidate for candidate in video.candidates if candidate.task_family == "final_state"), None)
        if final_candidate is None:
            continue
        final_segment = find_segment(
            timeline,
            final_candidate.source_label,
            final_candidate.source_span.get("start"),
            final_candidate.source_span.get("end"),
            final_candidate.answer,
        )
        if final_segment is None:
            raise RuntimeError(f"Unable to match final state segment for {final_candidate.qa_id}")
        final_state_frame = get_final_state_frame(timeline)
        required_ui, required_physical, support_segments = collect_final_support_evidence(
            timeline,
            final_segment,
            final_state_frame,
        )
        anchor_start = first_action.start if first_action and first_action.start is not None else 0
        anchor_frame = clamp_frame(anchor_start - 1, timeline.total_frames)
        sample_id = sample_id_from_candidate(final_candidate, "final_state_video")
        anchor_frame_path = extract_anchor_frame(asset_writer, timeline, repo_root, output_frames_dir, sample_id, anchor_frame)
        source_spans = [
            build_source_span(final_candidate.source_label, final_candidate.answer, final_candidate.source_span.get("start"), final_candidate.source_span.get("end"))
        ]
        if first_action is not None:
            source_spans.insert(
                0,
                build_source_span("action_description", first_action.action_description, first_action.start, first_action.end),
            )
        for segment in support_segments[:6]:
            source_spans.append(build_source_span(segment.label, segment.text, segment.start, segment.end))
        quality_flags: List[str] = []
        if not required_ui:
            quality_flags.append("missing_required_evidence_ui")
        if not required_physical:
            quality_flags.append("missing_required_evidence_physical")
        if final_state_frame is None:
            quality_flags.append("missing_final_state_frame")
        if timeline.local_video_path and "30fps" in timeline.local_video_path.parts:
            quality_flags.append("fallback_video_source")
        samples.append(
            build_base_sample(
                sample_id=sample_id,
                candidate=final_candidate,
                inventory=video,
                timeline=timeline,
                source_video=relative_repo_path(timeline.local_video_path, repo_root),
                task_type="final_state_video",
                device_family=device_family,
                task_intent=task_intent,
                goal_slots=goal_slots,
                anchor_frame=anchor_frame,
                anchor_frame_path=anchor_frame_path,
                anchor_source_type="action" if first_action is not None else "frame",
                anchor_source_span=build_source_span(
                    "action_description",
                    first_action.action_description,
                    first_action.start,
                    first_action.end,
                )
                if first_action is not None
                else None,
                prompt_en=build_prompt_en_final(
                    device_family,
                    video.main_task,
                    required_ui,
                    required_physical,
                    video.main_verification,
                ),
                prompt_zh=build_prompt_zh_final(
                    device_family,
                    video.main_task,
                    required_ui,
                    required_physical,
                    video.main_verification,
                ),
                next_action=None,
                required_ui=required_ui,
                required_physical=required_physical,
                temporal_stages=None,
                stop_condition=video.main_verification,
                overall_success_condition=video.main_verification,
                verification_action_hint=nearest_verification_hint(timeline, int(final_candidate.source_span.get("start") or 0)),
                final_state_frame=final_state_frame,
                correction_actions=None,
                expected_final_state=final_candidate.answer,
                error_action=None,
                error_state=None,
                correction_action=None,
                post_fix_state=None,
                source_spans=source_spans,
                quality_flags=quality_flags,
            )
        )
    return samples


def build_recovery_samples(
    repo_root: Path,
    output_frames_dir: Path,
    asset_writer: AssetWriter,
    inventory: Dict[str, VideoInventory],
    timelines: Dict[str, VideoTimeline],
) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    for data_id in sorted(inventory):
        video = inventory[data_id]
        timeline = timelines[data_id]
        wrong_actions = [candidate for candidate in video.candidates if candidate.task_family == "recovery" and candidate.qa_type == "error_action"]
        fix_actions = [candidate for candidate in video.candidates if candidate.task_family == "recovery" and candidate.qa_type == "correction_action"]
        if not wrong_actions or not fix_actions:
            continue
        device_family = infer_device_family(video.scenario_family, video.main_task)
        task_intent = infer_task_intent(video.main_task, video.scenario_family)
        action_answers = [candidate.answer for candidate in video.candidates if candidate.task_family == "action"]
        goal_slots = infer_goal_slots(video.main_task, action_answers)
        wrong_action = sorted(wrong_actions, key=lambda candidate: (span_sort_key(candidate.source_span), candidate.qa_id))[0]
        fix_actions = sorted(fix_actions, key=lambda candidate: (span_sort_key(candidate.source_span), candidate.qa_id))
        final_candidate = next((candidate for candidate in video.candidates if candidate.task_family == "final_state"), None)
        if final_candidate is None:
            continue
        final_segment = find_segment(
            timeline,
            final_candidate.source_label,
            final_candidate.source_span.get("start"),
            final_candidate.source_span.get("end"),
            final_candidate.answer,
        )
        if final_segment is None:
            raise RuntimeError(f"Unable to match final state segment for {final_candidate.qa_id}")
        post_fix_candidates = [
            candidate
            for candidate in video.candidates
            if candidate.task_family in {"verification_state", "final_state"}
            and int(candidate.source_span.get("start") or 0) >= int(fix_actions[-1].source_span.get("end") or 0)
        ]
        post_fix_candidates.sort(key=lambda candidate: (span_sort_key(candidate.source_span), candidate.qa_id))
        post_fix_state = post_fix_candidates[0].answer if post_fix_candidates else None
        error_state = choose_error_state(timeline, wrong_action, fix_actions[0])
        required_ui, required_physical, support_segments = collect_final_support_evidence(
            timeline,
            final_segment,
            get_final_state_frame(timeline),
        )
        anchor_frame = clamp_frame(int(wrong_action.source_span.get("end") or wrong_action.source_span.get("start") or 0), timeline.total_frames)
        correction_chain = [candidate.answer for candidate in fix_actions]
        sample_id = f"{data_id}_recovery_video_{qa_numeric_suffix(wrong_action.qa_id)}"
        anchor_frame_path = extract_anchor_frame(asset_writer, timeline, repo_root, output_frames_dir, sample_id, anchor_frame)
        source_spans = [
            build_source_span("action_description", wrong_action.answer, wrong_action.source_span.get("start"), wrong_action.source_span.get("end"))
        ]
        for fix_action in fix_actions:
            source_spans.append(
                build_source_span("action_description", fix_action.answer, fix_action.source_span.get("start"), fix_action.source_span.get("end"))
            )
        if post_fix_candidates:
            post_fix_candidate = post_fix_candidates[0]
            source_spans.append(
                build_source_span(
                    post_fix_candidate.source_label,
                    post_fix_candidate.answer,
                    post_fix_candidate.source_span.get("start"),
                    post_fix_candidate.source_span.get("end"),
                )
            )
        for segment in support_segments[:4]:
            source_spans.append(build_source_span(segment.label, segment.text, segment.start, segment.end))
        quality_flags: List[str] = []
        if error_state is None:
            quality_flags.append("missing_error_state")
        if post_fix_state is None:
            quality_flags.append("missing_post_fix_state")
        if not required_ui:
            quality_flags.append("missing_required_evidence_ui")
        if not required_physical:
            quality_flags.append("missing_required_evidence_physical")
        if timeline.local_video_path and "30fps" in timeline.local_video_path.parts:
            quality_flags.append("fallback_video_source")
        temporal_stages = list_or_none([wrong_action.answer, *correction_chain, post_fix_state or "null"])
        samples.append(
            build_base_sample(
                sample_id=sample_id,
                candidate=wrong_action,
                inventory=video,
                timeline=timeline,
                source_video=relative_repo_path(timeline.local_video_path, repo_root),
                task_type="recovery_video",
                device_family=device_family,
                task_intent=task_intent,
                goal_slots=goal_slots,
                anchor_frame=anchor_frame,
                anchor_frame_path=anchor_frame_path,
                anchor_source_type="recovery_error_action",
                anchor_source_span=build_source_span(
                    "action_description",
                    wrong_action.answer,
                    wrong_action.source_span.get("start"),
                    wrong_action.source_span.get("end"),
                ),
                prompt_en=build_prompt_en_recovery(
                    device_family,
                    video.main_task,
                    wrong_action.answer,
                    error_state,
                    correction_chain,
                    post_fix_state,
                    required_ui,
                    required_physical,
                ),
                prompt_zh=build_prompt_zh_recovery(
                    device_family,
                    video.main_task,
                    wrong_action.answer,
                    error_state,
                    correction_chain,
                    post_fix_state,
                    required_ui,
                    required_physical,
                ),
                next_action=None,
                required_ui=required_ui,
                required_physical=required_physical,
                temporal_stages=temporal_stages,
                stop_condition=video.main_verification,
                overall_success_condition=video.main_verification,
                verification_action_hint=nearest_verification_hint(timeline, int(fix_actions[-1].source_span.get("end") or 0)),
                final_state_frame=get_final_state_frame(timeline),
                correction_actions=correction_chain,
                expected_final_state=final_candidate.answer,
                error_action=wrong_action.answer,
                error_state=error_state,
                correction_action=" then ".join(correction_chain),
                post_fix_state=post_fix_state,
                source_spans=source_spans,
                quality_flags=quality_flags,
            )
        )
    return samples


def write_jsonl(output_path: Path, samples: Sequence[Dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")


def write_summary(output_path: Path, samples: Sequence[Dict[str, Any]]) -> None:
    task_counts = Counter(sample["task_type"] for sample in samples)
    video_counts = Counter(sample["source_video"] for sample in samples)
    flag_counts = Counter()
    for sample in samples:
        flag_counts.update(sample.get("quality_flags") or [])
    lines = [
        "# Single Image Generation Prompt v2 Summary",
        "",
        f"- Prompt version: `{PROMPT_VERSION}`",
        f"- Total samples: `{len(samples)}`",
        f"- Unique source videos: `{len(video_counts)}`",
        "",
        "## Task Counts",
        "",
    ]
    for task_type in ("state_transition_video", "final_state_video", "verification_state_video", "recovery_video"):
        lines.append(f"- `{task_type}`: `{task_counts.get(task_type, 0)}`")
    lines.extend(["", "## Video Coverage", ""])
    for video_path, count in sorted(video_counts.items()):
        lines.append(f"- `{video_path}`: `{count}`")
    lines.extend(["", "## Quality Flags", ""])
    if flag_counts:
        for flag, count in flag_counts.most_common():
            lines.append(f"- `{flag}`: `{count}`")
    else:
        lines.append("- No quality flags.")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_readme(output_path: Path, output_root: Path, samples: Sequence[Dict[str, Any]]) -> None:
    task_counts = Counter(sample["task_type"] for sample in samples)
    lines = [
        "# Single Image Generation Prompt v2 Full",
        "",
        "- Source inventory: `annotations/0421/switch/switch_all.qa_candidates.json`",
        "- Source timelines: raw `Action_1` export and raw `Action_2` export under `annotations/0421/switch/`",
        "- Output dataset: `dataset.jsonl`",
        "- Anchor frames: `frames/{sample_id}.jpg`",
        "",
        "## Task Counts",
        "",
        f"- `state_transition_video`: `{task_counts.get('state_transition_video', 0)}`",
        f"- `final_state_video`: `{task_counts.get('final_state_video', 0)}`",
        f"- `verification_state_video`: `{task_counts.get('verification_state_video', 0)}`",
        f"- `recovery_video`: `{task_counts.get('recovery_video', 0)}`",
        "",
        "## Key Fields",
        "",
        "- `prompt_version`: fixed as `single_image_gen_prompt_v2_full`",
        "- `required_evidence_ui` / `required_evidence_physical`: structured visible evidence lists, or `null`",
        "- `anchor_source_type` / `anchor_source_span`: where the anchor frame is derived from",
        "- `verification_action_hint`: metadata-only hint, not written into verification prompts",
        "- `final_state_frame`: raw `is_final_state` frame when available",
        "- `correction_actions`: ordered recovery correction chain",
        "",
        "## Notes",
        "",
        "- This directory is versioned output and does not overwrite any `30fps` artifacts.",
        "- `state_transition_video` uses local stop conditions rather than global task completion.",
        "- `verification_state_video` keeps the 86 observable success-signal samples from the QA inventory.",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def prune_stale_frames(output_root: Path, samples: Sequence[Dict[str, Any]], repo_root: Path) -> None:
    frames_dir = output_root / "frames"
    valid_paths = {
        (repo_root / sample["anchor_frame_path"]).resolve()
        for sample in samples
    }
    for frame_path in frames_dir.glob("*.jpg"):
        resolved = frame_path.resolve()
        if resolved not in valid_paths:
            frame_path.unlink()


def validate_samples(samples: Sequence[Dict[str, Any]], output_root: Path, repo_root: Path) -> None:
    task_counts = Counter(sample["task_type"] for sample in samples)
    for task_type, expected_count in EXPECTED_TASK_COUNTS.items():
        actual = task_counts.get(task_type, 0)
        if actual != expected_count:
            raise RuntimeError(f"Unexpected count for {task_type}: expected {expected_count}, got {actual}")

    for sample in samples:
        frame_path = repo_root / sample["anchor_frame_path"]
        if not frame_path.exists():
            raise RuntimeError(f"Missing anchor frame: {frame_path}")
        if "Please:" not in sample["prompt_en"] or "Do not:" not in sample["prompt_en"]:
            raise RuntimeError(f"Prompt missing required sections: {sample['sample_id']}")
        if sample["required_evidence_ui"] is not None and not isinstance(sample["required_evidence_ui"], list):
            raise RuntimeError(f"required_evidence_ui must be list or null: {sample['sample_id']}")
        if sample["required_evidence_physical"] is not None and not isinstance(sample["required_evidence_physical"], list):
            raise RuntimeError(f"required_evidence_physical must be list or null: {sample['sample_id']}")
        if not frame_path.is_relative_to(output_root):
            raise RuntimeError(f"Anchor frame path escapes output root: {sample['sample_id']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the 22-video single-image generation prompt v2 dataset.")
    parser.add_argument(
        "--annotation-root",
        type=Path,
        default=Path("annotations") / "0421" / "switch",
        help="Root directory for SWITCH annotations.",
    )
    parser.add_argument(
        "--fallback-video-dir",
        type=Path,
        default=Path("30fps"),
        help="Fallback directory for local MP4 files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("annotations") / "0421" / "switch" / PROMPT_VERSION,
        help="Output directory for the v2 dataset.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    annotation_root = (repo_root / args.annotation_root).resolve()
    fallback_video_dir = (repo_root / args.fallback_video_dir).resolve()
    output_root = (repo_root / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    frames_dir = output_root / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    inventory = load_inventory(annotation_root)
    video_index = build_video_index(annotation_root, fallback_video_dir)
    timelines = load_timelines(annotation_root, video_index)
    missing_timelines = sorted(set(inventory) - set(timelines))
    if missing_timelines:
        raise RuntimeError(f"Missing timelines for data ids: {missing_timelines}")
    missing_videos = sorted(data_id for data_id, timeline in timelines.items() if timeline.local_video_path is None)
    if missing_videos:
        raise RuntimeError(f"Missing local videos for data ids: {missing_videos}")

    asset_writer = AssetWriter()
    samples: List[Dict[str, Any]] = []
    samples.extend(build_state_transition_samples(repo_root, frames_dir, asset_writer, inventory, timelines))
    samples.extend(build_final_state_samples(repo_root, frames_dir, asset_writer, inventory, timelines))
    samples.extend(build_verification_state_samples(repo_root, frames_dir, asset_writer, inventory, timelines))
    samples.extend(build_recovery_samples(repo_root, frames_dir, asset_writer, inventory, timelines))

    samples.sort(key=lambda sample: (sample["sample_id"]))

    dataset_path = output_root / "dataset.jsonl"
    summary_path = output_root / "summary.md"
    readme_path = output_root / "README.md"
    write_jsonl(dataset_path, samples)
    write_summary(summary_path, samples)
    write_readme(readme_path, output_root, samples)
    prune_stale_frames(output_root, samples, repo_root)
    validate_samples(samples, output_root, repo_root)

    print(f"Wrote dataset: {dataset_path}")
    print(f"Wrote summary: {summary_path}")
    print(f"Wrote readme: {readme_path}")
    print(f"Generated samples: {len(samples)}")


if __name__ == "__main__":
    main()
