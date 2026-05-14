#此脚本针对D:\Search\BAAI\SWITCH\annotations\latest\灯光与照明控制-问卷_1
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

import cv2


SOURCE_FOLDER_NAME = "\u706f\u5149\u4e0e\u7167\u660e\u63a7\u5236-\u95ee\u5377_1"
SOURCE_JSON_NAME = "Action_5_Light_1.json"
OUTPUT_NAME = "hf_innovative_qa_v2_multiform_lighting_v1"
SOURCE_FILE = SOURCE_JSON_NAME
SCHEMA_VERSION = "switch-hf-innovative-qa-v2-multiform-lighting-v1"
SCENARIO_FAMILY = "lighting_control"
LABELS = ["A", "B", "C", "D"]
LABELS5 = ["A", "B", "C", "D", "E"]
TMP_CLIP_DIR = Path.cwd() / ".codex_tmp" / "lighting_qa_clip_tmp"
ACTION_IMG_FIRST_ACTION_OFFSET_MIN = 40
ACTION_IMG_FIRST_ACTION_OFFSET_MAX = 50
ACTION_VIDEO_FIRST_ACTION_END_OFFSET_MIN = 50
ACTION_VIDEO_FIRST_ACTION_END_OFFSET_MAX = 60
ACTION_VIDEO_FIRST_ACTION_CLIP_LENGTH = 60


@dataclass
class ActionSegment:
    data_id: str
    video_name: str
    start: int
    end: int
    step_id: str
    action_type: str
    requirement: str
    description: str
    anchor_frame: int

    @property
    def origin_qa_id(self) -> str:
        return f"{self.data_id}_action_{int(self.step_id or 0):03d}"

    @property
    def is_verification(self) -> bool:
        return "verification" in normalize_text(self.action_type)


@dataclass
class UiChange:
    data_id: str
    video_name: str
    frame: int
    text: str

    @property
    def origin_qa_id(self) -> str:
        return f"{self.data_id}_state_ui_{self.frame:06d}"


@dataclass
class VideoRecord:
    data_id: str
    video_name: str
    video_path: Path
    fps: float
    total_frames: int
    duration: float
    overall_requirement: str
    overall_verification: str
    final_state_frame: int
    actions: list[ActionSegment] = field(default_factory=list)
    ui_changes: list[UiChange] = field(default_factory=list)

    @property
    def full_span(self) -> dict:
        return {"start": 1, "end": self.total_frames}

    @property
    def last_action(self) -> ActionSegment | None:
        if not self.actions:
            return None
        return sorted(self.actions, key=lambda x: (x.end, x.start))[-1]


def normalize_text(value: str) -> str:
    return " ".join((value or "").lower().split())


def stable_int(value: str) -> int:
    return int(hashlib.md5(value.encode("utf-8")).hexdigest()[:8], 16)


def clean_answer(value: str) -> str:
    return " ".join((value or "").strip().split())


def sentence(value: str) -> str:
    value = clean_answer(value)
    if not value:
        return value
    return value if value.endswith((".", "?", "!")) else value + "."


def aliases(value: str) -> list[str]:
    answer = clean_answer(value)
    alias_list = [answer]
    fixed = answer.replace("botton", "button").replace("motton", "button")
    if fixed != answer:
        alias_list.append(fixed)
    return alias_list


def video_data_key(item: dict) -> str:
    for key, value in item.get("data", {}).items():
        if key == "meta":
            continue
        if isinstance(value, str) and value.lower().split("?")[0].endswith((".mp4", ".mov", ".avi", ".mkv")):
            return key
    return "Action_5_Light_1"


def video_name_from_url(value: str) -> str:
    parsed = urlparse(value)
    return Path(parsed.path).name or Path(value).name


def first_text(result: dict) -> str:
    text = (result.get("meta") or {}).get("text", [])
    if isinstance(text, list):
        return clean_answer("; ".join(str(x) for x in text if x is not None))
    if text is None:
        return ""
    return clean_answer(str(text))


def parse_records(source_json: Path, source_dir: Path) -> list[VideoRecord]:
    data = json.loads(source_json.read_text(encoding="utf-8"))
    records: list[VideoRecord] = []

    for item in data:
        key = video_data_key(item)
        video_name = video_name_from_url(item.get("data", {}).get(key, ""))
        video_path = source_dir / video_name
        meta = item.get("data", {}).get("meta", {}).get(key, {})
        fps = float(meta.get("fps") or 30.0)
        total_frames = int(round(float(meta.get("total_frames") or 0)))
        duration = float(meta.get("duration") or (total_frames / fps if fps else 0.0))
        result_list = (item.get("annotations") or [{}])[0].get("result") or []

        scalar_labels: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
        for result in result_list:
            labels = result.get("value", {}).get("timelinelabels", [])
            ranges = result.get("value", {}).get("ranges", [])
            text = first_text(result)
            for range_value in ranges:
                start = int(round(range_value.get("start", 0)))
                end = int(round(range_value.get("end", 0)))
                for label in labels:
                    scalar_labels[label].append((start, end, text))

        data_id = next((text for _, _, text in scalar_labels.get("data_id", []) if text), Path(video_name).stem)
        overall_requirement = next((text for _, _, text in scalar_labels.get("overall_requirement", []) if text), "")
        overall_verification = next((text for _, _, text in scalar_labels.get("overall_verification", []) if text), "")
        final_state_frame = next((start for start, end, _ in scalar_labels.get("is_final_state", []) if start == end), total_frames)

        span_starts = sorted({start for start, end, _ in scalar_labels.get("action_description", [])})
        actions: list[ActionSegment] = []
        for start in span_starts:
            desc_row = next((row for row in scalar_labels["action_description"] if row[0] == start), None)
            if not desc_row:
                continue
            _, end, description = desc_row
            action_type = next((text for s, e, text in scalar_labels.get("action-type", []) if s == start and e == end), "")
            requirement = next((text for s, e, text in scalar_labels.get("action_requirement", []) if s == start and e == end), "")
            step_id = next((text for s, e, text in scalar_labels.get("action_step_id", []) if s == start and e == end), "")
            anchor = next((s for s, e, _ in scalar_labels.get("action_anchor", []) if start <= s <= end and s == e), (start + end) // 2)
            actions.append(
                ActionSegment(
                    data_id=data_id,
                    video_name=video_name,
                    start=start,
                    end=end,
                    step_id=step_id,
                    action_type=action_type,
                    requirement=requirement,
                    description=description,
                    anchor_frame=anchor,
                )
            )

        ui_changes = [
            UiChange(data_id=data_id, video_name=video_name, frame=start, text=text)
            for start, end, text in scalar_labels.get("ui_change", [])
            if start == end and text
        ]

        records.append(
            VideoRecord(
                data_id=data_id,
                video_name=video_name,
                video_path=video_path,
                fps=fps,
                total_frames=total_frames,
                duration=duration,
                overall_requirement=overall_requirement,
                overall_verification=overall_verification,
                final_state_frame=final_state_frame,
                actions=sorted(actions, key=lambda x: (int(x.step_id or 0), x.start)),
                ui_changes=ui_changes,
            )
        )

    return records


def clamp_frame(frame: int, total_frames: int) -> int:
    return max(1, min(int(frame), max(1, total_frames)))


def previous_action_segment(record: VideoRecord, current_action: ActionSegment) -> ActionSegment | None:
    for index, action in enumerate(record.actions):
        if action.origin_qa_id == current_action.origin_qa_id:
            return record.actions[index - 1] if index > 0 else None
    return None


def select_action_img2txt_query_frame(record: VideoRecord, action: ActionSegment) -> tuple[int, str, str | None]:
    previous = previous_action_segment(record, action)
    if previous is not None:
        frame = clamp_frame(min(previous.anchor_frame, action.start - 1), record.total_frames)
        return frame, "previous_action_anchor", previous.origin_qa_id

    offset_span = ACTION_IMG_FIRST_ACTION_OFFSET_MAX - ACTION_IMG_FIRST_ACTION_OFFSET_MIN + 1
    offset = ACTION_IMG_FIRST_ACTION_OFFSET_MIN + (stable_int(action.origin_qa_id) % offset_span)
    candidate = min(action.anchor_frame - offset, action.start - 1)
    frame = clamp_frame(candidate, record.total_frames)
    return frame, f"pre_action_offset_{offset}", None


def select_first_action_pre_video_range(
    record: VideoRecord, action: ActionSegment
) -> tuple[int, int, str]:
    offset_span = ACTION_VIDEO_FIRST_ACTION_END_OFFSET_MAX - ACTION_VIDEO_FIRST_ACTION_END_OFFSET_MIN + 1
    offset = ACTION_VIDEO_FIRST_ACTION_END_OFFSET_MIN + (stable_int(action.origin_qa_id) % offset_span)
    end = clamp_frame(min(action.anchor_frame - offset, action.start - 1), record.total_frames)
    start = clamp_frame(end - ACTION_VIDEO_FIRST_ACTION_CLIP_LENGTH + 1, record.total_frames)
    if end < start:
        end = start
    return start, end, f"pre_action_offset_{offset}"


def select_final_state_img2img_query_frame(
    record: VideoRecord, candidate_frame: int
) -> tuple[int, str, str | None]:
    if len(record.actions) >= 2:
        previous = record.actions[-2]
        frame = clamp_frame(min(previous.anchor_frame, candidate_frame - 1), record.total_frames)
        return frame, "previous_action_anchor", previous.origin_qa_id

    if record.last_action is not None:
        frame = clamp_frame(min(record.last_action.anchor_frame, candidate_frame - 1), record.total_frames)
        return frame, "last_action_anchor_fallback", record.last_action.origin_qa_id

    frame = clamp_frame(candidate_frame - 15, record.total_frames)
    return frame, "pre_final_state_default", None


def select_final_state_video2img_query_range(
    record: VideoRecord, candidate_frame: int
) -> tuple[int, int, str, str | None]:
    if len(record.actions) >= 2:
        action = record.actions[-2]
        start = clamp_frame(action.start, record.total_frames)
        end = clamp_frame(min(action.end, candidate_frame - 1), record.total_frames)
        if end < start:
            end = start
        return start, end, "penultimate_action_segment", action.origin_qa_id

    if record.last_action is not None:
        action = record.last_action
        start = clamp_frame(action.start, record.total_frames)
        end = clamp_frame(min(action.end, candidate_frame - 1), record.total_frames)
        if end < start:
            end = start
        return start, end, "last_action_segment_fallback", action.origin_qa_id

    end = clamp_frame(candidate_frame - 10, record.total_frames)
    start = clamp_frame(end - 45, record.total_frames)
    return start, end, "pre_final_state_default", None


def select_action_video2txt_query_range(
    record: VideoRecord, action: ActionSegment
) -> tuple[int, int, str, str | None]:
    previous = previous_action_segment(record, action)
    if previous is not None:
        start = clamp_frame(previous.start, record.total_frames)
        end = clamp_frame(min(previous.end, action.start - 1), record.total_frames)
        if end < start:
            end = start
        return start, end, "previous_action_segment", previous.origin_qa_id

    start, end, rule = select_first_action_pre_video_range(record, action)
    return start, end, rule, None


def select_action_video2video_query_range(
    record: VideoRecord, action: ActionSegment
) -> tuple[int, int, str, str | None]:
    previous = previous_action_segment(record, action)
    if previous is not None:
        start = clamp_frame(previous.start, record.total_frames)
        end = clamp_frame(min(previous.end, action.start - 1), record.total_frames)
        if end < start:
            end = start
        return start, end, "previous_action_segment", previous.origin_qa_id

    start, end, rule = select_first_action_pre_video_range(record, action)
    return start, end, rule, None


def read_frame(video_path: Path, frame: int) -> tuple[bool, object | None]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False, None
    total = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
    frame = clamp_frame(frame, total)
    zero_based = frame - 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, zero_based)
    ok, image = cap.read()
    if not ok or image is None:
        for fallback in range(min(zero_based, total - 1), max(-1, zero_based - 30), -1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fallback)
            ok, image = cap.read()
            if ok and image is not None:
                break
    cap.release()
    return bool(ok and image is not None), image


def write_frame(video_path: Path, frame: int, output_path: Path) -> None:
    ok, image = read_frame(video_path, frame)
    if not ok or image is None:
        raise RuntimeError(f"Could not read frame {frame} from {video_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise RuntimeError(f"Could not encode frame {frame} from {video_path}")
    encoded.tofile(str(output_path))
    if not output_path.exists():
        raise RuntimeError(f"Could not write image to {output_path}")


def write_clip(video_path: Path, start: int, end: int, output_path: Path, fps: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    TMP_CLIP_DIR.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video {video_path}")
    total = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
    width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0))
    height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0))
    start = clamp_frame(start, total)
    end = clamp_frame(max(start, end), total)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    tmp_path = TMP_CLIP_DIR / f"{stable_int(str(output_path))}_{output_path.name}"
    writer = cv2.VideoWriter(str(tmp_path), fourcc, float(fps or 30.0), (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create temporary clip {tmp_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start - 1)
    written = 0
    for _ in range(start, end + 1):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        writer.write(frame)
        written += 1
    writer.release()
    cap.release()
    if written <= 0 or not tmp_path.exists():
        raise RuntimeError(f"Could not write clip {output_path}")
    shutil.move(str(tmp_path), str(output_path))


def ensure_video_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copy2(src, dst)


def rel(path: Path, form_dir: Path) -> str:
    return path.relative_to(form_dir).as_posix()


def unique_texts(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        value = clean_answer(value)
        key = normalize_text(value)
        if value and key not in seen:
            out.append(value)
            seen.add(key)
    return out


def text_candidate(origin_qa_id: str, text: str, source_type: str = "annotation_text") -> dict:
    return {
        "origin_qa_id": origin_qa_id,
        "text": clean_answer(text),
        "source_type": source_type,
    }


def fallback_text_candidates() -> list[dict]:
    return [
        text_candidate("distractor::fallback::room_remains_dark", "Check whether the room remains dark", "fallback_text"),
        text_candidate("distractor::fallback::different_lighting_control", "Press a different lighting control", "fallback_text"),
        text_candidate("distractor::fallback::look_away", "Look away from the switch panel", "fallback_text"),
        text_candidate("distractor::fallback::wait_without_change", "Wait without changing the light state", "fallback_text"),
        text_candidate("distractor::fallback::wrong_light_on", "Confirm that the wrong light is on", "fallback_text"),
    ]


def normalize_text_candidate(value: object, fallback_origin: str) -> dict:
    if isinstance(value, dict):
        return text_candidate(
            str(value.get("origin_qa_id") or fallback_origin),
            str(value.get("text") or ""),
            str(value.get("source_type") or "annotation_text"),
        )
    return text_candidate(fallback_origin, str(value), "legacy_text_pool")


def unique_text_candidates(values: list[object]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for idx, value in enumerate(values):
        candidate = normalize_text_candidate(value, f"distractor::legacy_text_pool::{idx:03d}")
        key = normalize_text(candidate["text"])
        if candidate["text"] and key not in seen:
            out.append(candidate)
            seen.add(key)
    return out


def pick_distractor_text_candidates(correct: str, pool: list[object], origin: str, k: int = 3) -> list[dict]:
    candidates = [
        candidate
        for candidate in unique_text_candidates(pool + fallback_text_candidates())
        if normalize_text(candidate["text"]) != normalize_text(correct)
    ]
    annotation_candidates = [candidate for candidate in candidates if candidate["source_type"] != "fallback_text"]
    fallback_candidates = [candidate for candidate in candidates if candidate["source_type"] == "fallback_text"]
    rng = random.Random(stable_int(origin + "::text_options"))
    rng.shuffle(annotation_candidates)
    rng.shuffle(fallback_candidates)
    selected = annotation_candidates[:k]
    if len(selected) < k:
        selected.extend(fallback_candidates[: k - len(selected)])
    return selected


def build_text_options(
    correct: str,
    pool: list[object],
    origin: str,
    correct_origin_qa_id: str | None = None,
) -> tuple[str, dict[str, str], str, list[str], list[str]]:
    correct_candidate = text_candidate(correct_origin_qa_id or origin, correct, "correct_annotation_text")
    options = [correct_candidate] + pick_distractor_text_candidates(correct, pool, origin)
    rng = random.Random(stable_int(origin + "::shuffle"))
    rng.shuffle(options)
    option_map = {label: option["text"] for label, option in zip(LABELS, options)}
    gt = next(label for label, option in option_map.items() if normalize_text(option) == normalize_text(correct))
    option_origin_ids = [option["origin_qa_id"] for option in options]
    option_source_types = [option["source_type"] for option in options]
    return gt, option_map, sentence(option_map[gt]).rstrip("."), option_origin_ids, option_source_types


def ui_change_text_candidate(candidate: dict, source_type: str | None = None) -> dict:
    out = text_candidate(
        candidate["origin_qa_id"],
        candidate.get("answer") or candidate.get("text") or "",
        source_type or candidate.get("source_type", "ui_change_state_frame"),
    )
    out["category_name"] = candidate.get("category_name", category_key(SOURCE_FOLDER_NAME))
    out["data_id"] = candidate.get("data_id") or (candidate.get("record").data_id if candidate.get("record") is not None else "")
    return out


def text_difference_score(a: str, b: str) -> float:
    a_norm = normalize_text(a)
    b_norm = normalize_text(b)
    ratio_distance = 1.0 - difflib.SequenceMatcher(None, a_norm, b_norm).ratio()
    a_tokens = set(a_norm.split())
    b_tokens = set(b_norm.split())
    if not a_tokens or not b_tokens:
        return ratio_distance
    jaccard_distance = 1.0 - (len(a_tokens & b_tokens) / len(a_tokens | b_tokens))
    return 0.55 * ratio_distance + 0.45 * jaccard_distance


def action_candidate(
    origin_qa_id: str,
    text: str,
    source_type: str,
    category_name: str,
    data_id: str,
) -> dict:
    candidate = text_candidate(origin_qa_id, text, source_type)
    candidate["category_name"] = category_name
    candidate["data_id"] = data_id
    return candidate


def final_state_text_candidate(
    origin_qa_id: str,
    text: str,
    source_type: str,
    category_name: str,
    data_id: str,
) -> dict:
    candidate = text_candidate(origin_qa_id, text, source_type)
    candidate["category_name"] = category_name
    candidate["data_id"] = data_id
    return candidate


def unique_action_candidates(values: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for candidate in values:
        key = normalize_text(candidate.get("text", ""))
        if key and key not in seen:
            out.append(candidate)
            seen.add(key)
    return out


def unique_final_state_text_candidates(values: list[dict]) -> list[dict]:
    return unique_action_candidates(values)


def load_external_action_text_candidates(current_category_dir: Path, target: str = "different") -> list[dict]:
    if target not in {"same", "different"}:
        raise ValueError(f"Unsupported category target: {target}")
    latest_root = current_category_dir.parent
    current_category = category_key(current_category_dir.name)
    pool: list[dict] = []
    for category_dir in sorted(latest_root.iterdir()):
        if not category_dir.is_dir() or category_dir.resolve() == current_category_dir.resolve():
            continue
        category_name = category_key(category_dir.name)
        if target == "same" and category_name != current_category:
            continue
        if target == "different" and category_name == current_category:
            continue
        for json_path in raw_annotation_jsons_for_category(category_dir):
            try:
                records = parse_records(json_path, json_path.parent)
            except Exception:
                continue
            for record in records:
                for action in record.actions:
                    if action.description:
                        pool.append(
                            action_candidate(
                                f"{category_dir.name}::{action.origin_qa_id}",
                                action.description,
                                f"{target}_category_action",
                                category_name,
                                record.data_id,
                            )
                        )
            if pool:
                break
    return unique_action_candidates(pool)


def load_external_verification_action_text_candidates(current_category_dir: Path, target: str = "different") -> list[dict]:
    if target not in {"same", "different"}:
        raise ValueError(f"Unsupported category target: {target}")
    latest_root = current_category_dir.parent
    current_category = category_key(current_category_dir.name)
    pool: list[dict] = []
    for category_dir in sorted(latest_root.iterdir()):
        if not category_dir.is_dir() or category_dir.resolve() == current_category_dir.resolve():
            continue
        category_name = category_key(category_dir.name)
        if target == "same" and category_name != current_category:
            continue
        if target == "different" and category_name == current_category:
            continue
        for json_path in raw_annotation_jsons_for_category(category_dir):
            try:
                records = parse_records(json_path, json_path.parent)
            except Exception:
                continue
            for record in records:
                for action in record.actions:
                    if action.is_verification and action.description:
                        pool.append(
                            action_candidate(
                                f"{category_dir.name}::{action.data_id}_verify_{int(action.step_id or 0):03d}",
                                action.description,
                                f"{target}_category_verification_action",
                                category_name,
                                record.data_id,
                            )
                        )
            if pool:
                break
    return unique_action_candidates(pool)


def load_external_final_state_text_candidates(current_category_dir: Path, target: str = "different") -> list[dict]:
    if target not in {"same", "different"}:
        raise ValueError(f"Unsupported category target: {target}")
    latest_root = current_category_dir.parent
    current_category = category_key(current_category_dir.name)
    pool: list[dict] = []
    for category_dir in sorted(latest_root.iterdir()):
        if not category_dir.is_dir() or category_dir.resolve() == current_category_dir.resolve():
            continue
        category_name = category_key(category_dir.name)
        if target == "same" and category_name != current_category:
            continue
        if target == "different" and category_name == current_category:
            continue
        for json_path in raw_annotation_jsons_for_category(category_dir):
            try:
                records = parse_records(json_path, json_path.parent)
            except Exception:
                continue
            for record in records:
                if record.overall_verification:
                    pool.append(
                        final_state_text_candidate(
                            f"{category_dir.name}::{record.data_id}_final_001",
                            record.overall_verification,
                            f"{target}_category_final_state",
                            category_name,
                            record.data_id,
                        )
                    )
            if pool:
                break
    return unique_final_state_text_candidates(pool)


def build_action_video2txt_text_options(
    correct_action: ActionSegment,
    same_category_pool: list[dict],
    different_category_pool: list[dict],
    origin: str,
    current_category_name: str,
) -> tuple[str, dict[str, str], str, list[str], list[str], list[str], dict]:
    correct = action_candidate(
        origin,
        correct_action.description,
        "correct_annotation_text",
        current_category_name,
        correct_action.data_id,
    )
    correct_key = normalize_text(correct["text"])
    same_candidates = [
        dict(candidate, source_type="same_category_far_action")
        for candidate in same_category_pool
        if normalize_text(candidate["text"]) != correct_key
    ]
    same_other_video = [candidate for candidate in same_candidates if candidate.get("data_id") != correct_action.data_id]
    same_rank_pool = same_other_video or same_candidates
    same_rank_pool = sorted(
        unique_action_candidates(same_rank_pool),
        key=lambda candidate: (
            text_difference_score(correct["text"], candidate["text"]),
            stable_int(origin + "::same::" + candidate["origin_qa_id"]),
        ),
        reverse=True,
    )

    cross_candidates = [
        dict(candidate, source_type="different_category_action")
        for candidate in different_category_pool
        if normalize_text(candidate["text"]) != correct_key
    ]
    rng = random.Random(stable_int(origin + "::category_aware_text_options"))
    rng.shuffle(cross_candidates)
    cross_candidates = unique_action_candidates(cross_candidates)

    selected: list[dict] = []
    if same_rank_pool:
        selected.append(same_rank_pool[0])

    used_texts = {correct_key, *(normalize_text(candidate["text"]) for candidate in selected)}
    for candidate in cross_candidates:
        key = normalize_text(candidate["text"])
        if key in used_texts:
            continue
        selected.append(candidate)
        used_texts.add(key)
        if len([x for x in selected if x["source_type"] == "different_category_action"]) >= 2:
            break

    # Fill any remaining slots with real annotation candidates before using fallback.
    fill_pool = same_rank_pool[1:] + cross_candidates
    for candidate in fill_pool:
        if len(selected) >= 3:
            break
        key = normalize_text(candidate["text"])
        if key in used_texts:
            continue
        selected.append(candidate)
        used_texts.add(key)

    if len(selected) < 3:
        for candidate in fallback_text_candidates():
            if len(selected) >= 3:
                break
            selected.append(candidate)

    options = [correct] + selected[:3]
    rng.shuffle(options)
    option_map = {label: option["text"] for label, option in zip(LABELS, options)}
    gt = next(label for label, option in option_map.items() if normalize_text(option) == correct_key)
    option_origin_ids = [option["origin_qa_id"] for option in options]
    option_source_types = [option["source_type"] for option in options]
    option_category_names = [option.get("category_name", current_category_name) for option in options]
    strategy_meta = {
        "distractor_strategy": "category_aware_one_same_category_far_two_different_category",
        "same_category_distractor_count": sum(1 for x in option_source_types if x == "same_category_far_action"),
        "different_category_distractor_count": sum(1 for x in option_source_types if x == "different_category_action"),
        "fallback_distractor_count": sum(1 for x in option_source_types if x == "fallback_text"),
    }
    return gt, option_map, sentence(option_map[gt]).rstrip("."), option_origin_ids, option_source_types, option_category_names, strategy_meta


def build_verification_action_img2txt_text_options(
    correct_action: ActionSegment,
    same_category_pool: list[dict],
    different_category_pool: list[dict],
    origin: str,
    current_category_name: str,
) -> tuple[str, dict[str, str], str, list[str], list[str], list[str], dict]:
    correct = action_candidate(
        origin,
        correct_action.description,
        "correct_annotation_text",
        current_category_name,
        correct_action.data_id,
    )
    correct_key = normalize_text(correct["text"])
    same_candidates = [
        dict(candidate, source_type="same_category_verification_action")
        for candidate in same_category_pool
        if normalize_text(candidate["text"]) != correct_key
    ]
    same_other_video = [candidate for candidate in same_candidates if candidate.get("data_id") != correct_action.data_id]
    same_rank_pool = same_other_video or same_candidates
    same_rank_pool = sorted(
        unique_action_candidates(same_rank_pool),
        key=lambda candidate: (
            text_difference_score(correct["text"], candidate["text"]),
            stable_int(origin + "::same_verify::" + candidate["origin_qa_id"]),
        ),
        reverse=True,
    )

    cross_candidates = [
        dict(candidate, source_type="different_category_verification_action")
        for candidate in different_category_pool
        if normalize_text(candidate["text"]) != correct_key
    ]
    rng = random.Random(stable_int(origin + "::verification_action_img2txt_options"))
    rng.shuffle(cross_candidates)
    cross_candidates = unique_action_candidates(cross_candidates)

    selected: list[dict] = []
    if same_rank_pool:
        selected.append(same_rank_pool[0])

    used_texts = {correct_key, *(normalize_text(candidate["text"]) for candidate in selected)}
    for candidate in cross_candidates:
        key = normalize_text(candidate["text"])
        if key in used_texts:
            continue
        selected.append(candidate)
        used_texts.add(key)
        if len([x for x in selected if x["source_type"] == "different_category_verification_action"]) >= 2:
            break

    fill_pool = same_rank_pool[1:] + cross_candidates
    for candidate in fill_pool:
        if len(selected) >= 3:
            break
        key = normalize_text(candidate["text"])
        if key in used_texts:
            continue
        selected.append(candidate)
        used_texts.add(key)

    if len(selected) < 3:
        for candidate in fallback_text_candidates():
            if len(selected) >= 3:
                break
            selected.append(candidate)

    options = [correct] + selected[:3]
    rng.shuffle(options)
    option_map = {label: option["text"] for label, option in zip(LABELS, options)}
    gt = next(label for label, option in option_map.items() if normalize_text(option) == correct_key)
    option_origin_ids = [option["origin_qa_id"] for option in options]
    option_source_types = [option["source_type"] for option in options]
    option_category_names = [option.get("category_name", current_category_name) for option in options]
    strategy_meta = {
        "distractor_strategy": "category_aware_one_same_category_two_different_category",
        "same_category_distractor_count": sum(1 for x in option_source_types if x == "same_category_verification_action"),
        "different_category_distractor_count": sum(1 for x in option_source_types if x == "different_category_verification_action"),
        "fallback_distractor_count": sum(1 for x in option_source_types if x == "fallback_text"),
    }
    return gt, option_map, sentence(option_map[gt]).rstrip("."), option_origin_ids, option_source_types, option_category_names, strategy_meta


def build_final_state_img2txt_text_options(
    correct_record: VideoRecord,
    same_category_pool: list[dict],
    different_category_pool: list[dict],
    origin: str,
    current_category_name: str,
) -> tuple[str, dict[str, str], str, list[str], list[str], list[str], dict]:
    correct = final_state_text_candidate(
        origin,
        correct_record.overall_verification,
        "correct_annotation_text",
        current_category_name,
        correct_record.data_id,
    )
    correct_key = normalize_text(correct["text"])
    same_candidates = [
        dict(candidate, source_type="same_category_far_final_state")
        for candidate in same_category_pool
        if normalize_text(candidate["text"]) != correct_key
    ]
    same_other_video = [candidate for candidate in same_candidates if candidate.get("data_id") != correct_record.data_id]
    same_rank_pool = same_other_video or same_candidates
    same_rank_pool = sorted(
        unique_final_state_text_candidates(same_rank_pool),
        key=lambda candidate: (
            text_difference_score(correct["text"], candidate["text"]),
            stable_int(origin + "::same_final::" + candidate["origin_qa_id"]),
        ),
        reverse=True,
    )

    cross_candidates = [
        dict(candidate, source_type="different_category_final_state")
        for candidate in different_category_pool
        if normalize_text(candidate["text"]) != correct_key
    ]
    rng = random.Random(stable_int(origin + "::final_state_category_aware_text_options"))
    rng.shuffle(cross_candidates)
    cross_candidates = unique_final_state_text_candidates(cross_candidates)

    selected: list[dict] = []
    if same_rank_pool:
        selected.append(same_rank_pool[0])

    used_texts = {correct_key, *(normalize_text(candidate["text"]) for candidate in selected)}
    for candidate in cross_candidates:
        key = normalize_text(candidate["text"])
        if key in used_texts:
            continue
        selected.append(candidate)
        used_texts.add(key)
        if len([x for x in selected if x["source_type"] == "different_category_final_state"]) >= 2:
            break

    fill_pool = same_rank_pool[1:] + cross_candidates
    for candidate in fill_pool:
        if len(selected) >= 3:
            break
        key = normalize_text(candidate["text"])
        if key in used_texts:
            continue
        selected.append(candidate)
        used_texts.add(key)

    if len(selected) < 3:
        for candidate in fallback_text_candidates():
            if len(selected) >= 3:
                break
            selected.append(candidate)

    options = [correct] + selected[:3]
    rng.shuffle(options)
    option_map = {label: option["text"] for label, option in zip(LABELS, options)}
    gt = next(label for label, option in option_map.items() if normalize_text(option) == correct_key)
    option_origin_ids = [option["origin_qa_id"] for option in options]
    option_source_types = [option["source_type"] for option in options]
    option_category_names = [option.get("category_name", current_category_name) for option in options]
    strategy_meta = {
        "distractor_strategy": "category_aware_one_same_category_far_two_different_category",
        "same_category_distractor_count": sum(1 for x in option_source_types if x == "same_category_far_final_state"),
        "different_category_distractor_count": sum(1 for x in option_source_types if x == "different_category_final_state"),
        "fallback_distractor_count": sum(1 for x in option_source_types if x == "fallback_text"),
    }
    return gt, option_map, sentence(option_map[gt]).rstrip("."), option_origin_ids, option_source_types, option_category_names, strategy_meta


def with_options(query: str, option_map: dict[str, str]) -> str:
    lines = [query.rstrip()]
    for label in LABELS:
        lines.append(f"{label}. {sentence(option_map[label])}")
    return "\n".join(lines) + "\n"


def build_vqa_state_final_state_text_options(
    correct_origin: str,
    correct_answer: str,
    current_data_id: str,
    same_category_final_state_pool: list[dict],
    same_category_other_state_pool: list[dict],
    different_category_state_pool: list[dict],
    current_category_name: str,
) -> tuple[str, dict[str, str], list[str], list[str], list[str], list[str], list[float | None]]:
    correct = final_state_text_candidate(
        correct_origin,
        correct_answer,
        "correct_annotation_text",
        current_category_name,
        current_data_id,
    )
    correct_key = normalize_text(correct_answer)
    used_texts = {correct_key}
    options = [correct]

    def ranked(candidates: list[dict], source_type: str) -> list[dict]:
        prepared = []
        for candidate in candidates:
            text = candidate.get("text", "")
            key = normalize_text(text)
            if not key or key in used_texts or key == correct_key:
                continue
            item = dict(candidate, source_type=source_type)
            item["semantic_distance_to_correct"] = text_difference_score(text, correct_answer)
            prepared.append(item)
        prepared.sort(
            key=lambda item: (
                item.get("semantic_distance_to_correct", 0.0),
                stable_int(correct_origin + "::" + item["origin_qa_id"]),
            ),
            reverse=True,
        )
        return prepared

    same_final_candidates = ranked(same_category_final_state_pool, "same_category_final_state_distractor")
    same_other_candidates = ranked(same_category_other_state_pool, "same_category_other_state_distractor")
    different_candidates = ranked(different_category_state_pool, "different_category_state_distractor")

    for candidate_list in [same_final_candidates, same_other_candidates, different_candidates]:
        for candidate in candidate_list:
            key = normalize_text(candidate["text"])
            if key in used_texts:
                continue
            options.append(candidate)
            used_texts.add(key)
            break

    if len(options) < 4:
        fallback_candidates = ranked(
            same_final_candidates + same_other_candidates + different_candidates,
            "fallback_vqa_state_distractor",
        )
        for candidate in fallback_candidates:
            key = normalize_text(candidate["text"])
            if key in used_texts:
                continue
            options.append(candidate)
            used_texts.add(key)
            if len(options) == 4:
                break

    rng = random.Random(stable_int(correct_origin + "::vqa_state_final_state_options_shuffle"))
    rng.shuffle(options)
    option_map = {label: option["text"] for label, option in zip(LABELS, options)}
    gt = next(label for label, option in option_map.items() if normalize_text(option) == correct_key)
    option_origin_ids = [option["origin_qa_id"] for option in options]
    option_source_types = [option["source_type"] for option in options]
    option_category_names = [option.get("category_name", current_category_name) for option in options]
    option_data_ids = [option.get("data_id", "") for option in options]
    option_semantic_distances = [option.get("semantic_distance_to_correct") for option in options]
    return gt, option_map, option_origin_ids, option_source_types, option_category_names, option_data_ids, option_semantic_distances


def build_vqa_state_ui_change_text_options(
    correct_origin: str,
    correct_answer: str,
    current_data_id: str,
    same_category_ui_pool: list[dict],
    same_category_other_state_pool: list[dict],
    different_category_state_pool: list[dict],
    current_category_name: str,
) -> tuple[str, dict[str, str], list[str], list[str], list[str], list[str], list[float | None]]:
    correct = final_state_text_candidate(
        correct_origin,
        correct_answer,
        "correct_annotation_text",
        current_category_name,
        current_data_id,
    )
    correct_key = normalize_text(correct_answer)
    used_texts = {correct_key}
    options = [correct]

    def ranked(candidates: list[dict], source_type: str) -> list[dict]:
        prepared = []
        for candidate in candidates:
            text = candidate.get("text", "")
            key = normalize_text(text)
            if not key or key in used_texts or key == correct_key:
                continue
            item = dict(candidate, source_type=source_type)
            item["semantic_distance_to_correct"] = text_difference_score(text, correct_answer)
            prepared.append(item)
        prepared.sort(
            key=lambda item: (
                item.get("semantic_distance_to_correct", 0.0),
                stable_int(correct_origin + "::" + item["origin_qa_id"]),
            ),
            reverse=True,
        )
        return prepared

    same_ui_candidates = ranked(same_category_ui_pool, "same_category_ui_change_distractor")
    same_other_candidates = ranked(same_category_other_state_pool, "same_category_other_state_distractor")
    different_candidates = ranked(different_category_state_pool, "different_category_state_distractor")

    for candidate_list in [same_ui_candidates, same_other_candidates, different_candidates]:
        for candidate in candidate_list:
            key = normalize_text(candidate["text"])
            if key in used_texts:
                continue
            options.append(candidate)
            used_texts.add(key)
            break

    if len(options) < 4:
        fallback_candidates = ranked(
            same_ui_candidates + same_other_candidates + different_candidates,
            "fallback_vqa_state_distractor",
        )
        for candidate in fallback_candidates:
            key = normalize_text(candidate["text"])
            if key in used_texts:
                continue
            options.append(candidate)
            used_texts.add(key)
            if len(options) == 4:
                break

    rng = random.Random(stable_int(correct_origin + "::vqa_state_ui_change_options_shuffle"))
    rng.shuffle(options)
    option_map = {label: option["text"] for label, option in zip(LABELS, options)}
    gt = next(label for label, option in option_map.items() if normalize_text(option) == correct_key)
    option_origin_ids = [option["origin_qa_id"] for option in options]
    option_source_types = [option["source_type"] for option in options]
    option_category_names = [option.get("category_name", current_category_name) for option in options]
    option_data_ids = [option.get("data_id", "") for option in options]
    option_semantic_distances = [option.get("semantic_distance_to_correct") for option in options]
    return gt, option_map, option_origin_ids, option_source_types, option_category_names, option_data_ids, option_semantic_distances


def with_multiselect_options(query: str, options: list[dict]) -> str:
    lines = [query.rstrip()]
    for label, option in zip(LABELS5, options):
        lines.append(f"{label}. {sentence(option['answer'])}")
    return "\n".join(lines) + "\n"


def base_item(
    origin_qa_id: str,
    query: str,
    gt: str,
    task_family: str,
    qa_type: str,
    capability_level: str,
    prompt_variant: str,
    rewrite_type: str,
    semantic_anchor: str,
    output_schema: dict,
    canonical_answer: str,
    source_span: dict,
) -> dict:
    return {
        "origin_qa_id": origin_qa_id,
        "query": query,
        "GT": gt,
        "task_family": task_family,
        "qa_type": qa_type,
        "capability_level": capability_level,
        "scenario_family": SCENARIO_FAMILY,
        "slice_tags": ["lighting_control", "two_step", "state_aligned"],
        "question_zh": "",
        "answer_explanation_zh": "",
        "prompt_variant": prompt_variant,
        "rewrite_type": rewrite_type,
        "semantic_anchor": semantic_anchor,
        "output_schema": output_schema,
        "canonical_answer": clean_answer(canonical_answer),
        "alias_list": aliases(canonical_answer),
        "numeric_slots": None,
        "source_file": SOURCE_FILE,
        "source_span": source_span,
        "notes": "Generated from Action_5_Light_1 Label Studio annotations. Single-image verification uses visible state frames when available.",
    }


def assign_ids(data: list[dict]) -> list[dict]:
    for idx, item in enumerate(data):
        item["id"] = idx
    return data


def form_dir(output_root: Path, task_family: str, form: str) -> Path:
    path = output_root / task_family / form
    (path / "imgs").mkdir(parents=True, exist_ok=True)
    (path / "videos").mkdir(parents=True, exist_ok=True)
    return path


def write_dataset(output_root: Path, task_family: str, form: str, vqa: list[dict], openqa: list[dict]) -> None:
    path = form_dir(output_root, task_family, form)
    (path / "vqa.json").write_text(json.dumps({"data": assign_ids(vqa)}, ensure_ascii=False, indent=2), encoding="utf-8")
    (path / "openqa.json").write_text(json.dumps({"data": assign_ids(openqa)}, ensure_ascii=False, indent=2), encoding="utf-8")


def all_action_pool(records: list[VideoRecord], verification_only: bool | None = None) -> list[ActionSegment]:
    actions = [action for record in records for action in record.actions]
    if verification_only is True:
        return [action for action in actions if action.is_verification]
    if verification_only is False:
        return [action for action in actions if not action.is_verification]
    return actions


def state_pool(records: list[VideoRecord]) -> list[tuple[str, str, Path, int, str, dict]]:
    pool: list[tuple[str, str, Path, int, str, dict]] = []
    for record in records:
        pool.append(
            (
                f"{record.data_id}_final_state",
                clean_answer(record.overall_verification),
                record.video_path,
                record.final_state_frame,
                "is_final_state_frame",
                {"start": record.final_state_frame, "end": record.final_state_frame},
            )
        )
        for ui in record.ui_changes:
            pool.append(
                (
                    ui.origin_qa_id,
                    clean_answer(ui.text),
                    record.video_path,
                    ui.frame,
                    "ui_change_state_frame",
                    {"start": ui.frame, "end": ui.frame},
                )
            )
    return pool


def visual_candidates(records: list[VideoRecord], mode: str) -> list[dict]:
    candidates = []
    if mode == "final_state":
        for record in records:
            candidates.append(
                {
                    "origin_qa_id": f"{record.data_id}_final_state",
                    "answer": record.overall_verification,
                    "video_path": record.video_path,
                    "video_name": record.video_name,
                    "frame": record.final_state_frame,
                    "source_type": "is_final_state_frame",
                    "source_span": {"start": record.final_state_frame, "end": record.final_state_frame},
                    "record": record,
                }
            )
    elif mode == "verification_state":
        for record in records:
            for ui in record.ui_changes:
                candidates.append(
                    {
                        "origin_qa_id": ui.origin_qa_id,
                        "answer": ui.text,
                        "video_path": record.video_path,
                        "video_name": record.video_name,
                        "frame": ui.frame,
                        "source_type": "ui_change_state_frame",
                        "source_span": {"start": ui.frame, "end": ui.frame},
                        "record": record,
                    }
                )
    return candidates


def choose_visual_options(correct: dict, candidates: list[dict], origin: str) -> list[dict]:
    distractors = [c for c in candidates if c["origin_qa_id"] != correct["origin_qa_id"]]
    rng = random.Random(stable_int(origin + "::visual_options"))
    rng.shuffle(distractors)
    options = [correct] + distractors[:3]
    rng.shuffle(options)
    return options


def normalize_category_name(name: str) -> str:
    return category_key(name)


def raw_annotation_jsons_for_category(category_dir: Path) -> list[Path]:
    excluded_names = {
        "video_fps_report.json",
        "annotation_schema_report.json",
        "dataset_manifest.json",
        "form_matrix.json",
        "vqa.json",
        "openqa.json",
    }
    results: list[Path] = []
    for path in category_dir.rglob("*.json"):
        if path.name in excluded_names:
            continue
        if any(part.startswith("hf_") for part in path.parts):
            continue
        if "single_frame_results" in path.parts:
            continue
        results.append(path)
    return sorted(results)


def category_key(name: str) -> str:
    base = name
    prefix, sep, suffix = base.rpartition("_")
    if sep and suffix.isdigit():
        base = prefix
    return base.split("-", 1)[0]


def load_external_final_state_candidates(current_source_base: Path, target: str = "different") -> list[dict]:
    if target not in {"same", "different"}:
        raise ValueError(f"Unsupported category target: {target}")
    latest_root = current_source_base.parent
    current_category = category_key(current_source_base.name)
    pool: list[dict] = []
    for category_dir in latest_root.iterdir():
        if not category_dir.is_dir():
            continue
        category_name = category_key(category_dir.name)
        if category_dir.resolve() == current_source_base.resolve():
            continue
        if target == "same" and category_name != current_category:
            continue
        if target == "different" and category_name == current_category:
            continue
        for json_path in raw_annotation_jsons_for_category(category_dir):
            try:
                records = parse_records(json_path, json_path.parent)
            except Exception:
                continue
            if not records:
                continue
            for candidate in visual_candidates(records, "final_state"):
                enriched = dict(candidate)
                enriched["category_name"] = category_name
                pool.append(enriched)
            break
    return pool


def load_external_ui_change_candidates(current_source_base: Path, target: str) -> list[dict]:
    if target not in {"same", "different"}:
        raise ValueError(f"Unsupported category target: {target}")
    latest_root = current_source_base.parent
    current_category = category_key(current_source_base.name)
    pool: list[dict] = []
    seen: set[str] = set()
    for category_dir in sorted(latest_root.iterdir()):
        if not category_dir.is_dir() or category_dir.resolve() == current_source_base.resolve():
            continue
        category_name = category_key(category_dir.name)
        if target == "same" and category_name != current_category:
            continue
        if target == "different" and category_name == current_category:
            continue
        for json_path in raw_annotation_jsons_for_category(category_dir):
            try:
                records = parse_records(json_path, json_path.parent)
            except Exception:
                continue
            for record in records:
                for ui in record.ui_changes:
                    origin = f"{category_dir.name}::{ui.origin_qa_id}"
                    if origin in seen:
                        continue
                    pool.append(
                        {
                            "origin_qa_id": origin,
                            "answer": clean_answer(ui.text),
                            "video_path": record.video_path,
                            "video_name": record.video_name,
                            "frame": ui.frame,
                            "source_type": "ui_change_state_frame",
                            "source_span": {"start": ui.frame, "end": ui.frame},
                            "record": record,
                            "category_name": category_name,
                            "data_id": record.data_id,
                        }
                    )
                    seen.add(origin)
            if pool:
                break
    return pool


def simple_frame_distance(video_path: Path, frame_a: int, frame_b: int) -> float:
    ok_a, image_a = read_frame(video_path, frame_a)
    ok_b, image_b = read_frame(video_path, frame_b)
    if not ok_a or not ok_b or image_a is None or image_b is None:
        return 0.0
    image_a = cv2.resize(image_a, (32, 32), interpolation=cv2.INTER_AREA)
    image_b = cv2.resize(image_b, (32, 32), interpolation=cv2.INTER_AREA)
    gray_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY).astype("float32")
    gray_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY).astype("float32")
    return float(abs(gray_a - gray_b).mean() / 255.0)


def select_verification_state_same_video_wrong_frame(
    record: VideoRecord,
    correct_frames: set[int],
    query_frame: int,
    category_name: str,
) -> dict | None:
    raw_candidates: list[int] = []
    non_verification_actions = [action for action in record.actions if not action.is_verification]
    if non_verification_actions:
        first_action = non_verification_actions[0]
        raw_candidates.extend(
            [
                first_action.start - 60,
                first_action.start - 45,
                first_action.start - 30,
                first_action.anchor_frame - 70,
                first_action.anchor_frame - 55,
            ]
        )
    earliest_correct = min(correct_frames)
    raw_candidates.extend([earliest_correct - 120, earliest_correct - 90, earliest_correct - 60])

    scored: list[tuple[float, int]] = []
    seen: set[int] = set()
    for raw_frame in raw_candidates:
        frame = clamp_frame(raw_frame, record.total_frames)
        if frame in seen or frame == query_frame or frame in correct_frames:
            continue
        if any(abs(frame - correct_frame) < 20 for correct_frame in correct_frames):
            continue
        seen.add(frame)
        distance = min(simple_frame_distance(record.video_path, frame, correct_frame) for correct_frame in correct_frames)
        scored.append((distance, frame))
    if not scored:
        fallback = select_same_video_obvious_distractor(
            record,
            max(correct_frames),
            forbidden_frames=set(correct_frames) | {query_frame},
            category_name=category_name,
        )
        if fallback is None:
            return None
        fallback["source_type"] = "same_video_obvious_wrong_verification_state"
        fallback["answer"] = "Same-video frame that does not show the requested verification evidence"
        fallback["data_id"] = record.data_id
        return fallback

    distance, frame = max(scored, key=lambda item: (item[0], abs(item[1] - earliest_correct)))
    return {
        "origin_qa_id": f"{record.data_id}_same_video_wrong_verification_state_{frame:06d}",
        "answer": "Same-video frame that does not show the requested verification evidence",
        "video_path": record.video_path,
        "video_name": record.video_name,
        "frame": frame,
        "source_type": "same_video_obvious_wrong_verification_state",
        "source_span": {"start": frame, "end": frame},
        "record": record,
        "category_name": category_name,
        "data_id": record.data_id,
        "perceptual_distance": distance,
    }


def select_same_video_obvious_distractor(
    record: VideoRecord,
    target_frame: int,
    forbidden_frames: set[int],
    category_name: str,
    forbidden_ranges: list[tuple[int, int]] | None = None,
) -> dict | None:
    raw_candidates: list[int] = []
    if record.actions:
        first_action = record.actions[0]
        raw_candidates.extend(
            [
                min(first_action.anchor_frame - ACTION_IMG_FIRST_ACTION_OFFSET_MAX, first_action.start - 1),
                min(first_action.anchor_frame - ACTION_IMG_FIRST_ACTION_OFFSET_MIN, first_action.start - 1),
                first_action.start - 20,
                first_action.start - 1,
                first_action.anchor_frame,
            ]
        )
    raw_candidates.extend([target_frame - 120, target_frame - 90, target_frame - 60])

    seen: set[int] = set()
    for raw_frame in raw_candidates:
        frame = clamp_frame(raw_frame, record.total_frames)
        if frame in seen or frame in forbidden_frames:
            continue
        if forbidden_ranges and any(start <= frame <= end for start, end in forbidden_ranges):
            continue
        seen.add(frame)
        if abs(frame - target_frame) < 30:
            continue
        return {
            "origin_qa_id": f"{record.data_id}_same_video_distractor_{frame:06d}",
            "answer": "",
            "video_path": record.video_path,
            "video_name": record.video_name,
            "frame": frame,
            "source_type": "same_video_obvious_wrong_frame",
            "source_span": {"start": frame, "end": frame},
            "record": record,
            "category_name": category_name,
        }
    return None


def choose_final_state_img2img_options(
    correct: dict,
    current_candidates: list[dict],
    external_candidates: list[dict],
    origin: str,
    query_frame: int,
    current_category_name: str,
    forbidden_ranges: list[tuple[int, int]] | None = None,
) -> list[dict]:
    rng = random.Random(stable_int(origin + "::final_state_img2img_options"))
    correct_record = correct["record"]

    options = [dict(correct, category_name=current_category_name, source_type="correct_final_state")]

    same_video = select_same_video_obvious_distractor(
        correct_record,
        correct["frame"],
        forbidden_frames={correct["frame"], query_frame},
        category_name=current_category_name,
        forbidden_ranges=forbidden_ranges,
    )
    if same_video is not None:
        options.append(same_video)

    same_category_candidates = [
        dict(candidate, category_name=current_category_name, source_type="same_category_other_video")
        for candidate in current_candidates
        if candidate["origin_qa_id"] != correct["origin_qa_id"] and candidate["record"].data_id != correct_record.data_id
    ]
    rng.shuffle(same_category_candidates)
    if same_category_candidates:
        options.append(same_category_candidates[0])

    different_category_candidates = [dict(candidate) for candidate in external_candidates]
    rng.shuffle(different_category_candidates)
    if different_category_candidates:
        external = different_category_candidates[0]
        external["source_type"] = "different_category_other_video"
        options.append(external)

    used_ids = {option["origin_qa_id"] for option in options}
    if len(options) < 4:
        fallback_pool = [
            dict(candidate, category_name=current_category_name, source_type="fallback_same_category")
            for candidate in current_candidates
            if candidate["origin_qa_id"] != correct["origin_qa_id"] and candidate["origin_qa_id"] not in used_ids
        ]
        fallback_pool.extend(
            [
                dict(candidate, source_type="fallback_different_category")
                for candidate in external_candidates
                if candidate["origin_qa_id"] not in used_ids
            ]
        )
        rng.shuffle(fallback_pool)
        for candidate in fallback_pool:
            options.append(candidate)
            used_ids.add(candidate["origin_qa_id"])
            if len(options) == 4:
                break

    rng.shuffle(options)
    return options


def verification_multiselect_targets(records: list[VideoRecord]) -> list[dict]:
    targets: list[dict] = []
    for record in records:
        if not record.ui_changes or not record.final_state_frame:
            continue
        ui = record.ui_changes[0]
        targets.append(
            {
                "origin_qa_id": f"{record.data_id}_verification_state_multiselect",
                "record": record,
                "correct_targets": [
                    {
                        "origin_qa_id": ui.origin_qa_id,
                        "answer": clean_answer(ui.text),
                        "video_path": record.video_path,
                        "video_name": record.video_name,
                        "frame": ui.frame,
                        "source_type": "ui_change_state_frame",
                        "source_span": {"start": ui.frame, "end": ui.frame},
                        "verification_signal_type": "ui_signal",
                        "record": record,
                        "category_name": category_key(SOURCE_FOLDER_NAME),
                        "data_id": record.data_id,
                    },
                    {
                        "origin_qa_id": f"{record.data_id}_physical_world_state_from_is_final_state",
                        "answer": clean_answer(record.overall_verification),
                        "video_path": record.video_path,
                        "video_name": record.video_name,
                        "frame": record.final_state_frame,
                        "source_type": "is_final_state_as_physical_world_state",
                        "source_span": {"start": record.final_state_frame, "end": record.final_state_frame},
                        "verification_signal_type": "physical_world_state",
                        "record": record,
                        "category_name": category_key(SOURCE_FOLDER_NAME),
                        "data_id": record.data_id,
                    },
                ],
            }
        )
    return targets


def verification_option_pool(records: list[VideoRecord]) -> list[dict]:
    pool: list[dict] = []
    for target in verification_multiselect_targets(records):
        pool.extend(target["correct_targets"])
    return pool


def choose_multiselect_options(correct_targets: list[dict], pool: list[dict], origin: str) -> list[dict]:
    correct_ids = {target["origin_qa_id"] for target in correct_targets}
    distractors = [candidate for candidate in pool if candidate["origin_qa_id"] not in correct_ids]
    rng = random.Random(stable_int(origin + "::multiselect_options"))
    rng.shuffle(distractors)
    options = [dict(target, correct=True) for target in correct_targets]
    options.extend(dict(target, correct=False) for target in distractors[:3])
    rng.shuffle(options)
    return options


def choose_verification_state_img2img_options(
    correct_targets: list[dict],
    pool: list[dict],
    origin: str,
    record: VideoRecord,
    query_frame: int,
    current_category_name: str,
    same_category_ui_pool: list[dict],
    different_category_final_state_pool: list[dict],
) -> list[dict]:
    rng = random.Random(stable_int(origin + "::verification_state_img2img_category_options"))
    correct_ids = {target["origin_qa_id"] for target in correct_targets}
    used_ids = set(correct_ids)
    options = [dict(target, correct=True, category_name=current_category_name, data_id=record.data_id) for target in correct_targets]

    same_category_ui_candidates = [
        dict(
            candidate,
            correct=False,
            source_type="same_category_ui_change_distractor",
            category_name=current_category_name,
            data_id=candidate.get("data_id") or candidate.get("record").data_id,
        )
        for candidate in pool
        if candidate.get("source_type") == "ui_change_state_frame"
        and candidate["origin_qa_id"] not in used_ids
        and candidate.get("record") is not None
        and candidate["record"].data_id != record.data_id
    ]
    same_category_ui_candidates.extend(
        dict(
            candidate,
            correct=False,
            source_type="same_category_ui_change_distractor",
            category_name=candidate.get("category_name", current_category_name),
            data_id=candidate.get("data_id") or candidate.get("record").data_id,
        )
        for candidate in same_category_ui_pool
        if candidate["origin_qa_id"] not in used_ids
        and candidate.get("record") is not None
        and candidate["record"].data_id != record.data_id
    )
    rng.shuffle(same_category_ui_candidates)
    for selected in same_category_ui_candidates:
        if len([option for option in options if option.get("source_type") == "same_category_ui_change_distractor"]) >= 2:
            break
        if selected["origin_qa_id"] in used_ids:
            continue
        options.append(selected)
        used_ids.add(selected["origin_qa_id"])

    different_category_final_state_candidates = [
        dict(
            candidate,
            correct=False,
            source_type="different_category_is_final_state_distractor",
            data_id=candidate.get("data_id") or candidate.get("record").data_id,
        )
        for candidate in different_category_final_state_pool
        if candidate["origin_qa_id"] not in used_ids and candidate.get("record") is not None
    ]
    rng.shuffle(different_category_final_state_candidates)
    if different_category_final_state_candidates:
        selected = different_category_final_state_candidates[0]
        options.append(selected)
        used_ids.add(selected["origin_qa_id"])

    if len(options) < 5:
        fallback_pool = [
            dict(candidate, correct=False, source_type=f"fallback_{candidate.get('source_type', 'verification_state')}")
            for candidate in pool
            if candidate["origin_qa_id"] not in used_ids
        ]
        rng.shuffle(fallback_pool)
        for candidate in fallback_pool:
            options.append(candidate)
            used_ids.add(candidate["origin_qa_id"])
            if len(options) == 5:
                break

    rng.shuffle(options)
    return options[:5]


def choose_verification_state_img2txt_options(
    correct_targets: list[dict],
    origin: str,
    record: VideoRecord,
    current_category_name: str,
    different_category_ui_pool: list[dict],
    different_category_final_state_pool: list[dict],
) -> list[dict]:
    rng = random.Random(stable_int(origin + "::verification_state_img2txt_cross_category_options"))
    used_ids = {target["origin_qa_id"] for target in correct_targets}
    options = [dict(target, correct=True, category_name=current_category_name, data_id=record.data_id) for target in correct_targets]

    ui_candidates = [
        dict(
            candidate,
            correct=False,
            source_type="different_category_ui_change_distractor",
            data_id=candidate.get("data_id") or candidate.get("record").data_id,
        )
        for candidate in different_category_ui_pool
        if candidate["origin_qa_id"] not in used_ids
    ]
    final_state_candidates = [
        dict(
            candidate,
            correct=False,
            source_type="different_category_is_final_state_distractor",
            data_id=candidate.get("data_id") or candidate.get("record").data_id,
        )
        for candidate in different_category_final_state_pool
        if candidate["origin_qa_id"] not in used_ids
    ]
    rng.shuffle(ui_candidates)
    rng.shuffle(final_state_candidates)

    for candidate in ui_candidates[:1]:
        options.append(candidate)
        used_ids.add(candidate["origin_qa_id"])

    for candidate in final_state_candidates:
        if candidate["origin_qa_id"] in used_ids:
            continue
        options.append(candidate)
        used_ids.add(candidate["origin_qa_id"])
        if len([option for option in options if option.get("source_type") == "different_category_is_final_state_distractor"]) >= 2:
            break

    if len(options) < 5:
        filler_pool = [candidate for candidate in ui_candidates + final_state_candidates if candidate["origin_qa_id"] not in used_ids]
        rng.shuffle(filler_pool)
        for candidate in filler_pool:
            options.append(candidate)
            used_ids.add(candidate["origin_qa_id"])
            if len(options) == 5:
                break

    rng.shuffle(options)
    return options[:5]


def choose_verification_state_video2txt_options(
    correct_targets: list[dict],
    pool: list[dict],
    origin: str,
    record: VideoRecord,
    current_category_name: str,
    same_category_external_pool: list[dict],
    different_category_ui_pool: list[dict],
    different_category_final_state_pool: list[dict],
) -> list[dict]:
    rng = random.Random(stable_int(origin + "::verification_state_video2txt_mixed_category_options"))
    correct_texts = [target["answer"] for target in correct_targets]
    used_ids = {target["origin_qa_id"] for target in correct_targets}
    options = [dict(target, correct=True, category_name=current_category_name, data_id=record.data_id) for target in correct_targets]

    same_category_candidates = [
        dict(
            candidate,
            correct=False,
            source_type="same_category_far_verification_state_distractor",
            category_name=current_category_name,
            data_id=candidate.get("data_id") or candidate.get("record").data_id,
            semantic_distance_to_correct=min(text_difference_score(candidate["answer"], correct) for correct in correct_texts),
        )
        for candidate in pool
        if candidate["origin_qa_id"] not in used_ids
        and candidate.get("record") is not None
        and candidate["record"].data_id != record.data_id
    ]
    same_category_candidates.extend(
        dict(
            candidate,
            correct=False,
            source_type="same_category_far_verification_state_distractor",
            category_name=candidate.get("category_name", current_category_name),
            data_id=candidate.get("data_id") or candidate.get("record").data_id,
            semantic_distance_to_correct=min(text_difference_score(candidate["answer"], correct) for correct in correct_texts),
        )
        for candidate in same_category_external_pool
        if candidate["origin_qa_id"] not in used_ids
        and candidate.get("record") is not None
        and candidate["record"].data_id != record.data_id
    )
    same_category_candidates.sort(
        key=lambda candidate: (
            candidate.get("semantic_distance_to_correct", 0.0),
            stable_int(origin + "::" + candidate["origin_qa_id"]),
        ),
        reverse=True,
    )
    if same_category_candidates:
        selected = same_category_candidates[0]
        options.append(selected)
        used_ids.add(selected["origin_qa_id"])

    different_category_candidates = []
    different_category_candidates.extend(
        dict(
            candidate,
            correct=False,
            source_type="different_category_ui_change_distractor",
            data_id=candidate.get("data_id") or candidate.get("record").data_id,
        )
        for candidate in different_category_ui_pool
        if candidate["origin_qa_id"] not in used_ids
    )
    different_category_candidates.extend(
        dict(
            candidate,
            correct=False,
            source_type="different_category_is_final_state_distractor",
            data_id=candidate.get("data_id") or candidate.get("record").data_id,
        )
        for candidate in different_category_final_state_pool
        if candidate["origin_qa_id"] not in used_ids
    )
    rng.shuffle(different_category_candidates)
    for candidate in different_category_candidates:
        if candidate["origin_qa_id"] in used_ids:
            continue
        options.append(candidate)
        used_ids.add(candidate["origin_qa_id"])
        if len(options) == 5:
            break

    if len(options) < 5:
        fallback_pool = [
            dict(candidate, correct=False, source_type=f"fallback_{candidate.get('source_type', 'verification_state')}")
            for candidate in pool
            if candidate["origin_qa_id"] not in used_ids
        ]
        rng.shuffle(fallback_pool)
        for candidate in fallback_pool:
            options.append(candidate)
            used_ids.add(candidate["origin_qa_id"])
            if len(options) == 5:
                break

    rng.shuffle(options)
    return options[:5]


def choose_action_video_options(correct: ActionSegment, actions: list[ActionSegment], origin: str) -> list[ActionSegment]:
    distractors = [a for a in actions if a.origin_qa_id != correct.origin_qa_id]
    rng = random.Random(stable_int(origin + "::action_video_options"))
    rng.shuffle(distractors)
    options = [correct] + distractors[:3]
    rng.shuffle(options)
    return options


def spans_overlap(start_a: int, end_a: int, start_b: int, end_b: int) -> bool:
    return max(start_a, start_b) <= min(end_a, end_b)


def select_action_img2video_correct_range(
    record: VideoRecord, action: ActionSegment, query_frame: int
) -> tuple[int, int, str]:
    start = clamp_frame(min(query_frame + 10, action.end), record.total_frames)
    end = clamp_frame(action.end, record.total_frames)
    if start > end:
        start = clamp_frame(min(action.start, end), record.total_frames)
    return start, end, "query_plus_10_to_action_end"


def select_verification_img2video_query_frame(
    record: VideoRecord, action: ActionSegment
) -> tuple[int, str, str | None]:
    previous = previous_action_segment(record, action)
    if previous is not None:
        return clamp_frame(previous.anchor_frame, record.total_frames), "previous_action_anchor", previous.origin_qa_id
    return clamp_frame(max(1, action.start - 10), record.total_frames), "pre_verification_action_fallback", None


def select_verification_img2video_correct_range(
    record: VideoRecord, action: ActionSegment, query_frame: int
) -> tuple[int, int, str]:
    start = clamp_frame(query_frame + 10, record.total_frames)
    end = clamp_frame(action.end, record.total_frames)
    if start > end:
        start = clamp_frame(action.start, record.total_frames)
    return start, end, "query_plus_10_to_following_verification_action_end"


def select_verification_video2video_query_range(
    record: VideoRecord, action: ActionSegment
) -> tuple[int, int, str, str | None]:
    previous = previous_action_segment(record, action)
    if previous is not None:
        start = clamp_frame(previous.start, record.total_frames)
        end = clamp_frame(min(previous.end, action.start - 1), record.total_frames)
        if end < start:
            end = start
        return start, end, "penultimate_action_segment", previous.origin_qa_id
    start, end, rule = select_first_action_pre_video_range(record, action)
    return start, end, rule, None


def select_verification_video2video_correct_range(
    record: VideoRecord, action: ActionSegment, query_end: int
) -> tuple[int, int, str]:
    start = clamp_frame(query_end + 10, record.total_frames)
    end = clamp_frame(action.end, record.total_frames)
    if start > end:
        start = clamp_frame(action.start, record.total_frames)
        return start, end, "verification_action_span_fallback"
    return start, end, "query_end_plus10_to_verification_end"


def select_verification_state_video2txt_query_range(record: VideoRecord) -> tuple[int, int, str, str | None]:
    first_action = next((action for action in record.actions if not action.is_verification), record.actions[0] if record.actions else None)
    if first_action is None:
        end = clamp_frame(45, record.total_frames)
        start = clamp_frame(end - 45, record.total_frames)
        return start, end, "default_initial_context_clip", None
    end = clamp_frame(first_action.anchor_frame - 30, record.total_frames)
    if end >= first_action.anchor_frame:
        end = clamp_frame(first_action.start - 1, record.total_frames)
    start = clamp_frame(end - 45, record.total_frames)
    if start > end:
        start = end
    return start, end, "first_action_anchor_minus_30_context", first_action.origin_qa_id


def choose_action_img2video_options(
    correct: ActionSegment,
    record: VideoRecord,
    actions: list[ActionSegment],
    origin: str,
    correct_range: tuple[int, int],
) -> list[dict]:
    correct_start, correct_end = correct_range
    rng = random.Random(stable_int(origin + "::action_img2video_options"))

    same_video_candidates = [
        action
        for action in actions
        if action.origin_qa_id != correct.origin_qa_id
        and action.data_id == correct.data_id
        and not spans_overlap(action.start, action.end, correct_start, correct_end)
    ]
    rng.shuffle(same_video_candidates)

    other_video_candidates = [
        action
        for action in actions
        if action.origin_qa_id != correct.origin_qa_id and action.data_id != correct.data_id
    ]
    rng.shuffle(other_video_candidates)

    options = [
        {
            "action": correct,
            "clip_start": correct_start,
            "clip_end": correct_end,
            "source_type": "correct_query_plus10_to_action_end",
        }
    ]

    used_origin_ids = {correct.origin_qa_id}

    if same_video_candidates:
        candidate = same_video_candidates[0]
        options.append(
            {
                "action": candidate,
                "clip_start": candidate.start,
                "clip_end": candidate.end,
                "source_type": "same_video_nonoverlap",
            }
        )
        used_origin_ids.add(candidate.origin_qa_id)

    for candidate in other_video_candidates:
        if candidate.origin_qa_id in used_origin_ids:
            continue
        options.append(
            {
                "action": candidate,
                "clip_start": candidate.start,
                "clip_end": candidate.end,
                "source_type": "other_video",
            }
        )
        used_origin_ids.add(candidate.origin_qa_id)
        if len(options) == 4:
            break

    if len(options) < 4:
        fallback_candidates = [
            action for action in actions if action.origin_qa_id not in used_origin_ids
        ]
        rng.shuffle(fallback_candidates)
        for candidate in fallback_candidates:
            options.append(
                {
                    "action": candidate,
                    "clip_start": candidate.start,
                    "clip_end": candidate.end,
                    "source_type": "fallback_any_video",
                }
            )
            used_origin_ids.add(candidate.origin_qa_id)
            if len(options) == 4:
                break

    rng.shuffle(options)
    return options


def action_video_candidate(action: ActionSegment, record: VideoRecord, category_name: str) -> dict:
    return {"action": action, "record": record, "category_name": category_name}


def load_external_verification_action_video_candidates(current_category_dir: Path, target: str) -> list[dict]:
    if target not in {"same", "different"}:
        raise ValueError(f"Unsupported category target: {target}")
    latest_root = current_category_dir.parent
    current_category = category_key(current_category_dir.name)
    pool: list[dict] = []
    seen: set[str] = set()
    for category_dir in sorted(latest_root.iterdir()):
        if not category_dir.is_dir() or category_dir.resolve() == current_category_dir.resolve():
            continue
        category_name = category_key(category_dir.name)
        if target == "same" and category_name != current_category:
            continue
        if target == "different" and category_name == current_category:
            continue
        for json_path in raw_annotation_jsons_for_category(category_dir):
            try:
                records = parse_records(json_path, json_path.parent)
            except Exception:
                continue
            for record in records:
                for action in record.actions:
                    if not action.is_verification:
                        continue
                    key = f"{category_dir.name}::{action.origin_qa_id}"
                    if key in seen:
                        continue
                    candidate = action_video_candidate(action, record, category_name)
                    candidate["origin_qa_id"] = key
                    pool.append(candidate)
                    seen.add(key)
            if pool:
                break
    return pool


def nonoverlap_subrange(
    start: int,
    end: int,
    forbidden_start: int,
    forbidden_end: int,
    min_length: int = 5,
) -> tuple[int, int] | None:
    candidates: list[tuple[int, int]] = []
    if start < forbidden_start:
        candidates.append((start, min(end, forbidden_start - 1)))
    if end > forbidden_end:
        candidates.append((max(start, forbidden_end + 1), end))
    candidates = [(s, e) for s, e in candidates if e - s + 1 >= min_length]
    if not candidates:
        return None
    return max(candidates, key=lambda span: span[1] - span[0])


def same_video_nonoverlap_window(
    record: VideoRecord,
    forbidden_start: int,
    forbidden_end: int,
    origin: str,
) -> dict | None:
    clip_len = max(12, min(60, forbidden_end - forbidden_start + 1))
    before_end = forbidden_start - 1
    before_start = before_end - clip_len + 1
    if before_start >= 1:
        return {
            "action": None,
            "record": record,
            "clip_start": before_start,
            "clip_end": before_end,
            "source_type": "same_video_nonoverlap_window",
            "origin_qa_id": f"{origin}_same_video_window_{before_start:06d}_{before_end:06d}",
            "category_name": category_key(record.video_name),
        }
    after_start = forbidden_end + 1
    after_end = after_start + clip_len - 1
    if after_end <= record.total_frames:
        return {
            "action": None,
            "record": record,
            "clip_start": after_start,
            "clip_end": after_end,
            "source_type": "same_video_nonoverlap_window",
            "origin_qa_id": f"{origin}_same_video_window_{after_start:06d}_{after_end:06d}",
            "category_name": category_key(record.video_name),
        }
    return None


def choose_verification_action_img2video_options(
    correct: ActionSegment,
    record: VideoRecord,
    records: list[VideoRecord],
    origin: str,
    correct_range: tuple[int, int],
    current_category_name: str,
    same_category_external_pool: list[dict],
    different_category_pool: list[dict],
    correct_source_type: str = "correct_query_plus10_to_following_verification_action_end",
) -> list[dict]:
    correct_start, correct_end = correct_range
    rng = random.Random(stable_int(origin + "::verification_action_img2video_options"))
    options: list[dict] = [
        {
            "action": correct,
            "record": record,
            "clip_start": correct_start,
            "clip_end": correct_end,
            "source_type": correct_source_type,
            "category_name": current_category_name,
        }
    ]
    used_origin_ids = {correct.origin_qa_id}

    same_video_candidates: list[dict] = []
    for action in record.actions:
        if action.origin_qa_id == correct.origin_qa_id:
            continue
        subrange = nonoverlap_subrange(action.start, action.end, correct_start, correct_end)
        if subrange is None:
            continue
        same_video_candidates.append(
            {
                "action": action,
                "record": record,
                "clip_start": subrange[0],
                "clip_end": subrange[1],
                "source_type": "same_video_nonoverlap",
                "category_name": current_category_name,
            }
        )
    rng.shuffle(same_video_candidates)
    if same_video_candidates:
        same_video = same_video_candidates[0]
    else:
        same_video = same_video_nonoverlap_window(record, correct_start, correct_end, origin)
        if same_video is not None:
            same_video["category_name"] = current_category_name
    if same_video is not None:
        options.append(same_video)
        if same_video.get("action") is not None:
            used_origin_ids.add(same_video["action"].origin_qa_id)
        elif same_video.get("origin_qa_id"):
            used_origin_ids.add(same_video["origin_qa_id"])

    same_category_candidates = [
        {
            "action": action,
            "record": candidate_record,
            "clip_start": action.start,
            "clip_end": action.end,
            "source_type": "same_category_other_video",
            "category_name": current_category_name,
        }
        for candidate_record in records
        for action in candidate_record.actions
        if action.is_verification
        and action.origin_qa_id != correct.origin_qa_id
        and action.data_id != correct.data_id
    ]
    same_category_candidates.extend(
        {
            "action": candidate["action"],
            "record": candidate["record"],
            "clip_start": candidate["action"].start,
            "clip_end": candidate["action"].end,
            "source_type": "same_category_other_video",
            "category_name": candidate.get("category_name", current_category_name),
            "origin_qa_id": candidate.get("origin_qa_id"),
        }
        for candidate in same_category_external_pool
        if candidate["action"].origin_qa_id not in used_origin_ids
        and candidate["action"].data_id != correct.data_id
    )
    rng.shuffle(same_category_candidates)
    for candidate in same_category_candidates:
        origin_id = candidate.get("origin_qa_id") or candidate["action"].origin_qa_id
        if origin_id in used_origin_ids:
            continue
        options.append(candidate)
        used_origin_ids.add(origin_id)
        break

    different_category_candidates = [
        {
            "action": candidate["action"],
            "record": candidate["record"],
            "clip_start": candidate["action"].start,
            "clip_end": candidate["action"].end,
            "source_type": "different_category_other_video",
            "category_name": candidate.get("category_name"),
            "origin_qa_id": candidate.get("origin_qa_id"),
        }
        for candidate in different_category_pool
        if candidate["action"].origin_qa_id not in used_origin_ids
    ]
    rng.shuffle(different_category_candidates)
    for candidate in different_category_candidates:
        origin_id = candidate.get("origin_qa_id") or candidate["action"].origin_qa_id
        if origin_id in used_origin_ids:
            continue
        options.append(candidate)
        used_origin_ids.add(origin_id)
        break

    fallback_candidates = [
        {
            "action": action,
            "record": candidate_record,
            "clip_start": action.start,
            "clip_end": action.end,
            "source_type": "fallback_other_verification_action",
            "category_name": current_category_name,
        }
        for candidate_record in records
        for action in candidate_record.actions
        if action.is_verification and action.origin_qa_id not in used_origin_ids
    ]
    rng.shuffle(fallback_candidates)
    for candidate in fallback_candidates:
        if len(options) >= 4:
            break
        options.append(candidate)
        used_origin_ids.add(candidate["action"].origin_qa_id)

    rng.shuffle(options)
    return options[:4]


def choose_action_video2video_options(
    correct: ActionSegment,
    actions: list[ActionSegment],
    origin: str,
) -> list[dict]:
    rng = random.Random(stable_int(origin + "::action_video2video_options"))

    same_video_candidates = [
        action
        for action in actions
        if action.origin_qa_id != correct.origin_qa_id
        and action.data_id == correct.data_id
        and not spans_overlap(action.start, action.end, correct.start, correct.end)
    ]
    rng.shuffle(same_video_candidates)

    other_video_candidates = [
        action
        for action in actions
        if action.origin_qa_id != correct.origin_qa_id and action.data_id != correct.data_id
    ]
    rng.shuffle(other_video_candidates)

    options = [
        {
            "action": correct,
            "clip_start": correct.start,
            "clip_end": correct.end,
            "source_type": "correct_full_action_span",
        }
    ]
    used_origin_ids = {correct.origin_qa_id}

    if same_video_candidates:
        candidate = same_video_candidates[0]
        options.append(
            {
                "action": candidate,
                "clip_start": candidate.start,
                "clip_end": candidate.end,
                "source_type": "same_video_nonoverlap",
            }
        )
        used_origin_ids.add(candidate.origin_qa_id)

    for candidate in other_video_candidates:
        if candidate.origin_qa_id in used_origin_ids:
            continue
        options.append(
            {
                "action": candidate,
                "clip_start": candidate.start,
                "clip_end": candidate.end,
                "source_type": "other_video",
            }
        )
        used_origin_ids.add(candidate.origin_qa_id)
        if len(options) == 4:
            break

    if len(options) < 4:
        fallback_candidates = [
            action for action in actions if action.origin_qa_id not in used_origin_ids
        ]
        rng.shuffle(fallback_candidates)
        for candidate in fallback_candidates:
            options.append(
                {
                    "action": candidate,
                    "clip_start": candidate.start,
                    "clip_end": candidate.end,
                    "source_type": "fallback_any_video",
                }
            )
            used_origin_ids.add(candidate.origin_qa_id)
            if len(options) == 4:
                break

    rng.shuffle(options)
    return options


def add_option_text_fields(
    item: dict,
    option_map: dict[str, str],
    option_origin_ids: list[str],
    option_source_types: list[str] | None = None,
) -> None:
    for label in LABELS:
        item[f"option_{label.lower()}"] = sentence(option_map[label])
    item["option_origin_qa_ids"] = option_origin_ids
    item["option_source_types"] = option_source_types or ["annotation_text"] * len(option_origin_ids)


def add_multiselect_text_option_fields(item: dict, options: list[dict]) -> None:
    item["option_origin_qa_ids"] = []
    item["option_source_types"] = []
    item["option_category_names"] = []
    item["option_data_ids"] = []
    item["option_video_names"] = []
    item["option_answer_texts"] = []
    item["option_correctness"] = []
    for label, option in zip(LABELS5, options):
        item[f"option_{label.lower()}"] = sentence(option["answer"])
        item["option_origin_qa_ids"].append(option["origin_qa_id"])
        item["option_source_types"].append(option["source_type"])
        item["option_category_names"].append(option.get("category_name", category_key(SOURCE_FOLDER_NAME)))
        item["option_data_ids"].append(option.get("data_id") or (option.get("record").data_id if option.get("record") is not None else ""))
        item["option_video_names"].append(option.get("video_name", ""))
        item["option_answer_texts"].append(clean_answer(option["answer"]))
        if any("semantic_distance_to_correct" in option for option in options):
            item.setdefault("option_semantic_distances", []).append(option.get("semantic_distance_to_correct"))
        item["option_correctness"].append(bool(option.get("correct")))
    correct_answers = [label for label, option in zip(LABELS5, options) if option.get("correct")]
    item["GT"] = correct_answers
    item["correct_answers"] = correct_answers
    item["answer_mode"] = "multi_select"


def build_task_forms(records: list[VideoRecord], output_root: Path) -> None:
    form = form_dir(output_root, "vqa_task", "video2txt")
    answer_pool = [
        text_candidate(f"{record.data_id}_task_001", record.overall_requirement, "vqa_task")
        for record in records
    ]
    vqa: list[dict] = []
    openqa: list[dict] = []
    for record in records:
        origin = f"{record.data_id}_task_001"
        video_out = form / "videos" / f"{record.data_id}.mp4"
        ensure_video_copy(record.video_path, video_out)
        query_stem = "What overall task is being carried out from start to finish in this video? Answer with a short task phrase."
        gt, option_map, _, option_origin_ids, option_source_types = build_text_options(
            record.overall_requirement,
            answer_pool,
            origin,
            correct_origin_qa_id=origin,
        )
        item = base_item(
            origin,
            with_options(query_stem, option_map),
            gt,
            "vqa_task",
            "task",
            "L2",
            "task_goal_full_video__task_phrase_mcq",
            "keyword",
            "task_summary",
            {"type": "choice", "labels": LABELS, "choice_modality": "text"},
            record.overall_requirement,
            record.full_span,
        )
        item.update({"query_video_path": rel(video_out, form), "correct_answer": clean_answer(record.overall_requirement)})
        add_option_text_fields(item, option_map, option_origin_ids, option_source_types)
        vqa.append(item)

        open_item = base_item(
            origin,
            query_stem,
            clean_answer(record.overall_requirement),
            "vqa_task",
            "task",
            "L2",
            "task_goal_full_video__task_phrase_open",
            "structured_short_answer",
            "task_summary",
            {"type": "structured_short_answer", "value": "task_summary"},
            record.overall_requirement,
            record.full_span,
        )
        open_item.update({"query_video_path": rel(video_out, form), "answer_type": "structured_short_answer"})
        openqa.append(open_item)
    write_dataset(output_root, "vqa_task", "video2txt", vqa, openqa)


def build_action_text_forms(records: list[VideoRecord], output_root: Path, task_family: str, verification_only: bool | None) -> None:
    actions = all_action_pool(records, verification_only)
    def action_text_origin(action: ActionSegment) -> str:
        return action.origin_qa_id if task_family == "action" else f"{action.data_id}_verify_{int(action.step_id or 0):03d}"

    current_category_name = category_key(output_root.parent.name)
    answer_pool = [
        action_candidate(action_text_origin(action), action.description, task_family, current_category_name, action.data_id)
        for action in actions
    ]
    same_category_external_action_pool = (
        load_external_action_text_candidates(output_root.parent, target="same") if task_family == "action" else []
    )
    external_action_pool = (
        load_external_action_text_candidates(output_root.parent, target="different") if task_family == "action" else []
    )
    same_category_external_verification_action_pool = (
        load_external_verification_action_text_candidates(output_root.parent, target="same")
        if task_family == "verification_action"
        else []
    )
    external_verification_action_pool = (
        load_external_verification_action_text_candidates(output_root.parent, target="different")
        if task_family == "verification_action"
        else []
    )
    for current_form in ["video2txt", "img2txt"]:
        form = form_dir(output_root, task_family, current_form)
        vqa: list[dict] = []
        openqa: list[dict] = []
        for action in actions:
            record = next(r for r in records if r.data_id == action.data_id)
            origin = action.origin_qa_id if task_family == "action" else f"{action.data_id}_verify_{int(action.step_id or 0):03d}"
            if task_family == "verification_action":
                query_stem = f'What should the operator check to confirm success in "{record.overall_requirement}"? Answer with a short phrase.'
                prompt_variant = "verify_action_operator_check__short_phrase"
                semantic_anchor = "verification_check"
                qa_type = "verification_action"
                capability = "L4"
            elif action.step_id == "1":
                query_stem = f'What is the first key action needed to start "{record.overall_requirement}"? Answer with a brief action phrase.'
                prompt_variant = "action_first_needed__short_phrase"
                semantic_anchor = "first_required_action"
                qa_type = "action_step"
                capability = "L2"
            else:
                previous = next((a.description for a in record.actions if int(a.step_id or 0) == int(action.step_id or 0) - 1), "")
                query_stem = f'After "{previous}", what does the operator do next in "{record.overall_requirement}"? Answer with a short action phrase.'
                prompt_variant = "action_next_after_previous__short_phrase"
                semantic_anchor = "next_action_after_previous"
                qa_type = "action_step"
                capability = "L2"

            option_category_names = None
            distractor_strategy_meta = {}
            if task_family == "action" and current_form == "video2txt":
                (
                    gt,
                    option_map,
                    _,
                    option_origin_ids,
                    option_source_types,
                    option_category_names,
                    distractor_strategy_meta,
                ) = build_action_video2txt_text_options(
                    action,
                    answer_pool + same_category_external_action_pool,
                    external_action_pool,
                    origin,
                    current_category_name,
                )
            elif task_family == "verification_action" and current_form in {"video2txt", "img2txt"}:
                (
                    gt,
                    option_map,
                    _,
                    option_origin_ids,
                    option_source_types,
                    option_category_names,
                    distractor_strategy_meta,
                ) = build_verification_action_img2txt_text_options(
                    action,
                    answer_pool + same_category_external_verification_action_pool,
                    external_verification_action_pool,
                    origin,
                    current_category_name,
                )
            else:
                gt, option_map, _, option_origin_ids, option_source_types = build_text_options(
                    action.description,
                    answer_pool,
                    origin,
                    correct_origin_qa_id=origin,
                )
            item = base_item(
                origin,
                with_options(query_stem, option_map),
                gt,
                task_family,
                qa_type,
                capability,
                prompt_variant,
                "keyword",
                semantic_anchor,
                {"type": "choice", "labels": LABELS, "choice_modality": "text"},
                action.description,
                {"start": action.start, "end": action.end},
            )
            item.update(
                {
                    "correct_answer": clean_answer(action.description),
                    "action_step_id": action.step_id,
                    "action_type": action.action_type,
                    "action_requirement": action.requirement,
                    "action_anchor_frame": action.anchor_frame,
                }
            )
            if current_form == "video2txt":
                if task_family == "action":
                    query_start, query_end, query_rule, query_from_action = select_action_video2txt_query_range(record, action)
                    video_out = form / "videos" / f"{origin}_query_video.mp4"
                    write_clip(record.video_path, query_start, query_end, video_out, record.fps)
                    item["query_video_path"] = rel(video_out, form)
                    item["query_source_range"] = {"start_frame": query_start, "end_frame": query_end}
                    item["query_source_rule"] = query_rule
                    item["query_source_action_qa_id"] = query_from_action
                else:
                    video_out = form / "videos" / f"{record.data_id}.mp4"
                    ensure_video_copy(record.video_path, video_out)
                    item["query_video_path"] = rel(video_out, form)
            else:
                query_frame, query_rule, query_from_action = select_action_img2txt_query_frame(record, action) if task_family == "action" else (action.anchor_frame, "current_action_anchor", None)
                img_out = form / "imgs" / f"{origin}_query_img.jpg"
                write_frame(record.video_path, query_frame, img_out)
                item["query_img_path"] = rel(img_out, form)
                item["query_source_frame"] = query_frame
                item["query_source_rule"] = query_rule
                item["query_source_action_qa_id"] = query_from_action
            add_option_text_fields(item, option_map, option_origin_ids, option_source_types)
            if option_category_names is not None:
                item["option_category_names"] = option_category_names
            if distractor_strategy_meta:
                item.update(distractor_strategy_meta)
            vqa.append(item)

            open_item = base_item(
                origin,
                query_stem,
                clean_answer(action.description),
                task_family,
                qa_type,
                capability,
                prompt_variant.replace("mcq", "open"),
                "keyword",
                semantic_anchor,
                {"type": "keyword", "value": "action_phrase"},
                action.description,
                {"start": action.start, "end": action.end},
            )
            open_item.update(
                {
                    "answer_type": "keyword",
                    "action_step_id": action.step_id,
                    "action_type": action.action_type,
                    "action_requirement": action.requirement,
                    "action_anchor_frame": action.anchor_frame,
                }
            )
            if current_form == "video2txt":
                open_item["query_video_path"] = item["query_video_path"]
                if task_family == "action":
                    open_item["query_source_range"] = item["query_source_range"]
                    open_item["query_source_rule"] = item["query_source_rule"]
                    open_item["query_source_action_qa_id"] = item["query_source_action_qa_id"]
            else:
                open_item["query_img_path"] = item["query_img_path"]
                open_item["query_source_frame"] = item["query_source_frame"]
                open_item["query_source_rule"] = item["query_source_rule"]
                open_item["query_source_action_qa_id"] = item["query_source_action_qa_id"]
            openqa.append(open_item)
        write_dataset(output_root, task_family, current_form, vqa, openqa)


def build_action_video_forms(records: list[VideoRecord], output_root: Path, task_family: str, verification_only: bool | None) -> None:
    actions = all_action_pool(records, verification_only)
    current_category_name = category_key(output_root.parent.name)
    same_category_external_verification_video_pool = (
        load_external_verification_action_video_candidates(output_root.parent, target="same")
        if task_family == "verification_action"
        else []
    )
    different_category_external_verification_video_pool = (
        load_external_verification_action_video_candidates(output_root.parent, target="different")
        if task_family == "verification_action"
        else []
    )
    for current_form in ["img2video", "video2video"]:
        form = form_dir(output_root, task_family, current_form)
        vqa: list[dict] = []
        for action in actions:
            record = next(r for r in records if r.data_id == action.data_id)
            origin = action.origin_qa_id if task_family == "action" else f"{action.data_id}_verify_{int(action.step_id or 0):03d}"
            if task_family == "verification_action":
                if current_form == "img2video":
                    query_text = f'Based on this frame from the previous action, which video option shows the following observation step that verifies success for "{record.overall_requirement}"?'
                elif current_form == "video2video":
                    query_text = f'Watch the penultimate action clip. Which later video option shows the observation step that verifies success for "{record.overall_requirement}"?'
                else:
                    query_text = f'Which video option shows the observation step that verifies success for "{record.overall_requirement}"?'
                semantic_anchor = "verification_check"
                prompt_variant = "verify_action_video_option"
                qa_type = "verification_action"
                capability = "L4"
            elif action.step_id == "1":
                query_text = f'What is the first key action needed to start "{record.overall_requirement}"? Choose the best matching video option.'
                semantic_anchor = "first_required_action"
                prompt_variant = "action_first_needed__video_option"
                qa_type = "action_step"
                capability = "L2"
            else:
                previous = next((a.description for a in record.actions if int(a.step_id or 0) == int(action.step_id or 0) - 1), "")
                query_text = f'After "{previous}", what does the operator do next in "{record.overall_requirement}"? Choose the best matching video option.'
                semantic_anchor = "next_action_after_previous"
                prompt_variant = "action_next_after_previous__video_option"
                qa_type = "action_step"
                capability = "L2"

            item = base_item(
                origin,
                query_text,
                "",
                task_family,
                qa_type,
                capability,
                prompt_variant,
                "choice",
                semantic_anchor,
                {"type": "choice", "labels": LABELS, "choice_modality": "video"},
                action.description,
                {"start": action.start, "end": action.end},
            )
            if current_form == "img2video":
                if task_family == "action":
                    query_frame, query_rule, query_from_action = select_action_img2txt_query_frame(record, action)
                elif task_family == "verification_action":
                    query_frame, query_rule, query_from_action = select_verification_img2video_query_frame(record, action)
                else:
                    query_frame, query_rule, query_from_action = action.anchor_frame, "current_action_anchor", None
                query_img = form / "imgs" / f"{origin}_query_img.jpg"
                write_frame(record.video_path, query_frame, query_img)
                item["query_img_path"] = rel(query_img, form)
                item["query_source_frame"] = query_frame
                item["query_source_rule"] = query_rule
                item["query_source_action_qa_id"] = query_from_action
            else:
                if task_family == "action":
                    query_start, query_end, query_rule, query_from_action = select_action_video2video_query_range(record, action)
                elif task_family == "verification_action":
                    query_start, query_end, query_rule, query_from_action = select_verification_video2video_query_range(record, action)
                else:
                    query_start = clamp_frame(action.start - 45, record.total_frames)
                    query_end = clamp_frame(action.start - 1, record.total_frames)
                    if query_end < query_start:
                        query_end = action.start
                    query_rule = "pre_action_window_45"
                    query_from_action = None
                query_video = form / "videos" / f"{origin}_query_video.mp4"
                write_clip(record.video_path, query_start, query_end, query_video, record.fps)
                item["query_video_path"] = rel(query_video, form)
                item["query_source_range"] = {"start_frame": query_start, "end_frame": query_end}
                item["query_source_rule"] = query_rule
                item["query_source_action_qa_id"] = query_from_action

            if current_form == "img2video" and task_family == "action":
                correct_start, correct_end, correct_rule = select_action_img2video_correct_range(record, action, query_frame)
                options = choose_action_img2video_options(
                    action,
                    record,
                    actions,
                    origin,
                    (correct_start, correct_end),
                )
                item["correct_option_source_range"] = {
                    "start_frame": correct_start,
                    "end_frame": correct_end,
                }
                item["correct_option_source_rule"] = correct_rule
            elif current_form == "img2video" and task_family == "verification_action":
                correct_start, correct_end, correct_rule = select_verification_img2video_correct_range(
                    record,
                    action,
                    query_frame,
                )
                options = choose_verification_action_img2video_options(
                    action,
                    record,
                    records,
                    origin,
                    (correct_start, correct_end),
                    current_category_name,
                    same_category_external_verification_video_pool,
                    different_category_external_verification_video_pool,
                )
                item["correct_option_source_range"] = {
                    "start_frame": correct_start,
                    "end_frame": correct_end,
                }
                item["correct_option_source_rule"] = correct_rule
            elif current_form == "video2video" and task_family == "verification_action":
                correct_start, correct_end, correct_rule = select_verification_video2video_correct_range(
                    record,
                    action,
                    item["query_source_range"]["end_frame"],
                )
                options = choose_verification_action_img2video_options(
                    action,
                    record,
                    records,
                    origin,
                    (correct_start, correct_end),
                    current_category_name,
                    same_category_external_verification_video_pool,
                    different_category_external_verification_video_pool,
                    correct_source_type="correct_query_end_plus10_to_verification_end",
                )
                item["correct_option_source_range"] = {
                    "start_frame": correct_start,
                    "end_frame": correct_end,
                }
                item["correct_option_source_rule"] = correct_rule
            elif current_form == "video2video" and task_family == "action":
                options = choose_action_video2video_options(action, actions, origin)
                item["correct_option_source_range"] = {
                    "start_frame": action.start,
                    "end_frame": action.end,
                }
                item["correct_option_source_rule"] = "full_action_span"
            else:
                options = [
                    {
                        "action": option_action,
                        "clip_start": option_action.start,
                        "clip_end": option_action.end,
                        "source_type": "full_action_span",
                    }
                    for option_action in choose_action_video_options(action, actions, origin)
                ]
            option_paths = []
            option_ranges = []
            option_origin_ids = []
            option_source_types = []
            option_category_names = []
            option_video_names = []
            option_data_ids = []
            for label, option_meta in zip(LABELS, options):
                option_action = option_meta.get("action")
                option_record = option_meta.get("record")
                if option_record is None and option_action is not None:
                    option_record = next(r for r in records if r.data_id == option_action.data_id)
                if option_record is None:
                    raise RuntimeError(f"Missing option record for {origin} option {label}")
                option_video = form / "videos" / f"option_clip_{origin}_{label}.mp4"
                write_clip(
                    option_record.video_path,
                    option_meta["clip_start"],
                    option_meta["clip_end"],
                    option_video,
                    option_record.fps,
                )
                option_paths.append(rel(option_video, form))
                option_ranges.append(
                    {
                        "start_frame": option_meta["clip_start"],
                        "end_frame": option_meta["clip_end"],
                    }
                )
                option_origin_id = option_meta.get("origin_qa_id")
                if option_origin_id is None and option_action is not None:
                    option_origin_id = option_action.origin_qa_id
                option_origin_ids.append(option_origin_id or f"{origin}_{label}")
                option_source_types.append(option_meta["source_type"])
                option_category_names.append(option_meta.get("category_name", current_category_name))
                option_video_names.append(option_record.video_name)
                option_data_ids.append(option_action.data_id if option_action is not None else option_record.data_id)
                if option_meta["source_type"].startswith("correct"):
                    item["GT"] = label
            item.update(
                {
                    "option_videos_path": option_paths,
                    "option_source_ranges": option_ranges,
                    "option_origin_qa_ids": option_origin_ids,
                    "option_source_types": option_source_types,
                    "option_category_names": option_category_names,
                    "option_video_names": option_video_names,
                    "option_data_ids": option_data_ids,
                    "canonical_answer": clean_answer(action.description),
                    "alias_list": aliases(action.description),
                    "correct_answer": clean_answer(action.description),
                    "action_step_id": action.step_id,
                    "action_type": action.action_type,
                    "action_requirement": action.requirement,
                }
            )
            vqa.append(item)
        write_dataset(output_root, task_family, current_form, vqa, [])


def build_vqa_state(records: list[VideoRecord], output_root: Path) -> None:
    form = form_dir(output_root, "vqa_state", "img2txt")
    current_category_name = category_key(output_root.parent.name)
    pool = state_pool(records)
    answer_pool = [
        text_candidate(f"{origin}_vqa_state", answer, source_label)
        for origin, answer, _, _, source_label, _ in pool
    ]
    current_final_state_pool = [
        final_state_text_candidate(
            f"{record.data_id}_final_state_vqa_state",
            record.overall_verification,
            "is_final_state_frame",
            current_category_name,
            record.data_id,
        )
        for record in records
    ]
    current_ui_change_pool = [
        final_state_text_candidate(
            f"{ui.origin_qa_id}_vqa_state",
            ui.text,
            "ui_change_state_frame",
            current_category_name,
            record.data_id,
        )
        for record in records
        for ui in record.ui_changes
    ]
    same_category_final_state_pool = current_final_state_pool + load_external_final_state_text_candidates(
        output_root.parent,
        target="same",
    )
    same_category_other_state_pool = current_ui_change_pool + [
        ui_change_text_candidate(candidate)
        for candidate in load_external_ui_change_candidates(output_root.parent, target="same")
    ]
    different_category_state_pool = load_external_final_state_text_candidates(
        output_root.parent,
        target="different",
    ) + [
        ui_change_text_candidate(candidate)
        for candidate in load_external_ui_change_candidates(output_root.parent, target="different")
    ]

    def data_id_from_state_origin(value: str) -> str:
        if "_final_state" in value:
            return value.split("_final_state", 1)[0]
        if "_state_ui_" in value:
            return value.split("_state_ui_", 1)[0]
        return value.split("_", 1)[0]

    vqa: list[dict] = []
    openqa: list[dict] = []
    for origin, answer, video_path, frame, source_label, source_span in pool:
        query_stem = "Based on this frame, which description best matches the current visible lighting state? Answer with a brief state phrase."
        vqa_origin = f"{origin}_vqa_state"
        option_category_names = None
        option_data_ids = None
        option_semantic_distances = None
        if source_label == "is_final_state_frame":
            (
                gt,
                option_map,
                option_origin_ids,
                option_source_types,
                option_category_names,
                option_data_ids,
                option_semantic_distances,
            ) = build_vqa_state_final_state_text_options(
                vqa_origin,
                answer,
                data_id_from_state_origin(origin),
                same_category_final_state_pool,
                same_category_other_state_pool,
                different_category_state_pool,
                current_category_name,
            )
        elif source_label == "ui_change_state_frame":
            same_category_ui_pool_for_current = [
                candidate
                for candidate in same_category_other_state_pool
                if candidate.get("source_type") == "ui_change_state_frame"
                and candidate.get("data_id") != data_id_from_state_origin(origin)
            ]
            same_category_other_pool_for_current = [
                candidate
                for candidate in same_category_final_state_pool
                if candidate.get("data_id") != data_id_from_state_origin(origin)
            ]
            (
                gt,
                option_map,
                option_origin_ids,
                option_source_types,
                option_category_names,
                option_data_ids,
                option_semantic_distances,
            ) = build_vqa_state_ui_change_text_options(
                vqa_origin,
                answer,
                data_id_from_state_origin(origin),
                same_category_ui_pool_for_current,
                same_category_other_pool_for_current,
                different_category_state_pool,
                current_category_name,
            )
        else:
            gt, option_map, _, option_origin_ids, option_source_types = build_text_options(
                answer,
                answer_pool,
                vqa_origin,
                correct_origin_qa_id=vqa_origin,
            )
        img_out = form / "imgs" / f"{origin}_vqa_state_query_img.jpg"
        write_frame(video_path, frame, img_out)
        item = base_item(
            f"{origin}_vqa_state",
            with_options(query_stem, option_map),
            gt,
            "vqa_state",
            "state_description",
            "L2",
            "vqa_state_visible_lighting__mcq",
            "keyword",
            "current_visible_state",
            {"type": "choice", "labels": LABELS, "choice_modality": "text"},
            answer,
            source_span,
        )
        item.update({"query_img_path": rel(img_out, form), "query_source_frame": frame, "source_label": source_label, "correct_answer": answer})
        add_option_text_fields(item, option_map, option_origin_ids, option_source_types)
        if option_category_names is not None:
            item["option_category_names"] = option_category_names
            item["option_data_ids"] = option_data_ids
            item["option_semantic_distances"] = option_semantic_distances
            item["distractor_strategy"] = (
                "is_final_state_two_same_category_one_different_category_far_text"
                if source_label == "is_final_state_frame"
                else "ui_change_one_same_ui_one_same_other_one_different_category_far_text"
            )
        vqa.append(item)

        open_item = base_item(
            f"{origin}_vqa_state",
            query_stem,
            answer,
            "vqa_state",
            "state_description",
            "L2",
            "vqa_state_visible_lighting__open",
            "keyword",
            "current_visible_state",
            {"type": "keyword", "value": "state_phrase"},
            answer,
            source_span,
        )
        open_item.update({"query_img_path": rel(img_out, form), "query_source_frame": frame, "source_label": source_label, "answer_type": "keyword"})
        openqa.append(open_item)
    write_dataset(output_root, "vqa_state", "img2txt", vqa, openqa)


def build_final_state_text_forms(records: list[VideoRecord], output_root: Path) -> None:
    current_category_name = category_key(output_root.parent.name)
    answer_pool = [
        final_state_text_candidate(
            f"{record.data_id}_final_001",
            record.overall_verification,
            "final_state",
            current_category_name,
            record.data_id,
        )
        for record in records
    ]
    same_category_external_final_state_text_pool = load_external_final_state_text_candidates(
        output_root.parent,
        target="same",
    )
    external_final_state_text_pool = load_external_final_state_text_candidates(
        output_root.parent,
        target="different",
    )
    for current_form in ["video2txt", "img2txt"]:
        form = form_dir(output_root, "final_state", current_form)
        vqa: list[dict] = []
        openqa: list[dict] = []
        for record in records:
            origin = f"{record.data_id}_final_001"
            query_stem = f'What visible outcome confirms that "{record.overall_requirement}" has succeeded? Answer with a short phrase.'
            option_category_names = None
            distractor_strategy_meta = {}
            if current_form in {"img2txt", "video2txt"}:
                (
                    gt,
                    option_map,
                    _,
                    option_origin_ids,
                    option_source_types,
                    option_category_names,
                    distractor_strategy_meta,
                ) = build_final_state_img2txt_text_options(
                    record,
                    answer_pool + same_category_external_final_state_text_pool,
                    external_final_state_text_pool,
                    origin,
                    current_category_name,
                )
            else:
                gt, option_map, _, option_origin_ids, option_source_types = build_text_options(
                    record.overall_verification,
                    answer_pool,
                    origin,
                    correct_origin_qa_id=origin,
                )
            item = base_item(
                origin,
                with_options(query_stem, option_map),
                gt,
                "final_state",
                "final_state",
                "L3",
                "final_state_visible_outcome__mcq",
                "keyword",
                "visible_outcome",
                {"type": "choice", "labels": LABELS, "choice_modality": "text"},
                record.overall_verification,
                {"start": record.final_state_frame, "end": record.final_state_frame},
            )
            item.update({"correct_answer": record.overall_verification, "is_final_state_frame": record.final_state_frame, "final_state_gt_mode": "is_final_state_frame"})
            if current_form == "video2txt":
                video_out = form / "videos" / f"{record.data_id}.mp4"
                ensure_video_copy(record.video_path, video_out)
                item["query_video_path"] = rel(video_out, form)
            else:
                img_out = form / "imgs" / f"{origin}_query_img.jpg"
                write_frame(record.video_path, record.final_state_frame, img_out)
                item["query_img_path"] = rel(img_out, form)
                item["query_source_frame"] = record.final_state_frame
            add_option_text_fields(item, option_map, option_origin_ids, option_source_types)
            if option_category_names is not None:
                item["option_category_names"] = option_category_names
            if distractor_strategy_meta:
                item.update(distractor_strategy_meta)
            vqa.append(item)

            open_item = base_item(
                origin,
                query_stem,
                record.overall_verification,
                "final_state",
                "final_state",
                "L3",
                "final_state_visible_outcome__open",
                "keyword",
                "visible_outcome",
                {"type": "keyword", "value": "final_state_phrase"},
                record.overall_verification,
                {"start": record.final_state_frame, "end": record.final_state_frame},
            )
            open_item.update({"answer_type": "keyword", "is_final_state_frame": record.final_state_frame, "final_state_gt_mode": "is_final_state_frame"})
            if current_form == "video2txt":
                open_item["query_video_path"] = item["query_video_path"]
            else:
                open_item["query_img_path"] = item["query_img_path"]
                open_item["query_source_frame"] = record.final_state_frame
            openqa.append(open_item)
        write_dataset(output_root, "final_state", current_form, vqa, openqa)


def build_visual_state_choice_forms(records: list[VideoRecord], output_root: Path, task_family: str, mode: str) -> None:
    candidates = visual_candidates(records, mode)
    external_final_state_candidates = load_external_final_state_candidates(output_root.parent) if task_family == "final_state" else []
    current_category_name = category_key(output_root.parent.name)
    if not candidates:
        for current_form in ["img2img", "video2img"]:
            write_dataset(output_root, task_family, current_form, [], [])
        return
    for current_form in ["img2img", "video2img"]:
        form = form_dir(output_root, task_family, current_form)
        vqa: list[dict] = []
        for candidate in candidates:
            record = candidate["record"]
            origin = candidate["origin_qa_id"]
            if task_family == "final_state":
                query_text_img = f'This frame is captured before the result appears. After the operator completes "{record.last_action.description if record.last_action else record.overall_requirement}", which image best shows the final state of "{record.overall_requirement}"?'
                query_text_vid = f'Watch the clip up to just before the outcome is revealed. Which image best represents the future result of "{record.overall_requirement}"?'
                subtask_role = "snapshot_transition_prediction" if current_form == "img2img" else "future_result_prediction"
                query_temporal_role = "after_last_action_before_result" if current_form == "img2img" else "context_before_result_reveal"
                prompt_variant = f"lighting_final_state_{current_form}__strict_future"
                semantic_anchor = "visible_outcome"
            else:
                query_text_img = f'Based on this pre-check frame, which image best shows the visible signal that "{record.overall_requirement}" worked?'
                query_text_vid = f'Watch the context clip before the visible signal. Which image shows the success signal for "{record.overall_requirement}"?'
                subtask_role = "verification_state_prediction"
                query_temporal_role = "context_before_success_signal"
                prompt_variant = f"lighting_verification_state_{current_form}__state_signal"
                semantic_anchor = "visible_success_signal"

            query_text = query_text_img if current_form == "img2img" else query_text_vid
            item = base_item(
                origin,
                query_text,
                "",
                task_family,
                "final_state" if task_family == "final_state" else "verification_state",
                "L3" if task_family == "final_state" else "L4",
                prompt_variant,
                "choice",
                semantic_anchor,
                {"type": "choice", "labels": LABELS, "choice_modality": "image"},
                candidate["answer"],
                candidate["source_span"],
            )
            if current_form == "img2img":
                if task_family == "final_state":
                    query_frame, query_rule, query_from_action = select_final_state_img2img_query_frame(record, candidate["frame"])
                else:
                    query_frame = clamp_frame((record.last_action.anchor_frame if record.last_action else candidate["frame"] - 15), record.total_frames)
                    if query_frame >= candidate["frame"]:
                        query_frame = clamp_frame(candidate["frame"] - 10, record.total_frames)
                    query_rule = "last_action_anchor"
                    query_from_action = record.last_action.origin_qa_id if record.last_action else None
                query_img = form / "imgs" / f"{origin}_query_img.jpg"
                write_frame(record.video_path, query_frame, query_img)
                item["query_img_path"] = rel(query_img, form)
                item["query_source_frame"] = query_frame
                item["query_source_rule"] = query_rule
                item["query_source_action_qa_id"] = query_from_action
                future_gap = candidate["frame"] - query_frame
            else:
                if task_family == "final_state":
                    query_start, query_end, query_rule, query_from_action = select_final_state_video2img_query_range(record, candidate["frame"])
                else:
                    query_end = clamp_frame(candidate["frame"] - 10, record.total_frames)
                    query_start = clamp_frame(query_end - 45, record.total_frames)
                    query_rule = "pre_target_context_clip"
                    query_from_action = None
                query_video = form / "videos" / f"{origin}_query_video.mp4"
                write_clip(record.video_path, query_start, query_end, query_video, record.fps)
                item["query_video_path"] = rel(query_video, form)
                item["query_source_range"] = {"start_frame": query_start, "end_frame": query_end}
                item["query_source_rule"] = query_rule
                item["query_source_action_qa_id"] = query_from_action
                future_gap = candidate["frame"] - query_end

            if task_family == "final_state" and current_form in {"img2img", "video2img"}:
                option_forbidden_ranges = [(query_start, query_end)] if current_form == "video2img" else None
                option_query_reference = query_end if current_form == "video2img" else query_frame
                options = choose_final_state_img2img_options(
                    candidate,
                    candidates,
                    external_final_state_candidates,
                    origin,
                    option_query_reference,
                    current_category_name,
                    forbidden_ranges=option_forbidden_ranges,
                )
            else:
                options = choose_visual_options(candidate, candidates, origin)
            option_paths = []
            option_frames = []
            option_origin_ids = []
            option_source_types = []
            option_category_names = []
            for label, option in zip(LABELS, options):
                option_img = form / "imgs" / f"option_img_{origin}_{label}.jpg"
                write_frame(option["video_path"], option["frame"], option_img)
                option_paths.append(rel(option_img, form))
                option_frames.append(option["frame"])
                option_origin_ids.append(option["origin_qa_id"])
                option_source_types.append(option["source_type"])
                option_category_names.append(option.get("category_name", current_category_name))
                if option["origin_qa_id"] == candidate["origin_qa_id"]:
                    item["GT"] = label
            item.update(
                {
                    "option_imgs_path": option_paths,
                    "option_source_frames": option_frames,
                    "option_origin_qa_ids": option_origin_ids,
                    "option_source_types": option_source_types,
                    "option_category_names": option_category_names,
                    "source_label": candidate["source_type"],
                    "is_final_state_frame": record.final_state_frame,
                    "gt_frame_source_label": candidate["source_type"],
                    "gt_frame_index": candidate["frame"],
                    "subtask_role": subtask_role,
                    "query_temporal_role": query_temporal_role,
                    "gt_temporal_role": "stable_post_result" if task_family == "final_state" else "post_change_state",
                    "strict_future_pass": future_gap > 0,
                    "future_gap_frames": future_gap,
                    "final_state_logic_version": "lighting_is_final_state_frame_v1" if task_family == "final_state" else None,
                    "verification_logic_version": "lighting_ui_change_state_v1" if task_family == "verification_state" else None,
                }
            )
            vqa.append(item)
        write_dataset(output_root, task_family, current_form, vqa, [])


def build_verification_state_text_forms(records: list[VideoRecord], output_root: Path) -> None:
    targets = verification_multiselect_targets(records)
    pool = verification_option_pool(records)
    current_category_name = category_key(output_root.parent.name)
    same_category_ui_pool = load_external_ui_change_candidates(output_root.parent, target="same")
    same_category_final_state_pool = load_external_final_state_candidates(output_root.parent, target="same")
    same_category_state_pool = same_category_ui_pool + same_category_final_state_pool
    different_category_ui_pool = load_external_ui_change_candidates(output_root.parent, target="different")
    different_category_final_state_pool = load_external_final_state_candidates(output_root.parent)
    for current_form in ["video2txt", "img2txt"]:
        form = form_dir(output_root, "verification_state", current_form)
        vqa: list[dict] = []
        openqa: list[dict] = []
        for target in targets:
            record = target["record"]
            origin = target["origin_qa_id"]
            correct_targets = target["correct_targets"]
            ui_target = next(t for t in correct_targets if t["verification_signal_type"] == "ui_signal")
            physical_target = next(t for t in correct_targets if t["verification_signal_type"] == "physical_world_state")
            if current_form == "img2txt":
                options = choose_verification_state_img2txt_options(
                    correct_targets,
                    origin,
                    record,
                    current_category_name,
                    different_category_ui_pool,
                    different_category_final_state_pool,
                )
            else:
                options = choose_verification_state_video2txt_options(
                    correct_targets,
                    pool,
                    origin,
                    record,
                    current_category_name,
                    same_category_state_pool,
                    different_category_ui_pool,
                    different_category_final_state_pool,
                )
            query_stem = f'Which visible states together verify that "{record.overall_requirement}" worked? Select all applicable options.'
            item = base_item(
                origin,
                with_multiselect_options(query_stem, options),
                [],
                "verification_state",
                "verification_state",
                "L4",
                "verify_state_ui_plus_physical__multiselect_text",
                "multi_select",
                "visible_success_signals",
                {
                    "type": "multi_select",
                    "labels": LABELS5,
                    "choice_modality": "text",
                    "min_correct": 2,
                    "max_correct": 2,
                    "required_signal_types": ["ui_signal", "physical_world_state"],
                },
                f"UI signal: {ui_target['answer']}; physical-world state: {physical_target['answer']}",
                {
                    "start": min(ui_target["frame"], physical_target["frame"]),
                    "end": max(ui_target["frame"], physical_target["frame"]),
                },
            )
            item.update(
                {
                    "correct_answer": {
                        "ui_signal": ui_target["answer"],
                        "physical_world_state": physical_target["answer"],
                    },
                    "source_label": "ui_change_plus_is_final_state",
                    "verification_target_mode": "multi_signal_ui_plus_physical_state",
                    "ui_change_frame": ui_target["frame"],
                    "is_final_state_frame": physical_target["frame"],
                    "physical_world_state_source": "is_final_state",
                    "verification_logic_version": "lighting_multiselect_ui_plus_physical_v1",
                }
            )
            if current_form == "video2txt":
                query_start, query_end, query_rule, query_from_action = select_verification_state_video2txt_query_range(record)
                video_out = form / "videos" / f"{origin}_query_video.mp4"
                write_clip(record.video_path, query_start, query_end, video_out, record.fps)
                item["query_video_path"] = rel(video_out, form)
                item["query_source_range"] = {"start_frame": query_start, "end_frame": query_end}
                item["query_source_rule"] = query_rule
                item["query_source_action_qa_id"] = query_from_action
                item["future_gap_frames"] = min(ui_target["frame"], physical_target["frame"]) - query_end
                item["strict_future_pass"] = item["future_gap_frames"] > 0
            else:
                query_frame = clamp_frame(min(ui_target["frame"], physical_target["frame"]) - 8, record.total_frames)
                img_out = form / "imgs" / f"{origin}_query_img.jpg"
                write_frame(record.video_path, query_frame, img_out)
                item["query_img_path"] = rel(img_out, form)
                item["query_source_frame"] = query_frame
            add_multiselect_text_option_fields(item, options)
            vqa.append(item)

            open_item = base_item(
                origin,
                f'List both the UI-side signal and the physical-world state that verify "{record.overall_requirement}" worked.',
                f"UI signal: {ui_target['answer']}; physical-world state: {physical_target['answer']}",
                "verification_state",
                "verification_state",
                "L4",
                "verify_state_ui_plus_physical__structured_open",
                "structured_short_answer",
                "visible_success_signals",
                {
                    "type": "structured_short_answer",
                    "fields": ["ui_signal", "physical_world_state"],
                },
                f"UI signal: {ui_target['answer']}; physical-world state: {physical_target['answer']}",
                {
                    "start": min(ui_target["frame"], physical_target["frame"]),
                    "end": max(ui_target["frame"], physical_target["frame"]),
                },
            )
            open_item.update(
                {
                    "GT": {
                        "ui_signal": ui_target["answer"],
                        "physical_world_state": physical_target["answer"],
                    },
                    "answer_type": "structured_short_answer",
                    "source_label": "ui_change_plus_is_final_state",
                    "verification_target_mode": "multi_signal_ui_plus_physical_state",
                    "ui_change_frame": ui_target["frame"],
                    "is_final_state_frame": physical_target["frame"],
                    "physical_world_state_source": "is_final_state",
                    "verification_logic_version": "lighting_multiselect_ui_plus_physical_v1",
                }
            )
            if current_form == "video2txt":
                open_item["query_video_path"] = item["query_video_path"]
                open_item["query_source_range"] = item["query_source_range"]
                open_item["query_source_rule"] = item["query_source_rule"]
                open_item["query_source_action_qa_id"] = item["query_source_action_qa_id"]
            else:
                open_item["query_img_path"] = item["query_img_path"]
                open_item["query_source_frame"] = item["query_source_frame"]
            openqa.append(open_item)
        write_dataset(output_root, "verification_state", current_form, vqa, openqa)


def build_verification_state_visual_forms(records: list[VideoRecord], output_root: Path) -> None:
    targets = verification_multiselect_targets(records)
    pool = verification_option_pool(records)
    current_category_name = category_key(output_root.parent.name)
    same_category_ui_pool = load_external_ui_change_candidates(output_root.parent, target="same")
    different_category_final_state_pool = load_external_final_state_candidates(output_root.parent)
    for current_form in ["img2img", "video2img"]:
        form = form_dir(output_root, "verification_state", current_form)
        vqa: list[dict] = []
        for target in targets:
            record = target["record"]
            origin = target["origin_qa_id"]
            correct_targets = target["correct_targets"]
            ui_target = next(t for t in correct_targets if t["verification_signal_type"] == "ui_signal")
            physical_target = next(t for t in correct_targets if t["verification_signal_type"] == "physical_world_state")
            query_text_img = f'After this lighting command, which images show the evidence that "{record.overall_requirement}" worked? Select all applicable options.'
            query_text_vid = f'Watch the context before the verification signals appear. Which images show the evidence that "{record.overall_requirement}" worked? Select all applicable options.'
            query_text = query_text_img if current_form == "img2img" else query_text_vid
            item = base_item(
                origin,
                query_text,
                [],
                "verification_state",
                "verification_state",
                "L4",
                f"verify_state_ui_plus_physical_{current_form}__multiselect_image",
                "multi_select",
                "visible_success_signals",
                {
                    "type": "multi_select",
                    "labels": LABELS5,
                    "choice_modality": "image",
                    "min_correct": 2,
                    "max_correct": 2,
                    "required_signal_types": ["ui_signal", "physical_world_state"],
                },
                f"UI signal: {ui_target['answer']}; physical-world state: {physical_target['answer']}",
                {
                    "start": min(ui_target["frame"], physical_target["frame"]),
                    "end": max(ui_target["frame"], physical_target["frame"]),
                },
            )
            action = next((a for a in record.actions if not a.is_verification), record.actions[0] if record.actions else None)
            if current_form == "img2img":
                query_frame = clamp_frame(min(ui_target["frame"], physical_target["frame"]) - 8, record.total_frames)
                query_img = form / "imgs" / f"{origin}_query_img.jpg"
                write_frame(record.video_path, query_frame, query_img)
                item["query_img_path"] = rel(query_img, form)
                item["query_source_frame"] = query_frame
                future_gap = min(ui_target["frame"], physical_target["frame"]) - query_frame
                options = choose_verification_state_img2img_options(
                    correct_targets,
                    pool,
                    origin,
                    record,
                    query_frame,
                    current_category_name,
                    same_category_ui_pool,
                    different_category_final_state_pool,
                )
            else:
                query_end = clamp_frame((action.end if action else ui_target["frame"] - 10), record.total_frames)
                if query_end >= min(ui_target["frame"], physical_target["frame"]):
                    query_end = clamp_frame(min(ui_target["frame"], physical_target["frame"]) - 8, record.total_frames)
                query_start = clamp_frame(query_end - 45, record.total_frames)
                query_video = form / "videos" / f"{origin}_query_video.mp4"
                write_clip(record.video_path, query_start, query_end, query_video, record.fps)
                item["query_video_path"] = rel(query_video, form)
                item["query_source_range"] = {"start_frame": query_start, "end_frame": query_end}
                future_gap = min(ui_target["frame"], physical_target["frame"]) - query_end
                options = choose_verification_state_img2img_options(
                    correct_targets,
                    pool,
                    origin,
                    record,
                    query_end,
                    current_category_name,
                    same_category_ui_pool,
                    different_category_final_state_pool,
                )

            option_paths = []
            option_frames = []
            option_origin_ids = []
            option_source_types = []
            option_category_names = []
            option_data_ids = []
            option_video_names = []
            option_answer_texts = []
            option_correctness = []
            option_perceptual_distances = []
            gt_labels = []
            for label, option in zip(LABELS5, options):
                option_img = form / "imgs" / f"option_img_{origin}_{label}.jpg"
                write_frame(option["video_path"], option["frame"], option_img)
                option_paths.append(rel(option_img, form))
                option_frames.append(option["frame"])
                option_origin_ids.append(option["origin_qa_id"])
                option_source_types.append(option["source_type"])
                option_category_names.append(option.get("category_name", current_category_name))
                option_data_ids.append(option.get("data_id") or option.get("record").data_id if option.get("record") is not None else "")
                option_video_names.append(option.get("video_name", ""))
                option_answer_texts.append(clean_answer(option["answer"]))
                option_perceptual_distances.append(option.get("perceptual_distance"))
                is_correct = bool(option.get("correct"))
                option_correctness.append(is_correct)
                if is_correct:
                    gt_labels.append(label)
            item.update(
                {
                    "GT": gt_labels,
                    "correct_answers": gt_labels,
                    "answer_mode": "multi_select",
                    "correct_answer": {
                        "ui_signal": ui_target["answer"],
                        "physical_world_state": physical_target["answer"],
                    },
                    "option_imgs_path": option_paths,
                    "option_source_frames": option_frames,
                    "option_origin_qa_ids": option_origin_ids,
                    "option_source_types": option_source_types,
                    "option_category_names": option_category_names,
                    "option_data_ids": option_data_ids,
                    "option_video_names": option_video_names,
                    "option_answer_texts": option_answer_texts,
                    "option_perceptual_distances": option_perceptual_distances,
                    "option_correctness": option_correctness,
                    "source_label": "ui_change_plus_is_final_state",
                    "ui_change_frame": ui_target["frame"],
                    "is_final_state_frame": physical_target["frame"],
                    "physical_world_state_source": "is_final_state",
                    "verification_target_mode": "multi_signal_ui_plus_physical_state",
                    "verification_logic_version": "lighting_multiselect_ui_plus_physical_v1",
                    "subtask_role": "verification_state_multiselect",
                    "query_temporal_role": "command_context_before_verification_signals",
                    "gt_temporal_role": "post_change_ui_and_physical_state",
                    "strict_future_pass": future_gap > 0,
                    "future_gap_frames": future_gap,
                }
            )
            vqa.append(item)
        write_dataset(output_root, "verification_state", current_form, vqa, [])


def write_empty_recovery(output_root: Path) -> None:
    form = form_dir(output_root, "recovery", "video2txt")
    note = {
        "data": [],
        "notes": "No recovery samples generated because Action_5_Light_1 has no Abnormal/Wrong action/Correction action chain annotations.",
    }
    (form / "vqa.json").write_text(json.dumps(note, ensure_ascii=False, indent=2), encoding="utf-8")
    (form / "openqa.json").write_text(json.dumps(note, ensure_ascii=False, indent=2), encoding="utf-8")


def infer_modalities(form_name: str) -> tuple[str, str]:
    input_name, output_name = form_name.split("2", 1)
    input_mapping = {"img": "image", "video": "video", "txt": "text"}
    output_mapping = {"img": "img", "video": "video", "txt": "text"}
    return input_mapping.get(input_name, input_name), output_mapping.get(output_name, output_name)


def collect_asset_counts(items: list[dict]) -> dict:
    query_videos: set[str] = set()
    option_videos: set[str] = set()
    query_imgs: set[str] = set()
    option_imgs: set[str] = set()
    for item in items:
        if item.get("query_video_path"):
            query_videos.add(item["query_video_path"])
        if item.get("query_img_path"):
            query_imgs.add(item["query_img_path"])
        option_videos.update(item.get("option_videos_path", []) or [])
        option_imgs.update(item.get("option_imgs_path", []) or [])
    return {
        "query_videos": len(query_videos),
        "option_videos": len(option_videos),
        "query_imgs": len(query_imgs),
        "option_imgs": len(option_imgs),
    }


def validate_assets(output_root: Path) -> list[str]:
    missing = []
    for json_path in output_root.rglob("*.json"):
        if json_path.name not in {"vqa.json", "openqa.json"}:
            continue
        form_dir_path = json_path.parent
        data = json.loads(json_path.read_text(encoding="utf-8")).get("data", [])
        for item in data:
            paths = []
            for key in ["query_video_path", "query_img_path"]:
                if item.get(key):
                    paths.append(item[key])
            paths.extend(item.get("option_videos_path", []) or [])
            paths.extend(item.get("option_imgs_path", []) or [])
            for value in paths:
                if not (form_dir_path / value).exists():
                    missing.append(f"{json_path}: {value}")
    return missing


def write_manifest(output_root: Path, records: list[VideoRecord]) -> None:
    task_families = {}
    for family_dir in sorted(p for p in output_root.iterdir() if p.is_dir()):
        forms = {}
        for form_path in sorted(p for p in family_dir.iterdir() if p.is_dir()):
            vqa_path = form_path / "vqa.json"
            openqa_path = form_path / "openqa.json"
            vqa_items = json.loads(vqa_path.read_text(encoding="utf-8")).get("data", []) if vqa_path.exists() else []
            openqa_items = json.loads(openqa_path.read_text(encoding="utf-8")).get("data", []) if openqa_path.exists() else []
            input_modality, output_modality = infer_modalities(form_path.name)
            forms[form_path.name] = {
                "input_modality": input_modality,
                "output_modality": output_modality,
                "mcq": len(vqa_items),
                "openqa": len(openqa_items),
                "asset_counts": collect_asset_counts(vqa_items + openqa_items),
                "candidate_recovery_future_metadata_only": family_dir.name == "recovery",
            }
        task_families[family_dir.name] = forms

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source_json": SOURCE_JSON_NAME,
        "num_videos": len(records),
        "task_families": list(task_families.keys()),
        "dependency": "opencv-python-headless",
        "form_matrix": task_families,
        "annotation_summary": {
            "action_segments": sum(len(r.actions) for r in records),
            "verification_action_segments": sum(len([a for a in r.actions if a.is_verification]) for r in records),
            "ui_change_frames": sum(len(r.ui_changes) for r in records),
            "is_final_state_frames": sum(1 for r in records if r.final_state_frame),
            "verification_state_multiselect_samples": len(verification_multiselect_targets(records)),
        },
        "generation_notes": [
            "Category-aware sampling groups sibling batch folders by normalized category key, so folders such as 灯光与照明控制-问卷 and 灯光与照明控制-问卷_1 are treated as the same category.",
            "For action/video2txt, text distractors use category-aware sampling: one semantically distant same-category action plus two different-category actions from annotations/latest.",
            "For verification_action/video2txt and verification_action/img2txt, text distractors use category-aware sampling: one same-category verification action plus two different-category verification actions from annotations/latest.",
            "For verification_action/img2video, query frames use the previous action anchor; the correct option runs from query_frame+10 to the current verification action end, with same-video non-overlap, same-category other-video, and different-category other-video distractors.",
            "For verification_action/video2video, query clips use the penultimate action segment; the correct option runs from query_clip_end+10 to the verification action end, with same-video non-overlap, same-category other-video, and different-category other-video distractors.",
            "For final_state/img2txt and final_state/video2txt, text distractors use category-aware sampling: one semantically distant same-category final state plus two different-category final states from annotations/latest.",
            "For final_state/video2img, query clips are taken from the penultimate action segment; image options include one same-video wrong frame, one same-category final state, and one different-category final state.",
            "Final-state visual GT uses explicit is_final_state frame.",
            "Verification-state samples are five-option multi-select questions when ui_change and is_final_state are both available.",
            "For verification_state/img2txt, all distractor text options are sampled from different normalized categories.",
            "For verification_state/video2txt, distractor text options use one semantically distant same-category state and two states from different normalized categories.",
            "For verification_state/video2txt, query clips end at the first action anchor frame minus 30 frames.",
            "For verification_state/img2img and verification_state/video2img, distractors are fixed as two same-category ui_change frames and one different-category is_final_state frame.",
            "For verification_state, ui_change is treated as the UI-side success signal and is_final_state is treated as the physical-world state verification.",
            "Recovery is intentionally empty; no abnormal/correction chain exists in this package.",
            "For action/img2txt, the query image uses the previous action anchor frame when available; if no previous action exists, it falls back to a deterministic frame 30-40 frames before the current action and before the current action span.",
            "For vqa_state/img2txt is_final_state samples, distractors use one same-category final_state, one semantically distant same-category non-final-state annotation, and one different-category state annotation.",
            "For vqa_state/img2txt ui_change samples, distractors use one same-category ui_change, one semantically distant same-category non-ui annotation, and one different-category state annotation.",
        ],
    }
    (output_root / "dataset_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_root / "form_matrix.json").write_text(
        json.dumps(
            {
                "schema_version": "switch-hf-innovative-qa-v2-multiform",
                "dependency": "opencv-python-headless",
                "task_families": task_families,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    readme = [
        "# Lighting QA v2 Multiform v1",
        "",
        "Generated from `Action_5_Light_1.json` using the state-aligned SWITCH v2 QA layout.",
        "",
        "## Key decisions",
        "",
        "- `is_final_state` frames are used as whole-video final-state GT.",
        "- Category-aware candidate pools normalize sibling batch folders into one category, e.g. `灯光与照明控制-问卷` and `灯光与照明控制-问卷_1` are both `灯光与照明控制`.",
        "- `action/video2txt` text distractors use one semantically distant same-category action and two actions from other `annotations/latest` category folders.",
        "- `verification_action/video2txt` and `verification_action/img2txt` text distractors use one same-category verification action and two verification actions from other `annotations/latest` category folders.",
        "- `verification_action/img2video` uses the previous action anchor as the query frame. Its correct video option spans from `query_source_frame + 10` to the current verification action end, with one non-overlapping same-video distractor, one same-category other-video distractor, and one different-category other-video distractor.",
        "- `verification_action/video2video` uses the penultimate action segment as the query clip. Its correct video option spans from `query_source_range.end_frame + 10` to the current verification action end, with one non-overlapping same-video distractor, one same-category other-video distractor, and one different-category other-video distractor.",
        "- `final_state/img2txt` and `final_state/video2txt` text distractors use one semantically distant same-category final state and two final states from other `annotations/latest` category folders.",
        "- `final_state/video2img` query clips come from the penultimate action segment; options include same-video wrong, same-category, and different-category frames.",
        "- `verification_state` is generated as a five-option multi-select task when both `ui_change` and `is_final_state` exist.",
        "- `verification_state/img2txt` keeps the two current-video correct options, while all three distractor text options come from different normalized categories.",
        "- `verification_state/video2txt` keeps the two current-video correct options, adds one semantically distant same-category state distractor, and adds two distractors from different normalized categories.",
        "- `verification_state/video2txt` query clips end at `first_action.action_anchor - 30` and use the preceding 45-frame context.",
        "- `verification_state/img2img` and `verification_state/video2img` use fixed distractor roles: two same-category `ui_change` frames and one different-category `is_final_state` frame.",
        "- In `verification_state`, `ui_change` is the UI-side verification signal and `is_final_state` is treated as the physical-world state verification.",
        "- `action_anchor` frames are representative frames inside action segments; their semantics come from the matching `action_description`, `action_requirement`, and `action-type` segment.",
        "- For `action/img2txt`, the query image uses the previous action anchor frame when available; otherwise it uses a frame 40-50 frames before the current `action_anchor_frame`, while still staying before the action span.",
        "- For `action/img2video`, the query image follows the same previous-action / 40-50-frame-before-anchor fallback rule. The correct option clip starts 10 frames after the query image and runs to the end of the current action. One distractor is sampled from the same video with a non-overlapping interval when available; the remaining distractors are sampled from other videos.",
        "- For `action/video2txt` and `action/video2video`, the query video uses the previous action segment when available. For first-step actions, both forms fall back to a short pre-action clip whose last frame is 50-60 frames before the current `action_anchor_frame`. In `action/video2video`, one distractor is sampled from the same video with a non-overlapping interval when available.",
        "- For `vqa_state/img2txt` is_final_state samples, distractors use one same-category final_state annotation, one semantically distant same-category non-final-state annotation, and one different-category state annotation.",
        "- For `vqa_state/img2txt` ui_change samples, distractors use one same-category ui_change annotation, one semantically distant same-category non-ui annotation, and one different-category state annotation.",
        "- Recovery is not generated because this package has no abnormal/wrong-action/correction chain annotations.",
        "",
        "## Counts",
        "",
        f"- Videos: {len(records)}",
        f"- Action segments: {manifest['annotation_summary']['action_segments']}",
        f"- Verification action segments: {manifest['annotation_summary']['verification_action_segments']}",
        f"- UI change frames: {manifest['annotation_summary']['ui_change_frames']}",
        f"- Final state frames: {manifest['annotation_summary']['is_final_state_frames']}",
        f"- Verification-state multi-select samples: {manifest['annotation_summary']['verification_state_multiselect_samples']}",
        "",
        "See `dataset_manifest.json` and `form_matrix.json` for per-form counts.",
    ]
    (output_root / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")


def build(source_dir: Path, source_json: Path, output_root: Path) -> None:
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    records = parse_records(source_json, source_dir)
    build_task_forms(records, output_root)
    build_vqa_state(records, output_root)
    build_action_text_forms(records, output_root, "action", verification_only=None)
    build_action_video_forms(records, output_root, "action", verification_only=None)
    build_final_state_text_forms(records, output_root)
    build_visual_state_choice_forms(records, output_root, "final_state", "final_state")
    build_action_text_forms(records, output_root, "verification_action", verification_only=True)
    build_action_video_forms(records, output_root, "verification_action", verification_only=True)
    build_verification_state_text_forms(records, output_root)
    build_verification_state_visual_forms(records, output_root)
    write_empty_recovery(output_root)
    missing = validate_assets(output_root)
    if missing:
        raise RuntimeError("Missing generated assets:\n" + "\n".join(missing[:20]))
    write_manifest(output_root, records)


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    default_source_dir = script_dir / SOURCE_FOLDER_NAME
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=default_source_dir)
    parser.add_argument("--source-json", type=Path, default=default_source_dir / SOURCE_JSON_NAME)
    parser.add_argument("--output-root", type=Path, default=script_dir / OUTPUT_NAME)
    args = parser.parse_args()
    build(args.source_dir, args.source_json, args.output_root)
    print(json.dumps({"output_root": str(args.output_root), "status": "ok"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
