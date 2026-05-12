#!/usr/bin/env python3
from __future__ import annotations

import csv
import html
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


ACTION_LABELS = {
    "action-type",
    "action_requirement",
    "action_description",
    "action_step_id",
}

SIGNAL_LABELS = [
    "ui_change",
    "view_change",
    "ui_state",
    "physical_world_change",
    "physical_world_state",
    "is_final_state",
]


@dataclass
class ActionEvent:
    start_frame: int
    end_frame: int
    action_type: str = ""
    action_requirement: str = ""
    action_description: str = ""
    action_step_id: str = ""


@dataclass
class ReviewEntry:
    entry_id: str
    category: str
    label: str
    start_frame: int
    end_frame: int
    span_type: str
    anchor_frame: int
    linked_step_id: str
    linked_action_type: str
    text: str
    action_requirement: str = ""
    action_description: str = ""
    action_type: str = ""
    image_path: Optional[str] = None
    clip_path: Optional[str] = None
    storyboard_path: Optional[str] = None


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
        frame = self.read_frame(video_path, frame_index)
        if not cv2.imwrite(str(output_path), frame):
            raise RuntimeError(f"Unable to write image: {output_path}")

    def extract_clip(self, video_path: Path, start_frame: int, end_frame: int, fps: float, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open video: {video_path}")
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
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


def safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text).strip("._") or "item"


def get_meta_text(result: Dict[str, Any]) -> str:
    meta = result.get("meta") or {}
    value = meta.get("text")
    if isinstance(value, list) and value:
        return str(value[0]).strip()
    if isinstance(value, str):
        return value.strip()
    return ""


def get_range(result: Dict[str, Any]) -> Tuple[int, int]:
    ranges = (result.get("value") or {}).get("ranges") or [{}]
    first = ranges[0]
    return int(first.get("start") or 0), int(first.get("end") or 0)


def resolve_item(annotation_root: Path, video_name: str) -> Tuple[Path, Dict[str, Any]]:
    json_paths = sorted(annotation_root.glob("SWITCH*Action_1.json"))
    json_paths.append(annotation_root / "SWITCH_Action_2.json")
    for json_path in json_paths:
        if not json_path.exists():
            continue
        items = json.loads(json_path.read_text(encoding="utf-8"))
        for item in items:
            data = item.get("data") or {}
            item_video = next(
                (
                    Path(value).name
                    for key, value in data.items()
                    if key != "meta" and isinstance(value, str) and value.lower().endswith(".mp4")
                ),
                None,
            )
            if item_video == video_name:
                return json_path, item
    raise FileNotFoundError(f"Could not find annotation item for {video_name}")


def resolve_video_path(annotation_root: Path, video_name: str) -> Path:
    for subdir in sorted(annotation_root.glob("SWITCHAction_*")):
        candidate = subdir / video_name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find local video for {video_name}")


def get_fps(item: Dict[str, Any]) -> float:
    meta = (item.get("data") or {}).get("meta") or {}
    for value in meta.values():
        if isinstance(value, dict) and "fps" in value:
            return float(value.get("fps") or 30.0)
    return 30.0


def parse_summary(item: Dict[str, Any]) -> Dict[str, Any]:
    summary = {
        "data_id": "",
        "frame_end": "",
        "overall_requirement": "",
        "overall_verification": "",
    }
    annotations = item.get("annotations") or []
    if not annotations:
        return summary
    for result in annotations[0].get("result", []):
        labels = (result.get("value") or {}).get("timelinelabels") or []
        if not labels:
            continue
        label = labels[0]
        text = get_meta_text(result)
        start_frame, end_frame = get_range(result)
        if label == "data_id":
            summary["data_id"] = text
        elif label == "frame_end":
            summary["frame_end"] = end_frame or start_frame
        elif label == "overall_requirement" and not summary["overall_requirement"]:
            summary["overall_requirement"] = text
        elif label == "overall_verification" and not summary["overall_verification"]:
            summary["overall_verification"] = text
    return summary


def parse_action_events(item: Dict[str, Any]) -> List[ActionEvent]:
    grouped: Dict[Tuple[int, int], ActionEvent] = {}
    annotations = item.get("annotations") or []
    if not annotations:
        return []
    for result in annotations[0].get("result", []):
        labels = (result.get("value") or {}).get("timelinelabels") or []
        if not labels:
            continue
        label = labels[0]
        if label not in ACTION_LABELS:
            continue
        start_frame, end_frame = get_range(result)
        key = (start_frame, end_frame)
        grouped.setdefault(key, ActionEvent(start_frame=start_frame, end_frame=end_frame))
        event = grouped[key]
        text = get_meta_text(result)
        if label == "action-type":
            event.action_type = text
        elif label == "action_requirement":
            event.action_requirement = text
        elif label == "action_description":
            event.action_description = text
        elif label == "action_step_id":
            event.action_step_id = text
    return sorted(grouped.values(), key=lambda event: (event.start_frame, event.end_frame))


def find_action_context(start_frame: int, end_frame: int, actions: List[ActionEvent]) -> Tuple[str, str]:
    for action in actions:
        if action.start_frame <= start_frame and end_frame <= action.end_frame:
            return action.action_step_id, action.action_type
    best_step = ""
    best_type = ""
    best_distance: Optional[int] = None
    midpoint = (start_frame + end_frame) // 2
    for action in actions:
        action_midpoint = (action.start_frame + action.end_frame) // 2
        distance = abs(action_midpoint - midpoint)
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_step = action.action_step_id
            best_type = action.action_type
    return best_step, best_type


def parse_signal_entries(item: Dict[str, Any], actions: List[ActionEvent]) -> List[ReviewEntry]:
    entries: List[ReviewEntry] = []
    annotations = item.get("annotations") or []
    if not annotations:
        return entries
    counters: Dict[str, int] = {}
    for result in annotations[0].get("result", []):
        labels = (result.get("value") or {}).get("timelinelabels") or []
        if not labels:
            continue
        label = labels[0]
        if label not in SIGNAL_LABELS:
            continue
        start_frame, end_frame = get_range(result)
        counters[label] = counters.get(label, 0) + 1
        step_id, action_type = find_action_context(start_frame, end_frame, actions)
        entries.append(
            ReviewEntry(
                entry_id=f"{label}_{counters[label]:03d}",
                category="signal",
                label=label,
                start_frame=start_frame,
                end_frame=end_frame,
                span_type="frame" if start_frame == end_frame else "segment",
                anchor_frame=(start_frame + end_frame) // 2,
                linked_step_id=step_id,
                linked_action_type=action_type,
                text=get_meta_text(result),
            )
        )
    return entries


def build_action_entries(actions: List[ActionEvent]) -> List[ReviewEntry]:
    entries: List[ReviewEntry] = []
    for index, action in enumerate(actions, start=1):
        entries.append(
            ReviewEntry(
                entry_id=f"action_{index:03d}",
                category="action",
                label="action_step",
                start_frame=action.start_frame,
                end_frame=action.end_frame,
                span_type="frame" if action.start_frame == action.end_frame else "segment",
                anchor_frame=(action.start_frame + action.end_frame) // 2,
                linked_step_id=action.action_step_id,
                linked_action_type=action.action_type,
                text=action.action_description or action.action_requirement or action.action_type,
                action_requirement=action.action_requirement,
                action_description=action.action_description,
                action_type=action.action_type,
            )
        )
    return entries


def make_storyboard(video_path: Path, start_frame: int, end_frame: int, output_path: Path, asset_writer: AssetWriter) -> None:
    frames: List[Any] = []
    probe_count = 4 if end_frame > start_frame else 1
    if probe_count == 1:
        frame_ids = [start_frame]
    else:
        frame_ids = [round(start_frame + (end_frame - start_frame) * idx / (probe_count - 1)) for idx in range(probe_count)]
    for frame_id in frame_ids:
        frame = asset_writer.read_frame(video_path, frame_id).copy()
        cv2.rectangle(frame, (8, 8), (180, 46), (20, 20, 20), thickness=-1)
        cv2.putText(frame, f"frame {frame_id}", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        frames.append(cv2.resize(frame, (240, 426)))
    storyboard = cv2.hconcat(frames) if len(frames) > 1 else frames[0]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), storyboard):
        raise RuntimeError(f"Unable to write storyboard: {output_path}")


def render_media_tag(path: str, media_type: str, css_class: str) -> str:
    escaped = html.escape(path)
    if media_type == "video":
        return f'<video class="{css_class}" controls preload="metadata" src="{escaped}"></video>'
    return f'<img class="{css_class}" src="{escaped}" alt="{escaped}">'


def build_output(video_name: str) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    annotation_root = repo_root / "annotations" / "0421" / "switch"
    source_json, item = resolve_item(annotation_root, video_name)
    local_video = resolve_video_path(annotation_root, video_name)
    fps = get_fps(item)
    summary = parse_summary(item)
    actions = parse_action_events(item)
    entries = build_action_entries(actions) + parse_signal_entries(item, actions)
    entries.sort(key=lambda entry: (entry.start_frame, entry.end_frame, entry.category, entry.label))

    output_root = annotation_root / f"{Path(video_name).stem}_key_annotation_review"
    if output_root.exists():
        shutil.rmtree(output_root)
    images_dir = output_root / "images"
    clips_dir = output_root / "clips"
    storyboards_dir = output_root / "storyboards"
    images_dir.mkdir(parents=True, exist_ok=True)
    clips_dir.mkdir(parents=True, exist_ok=True)
    storyboards_dir.mkdir(parents=True, exist_ok=True)

    asset_writer = AssetWriter()
    rendered_entries: List[ReviewEntry] = []
    for entry in entries:
        base_name = safe_name(f"{Path(video_name).stem}_{entry.entry_id}_{entry.label}")
        image_rel = f"images/{base_name}_anchor.jpg"
        asset_writer.extract_frame(local_video, entry.anchor_frame, output_root / image_rel)
        entry.image_path = image_rel
        if entry.span_type == "segment":
            clip_rel = f"clips/{base_name}.mp4"
            storyboard_rel = f"storyboards/{base_name}.jpg"
            asset_writer.extract_clip(local_video, entry.start_frame, entry.end_frame, fps, output_root / clip_rel)
            make_storyboard(local_video, entry.start_frame, entry.end_frame, output_root / storyboard_rel, asset_writer)
            entry.clip_path = clip_rel
            entry.storyboard_path = storyboard_rel
        rendered_entries.append(entry)

    json_payload = {
        "video_name": video_name,
        "source_file": source_json.name,
        "local_video_path": str(local_video),
        "fps": fps,
        "summary": summary,
        "entries": [asdict(entry) for entry in rendered_entries],
    }
    (output_root / "key_annotations.json").write_text(json.dumps(json_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_columns = [
        "entry_id",
        "category",
        "label",
        "start_frame",
        "end_frame",
        "span_type",
        "anchor_frame",
        "linked_step_id",
        "linked_action_type",
        "text",
        "action_requirement",
        "action_description",
        "action_type",
        "image_path",
        "clip_path",
        "storyboard_path",
    ]
    with (output_root / "key_annotations.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_columns)
        writer.writeheader()
        for entry in rendered_entries:
            writer.writerow({key: getattr(entry, key) for key in csv_columns})

    html_parts = [
        "<!doctype html>",
        "<html lang='en'>",
        "<head>",
        "<meta charset='utf-8'>",
        f"<title>{html.escape(video_name)} Key Annotation Review</title>",
        "<style>",
        "body{font-family:Arial,sans-serif;background:#f6f8fb;color:#1e2430;margin:24px;}",
        ".summary,.card{background:#fff;border:1px solid #d7dce5;border-radius:16px;padding:16px;margin-bottom:18px;box-shadow:0 4px 14px rgba(25,32,46,.05);}",
        ".meta{font-size:14px;line-height:1.5;}",
        ".pill{display:inline-block;padding:4px 10px;border-radius:999px;background:#e8f2ff;color:#245a9b;font-size:12px;font-weight:700;margin-right:8px;}",
        ".pill.action{background:#fff0e1;color:#9a5400;}",
        ".pill.signal{background:#e9f7ef;color:#1f7a4d;}",
        ".grid{display:grid;grid-template-columns:repeat(2,minmax(280px,1fr));gap:14px;align-items:start;}",
        "img.media,video.media{width:100%;border-radius:12px;border:1px solid #dde3eb;background:#000;}",
        ".kv{margin:6px 0;}",
        "code{background:#f1f4f8;padding:1px 5px;border-radius:6px;}",
        "</style>",
        "</head>",
        "<body>",
        f"<h1>{html.escape(video_name)} Key Annotation Review</h1>",
        "<div class='summary'>",
        f"<div class='kv'><b>Source file:</b> {html.escape(source_json.name)}</div>",
        f"<div class='kv'><b>Local video:</b> {html.escape(str(local_video))}</div>",
        f"<div class='kv'><b>FPS:</b> {fps}</div>",
        f"<div class='kv'><b>Data ID:</b> {html.escape(str(summary['data_id']))}</div>",
        f"<div class='kv'><b>Frame End:</b> {html.escape(str(summary['frame_end']))}</div>",
        f"<div class='kv'><b>Overall Requirement:</b> {html.escape(summary['overall_requirement'])}</div>",
        f"<div class='kv'><b>Overall Verification:</b> {html.escape(summary['overall_verification'])}</div>",
        "</div>",
    ]
    for entry in rendered_entries:
        pill_class = "action" if entry.category == "action" else "signal"
        html_parts.append("<article class='card'>")
        html_parts.append(
            f"<div class='meta'><span class='pill {pill_class}'>{html.escape(entry.category)}</span>"
            f"<span class='pill'>{html.escape(entry.label)}</span>"
            f"<b>frames {entry.start_frame}-{entry.end_frame}</b> | anchor <code>{entry.anchor_frame}</code></div>"
        )
        if entry.linked_step_id:
            html_parts.append(
                f"<div class='kv'><b>Linked step:</b> {html.escape(entry.linked_step_id)}"
                f" | <b>Action type:</b> {html.escape(entry.linked_action_type)}</div>"
            )
        html_parts.append(f"<div class='kv'><b>Text:</b> {html.escape(entry.text)}</div>")
        if entry.category == "action":
            html_parts.append(f"<div class='kv'><b>Action Requirement:</b> {html.escape(entry.action_requirement)}</div>")
            html_parts.append(f"<div class='kv'><b>Action Description:</b> {html.escape(entry.action_description)}</div>")
        html_parts.append("<div class='grid'>")
        html_parts.append("<div>")
        html_parts.append("<div class='meta'><b>Anchor frame</b></div>")
        html_parts.append(render_media_tag(entry.image_path or "", "image", "media"))
        html_parts.append("</div>")
        if entry.span_type == "segment":
            html_parts.append("<div>")
            html_parts.append("<div class='meta'><b>Segment clip</b></div>")
            html_parts.append(render_media_tag(entry.clip_path or "", "video", "media"))
            html_parts.append("</div>")
            html_parts.append("<div>")
            html_parts.append("<div class='meta'><b>Storyboard</b></div>")
            html_parts.append(render_media_tag(entry.storyboard_path or "", "image", "media"))
            html_parts.append("</div>")
        html_parts.append("</div>")
        html_parts.append("</article>")
    html_parts.extend(["</body>", "</html>"])
    (output_root / "index.html").write_text("\n".join(html_parts), encoding="utf-8")

    readme_lines = [
        f"# {video_name} Key Annotation Review",
        "",
        "- Purpose: inspect one source video with its key action / signal annotations",
        "- Main browser: `index.html`",
        "- Structured export: `key_annotations.json` / `key_annotations.csv`",
        "",
        "## Included categories",
        "",
        "- `action_step`: grouped from `action-type`, `action_requirement`, `action_description`, `action_step_id`",
        "- `ui_change`, `view_change`, `ui_state`, `physical_world_change`, `physical_world_state`, `is_final_state`",
        "",
        "## Notes",
        "",
        "- Single-frame labels are exported as anchor images.",
        "- Segment labels are exported as anchor image + video clip + storyboard.",
        "- Each signal row is linked to the nearest action step for quick inspection.",
    ]
    (output_root / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")
    return output_root


def main() -> None:
    video_name = sys.argv[1] if len(sys.argv) > 1 else "011.mp4"
    output_root = build_output(video_name)
    print(f"Wrote key annotation review to: {output_root}")


if __name__ == "__main__":
    main()
