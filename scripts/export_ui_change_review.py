#!/usr/bin/env python3
from __future__ import annotations

import csv
import html
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np


ACTION_LABELS = {
    "action-type",
    "action_requirement",
    "action_description",
    "action_step_id",
}

CONTEXT_LABELS = {
    "action_requirement",
    "action_description",
    "ui_change",
    "ui_state",
    "physical_world_change",
    "physical_world_state",
    "view_change",
    "overall_requirement",
    "overall_verification",
    "is_final_state",
}

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


def get_first_range(value: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    ranges = value.get("ranges") or []
    if not ranges:
        return None, None
    first = ranges[0]
    start = first.get("start")
    end = first.get("end")
    return int(start) if start is not None else None, int(end) if end is not None else None


def normalize_action_type(text: str) -> str:
    normalized = normalize_spaces(text).lower()
    return ACTION_TYPE_MAP.get(normalized, normalized)


def infer_video_name(item: Dict[str, Any]) -> str:
    data = item.get("data") or {}
    for key, value in data.items():
        if key == "meta":
            continue
        if isinstance(value, str) and value.lower().endswith(".mp4"):
            return Path(value).name
    return f"{item.get('id')}.mp4"


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

    def sort_key(event: ActionEvent) -> Tuple[int, int, int]:
        try:
            step_num = int(event.step_id)
        except (TypeError, ValueError):
            step_num = 10**9
        return (event.start or 0, event.end or 0, step_num)

    events.sort(key=sort_key)
    return events


def range_overlap(a_start: Optional[int], a_end: Optional[int], b_start: Optional[int], b_end: Optional[int]) -> bool:
    if a_start is None or a_end is None or b_start is None or b_end is None:
        return False
    return not (a_end < b_start or b_end < a_start)


def build_video_index(annotation_root: Path) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for subdir in sorted(annotation_root.glob("SWITCHAction_*")):
        if not subdir.is_dir():
            continue
        for video_path in subdir.rglob("*.mp4"):
            index.setdefault(video_path.name, video_path)
    return index


def resize_to_height(frame: Any, target_height: int) -> Any:
    height, width = frame.shape[:2]
    if height == target_height:
        return frame
    scale = target_height / float(height)
    target_width = max(1, int(round(width * scale)))
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)


def make_storyboard(
    asset_writer: AssetWriter,
    video_path: Path,
    frame_indices: List[int],
    labels: List[str],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        return
    panels: List[Any] = []
    target_height = 320
    header_height = 42
    for frame_index, label in zip(frame_indices, labels):
        frame = asset_writer.read_frame(video_path, frame_index)
        frame = resize_to_height(frame, target_height)
        header = np.full((header_height, frame.shape[1], 3), 248, dtype=np.uint8)
        cv2.putText(
            header,
            label,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (36, 42, 52),
            2,
            cv2.LINE_AA,
        )
        panels.append(np.vstack([header, frame]))
    storyboard = cv2.hconcat(panels)
    if not cv2.imwrite(str(output_path), storyboard):
        raise RuntimeError(f"Unable to write storyboard: {output_path}")


def safe_name(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = text.strip("._")
    return text or "item"


def render_context_lines(entry: Dict[str, Any]) -> List[str]:
    lines: List[str] = []
    if entry.get("overall_requirement"):
        lines.append(f"Task: {entry['overall_requirement']}")
    if entry.get("overall_verification"):
        lines.append(f"Overall verification: {entry['overall_verification']}")
    if entry.get("previous_action"):
        lines.append(f"Previous action: {entry['previous_action']}")
    if entry.get("overlapping_actions"):
        lines.append("Overlapping action(s): " + " | ".join(entry["overlapping_actions"]))
    if entry.get("next_action"):
        lines.append(f"Next action: {entry['next_action']}")
    if entry.get("next_state"):
        lines.append(f"Next state cue: {entry['next_state']}")
    if entry.get("distance_to_final_state_frame") is not None:
        lines.append(f"Distance to is_final_state frame: {entry['distance_to_final_state_frame']} frames")
    return lines


def write_html(entries: List[Dict[str, Any]], output_path: Path) -> None:
    single_count = sum(1 for entry in entries if entry["span_type"] == "single_frame")
    segment_count = len(entries) - single_count
    per_video = Counter(entry["video_name"] for entry in entries)

    def render_card(entry: Dict[str, Any]) -> str:
        media_html: List[str] = []
        anchor = entry.get("anchor_image_path")
        if anchor:
            media_html.append(
                f'<a href="{html.escape(anchor)}"><img class="anchor" src="{html.escape(anchor)}" alt="{html.escape(entry["uid"])}"></a>'
            )
        if entry["span_type"] == "segment" and entry.get("storyboard_path"):
            media_html.append(
                f'<a href="{html.escape(entry["storyboard_path"])}"><img class="storyboard" src="{html.escape(entry["storyboard_path"])}" alt="storyboard"></a>'
            )
        if entry["span_type"] == "segment" and entry.get("clip_path"):
            media_html.append(
                f'<video controls preload="metadata" src="{html.escape(entry["clip_path"])}"></video>'
            )

        meta_lines = [
            f"<b>{html.escape(entry['uid'])}</b>",
            f"Video: {html.escape(entry['video_name'])} | Scenario: {html.escape(entry['scenario_family'])}",
            f"Frames: {entry['start_frame']} - {entry['end_frame']} | Length: {entry['length_frames']} | Type: {html.escape(entry['span_type'])}",
            f"Label text: {html.escape(entry['text'])}",
        ]
        for line in render_context_lines(entry):
            meta_lines.append(html.escape(line))

        extra_lines: List[str] = []
        for ctx in entry.get("overlapping_annotations", [])[:8]:
            frame_text = ""
            if ctx["start"] is not None and ctx["end"] is not None:
                frame_text = f" ({ctx['start']}-{ctx['end']})"
            extra_lines.append(f"{ctx['label']}{frame_text}: {ctx['text']}")

        extra_html = ""
        if extra_lines:
            lis = "".join(f"<li>{html.escape(line)}</li>" for line in extra_lines)
            extra_html = f"<details><summary>Overlapping / nearby annotations</summary><ul>{lis}</ul></details>"

        return (
            '<article class="card">'
            f'<div class="media">{"".join(media_html)}</div>'
            f'<div class="meta">{"<br>".join(meta_lines)}{extra_html}</div>'
            "</article>"
        )

    cards_single = "\n".join(render_card(entry) for entry in entries if entry["span_type"] == "single_frame")
    cards_segment = "\n".join(render_card(entry) for entry in entries if entry["span_type"] == "segment")
    video_summary = "".join(
        f"<li>{html.escape(video)}: {count}</li>" for video, count in sorted(per_video.items())
    )

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>SWITCH UI Change Review</title>
  <style>
    body {{
      font-family: Arial, sans-serif;
      margin: 24px;
      background: #f7f8fa;
      color: #202632;
    }}
    h1, h2 {{
      margin-bottom: 8px;
    }}
    .summary {{
      background: white;
      border: 1px solid #d6dbe4;
      border-radius: 14px;
      padding: 16px 18px;
      margin-bottom: 20px;
    }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(480px, 1fr));
      gap: 18px;
      margin-bottom: 28px;
    }}
    .card {{
      background: white;
      border: 1px solid #d6dbe4;
      border-radius: 16px;
      padding: 14px;
      box-shadow: 0 4px 14px rgba(25, 32, 46, 0.06);
    }}
    .media {{
      display: grid;
      gap: 10px;
      margin-bottom: 12px;
    }}
    img {{
      max-width: 100%;
      border-radius: 10px;
      border: 1px solid #dfe4ec;
      background: #fafbfd;
    }}
    video {{
      width: 100%;
      border-radius: 10px;
      border: 1px solid #dfe4ec;
      background: #000;
    }}
    .meta {{
      line-height: 1.45;
      font-size: 14px;
    }}
    details {{
      margin-top: 10px;
    }}
    summary {{
      cursor: pointer;
      color: #1d5f87;
      font-weight: 600;
    }}
    ul {{
      margin-top: 8px;
      padding-left: 20px;
    }}
    .notes {{
      margin-top: 8px;
      font-size: 14px;
    }}
  </style>
</head>
<body>
  <h1>SWITCH UI Change Review</h1>
  <div class="summary">
    <div>Total entries: <b>{len(entries)}</b></div>
    <div>Single-frame entries: <b>{single_count}</b></div>
    <div>Segment entries: <b>{segment_count}</b></div>
    <div class="notes">Quick use hints: single-frame `ui_change` is often good for image-based state change recognition; segment `ui_change` is better for transition localization, temporal QA, and process supervision.</div>
    <details>
      <summary>Per-video counts</summary>
      <ul>{video_summary}</ul>
    </details>
  </div>
  <h2>Single-Frame UI Changes</h2>
  <section class="cards">{cards_single}</section>
  <h2>Segment UI Changes</h2>
  <section class="cards">{cards_segment}</section>
</body>
</html>
"""
    output_path.write_text(html_text, encoding="utf-8")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    annotation_root = repo_root / "annotations" / "0421" / "switch"
    output_root = annotation_root / "ui_change_review"
    images_dir = output_root / "images"
    clips_dir = output_root / "clips"
    storyboards_dir = output_root / "storyboards"
    output_root.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    clips_dir.mkdir(parents=True, exist_ok=True)
    storyboards_dir.mkdir(parents=True, exist_ok=True)

    source_jsons = [
        annotation_root / "SWITCH_帧Action_1.json",
        annotation_root / "SWITCH_Action_2.json",
    ]
    video_index = build_video_index(annotation_root)
    asset_writer = AssetWriter()
    entries: List[Dict[str, Any]] = []

    for source_path in source_jsons:
        items = json.loads(source_path.read_text(encoding="utf-8"))
        for item in items:
            video_name = infer_video_name(item)
            local_video = video_index.get(video_name)
            if local_video is None:
                continue
            media_meta = next(
                (
                    value
                    for value in (item.get("data", {}).get("meta") or {}).values()
                    if isinstance(value, dict) and "fps" in value
                ),
                {},
            )
            fps = float(media_meta.get("fps") or 30.0)
            total_frames = int(media_meta.get("total_frames") or 0)
            data_id = Path(video_name).stem
            segments = parse_segments(item)
            by_label: Dict[str, List[Segment]] = defaultdict(list)
            for segment in segments:
                by_label[segment.label].append(segment)
            overall_requirement = (by_label.get("overall_requirement") or [Segment("", "", None, None)])[0].text
            overall_verification = (by_label.get("overall_verification") or [Segment("", "", None, None)])[0].text
            scenario_family = infer_scenario_family(overall_requirement)
            action_events = build_action_events(segments)
            final_state_frame = None
            if by_label.get("is_final_state"):
                final_state_frame = by_label["is_final_state"][0].start

            ui_changes = by_label.get("ui_change") or []
            for index, segment in enumerate(ui_changes, start=1):
                if segment.start is None or segment.end is None:
                    continue
                start_frame = int(segment.start)
                end_frame = int(segment.end)
                length_frames = end_frame - start_frame
                span_type = "single_frame" if start_frame == end_frame else "segment"
                anchor_frame = start_frame if span_type == "single_frame" else int((start_frame + end_frame) // 2)
                uid = f"{data_id}_ui_change_{index:03d}"
                base_name = safe_name(uid)

                anchor_rel = f"images/{base_name}_anchor.jpg"
                asset_writer.extract_frame(local_video, anchor_frame, output_root / anchor_rel)

                clip_rel = None
                storyboard_rel = None
                if span_type == "segment":
                    clip_rel = f"clips/{base_name}.mp4"
                    storyboard_rel = f"storyboards/{base_name}.jpg"
                    asset_writer.extract_clip(local_video, start_frame, end_frame, fps, output_root / clip_rel)
                    mid_frame = int((start_frame + end_frame) // 2)
                    make_storyboard(
                        asset_writer,
                        local_video,
                        [start_frame, mid_frame, end_frame],
                        [f"start {start_frame}", f"mid {mid_frame}", f"end {end_frame}"],
                        output_root / storyboard_rel,
                    )

                overlapping_actions = [
                    event.action_description
                    for event in action_events
                    if range_overlap(start_frame, end_frame, event.start, event.end)
                ]
                previous_action = next(
                    (
                        event.action_description
                        for event in reversed(action_events)
                        if event.end is not None and event.end <= start_frame
                    ),
                    None,
                )
                next_action = next(
                    (
                        event.action_description
                        for event in action_events
                        if event.start is not None and event.start >= end_frame and event.action_description not in overlapping_actions
                    ),
                    None,
                )
                next_state = next(
                    (
                        f"{ctx.label}: {ctx.text}"
                        for ctx in segments
                        if ctx.label in {"ui_state", "physical_world_state"} and ctx.start is not None and ctx.start >= end_frame
                    ),
                    None,
                )

                overlapping_annotations: List[Dict[str, Any]] = []
                for ctx in segments:
                    if ctx.label not in CONTEXT_LABELS or ctx.label == "ui_change":
                        continue
                    if ctx.start is not None and ctx.end is not None and range_overlap(start_frame, end_frame, ctx.start, ctx.end):
                        overlapping_annotations.append(
                            {
                                "label": ctx.label,
                                "text": ctx.text,
                                "start": ctx.start,
                                "end": ctx.end,
                            }
                        )
                if not overlapping_annotations:
                    nearby = [
                        ctx
                        for ctx in segments
                        if ctx.label in CONTEXT_LABELS
                        and ctx.label != "ui_change"
                        and ctx.start is not None
                        and abs(ctx.start - anchor_frame) <= 90
                    ]
                    for ctx in nearby[:6]:
                        overlapping_annotations.append(
                            {
                                "label": ctx.label,
                                "text": ctx.text,
                                "start": ctx.start,
                                "end": ctx.end,
                            }
                        )

                entry = {
                    "uid": uid,
                    "source_file": source_path.name,
                    "data_id": data_id,
                    "video_name": video_name,
                    "video_local_path": local_video.relative_to(annotation_root).as_posix(),
                    "scenario_family": scenario_family,
                    "text": segment.text,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "length_frames": length_frames,
                    "span_type": span_type,
                    "anchor_frame": anchor_frame,
                    "fps": fps,
                    "total_frames": total_frames,
                    "overall_requirement": overall_requirement,
                    "overall_verification": overall_verification,
                    "is_final_state_frame": final_state_frame,
                    "distance_to_final_state_frame": (
                        None if final_state_frame is None else int(final_state_frame - end_frame)
                    ),
                    "previous_action": previous_action,
                    "overlapping_actions": overlapping_actions,
                    "next_action": next_action,
                    "next_state": next_state,
                    "overlapping_annotations": overlapping_annotations,
                    "anchor_image_path": anchor_rel,
                    "clip_path": clip_rel,
                    "storyboard_path": storyboard_rel,
                }
                entries.append(entry)

    entries.sort(key=lambda entry: (entry["video_name"], entry["start_frame"], entry["uid"]))

    json_path = output_root / "ui_change_segments.json"
    csv_path = output_root / "ui_change_segments.csv"
    html_path = output_root / "index.html"
    readme_path = output_root / "README.md"

    json_path.write_text(json.dumps({"data": entries}, ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "uid",
                "source_file",
                "data_id",
                "video_name",
                "video_local_path",
                "scenario_family",
                "text",
                "start_frame",
                "end_frame",
                "length_frames",
                "span_type",
                "anchor_frame",
                "fps",
                "total_frames",
                "overall_requirement",
                "overall_verification",
                "is_final_state_frame",
                "distance_to_final_state_frame",
                "previous_action",
                "overlapping_actions",
                "next_action",
                "next_state",
                "anchor_image_path",
                "clip_path",
                "storyboard_path",
            ],
        )
        writer.writeheader()
        for entry in entries:
            row = dict(entry)
            row["overlapping_actions"] = " | ".join(entry["overlapping_actions"])
            writer.writerow({key: row.get(key) for key in writer.fieldnames})

    write_html(entries, html_path)

    single_count = sum(1 for entry in entries if entry["span_type"] == "single_frame")
    segment_count = len(entries) - single_count
    notes = [
        "# UI Change Review",
        "",
        f"- Total ui_change entries: `{len(entries)}`",
        f"- Single-frame entries: `{single_count}`",
        f"- Segment entries: `{segment_count}`",
        f"- HTML gallery: `{html_path.name}`",
        f"- Metadata JSON: `{json_path.name}`",
        f"- Metadata CSV: `{csv_path.name}`",
        "",
        "## Quick Use Hints",
        "",
        "- Single-frame ui_change is usually better for image-based state change recognition, state verification, and lightweight MCQ distractor mining.",
        "- Segment ui_change is usually better for transition localization, video-to-text temporal QA, and process supervision.",
        "- Long ui_change segments often represent continuous page transitions and can be treated as temporal process cues rather than atomic final states.",
        "- `distance_to_final_state_frame` helps judge whether a ui_change is an early interaction cue or a late-stage confirmation cue.",
    ]
    readme_path.write_text("\n".join(notes), encoding="utf-8")

    print(f"Wrote UI change review package to: {output_root}")
    print(f"  - entries: {len(entries)}")
    print(f"  - single-frame: {single_count}")
    print(f"  - segment: {segment_count}")


if __name__ == "__main__":
    main()
