#!/usr/bin/env python3
from __future__ import annotations

import html
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2


@dataclass
class MediaRef:
    media_type: str  # image | video
    video_name: str
    start_frame: int
    end_frame: int
    label: str
    text: str
    correct: bool
    note: Optional[str] = None


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

    def extract_clip(self, video_path: Path, start_frame: int, end_frame: int, fps: float, output_path: Path) -> None:
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


def safe_name(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return text.strip("._") or "item"


def load_raw_items(annotation_root: Path) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    for json_path in [annotation_root / "SWITCH_帧Action_1.json", annotation_root / "SWITCH_Action_2.json"]:
        items = json.loads(json_path.read_text(encoding="utf-8"))
        for item in items:
            video_url = next(
                (
                    value
                    for key, value in (item.get("data") or {}).items()
                    if key != "meta" and isinstance(value, str) and value.endswith(".mp4")
                ),
                "",
            )
            if not video_url:
                continue
            index[Path(video_url).name] = {
                "source_file": json_path.name,
                "item": item,
            }
    return index


def load_raw_items_resolved(annotation_root: Path) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    json_paths = sorted(annotation_root.glob("SWITCH*Action_1.json"))
    json_paths.append(annotation_root / "SWITCH_Action_2.json")
    for json_path in json_paths:
        if not json_path.exists():
            continue
        items = json.loads(json_path.read_text(encoding="utf-8"))
        for item in items:
            video_url = next(
                (
                    value
                    for key, value in (item.get("data") or {}).items()
                    if key != "meta" and isinstance(value, str) and value.endswith(".mp4")
                ),
                "",
            )
            if not video_url:
                continue
            index[Path(video_url).name] = {
                "source_file": json_path.name,
                "item": item,
            }
    return index


def get_video_meta(item: Dict[str, Any]) -> Dict[str, Any]:
    meta = item.get("data", {}).get("meta") or {}
    for value in meta.values():
        if isinstance(value, dict) and "fps" in value:
            return value
    return {"fps": 30.0}


def build_video_path_index(annotation_root: Path) -> Dict[str, Path]:
    video_index: Dict[str, Path] = {}
    for subdir in sorted(annotation_root.glob("SWITCHAction_*")):
        if not subdir.is_dir():
            continue
        for video_path in subdir.rglob("*.mp4"):
            video_index[video_path.name] = video_path
    return video_index


def render_media_tag(relpath: str, media_type: str) -> str:
    escaped = html.escape(relpath)
    if media_type == "video":
        return f'<video controls preload="metadata" src="{escaped}"></video>'
    return f'<img src="{escaped}" alt="{escaped}">'


def build_samples() -> List[Dict[str, Any]]:
    return [
        {
            "sample_id": "pilot_final_state_is_final_frame_003",
            "title": "Pilot 1: Whole-Video Final State from is_final_state",
            "question": 'Watch the action clip immediately before the final completion state. Which frame should be used as the final-state ground-truth frame for the task "Take the elevator from the ninth floor to the tenth floor"?',
            "task_type": "final_state_gt_from_is_final_state",
            "answer_mode": "single_choice",
            "query": MediaRef(
                media_type="video",
                video_name="003.mp4",
                start_frame=520,
                end_frame=582,
                label="Query video",
                text='Pre-final action clip around step 3 ("Click the button for the tenth floor"), without the later verification scene.',
                correct=False,
            ),
            "options": [
                MediaRef("image", "003.mp4", 229, 229, "Option A", "physical_world_state @229: The elevator door is fully opened", False),
                MediaRef("image", "003.mp4", 640, 640, "Option B", "physical_world_state @640: The elevator door has closed", False),
                MediaRef("image", "003.mp4", 1052, 1052, "Option C", "physical_world_state @1052: The elevator arrives at the tenth floor", False, "Current pipeline tends to use this state-derived frame."),
                MediaRef("image", "003.mp4", 1430, 1430, "Option D", "is_final_state @1430", True, "Proposed final-state GT frame."),
            ],
            "correct_answers": ["D"],
            "rationale": "This prototype uses the explicitly annotated is_final_state frame as the final-state GT and uses the immediately preceding action clip as input, avoiding a query clip that is already too close to the terminal verification scene.",
            "source_notes": [
                "overall_requirement: Take the elevator from the ninth floor to the tenth floor",
                "overall_verification: Successfully reached the tenth floor",
                "pre-final action clip: action step 3 with immediate response (520-582)",
                "excluded later verification action: Check the surrounding environment and floor information (1269-1420)",
                "is_final_state: frame 1430",
            ],
        },
        {
            "sample_id": "pilot_verify_ui_frame_011",
            "title": "Pilot 2: Verification by Post-Change UI State Frame Selection",
            "question": 'During the verification step "View the homepage information of the machine screen", which frame shows the post-change interface state that confirms success?',
            "task_type": "verification_ui_state_frame_selection",
            "answer_mode": "single_choice",
            "query": MediaRef(
                media_type="video",
                video_name="011.mp4",
                start_frame=409,
                end_frame=538,
                label="Query video",
                text="verification action @409-538",
                correct=False,
            ),
            "options": [
                MediaRef("image", "011.mp4", 133, 133, "Option A", "ui_state @133: The machine page has been successfully changed", False),
                MediaRef("image", "011.mp4", 198, 198, "Option B", "ui_state @198: Enter the project query page", False),
                MediaRef("image", "011.mp4", 403, 403, "Option C", "ui_state @403: Enter the final diagnosis and treatment item query page", False),
                MediaRef("image", "011.mp4", 539, 539, "Option D", "ui_state @539: Return to the home screen interface", True),
            ],
            "correct_answers": ["D"],
            "rationale": "This sample treats the post-change ui_state frame as the single-image verification target. The query video covers the verification action, while the correct option is the resulting interface state after the transition completes.",
            "source_notes": [
                "overall_requirement: Utilize machines to view medical projects",
                "overall_verification: Successfully viewed other medical items",
                "verification action: View the homepage information of the machine screen (409-538)",
                "preceding ui_change: frame 456",
                "verification target ui_state: frame 539",
            ],
        },
        {
            "sample_id": "pilot_verify_physical_video_008",
            "title": "Pilot 3: Verification by Physical-World Change Video Selection",
            "question": 'During the verification step "Look down to check the ticket counter", which video segment shows the physical-world change that confirms success?',
            "task_type": "verification_physical_video_selection",
            "answer_mode": "single_choice",
            "query": MediaRef(
                media_type="video",
                video_name="008.mp4",
                start_frame=1217,
                end_frame=1261,
                label="Query video",
                text="verification action @1217-1261",
                correct=False,
            ),
            "options": [
                MediaRef("video", "008.mp4", 1261, 1331, "Option A", "physical_world_change @1261-1331: Ticket issued successfully", True),
                MediaRef("video", "005.mp4", 1098, 1207, "Option B", "physical_world_change @1098-1207: Print out a paper reservation form at the ticket counter", False),
                MediaRef("video", "003.mp4", 185, 228, "Option C", "physical_world_change @185-228: The elevator door opens", False),
                MediaRef("video", "006.mp4", 205, 294, "Option D", "physical_world_change @205-294: Hold the ID card in the sensing area", False),
            ],
            "correct_answers": ["A"],
            "rationale": "This sample treats segment-level physical_world_change as a video-selection target for verification evidence.",
            "source_notes": [
                "overall_requirement: Buy a two-yuan subway ticket from Kejiao South Station to Nanxiasu.",
                "overall_verification: Whether successfully got the subway ticket",
                "verification action: Look down to check the ticket counter (1217-1261)",
                "linked physical_world_change: 1261-1331",
            ],
        },
        {
            "sample_id": "pilot_multiselect_ui_plus_physical_003",
            "title": "Pilot 4: Multi-Select When UI Change and Physical Change Both Matter",
            "question": 'After the operator clicks the button for the tenth floor, which observable changes together indicate that the command has taken effect? Select all applicable options.',
            "task_type": "step_verification_multiselect_ui_and_physical",
            "answer_mode": "multi_select",
            "query": MediaRef(
                media_type="image",
                video_name="003.mp4",
                start_frame=537,
                end_frame=537,
                label="Query image",
                text="execute action @537: Click the button for the tenth floor",
                correct=False,
            ),
            "options": [
                MediaRef("image", "003.mp4", 556, 556, "Option A", "ui_change @556: The button for the tenth floor lights up", True),
                MediaRef("image", "003.mp4", 640, 640, "Option B", "physical_world_state @640: The elevator door has closed", True),
                MediaRef("image", "003.mp4", 147, 147, "Option C", "ui_change @147: The elevator up button is lit", False),
                MediaRef("image", "003.mp4", 206, 206, "Option D", "physical_world_change @185-228: The elevator door opens", False),
                MediaRef("image", "003.mp4", 993, 993, "Option E", "physical_world_change @993: The elevator door opens", False),
            ],
            "correct_answers": ["A", "B"],
            "rationale": "This pilot illustrates the proposed five-option multi-select format. The UI branch uses a visible post-change cue, and the physical branch uses a visible post-change state frame instead of the harder-to-read change point. Note: the current dataset does not provide an explicit verification-action segment with both ui_change and physical_world_change nearby, so this prototype uses an execute action step to demonstrate the format.",
            "source_notes": [
                "execute action: Click the button for the tenth floor (537)",
                "linked ui_change: 556",
                "preceding physical_world_change: 582",
                "verification target physical_world_state: 640",
                "This is a format prototype for 'both cues' rather than a literal verification-action example from the raw labels.",
            ],
        },
    ]


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    annotation_root = repo_root / "annotations" / "0421" / "switch"
    output_root = annotation_root / "change_driven_pilot_v1"
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    images_dir = output_root / "images"
    videos_dir = output_root / "videos"
    images_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)

    raw_index = load_raw_items_resolved(annotation_root)
    video_index = build_video_path_index(annotation_root)
    asset_writer = AssetWriter()
    samples = build_samples()

    rendered_samples: List[Dict[str, Any]] = []
    for sample in samples:
        query = sample["query"]
        query_video_path = video_index[query.video_name]
        raw_item = raw_index[query.video_name]["item"]
        video_meta = get_video_meta(raw_item)
        fps = float(video_meta.get("fps") or 30.0)
        query_filename = f"{safe_name(sample['sample_id'])}_query"
        if query.media_type == "image":
            query_rel = f"images/{query_filename}.jpg"
            asset_writer.extract_frame(query_video_path, query.start_frame, output_root / query_rel)
        else:
            query_rel = f"videos/{query_filename}.mp4"
            asset_writer.extract_clip(query_video_path, query.start_frame, query.end_frame, fps, output_root / query_rel)

        option_rows: List[Dict[str, Any]] = []
        for option in sample["options"]:
            option_video_path = video_index[option.video_name]
            option_prefix = f"{safe_name(sample['sample_id'])}_{safe_name(option.label.lower())}"
            if option.media_type == "image":
                rel = f"images/{option_prefix}.jpg"
                asset_writer.extract_frame(option_video_path, option.start_frame, output_root / rel)
            else:
                rel = f"videos/{option_prefix}.mp4"
                option_meta = get_video_meta(raw_index[option.video_name]["item"])
                option_fps = float(option_meta.get("fps") or 30.0)
                asset_writer.extract_clip(option_video_path, option.start_frame, option.end_frame, option_fps, output_root / rel)
            option_rows.append(
                {
                    "label": option.label,
                    "media_type": option.media_type,
                    "path": rel,
                    "video_name": option.video_name,
                    "start_frame": option.start_frame,
                    "end_frame": option.end_frame,
                    "text": option.text,
                    "correct": option.correct,
                    "note": option.note,
                }
            )

        rendered_samples.append(
            {
                "sample_id": sample["sample_id"],
                "title": sample["title"],
                "question": sample["question"],
                "task_type": sample["task_type"],
                "answer_mode": sample["answer_mode"],
                "query": {
                    "media_type": query.media_type,
                    "path": query_rel,
                    "video_name": query.video_name,
                    "start_frame": query.start_frame,
                    "end_frame": query.end_frame,
                    "text": query.text,
                },
                "options": option_rows,
                "correct_answers": sample["correct_answers"],
                "rationale": sample["rationale"],
                "source_notes": sample["source_notes"],
                "source_file": raw_index[query.video_name]["source_file"],
            }
        )

    json_path = output_root / "pilot_samples.json"
    json_path.write_text(json.dumps({"samples": rendered_samples}, ensure_ascii=False, indent=2), encoding="utf-8")

    html_parts: List[str] = [
        "<!doctype html>",
        "<html lang='en'>",
        "<head>",
        "<meta charset='utf-8'>",
        "<title>SWITCH Change-Driven Pilot</title>",
        "<style>",
        "body{font-family:Arial,sans-serif;background:#f7f8fa;color:#1f2430;margin:24px;}",
        ".card{background:#fff;border:1px solid #d7dce5;border-radius:16px;padding:16px;margin-bottom:22px;box-shadow:0 4px 14px rgba(25,32,46,.06);}",
        ".query,.options{display:grid;gap:12px;}",
        ".options.grid4{grid-template-columns:repeat(2,minmax(280px,1fr));}",
        ".options.grid5{grid-template-columns:repeat(2,minmax(280px,1fr));}",
        ".media{width:100%;border-radius:10px;border:1px solid #dfe4ec;background:#000;}",
        "img.media{background:#fafbfd;}",
        ".option{border:2px solid #d9e0e8;border-radius:12px;padding:10px;background:#fbfcfe;}",
        ".option.correct{border-color:#2d9158;box-shadow:inset 0 0 0 1px #2d9158;}",
        ".pill{display:inline-block;padding:5px 10px;border-radius:999px;background:#e8f2ff;color:#245a9b;font-size:12px;font-weight:700;margin-right:8px;}",
        ".pill.good{background:#e7f6ee;color:#2d9158;}",
        ".meta{font-size:14px;line-height:1.45;}",
        "video{max-width:100%;}",
        "details{margin-top:10px;}",
        "summary{cursor:pointer;font-weight:700;}",
        "</style>",
        "</head>",
        "<body>",
        "<h1>SWITCH Change-Driven Pilot v1</h1>",
        "<p>This pilot intentionally does not batch-generate the full dataset. It demonstrates four small, source-traceable prototypes based on <code>is_final_state</code>, <code>ui_change</code>, and <code>physical_world_change</code>.</p>",
    ]
    for sample in rendered_samples:
        grid_class = "grid5" if len(sample["options"]) == 5 else "grid4"
        html_parts.append("<article class='card'>")
        html_parts.append(f"<h2>{html.escape(sample['title'])}</h2>")
        html_parts.append(
            f"<div class='meta'><span class='pill'>{html.escape(sample['task_type'])}</span>"
            f"<span class='pill {'good' if sample['answer_mode']=='multi_select' else ''}'>{html.escape(sample['answer_mode'])}</span>"
            f" Source: {html.escape(sample['source_file'])}</div>"
        )
        html_parts.append(f"<p><b>Question:</b> {html.escape(sample['question'])}</p>")
        query = sample["query"]
        html_parts.append("<div class='query'>")
        html_parts.append(f"<div class='meta'><b>Query</b>: {html.escape(query['video_name'])} | frames {query['start_frame']}-{query['end_frame']} | {html.escape(query['text'])}</div>")
        html_parts.append(render_media_tag(query["path"], query["media_type"]).replace(">", " class='media'>", 1))
        html_parts.append("</div>")
        html_parts.append(f"<div class='options {grid_class}'>")
        for option in sample["options"]:
            classes = "option correct" if option["correct"] else "option"
            html_parts.append(f"<div class='{classes}'>")
            html_parts.append(f"<div class='meta'><b>{html.escape(option['label'])}</b> | {html.escape(option['video_name'])} | frames {option['start_frame']}-{option['end_frame']}</div>")
            html_parts.append(render_media_tag(option["path"], option["media_type"]).replace(">", " class='media'>", 1))
            html_parts.append(f"<div class='meta'>{html.escape(option['text'])}</div>")
            if option.get("note"):
                html_parts.append(f"<div class='meta'><i>{html.escape(option['note'])}</i></div>")
            html_parts.append("</div>")
        html_parts.append("</div>")
        html_parts.append(f"<p><b>Correct answer(s):</b> {', '.join(sample['correct_answers'])}</p>")
        html_parts.append(f"<p><b>Rationale:</b> {html.escape(sample['rationale'])}</p>")
        notes = "".join(f"<li>{html.escape(note)}</li>" for note in sample["source_notes"])
        html_parts.append(f"<details><summary>Source notes</summary><ul>{notes}</ul></details>")
        html_parts.append("</article>")
    html_parts.extend(["</body>", "</html>"])
    (output_root / "index.html").write_text("\n".join(html_parts), encoding="utf-8")

    readme = [
        "# SWITCH Change-Driven Pilot v1",
        "",
        "- Purpose: small prototype before large-scale generation",
        "- Output browser: `index.html`",
        "- Structured metadata: `pilot_samples.json`",
        "",
        "## Included Samples",
        "",
        "- `pilot_final_state_is_final_frame_003`: uses `is_final_state` as whole-video final-state GT",
        "- `pilot_verify_ui_frame_011`: uses single-frame post-change `ui_state` as verification evidence",
        "- `pilot_verify_physical_video_008`: uses segment-level `physical_world_change` as verification evidence",
        "- `pilot_multiselect_ui_plus_physical_003`: uses both `ui_change` and `physical_world_change` in a five-option multi-select prototype",
        "",
        "## Notes",
        "",
        "- The multi-select sample is intentionally a format prototype. This dataset slice does not contain a clean verification-action example with both `ui_change` and `physical_world_change` aligned nearby, so the demo uses an execute-action step to show the shape of the task.",
    ]
    (output_root / "README.md").write_text("\n".join(readme), encoding="utf-8")

    print(f"Wrote pilot package to: {output_root}")
    print(f"  - samples: {len(rendered_samples)}")


if __name__ == "__main__":
    main()
