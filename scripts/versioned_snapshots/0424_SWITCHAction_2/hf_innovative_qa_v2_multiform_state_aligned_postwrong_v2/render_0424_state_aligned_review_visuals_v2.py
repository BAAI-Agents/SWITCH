#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


DEFAULT_DATASET_ROOT = Path("annotations") / "0424" / "SWITCHAction_2" / "hf_innovative_qa_v2_multiform_state_aligned_v1"

IMAGE_SPECS = [
    {"family": "verification_state", "form": "img2img", "file": "vqa.json", "origin_qa_id": "011_state_007", "panel_title": "Verification State  img2img"},
    {"family": "verification_state", "form": "img2img", "file": "vqa.json", "origin_qa_id": "012_state_016", "panel_title": "Verification State  img2img"},
    {"family": "verification_state", "form": "img2img", "file": "vqa.json", "origin_qa_id": "014_state_008", "panel_title": "Verification State  img2img"},
    {"family": "verification_state", "form": "img2img", "file": "vqa.json", "origin_qa_id": "015_state_008", "panel_title": "Verification State  img2img"},
    {"family": "verification_state", "form": "img2img", "file": "vqa.json", "origin_qa_id": "027_state_011", "panel_title": "Verification State  img2img"},
    {"family": "final_state", "form": "img2img", "file": "vqa.json", "origin_qa_id": "013_final_002", "panel_title": "Final State  img2img"},
    {"family": "final_state", "form": "img2img", "file": "vqa.json", "origin_qa_id": "024_final_002", "panel_title": "Final State  img2img"},
    {"family": "final_state", "form": "img2img", "file": "vqa.json", "origin_qa_id": "030_final_002", "panel_title": "Final State  img2img"},
]

VIDEO_SPECS = [
    {"family": "action", "form": "video2video", "file": "vqa.json", "origin_qa_id": "011_action_005", "panel_title": "Action Matching  video2video", "segment_seconds": 6.0},
    {"family": "action", "form": "video2video", "file": "vqa.json", "origin_qa_id": "015_action_005", "panel_title": "Action Matching  video2video", "segment_seconds": 6.0},
    {"family": "verification_action", "form": "video2txt", "file": "openqa.json", "origin_qa_id": "014_verify_006", "panel_title": "Verification Action  video2txt", "segment_seconds": 6.0},
    {"family": "final_state", "form": "video2img", "file": "vqa.json", "origin_qa_id": "013_final_002", "panel_title": "Final State  video2img", "segment_seconds": 6.0},
    {"family": "recovery", "form": "video2txt", "file": "openqa.json", "origin_qa_id": "027_recovery_chain", "panel_title": "Recovery  video2txt", "segment_seconds": 7.0},
    {"family": "recovery", "form": "video2txt", "file": "openqa.json", "origin_qa_id": "054_recovery_chain", "panel_title": "Recovery  video2txt", "segment_seconds": 6.0},
]

SCENARIO_LABELS = {
    "elevator": "Elevator",
    "medical_kiosk": "Medical Kiosk",
    "subway_ticket": "Subway Ticket",
    "other": "Other",
}

FORM_ACCENTS = {
    "verification_state/img2img": (70, 122, 76),
    "final_state/img2img": (24, 133, 122),
    "action/video2video": (192, 104, 36),
    "verification_action/video2txt": (103, 92, 156),
    "final_state/video2img": (24, 133, 122),
    "recovery/video2txt": (167, 63, 94),
}

BG = (246, 246, 242)
PANEL_BG = (255, 255, 255)
PANEL_BORDER = (223, 224, 227)
TEXT = (28, 33, 41)
SUBTLE = (106, 114, 126)
QUERY_BG = (247, 249, 252)
OPTION_BG = (249, 250, 252)
CORRECT = (44, 143, 88)

FONT_REGULAR = Path(r"C:\Windows\Fonts\arial.ttf")
FONT_BOLD = Path(r"C:\Windows\Fonts\arialbd.ttf")
FONT_CJK = Path(r"C:\Windows\Fonts\simhei.ttf")

IMAGE_OUTPUT_NAME = "image_forms_overview.png"
VIDEO_OUTPUT_NAME = "video_forms_overview.mp4"
MANIFEST_OUTPUT_NAME = "review_visual_manifest.json"
OUTPUT_DIRNAME = "review_visuals_v2"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def regular_font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_REGULAR), size=size)


def bold_font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_BOLD), size=size)


def cjk_font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_CJK), size=size)


def text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont) -> int:
    if not text:
        return 0
    left, _, right, _ = draw.textbbox((0, 0), text, font=font)
    return right - left


def wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    max_width: int,
    max_lines: int | None = None,
) -> List[str]:
    words = text.split()
    if not words:
        return [""]
    lines: List[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if text_width(draw, candidate, font) <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    if max_lines and len(lines) > max_lines:
        lines = lines[:max_lines]
        last = lines[-1]
        while last and text_width(draw, f"{last}...", font) > max_width:
            last = " ".join(last.split()[:-1])
        lines[-1] = f"{last}..." if last else "..."
    return lines


def draw_wrapped_text(
    draw: ImageDraw.ImageDraw,
    position: Tuple[int, int],
    text: str,
    font: ImageFont.FreeTypeFont,
    fill: Tuple[int, int, int],
    max_width: int,
    line_gap: int = 4,
    max_lines: int | None = None,
) -> int:
    lines = wrap_text(draw, text, font, max_width, max_lines=max_lines)
    x, y = position
    bbox = draw.textbbox((0, 0), "Ag", font=font)
    line_height = bbox[3] - bbox[1]
    current_y = y
    for line in lines:
        draw.text((x, current_y), line, font=font, fill=fill)
        current_y += line_height + line_gap
    return current_y


def clean_query_text(text: str) -> str:
    return text.split("\nA.", 1)[0].strip()


def fit_image(image: Image.Image, width: int, height: int, bg_color: Tuple[int, int, int]) -> Image.Image:
    canvas = Image.new("RGB", (width, height), bg_color)
    src = image.convert("RGB")
    scale = min(width / src.width, height / src.height)
    new_size = (
        max(1, int(src.width * scale)),
        max(1, int(src.height * scale)),
    )
    resized = src.resize(new_size, Image.Resampling.LANCZOS)
    offset = ((width - new_size[0]) // 2, (height - new_size[1]) // 2)
    canvas.paste(resized, offset)
    return canvas


def read_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def read_video_all_frames(video_path: Path, max_seconds: float | None = None) -> Tuple[List[Image.Image], float]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    max_frames = None
    if max_seconds is not None:
        max_frames = max(1, int(round(max_seconds * fps)))
    frames: List[Image.Image] = []
    count = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(rgb))
        count += 1
        if max_frames is not None and count >= max_frames:
            break
    capture.release()
    if not frames:
        raise RuntimeError(f"No frames loaded from video: {video_path}")
    return frames, fps


def resolve_row(spec: Dict[str, Any], dataset_root: Path) -> Dict[str, Any]:
    json_path = dataset_root / spec["family"] / spec["form"] / spec["file"]
    rows = load_json(json_path)["data"]
    for row in rows:
        if row.get("origin_qa_id") == spec["origin_qa_id"]:
            return row
    raise KeyError(f"Missing {spec['origin_qa_id']} in {json_path}")


def draw_badge(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    text: str,
    bg_color: Tuple[int, int, int],
    font: ImageFont.FreeTypeFont,
    fg_color: Tuple[int, int, int] = (255, 255, 255),
) -> None:
    padding_x = 12
    padding_y = 6
    bbox = draw.textbbox((0, 0), text, font=font)
    width = bbox[2] - bbox[0] + padding_x * 2
    height = bbox[3] - bbox[1] + padding_y * 2
    draw.rounded_rectangle((x, y, x + width, y + height), radius=16, fill=bg_color)
    draw.text((x + padding_x, y + padding_y - 1), text, font=font, fill=fg_color)


def paste_asset(
    panel: Image.Image,
    draw: ImageDraw.ImageDraw,
    box: Tuple[int, int, int, int],
    image: Image.Image,
    label: str,
    accent: Tuple[int, int, int],
    correct: bool = False,
) -> None:
    x, y, w, h = box
    thumb = fit_image(image, w, h, QUERY_BG)
    panel.paste(thumb, (x, y))
    border = CORRECT if correct else PANEL_BORDER
    border_width = 5 if correct else 2
    draw.rounded_rectangle((x, y, x + w, y + h), radius=18, outline=border, width=border_width)
    draw_badge(draw, x + 14, y + 12, label, accent if not correct else CORRECT, bold_font(18))


def draw_image_panel(
    canvas: Image.Image,
    position: Tuple[int, int],
    size: Tuple[int, int],
    spec: Dict[str, Any],
    row: Dict[str, Any],
    form_dir: Path,
) -> None:
    panel = Image.new("RGB", size, PANEL_BG)
    draw = ImageDraw.Draw(panel)
    w, h = size
    accent = FORM_ACCENTS[f"{spec['family']}/{spec['form']}"]

    draw.rounded_rectangle((0, 0, w - 1, h - 1), radius=28, fill=PANEL_BG, outline=PANEL_BORDER, width=2)
    draw.rounded_rectangle((0, 0, w - 1, 10), radius=0, fill=accent)
    draw.text((22, 22), spec["panel_title"], font=cjk_font(26), fill=TEXT)
    draw.text((22, 58), row["origin_qa_id"], font=regular_font(18), fill=SUBTLE)
    draw_badge(draw, w - 190, 24, "Image -> Image", accent, bold_font(18))

    question_box = (20, 92, w - 20, 182)
    draw.rounded_rectangle(question_box, radius=18, fill=(248, 248, 245), outline=(232, 233, 236), width=2)
    draw.text((34, 108), "Question", font=bold_font(18), fill=accent)
    draw_wrapped_text(draw, (34, 140), clean_query_text(row["query"]), regular_font(22), TEXT, question_box[2] - question_box[0] - 28, line_gap=5, max_lines=2)

    query_image = read_image(form_dir / row["query_img_path"])
    paste_asset(panel, draw, (22, 204, 340, 250), query_image, "Query image", accent, correct=False)

    option_images = [read_image(form_dir / rel) for rel in row["option_imgs_path"]]
    options_x = 382
    options_y = 204
    draw.text((options_x, options_y - 28), "Image options", font=bold_font(18), fill=accent)
    box_w = 326
    box_h = 116
    gap_x = 18
    gap_y = 16
    for index, letter in enumerate("ABCD"):
        col = index % 2
        row_index = index // 2
        option_box = (
            options_x + col * (box_w + gap_x),
            options_y + row_index * (box_h + gap_y),
            box_w,
            box_h,
        )
        paste_asset(panel, draw, option_box, option_images[index], f"Option {letter}", accent, correct=(letter == row["GT"]))

    scenario_text = SCENARIO_LABELS.get(row["scenario_family"], row["scenario_family"])
    draw.text((22, h - 40), f"Scenario: {scenario_text}", font=regular_font(18), fill=SUBTLE)
    draw.text((w - 220, h - 40), f"Correct Option: {row['GT']}", font=bold_font(20), fill=CORRECT)
    canvas.paste(panel, position)


def render_image_overview(dataset_root: Path, output_path: Path) -> List[Dict[str, Any]]:
    panel_width = 1120
    panel_height = 500
    margin_x = 36
    margin_y = 24
    header_h = 120
    gutter_x = 28
    gutter_y = 24
    cols = 2
    rows = 4
    canvas_width = margin_x * 2 + cols * panel_width + (cols - 1) * gutter_x
    canvas_height = header_h + margin_y + rows * panel_height + (rows - 1) * gutter_y + margin_y
    canvas = Image.new("RGB", (canvas_width, canvas_height), BG)
    draw = ImageDraw.Draw(canvas)
    draw.text((36, 24), "0424 SWITCH State-Aligned Image Review", font=bold_font(34), fill=TEXT)
    draw.text(
        (40, 72),
        "Image-only checks built from img2img rows in hf_innovative_qa_v2_multiform_state_aligned_v1.",
        font=regular_font(20),
        fill=SUBTLE,
    )

    manifest_rows: List[Dict[str, Any]] = []
    for index, spec in enumerate(IMAGE_SPECS):
        row = resolve_row(spec, dataset_root)
        form_dir = dataset_root / spec["family"] / spec["form"]
        col = index % cols
        row_index = index // cols
        x = margin_x + col * (panel_width + gutter_x)
        y = header_h + margin_y + row_index * (panel_height + gutter_y)
        draw_image_panel(canvas, (x, y), (panel_width, panel_height), spec, row, form_dir)
        manifest_rows.append(
            {
                "origin_qa_id": spec["origin_qa_id"],
                "family": spec["family"],
                "form": spec["form"],
                "query_img_path": row["query_img_path"],
                "option_imgs_path": row["option_imgs_path"],
                "GT": row["GT"],
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG")
    return manifest_rows


def pil_to_bgr(image: Image.Image) -> np.ndarray:
    rgb = np.array(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def get_video_reader_frames(
    spec: Dict[str, Any],
    row: Dict[str, Any],
    dataset_root: Path,
    annotation_root: Path,
) -> Tuple[List[Image.Image], Dict[str, List[Image.Image]]]:
    form_dir = dataset_root / spec["family"] / spec["form"]
    query_frames: List[Image.Image]
    option_video_frames: Dict[str, List[Image.Image]] = {}

    if row.get("query_video_path") and spec["family"] in {"action", "final_state"}:
        query_frames, _ = read_video_all_frames(form_dir / row["query_video_path"], max_seconds=8.0)
    else:
        source_span = row["source_span"]
        data_id = row["origin_qa_id"].split("_", 1)[0]
        raw_video = annotation_root / f"{data_id}.mp4"
        start = int(source_span["start"])
        end = int(source_span["end"])
        capture = cv2.VideoCapture(str(raw_video))
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open raw video: {raw_video}")
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        start = max(0, min(start, total - 1))
        end = max(start, min(end, total - 1))
        query_frames = []
        capture.set(cv2.CAP_PROP_POS_FRAMES, start)
        current = start
        while current <= end:
            ok, frame = capture.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            query_frames.append(Image.fromarray(rgb))
            current += 1
        capture.release()
        if not query_frames:
            raise RuntimeError(f"No frames extracted from raw video for {row['origin_qa_id']}")

    if row.get("option_videos_path"):
        for letter, rel in zip("ABCD", row["option_videos_path"]):
            frames, _ = read_video_all_frames(form_dir / rel, max_seconds=8.0)
            option_video_frames[letter] = frames
    return query_frames, option_video_frames


def pick_frame(frames: Sequence[Image.Image], out_index: int, out_total: int, loop: bool = True) -> Image.Image:
    if not frames:
        raise ValueError("No frames available")
    if len(frames) == 1:
        return frames[0]
    if loop and len(frames) < out_total:
        return frames[out_index % len(frames)]
    ratio = out_index / max(1, out_total - 1)
    src_index = min(len(frames) - 1, int(round(ratio * (len(frames) - 1))))
    return frames[src_index]


def draw_text_box(
    draw: ImageDraw.ImageDraw,
    box: Tuple[int, int, int, int],
    title: str,
    body: str,
    accent: Tuple[int, int, int],
    fill_bg: Tuple[int, int, int] = (250, 250, 248),
) -> None:
    x, y, w, h = box
    draw.rounded_rectangle((x, y, x + w, y + h), radius=18, fill=fill_bg, outline=PANEL_BORDER, width=2)
    draw.text((x + 14, y + 12), title, font=bold_font(18), fill=accent)
    draw_wrapped_text(draw, (x + 14, y + 42), body, regular_font(22), TEXT, w - 28, line_gap=4, max_lines=6)


def render_video_segment_frame(
    spec: Dict[str, Any],
    row: Dict[str, Any],
    query_frame: Image.Image,
    option_video_frames: Dict[str, List[Image.Image]],
    option_image_frames: Dict[str, Image.Image],
    frame_index: int,
    total_frames: int,
    canvas_size: Tuple[int, int],
) -> Image.Image:
    w, h = canvas_size
    accent = FORM_ACCENTS[f"{spec['family']}/{spec['form']}"]
    canvas = Image.new("RGB", canvas_size, BG)
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((24, 18, w - 24, h - 20), radius=28, fill=PANEL_BG, outline=PANEL_BORDER, width=2)
    draw.rounded_rectangle((24, 18, w - 24, 28), radius=0, fill=accent)
    draw.text((44, 34), spec["panel_title"], font=cjk_font(28), fill=TEXT)
    draw.text((44, 72), row["origin_qa_id"], font=regular_font(18), fill=SUBTLE)
    modality = spec["form"].replace("2", " -> ").replace("video", "Video").replace("img", "Image").replace("txt", "Text")
    draw_badge(draw, w - 220, 30, modality, accent, bold_font(18))

    draw_text_box(draw, (40, 108, w - 80, 122), "Question", clean_query_text(row["query"]), accent)
    query_box = (40, 250, 620, 420)
    paste_asset(canvas, draw, query_box, query_frame, "Query video", accent, correct=False)

    if option_video_frames or option_image_frames:
        options_x = 700
        options_y = 250
        right_margin = 40
        gap_x = 18
        box_w = max(260, (w - options_x - right_margin - gap_x) // 2)
        box_h = 160
        gap_y = 18
        draw.text((options_x, options_y - 28), "Options", font=bold_font(18), fill=accent)
        for idx, letter in enumerate("ABCD"):
            col = idx % 2
            row_idx = idx // 2
            box = (options_x + col * (box_w + gap_x), options_y + row_idx * (box_h + gap_y), box_w, box_h)
            if letter in option_video_frames:
                option_frame = pick_frame(option_video_frames[letter], frame_index, total_frames, loop=True)
            else:
                option_frame = option_image_frames[letter]
            paste_asset(canvas, draw, box, option_frame, f"Option {letter}", accent, correct=(letter == row.get("GT")))
        footer = f"Scenario: {SCENARIO_LABELS.get(row['scenario_family'], row['scenario_family'])}    Correct: {row.get('GT', 'OpenQA')}"
        draw.text((40, h - 48), footer, font=regular_font(18), fill=SUBTLE)
    else:
        gt_payload = row.get("GT")
        gt_text = gt_payload if isinstance(gt_payload, str) else json.dumps(gt_payload, ensure_ascii=False, indent=2)
        draw_text_box(draw, (700, 250, 460, 190), "Answer", gt_text, accent, fill_bg=(247, 250, 248))
        draw_text_box(draw, (700, 458, 460, 212), "Canonical Answer", str(row.get("canonical_answer") or ""), accent, fill_bg=(249, 249, 251))
        footer = f"Scenario: {SCENARIO_LABELS.get(row['scenario_family'], row['scenario_family'])}"
        draw.text((40, h - 48), footer, font=regular_font(18), fill=SUBTLE)

    return canvas


def render_video_overview(dataset_root: Path, annotation_root: Path, output_path: Path) -> List[Dict[str, Any]]:
    canvas_size = (1600, 760)
    fps = 12
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        canvas_size,
    )
    if not writer.isOpened():
        raise RuntimeError(f"Unable to create writer: {output_path}")

    manifest_rows: List[Dict[str, Any]] = []
    try:
        for spec in VIDEO_SPECS:
            row = resolve_row(spec, dataset_root)
            form_dir = dataset_root / spec["family"] / spec["form"]
            query_frames, option_video_frames = get_video_reader_frames(spec, row, dataset_root, annotation_root)
            option_image_frames: Dict[str, Image.Image] = {}
            if row.get("option_imgs_path"):
                option_image_frames = {
                    letter: read_image(form_dir / rel)
                    for letter, rel in zip("ABCD", row["option_imgs_path"])
                }
            segment_frames = max(1, int(round(spec["segment_seconds"] * fps)))
            for out_index in range(segment_frames):
                query_frame = pick_frame(query_frames, out_index, segment_frames, loop=True)
                panel = render_video_segment_frame(
                    spec,
                    row,
                    query_frame,
                    option_video_frames,
                    option_image_frames,
                    out_index,
                    segment_frames,
                    canvas_size,
                )
                writer.write(pil_to_bgr(panel))
            manifest_rows.append(
                {
                    "origin_qa_id": spec["origin_qa_id"],
                    "family": spec["family"],
                    "form": spec["form"],
                    "segment_seconds": spec["segment_seconds"],
                    "GT": row.get("GT"),
                }
            )
    finally:
        writer.release()
    return manifest_rows


def main() -> None:
    dataset_root = DEFAULT_DATASET_ROOT
    annotation_root = dataset_root.parent
    output_root = dataset_root / OUTPUT_DIRNAME
    output_root.mkdir(parents=True, exist_ok=True)

    image_manifest = render_image_overview(dataset_root, output_root / IMAGE_OUTPUT_NAME)
    video_manifest = render_video_overview(dataset_root, annotation_root, output_root / VIDEO_OUTPUT_NAME)
    manifest = {
        "dataset_root": str(dataset_root),
        "image_output": IMAGE_OUTPUT_NAME,
        "video_output": VIDEO_OUTPUT_NAME,
        "image_samples": image_manifest,
        "video_samples": video_manifest,
    }
    (output_root / MANIFEST_OUTPUT_NAME).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote review visuals to: {output_root}")


if __name__ == "__main__":
    main()
