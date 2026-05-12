#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import cv2
from PIL import Image, ImageDraw, ImageFont


DEFAULT_DATASET_ROOT = Path("annotations") / "0421" / "switch" / "hf_innovative_qa_v2_multiform"

FONT_REGULAR = Path(r"C:\Windows\Fonts\arial.ttf")
FONT_BOLD = Path(r"C:\Windows\Fonts\arialbd.ttf")
FONT_CJK = Path(r"C:\Windows\Fonts\simhei.ttf")

BG = (246, 246, 242)
PANEL_BG = (255, 255, 255)
PANEL_BORDER = (223, 224, 227)
TEXT = (28, 33, 41)
SUBTLE = (106, 114, 126)
QUERY_BG = (247, 249, 252)
OPTION_BG = (249, 250, 252)
CORRECT = (44, 143, 88)

PANEL_SPECS = [
    {
        "family": "vqa_task",
        "form": "video2txt",
        "origin_qa_id": "008_task_001",
        "panel_title": "任务理解  video2txt",
        "family_label": "Task Understanding",
        "modality_label": "Video -> Text",
        "accent": (53, 105, 173),
    },
    {
        "family": "action",
        "form": "img2txt",
        "origin_qa_id": "004_action_003",
        "panel_title": "关键动作  img2txt",
        "family_label": "Key Action",
        "modality_label": "Image -> Text",
        "accent": (192, 104, 36),
    },
    {
        "family": "action",
        "form": "video2video",
        "origin_qa_id": "003_action_004",
        "panel_title": "动作匹配  video2video",
        "family_label": "Action Matching",
        "modality_label": "Video -> Video",
        "accent": (192, 104, 36),
    },
    {
        "family": "final_state",
        "form": "img2img",
        "origin_qa_id": "003_final_002",
        "panel_title": "最终状态  img2img",
        "family_label": "Final State",
        "modality_label": "Image -> Image",
        "accent": (24, 133, 122),
    },
    {
        "family": "final_state",
        "form": "video2img",
        "origin_qa_id": "006_final_002",
        "panel_title": "结果预测  video2img",
        "family_label": "Final State",
        "modality_label": "Video -> Image",
        "accent": (24, 133, 122),
    },
    {
        "family": "verification_state",
        "form": "img2img",
        "origin_qa_id": "008_state_008",
        "panel_title": "验证信号  img2img",
        "family_label": "Verification State",
        "modality_label": "Image -> Image",
        "accent": (70, 122, 76),
    },
]

SCENARIO_LABELS = {
    "elevator": "Elevator",
    "medical_kiosk": "Medical Kiosk",
    "subway_ticket": "Subway Ticket",
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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


def read_video_frame(path: Path, ratio: float = 0.45) -> Image.Image:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video for poster frame: {path}")
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    target = max(0, min(total_frames - 1, int(total_frames * ratio))) if total_frames > 0 else 0
    capture.set(cv2.CAP_PROP_POS_FRAMES, target)
    ok, frame = capture.read()
    if not ok and total_frames > 1:
        capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, total_frames // 2))
        ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Unable to read poster frame from {path}")
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)


def load_query_asset(form_dir: Path, row: Dict[str, Any]) -> Tuple[Image.Image, str]:
    if row.get("query_img_path"):
        return read_image(form_dir / row["query_img_path"]), "Query image"
    if row.get("query_video_path"):
        return read_video_frame(form_dir / row["query_video_path"]), "Query video"
    raise KeyError("Row has no query asset path")


def load_option_assets(form_dir: Path, row: Dict[str, Any]) -> Tuple[List[Image.Image], str]:
    if row.get("option_imgs_path"):
        return [read_image(form_dir / rel) for rel in row["option_imgs_path"]], "Image options"
    if row.get("option_videos_path"):
        return [read_video_frame(form_dir / rel) for rel in row["option_videos_path"]] , "Video options"
    raise KeyError("Row has no visual option assets")


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
    badge_font = bold_font(18)
    draw_badge(draw, x + 14, y + 12, label, accent if not correct else CORRECT, badge_font)


def draw_text_option(
    draw: ImageDraw.ImageDraw,
    box: Tuple[int, int, int, int],
    label: str,
    text: str,
    is_correct: bool,
) -> None:
    x, y, w, h = box
    bg = (237, 246, 240) if is_correct else OPTION_BG
    outline = CORRECT if is_correct else PANEL_BORDER
    draw.rounded_rectangle((x, y, x + w, y + h), radius=18, fill=bg, outline=outline, width=4 if is_correct else 2)
    draw_badge(draw, x + 14, y + 14, label, CORRECT if is_correct else (121, 127, 138), bold_font(18))
    text_x = x + 80
    text_y = y + 16
    draw_wrapped_text(draw, (text_x, text_y), text, regular_font(22), TEXT, w - 96, line_gap=4, max_lines=2)


def draw_panel(
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
    accent = spec["accent"]

    draw.rounded_rectangle((0, 0, w - 1, h - 1), radius=30, fill=PANEL_BG, outline=PANEL_BORDER, width=2)
    draw.rounded_rectangle((0, 0, w - 1, 10), radius=0, fill=accent)

    title_font = cjk_font(30)
    label_font = bold_font(20)
    meta_font = regular_font(18)
    body_font = regular_font(24)

    draw.text((28, 26), spec["panel_title"], font=title_font, fill=TEXT)
    draw.text((28, 66), spec["family_label"], font=regular_font(20), fill=SUBTLE)
    draw_badge(draw, w - 220, 28, spec["modality_label"], accent, label_font)

    question_box = (28, 104, w - 28, 224)
    draw.rounded_rectangle(question_box, radius=22, fill=(248, 248, 245), outline=(232, 233, 236), width=2)
    draw.text((44, 122), "Question", font=label_font, fill=accent)
    question_text = clean_query_text(row["query"])
    draw_wrapped_text(draw, (44, 156), question_text, body_font, TEXT, question_box[2] - question_box[0] - 34, line_gap=6, max_lines=3)

    query_image, query_label = load_query_asset(form_dir, row)
    query_box = (28, 250, 418, 240)
    paste_asset(panel, draw, query_box, query_image, query_label, accent, correct=False)

    options_box_x = 468
    options_box_y = 250
    correct_letter = row["GT"]

    if all(row.get(f"option_{letter.lower()}") for letter in "ABCD"):
        option_height = 60
        gap = 12
        for index, letter in enumerate("ABCD"):
            option_box = (
                options_box_x,
                options_box_y + index * (option_height + gap),
                w - options_box_x - 28,
                option_height,
            )
            draw_text_option(
                draw,
                option_box,
                letter,
                row[f"option_{letter.lower()}"],
                is_correct=(letter == correct_letter),
            )
    else:
        option_assets, option_label = load_option_assets(form_dir, row)
        draw.text((options_box_x, options_box_y - 30), option_label, font=label_font, fill=accent)
        box_w = 280
        box_h = 124
        gap_x = 18
        gap_y = 18
        for index, letter in enumerate("ABCD"):
            col = index % 2
            row_index = index // 2
            option_box = (
                options_box_x + col * (box_w + gap_x),
                options_box_y + row_index * (box_h + gap_y),
                box_w,
                box_h,
            )
            paste_asset(
                panel,
                draw,
                option_box,
                option_assets[index],
                f"Option {letter}",
                accent,
                correct=(letter == correct_letter),
            )

    footer_y = h - 48
    scenario_text = SCENARIO_LABELS.get(row["scenario_family"], row["scenario_family"])
    draw.text((28, footer_y), f"Scenario: {scenario_text}", font=meta_font, fill=SUBTLE)
    draw.text((w - 210, footer_y), f"Correct Option: {correct_letter}", font=bold_font(20), fill=CORRECT)

    canvas.paste(panel, position)


def find_row(dataset_root: Path, family: str, form: str, origin_qa_id: str) -> Dict[str, Any]:
    vqa_path = dataset_root / family / form / "vqa.json"
    data = load_json(vqa_path)["data"]
    for row in data:
        if row.get("origin_qa_id") == origin_qa_id:
            return row
    raise KeyError(f"Unable to find {origin_qa_id} in {vqa_path}")


def render_figure(dataset_root: Path, output_path: Path) -> None:
    canvas_width = 2360
    header_height = 132
    panel_width = 1120
    panel_height = 620
    margin_x = 40
    margin_y = 34
    gutter_x = 40
    gutter_y = 28
    canvas_height = header_height + margin_y + panel_height * 3 + gutter_y * 2 + margin_y

    canvas = Image.new("RGB", (canvas_width, canvas_height), BG)
    draw = ImageDraw.Draw(canvas)

    draw.text((44, 26), "SWITCH QA v2 题型示例图", font=cjk_font(42), fill=TEXT)
    draw.text(
        (48, 82),
        "绿色边框表示正确选项；视频题面与视频选项使用代表帧静态展示，示例均来自 hf_innovative_qa_v2_multiform。",
        font=cjk_font(22),
        fill=SUBTLE,
    )

    for index, spec in enumerate(PANEL_SPECS):
        row = find_row(dataset_root, spec["family"], spec["form"], spec["origin_qa_id"])
        form_dir = dataset_root / spec["family"] / spec["form"]
        col = index % 2
        row_index = index // 2
        x = margin_x + col * (panel_width + gutter_x)
        y = header_height + margin_y + row_index * (panel_height + gutter_y)
        draw_panel(canvas, (x, y), (panel_width, panel_height), spec, row, form_dir)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the SWITCH QA example overview figure.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Dataset root containing the family/form JSON and asset folders.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional explicit output PNG path. Defaults to <dataset-root>/qa_type_examples_overview.png",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root
    output_path = args.output_path or (dataset_root / "qa_type_examples_overview.png")
    render_figure(dataset_root, output_path)
    print(f"Wrote visualization figure to: {output_path}")


if __name__ == "__main__":
    main()
