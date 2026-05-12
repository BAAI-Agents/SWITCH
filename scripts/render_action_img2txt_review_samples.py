#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont


FONT_CANDIDATES = [
    Path("C:/Windows/Fonts/arial.ttf"),
    Path("C:/Windows/Fonts/segoeui.ttf"),
    Path("C:/Windows/Fonts/msyh.ttc"),
]
BOLD_FONT_CANDIDATES = [
    Path("C:/Windows/Fonts/arialbd.ttf"),
    Path("C:/Windows/Fonts/segoeuib.ttf"),
    Path("C:/Windows/Fonts/msyhbd.ttc"),
]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    candidates = BOLD_FONT_CANDIDATES if bold else FONT_CANDIDATES
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def text_width(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.ImageFont) -> int:
    box = draw.textbbox((0, 0), text, font=fnt)
    return box[2] - box[0]


def wrap_text(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.ImageFont, max_width: int) -> List[str]:
    words: List[str] = []
    for line in str(text).replace("\r", "").split("\n"):
        parts = line.split(" ")
        if parts:
            words.extend(parts)
        words.append("\n")
    lines: List[str] = []
    current = ""
    for word in words:
        if word == "\n":
            if current:
                lines.append(current)
                current = ""
            continue
        candidate = word if not current else f"{current} {word}"
        if text_width(draw, candidate, fnt) <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[int, int],
    text: str,
    fnt: ImageFont.ImageFont,
    fill: Tuple[int, int, int],
    max_width: int,
    line_gap: int = 6,
    max_lines: Optional[int] = None,
) -> int:
    x, y = xy
    lines = wrap_text(draw, text, fnt, max_width)
    if max_lines is not None and len(lines) > max_lines:
        lines = lines[:max_lines]
        if lines:
            lines[-1] = lines[-1].rstrip(". ") + "..."
    line_h = fnt.size + line_gap if hasattr(fnt, "size") else 22
    for line in lines:
        draw.text((x, y), line, font=fnt, fill=fill)
        y += line_h
    return y


def fit_image(path: Path, box: Tuple[int, int], bg: Tuple[int, int, int] = (248, 250, 252)) -> Image.Image:
    image = Image.open(path).convert("RGB")
    target_w, target_h = box
    image.thumbnail((target_w, target_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (target_w, target_h), bg)
    x = (target_w - image.width) // 2
    y = (target_h - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def strip_options_from_query(query: str) -> str:
    positions = [query.find(f"\n{label}. ") for label in "ABCD"]
    positions = [pos for pos in positions if pos >= 0]
    if not positions:
        return query.strip()
    return query[: min(positions)].strip()


def label_from_result(result: Dict[str, Any]) -> Optional[str]:
    labels = (result.get("value") or {}).get("timelinelabels") or []
    return labels[0] if labels else None


def text_from_result(result: Dict[str, Any]) -> Optional[str]:
    texts = (result.get("meta") or {}).get("text") or []
    return texts[0] if texts else None


def range_from_result(result: Dict[str, Any]) -> Dict[str, Optional[int]]:
    ranges = (result.get("value") or {}).get("ranges") or []
    if not ranges:
        return {"start": None, "end": None}
    first = ranges[0]
    return {"start": first.get("start"), "end": first.get("end")}


def build_raw_index(annotation_root: Path) -> Dict[str, Dict[str, Any]]:
    payload = load_json(annotation_root / "SWITCHAction_2.json")
    index: Dict[str, Dict[str, Any]] = {}
    for item in payload:
        annotations = item.get("annotations") or []
        if not annotations:
            continue
        results = annotations[0].get("result") or []
        data_id = None
        for result in results:
            if label_from_result(result) == "data_id":
                data_id = text_from_result(result)
                break
        if not data_id:
            continue
        labels: List[Dict[str, Any]] = []
        for result in results:
            label = label_from_result(result)
            if not label:
                continue
            span = range_from_result(result)
            labels.append(
                {
                    "label": label,
                    "start": span["start"],
                    "end": span["end"],
                    "text": text_from_result(result),
                }
            )
        index[data_id] = {
            "item_id": item.get("id"),
            "video_url": (item.get("data") or {}).get("Action_2"),
            "video_meta": ((item.get("data") or {}).get("meta") or {}).get("Action_2") or {},
            "labels": labels,
        }
    return index


def label_at_span(raw_info: Dict[str, Any], label: str, span: Dict[str, Any]) -> Optional[str]:
    start = span.get("start")
    end = span.get("end")
    for item in raw_info.get("labels", []):
        if item.get("label") == label and item.get("start") == start and item.get("end") == end:
            return item.get("text")
    return None


def labels_around(raw_info: Dict[str, Any], span: Dict[str, Any]) -> List[Dict[str, Any]]:
    start = int(span.get("start") or 0)
    end = int(span.get("end") or start)
    useful = {"ui_change", "view_change", "ui_state", "physical_world_change", "physical_world_state"}
    out = []
    for item in raw_info.get("labels", []):
        item_start = item.get("start")
        if item_start is None:
            continue
        if item.get("label") in useful and start - 5 <= int(item_start) <= end + 40:
            out.append(item)
    return out[:6]


def choose_samples(rows: Sequence[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row.get("scenario_family") or "other")].append(row)
    for bucket in buckets.values():
        bucket.sort(key=lambda row: (str(row.get("semantic_anchor")), int(row.get("id", 0))))
    chosen: List[Dict[str, Any]] = []
    scenarios = sorted(buckets.keys())
    while len(chosen) < limit and scenarios:
        next_scenarios = []
        for scenario in scenarios:
            bucket = buckets[scenario]
            if bucket:
                chosen.append(bucket.pop(0))
                if len(chosen) >= limit:
                    break
            if bucket:
                next_scenarios.append(scenario)
        scenarios = next_scenarios
    return chosen


def build_sample_info(
    row: Dict[str, Any],
    dataset_root: Path,
    form_root: Path,
    raw_index: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    data_id = str(row["origin_qa_id"]).split("_", 1)[0]
    raw_info = raw_index.get(data_id, {})
    span = row.get("source_span") or {}
    return {
        "id": row.get("id"),
        "origin_qa_id": row.get("origin_qa_id"),
        "scenario_family": row.get("scenario_family"),
        "semantic_anchor": row.get("semantic_anchor"),
        "prompt_variant": row.get("prompt_variant"),
        "rewrite_type": row.get("rewrite_type"),
        "query": row.get("query"),
        "question_only": strip_options_from_query(row.get("query") or ""),
        "query_img_path": row.get("query_img_path"),
        "query_img_abspath": str((form_root / row["query_img_path"]).resolve()),
        "query_source_frame": row.get("query_source_frame"),
        "source_span": span,
        "canonical_answer": row.get("canonical_answer"),
        "GT": row.get("GT"),
        "options": {label: row.get(f"option_{label.lower()}") for label in "ABCD"},
        "option_origin_qa_ids": row.get("option_origin_qa_ids"),
        "raw_annotation": {
            "data_id": data_id,
            "item_id": raw_info.get("item_id"),
            "video_meta": raw_info.get("video_meta"),
            "overall_requirement": next(
                (x.get("text") for x in raw_info.get("labels", []) if x.get("label") == "overall_requirement"),
                None,
            ),
            "overall_verification": next(
                (x.get("text") for x in raw_info.get("labels", []) if x.get("label") == "overall_verification"),
                None,
            ),
            "action_type": label_at_span(raw_info, "action-type", span),
            "action_requirement": label_at_span(raw_info, "action_requirement", span),
            "action_description": label_at_span(raw_info, "action_description", span),
            "action_step_id": label_at_span(raw_info, "action_step_id", span),
            "nearby_change_labels": labels_around(raw_info, span),
        },
        "construction": {
            "query_frame_rule": "pre_key_frame = source_span.start - max(1, int(fps * 0.4)), clamped to video range",
            "answer_rule": "answer = action_description from the action source span",
            "option_rule": "GT action record + 3 same-family distractors selected by qa_type/scenario/semantic_group score, then deterministic hash ordering",
        },
    }


def draw_badge(draw: ImageDraw.ImageDraw, xy: Tuple[int, int], text: str, fill: Tuple[int, int, int]) -> None:
    x, y = xy
    fnt = font(20, bold=True)
    pad_x, pad_y = 12, 6
    w = text_width(draw, text, fnt) + pad_x * 2
    h = 34
    draw.rounded_rectangle((x, y, x + w, y + h), radius=14, fill=fill)
    draw.text((x + pad_x, y + pad_y), text, font=fnt, fill=(255, 255, 255))


def render_card(sample: Dict[str, Any], output_path: Path) -> None:
    width, height = 1220, 720
    bg = (247, 249, 247)
    green = (34, 140, 119)
    dark = (30, 41, 59)
    muted = (100, 116, 139)
    card = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(card)
    draw.rounded_rectangle((18, 18, width - 18, height - 18), radius=22, fill=(255, 255, 255), outline=(220, 226, 232), width=2)
    draw.rectangle((18, 18, width - 18, 34), fill=green)

    draw.text((42, 58), "Action img2txt", font=font(34, bold=True), fill=dark)
    draw.text((42, 100), f"{sample['origin_qa_id']} | {sample['scenario_family']} | {sample['semantic_anchor']}", font=font(20), fill=muted)
    draw_badge(draw, (width - 225, 58), "Image -> Text", green)

    q_box = (42, 142, width - 42, 270)
    draw.rounded_rectangle(q_box, radius=18, fill=(250, 250, 248), outline=(225, 229, 235), width=2)
    draw.text((66, 162), "Question", font=font(22, bold=True), fill=green)
    draw_wrapped(draw, (66, 196), sample["question_only"], font(24), dark, q_box[2] - q_box[0] - 48, max_lines=3)

    image_box = (42, 306, 438, 620)
    draw.rounded_rectangle(image_box, radius=18, fill=(245, 247, 250), outline=(225, 229, 235), width=2)
    draw_badge(draw, (64, 326), "Query image", green)
    query_image = fit_image(Path(sample["query_img_abspath"]), (330, 260))
    card.paste(query_image, (75, 348))

    opt_x, opt_y = 470, 306
    option_w, option_h = 700, 64
    for i, label in enumerate("ABCD"):
        y = opt_y + i * (option_h + 18)
        is_gt = sample["GT"] == label
        fill = (235, 248, 242) if is_gt else (248, 250, 252)
        outline = (34, 140, 85) if is_gt else (225, 229, 235)
        draw.rounded_rectangle((opt_x, y, opt_x + option_w, y + option_h), radius=14, fill=fill, outline=outline, width=3 if is_gt else 2)
        draw_badge(draw, (opt_x + 14, y + 14), label, green if is_gt else (100, 116, 139))
        draw_wrapped(draw, (opt_x + 74, y + 14), sample["options"][label] or "", font(19), dark, option_w - 96, max_lines=2)

    meta_y = 640
    meta = (
        f"GT: {sample['GT']} | query_frame: {sample['query_source_frame']} | "
        f"action_span: {sample['source_span'].get('start')}-{sample['source_span'].get('end')} | "
        f"step: {sample['raw_annotation'].get('action_step_id')}"
    )
    draw.text((42, meta_y), meta, font=font(20, bold=True), fill=green)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    card.save(output_path)


def render_overview(card_paths: Sequence[Path], output_path: Path) -> None:
    thumbs = [Image.open(path).convert("RGB") for path in card_paths]
    thumb_w, thumb_h = 900, 532
    cols = 2
    rows = math.ceil(len(thumbs) / cols)
    header_h = 120
    gap = 28
    width = cols * thumb_w + (cols + 1) * gap
    height = header_h + rows * thumb_h + (rows + 1) * gap
    canvas = Image.new("RGB", (width, height), (241, 245, 241))
    draw = ImageDraw.Draw(canvas)
    draw.text((gap, 30), "SWITCH Action img2txt Review Samples", font=font(42, bold=True), fill=(30, 41, 59))
    draw.text((gap, 78), "Question + query image + text options; green option is GT.", font=font(22), fill=(100, 116, 139))
    for idx, thumb in enumerate(thumbs):
        thumb.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        x = gap + (idx % cols) * (thumb_w + gap)
        y = header_h + gap + (idx // cols) * (thumb_h + gap)
        bg = Image.new("RGB", (thumb_w, thumb_h), (255, 255, 255))
        bg.paste(thumb, ((thumb_w - thumb.width) // 2, (thumb_h - thumb.height) // 2))
        canvas.paste(bg, (x, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def build_markdown(samples: Sequence[Dict[str, Any]], output_root: Path) -> str:
    lines = ["# Action img2txt Review Samples", ""]
    lines.append("These samples come from `action/img2txt/vqa.json`.")
    lines.append("")
    lines.append("## Construction Summary")
    lines.append("")
    lines.append("- The source action record is created from `action_description` spans in `SWITCHAction_2.qa_candidates.json`.")
    lines.append("- `main_task` comes from `overall_requirement`; action step metadata comes from `action_step_id`.")
    lines.append("- Query image uses `pre_key_frame`: `source_span.start - max(1, int(fps * 0.4))`, clamped to the video range.")
    lines.append("- OpenQA answer is the action phrase; VQA options are the GT action plus three same-family distractors, ordered by a stable hash.")
    lines.append("")
    lines.append("## Samples")
    lines.append("")
    for sample in samples:
        card_name = f"{sample['origin_qa_id']}.png"
        lines.append(f"### {sample['origin_qa_id']}")
        lines.append("")
        lines.append(f"![{sample['origin_qa_id']}](cards/{card_name})")
        lines.append("")
        lines.append(f"- scenario: `{sample['scenario_family']}`")
        lines.append(f"- semantic_anchor: `{sample['semantic_anchor']}`")
        lines.append(f"- query_source_frame: `{sample['query_source_frame']}`")
        lines.append(f"- source_span: `{sample['source_span'].get('start')}-{sample['source_span'].get('end')}`")
        lines.append(f"- action_description: `{sample['raw_annotation'].get('action_description')}`")
        lines.append(f"- action_step_id: `{sample['raw_annotation'].get('action_step_id')}`")
        lines.append(f"- GT: `{sample['GT']}` / `{sample['canonical_answer']}`")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Render review samples for action/img2txt.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("annotations/0424/SWITCHAction_2/hf_innovative_qa_v2_multiform_state_aligned_postwrong_v2"),
    )
    parser.add_argument("--annotation-root", type=Path, default=Path("annotations/0424/SWITCHAction_2"))
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--output-dirname", default="review_samples_v1")
    args = parser.parse_args()

    form_root = args.dataset_root / "action" / "img2txt"
    output_root = form_root / args.output_dirname
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    rows = load_json(form_root / "vqa.json")["data"]
    raw_index = build_raw_index(args.annotation_root)
    chosen = choose_samples(rows, args.limit)
    samples = [build_sample_info(row, args.dataset_root, form_root, raw_index) for row in chosen]

    card_paths: List[Path] = []
    cards_dir = output_root / "cards"
    for sample in samples:
        card_path = cards_dir / f"{sample['origin_qa_id']}.png"
        render_card(sample, card_path)
        card_paths.append(card_path)

    render_overview(card_paths, output_root / "action_img2txt_review_overview.png")
    write_json(output_root / "selected_samples.json", samples)
    (output_root / "selected_samples.md").write_text(build_markdown(samples, output_root), encoding="utf-8")

    print(json.dumps({
        "output_root": str(output_root.resolve()),
        "overview": str((output_root / "action_img2txt_review_overview.png").resolve()),
        "sample_count": len(samples),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
