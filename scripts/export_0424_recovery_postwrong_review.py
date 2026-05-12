import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
from PIL import Image, ImageDraw, ImageFont


SELECTED_IDS = [
    "027_recovery_chain",
    "054_recovery_chain",
    "072_recovery_chain",
    "091_recovery_chain",
    "022_recovery_chain",
    "032_recovery_chain",
]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def remove_tree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/segoeui.ttf"),
        Path("C:/Windows/Fonts/msyh.ttc"),
    ]
    for candidate in candidates:
        if candidate.exists():
            try:
                return ImageFont.truetype(str(candidate), size)
            except OSError:
                pass
    return ImageFont.load_default()


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> List[str]:
    if not text:
        return [""]
    words = text.split()
    if not words:
        return [text]
    lines: List[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        bbox = draw.textbbox((0, 0), candidate, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def read_frame(video_path: Path, frame_index: int) -> Image.Image:
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(frame_index, 0))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Unable to read frame {frame_index} from {video_path}")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame)


def save_frame(video_path: Path, frame_index: int, output_path: Path) -> None:
    image = read_frame(video_path, frame_index)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=95)


def extract_clip(video_path: Path, start_frame: int, end_frame: int, output_path: Path) -> Dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 720)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1280)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    start_frame = max(0, min(start_frame, max(total_frames - 1, 0)))
    end_frame = max(start_frame, min(end_frame, max(total_frames - 1, 0)))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_idx = start_frame
    while frame_idx <= end_frame:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        writer.write(frame)
        frame_idx += 1
    cap.release()
    writer.release()
    return {
        "fps": fps,
        "width": width,
        "height": height,
        "total_frames": total_frames,
        "start_frame": start_frame,
        "end_frame": end_frame,
    }


def make_labeled_thumb(video_path: Path, frame_index: Optional[int], label: str, output_path: Path) -> Optional[Path]:
    if frame_index is None:
        return None
    image = read_frame(video_path, frame_index).convert("RGB")
    image.thumbnail((260, 420))
    canvas = Image.new("RGB", (300, 500), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(20)
    body_font = load_font(18)
    canvas.paste(image, ((300 - image.width) // 2, 56))
    draw.rounded_rectangle((16, 14, 180, 46), radius=14, fill=(175, 54, 90))
    draw.text((28, 21), label, fill="white", font=title_font)
    draw.text((16, 464), f"frame {frame_index}", fill=(70, 70, 70), font=body_font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=95)
    return output_path


def render_sample_card(sample: Dict[str, Any], thumb_paths: Sequence[Tuple[str, Optional[Path]]], output_path: Path) -> None:
    card = Image.new("RGB", (1760, 880), "white")
    draw = ImageDraw.Draw(card)
    title_font = load_font(34)
    subtitle_font = load_font(24)
    body_font = load_font(22)
    small_font = load_font(18)

    draw.rounded_rectangle((18, 18, 1742, 862), radius=24, outline=(210, 210, 210), width=2)
    draw.text((38, 34), f"{sample['origin_qa_id']}  Recovery Review", fill=(40, 40, 40), font=title_font)
    draw.text((38, 84), f"Scenario: {sample['scenario_family']}  |  Video: {sample['video_name']}", fill=(110, 110, 110), font=subtitle_font)

    left_box = (38, 132, 1060, 842)
    right_box = (1084, 132, 1722, 842)
    draw.rounded_rectangle(left_box, radius=24, outline=(220, 220, 220), width=2)
    draw.rounded_rectangle(right_box, radius=24, outline=(220, 220, 220), width=2)

    draw.text((56, 150), "Question", fill=(175, 54, 90), font=subtitle_font)
    y = 190
    for line in wrap_text(draw, sample["query"], body_font, 970):
        draw.text((56, y), line, fill=(35, 35, 35), font=body_font)
        y += 32

    y += 8
    draw.text((56, y), "GT", fill=(175, 54, 90), font=subtitle_font)
    y += 40
    gt_lines = [
        f"wrong_action: {sample['GT'].get('wrong_action')}",
        f"post_wrong_signal: {sample['GT'].get('post_wrong_signal')}",
        f"fix_steps: {' -> '.join(sample['GT'].get('fix_steps') or [])}",
        f"post_fix_signal: {sample['GT'].get('post_fix_signal')}",
    ]
    for line in gt_lines:
        for wrapped in wrap_text(draw, line, body_font, 970):
            draw.text((56, y), wrapped, fill=(35, 35, 35), font=body_font)
            y += 30

    if sample.get("recovery_chain_audit_note"):
        y += 10
        draw.text((56, y), "Audit note", fill=(175, 54, 90), font=subtitle_font)
        y += 40
        for line in wrap_text(draw, sample["recovery_chain_audit_note"], body_font, 970):
            draw.text((56, y), line, fill=(60, 60, 60), font=body_font)
            y += 30

    draw.text((1104, 150), "Key Frames", fill=(175, 54, 90), font=subtitle_font)
    thumb_slots = [
        (1104, 192),
        (1414, 192),
        (1104, 510),
        (1414, 510),
    ]
    for (label, thumb_path), (x, y_slot) in zip(thumb_paths, thumb_slots):
        if thumb_path is None:
            continue
        thumb = Image.open(thumb_path).convert("RGB")
        card.paste(thumb, (x, y_slot))

    extra_y = 796
    draw.text(
        (1104, extra_y),
        f"probe frame: {sample.get('probe_frame')}  |  clip: {sample['clip_start_frame']}-{sample['clip_end_frame']}",
        fill=(90, 90, 90),
        font=small_font,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    card.save(output_path, quality=95)


def render_overview(card_paths: Sequence[Path], output_path: Path) -> None:
    cards = [Image.open(path).convert("RGB") for path in card_paths]
    if not cards:
        return
    columns = 2
    card_w, card_h = cards[0].size
    rows = math.ceil(len(cards) / columns)
    canvas = Image.new("RGB", (columns * card_w + 60, rows * card_h + 60), (245, 247, 249))
    for index, card in enumerate(cards):
        row = index // columns
        col = index % columns
        x = 20 + col * card_w
        y = 20 + row * card_h
        canvas.paste(card, (x, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=95)


def span_anchor(span: Optional[Dict[str, Any]]) -> Optional[int]:
    if not span:
        return None
    start = span.get("start")
    if start is None:
        return None
    return int(start)


def build_sample(row: Dict[str, Any], video_path: Path, output_root: Path) -> Dict[str, Any]:
    origin_qa_id = row["origin_qa_id"]
    wrong_span = row.get("wrong_action_span") or {}
    post_wrong_span = row.get("post_wrong_signal_span") or {}
    fix_spans = row.get("fix_action_spans") or []
    post_fix_span = row.get("post_fix_signal_span") or {}

    wrong_end = int(wrong_span.get("end") or wrong_span.get("start") or 0)
    first_fix_start = int(fix_spans[0].get("start") or 0) if fix_spans else int(row["source_span"]["end"])
    probe_frame = max(wrong_end, (wrong_end + first_fix_start) // 2)
    post_wrong_frame = span_anchor(post_wrong_span)
    post_fix_frame = span_anchor(post_fix_span)
    wrong_frame = wrong_end
    fix_frame = int(fix_spans[0].get("start") or 0) if fix_spans else None

    clip_start = max(0, int(row["source_span"]["start"]) - 24)
    clip_end = max(int(row["source_span"]["end"]), post_fix_frame or 0) + 24
    clip_path = output_root / "clips" / f"{origin_qa_id}.mp4"
    clip_meta = extract_clip(video_path, clip_start, clip_end, clip_path)

    thumbs: List[Tuple[str, Optional[Path]]] = []
    thumb_specs = [
        ("wrong_action", wrong_frame),
        ("post_wrong_signal", post_wrong_frame),
        ("probe", probe_frame),
        ("post_fix_signal", post_fix_frame),
    ]
    for label, frame_index in thumb_specs:
        thumb_path = output_root / "thumbs" / f"{origin_qa_id}_{label}.jpg"
        saved = make_labeled_thumb(video_path, frame_index, label, thumb_path)
        thumbs.append((label, saved))

    card_path = output_root / "cards" / f"{origin_qa_id}.jpg"
    sample = {
        "origin_qa_id": origin_qa_id,
        "video_name": Path(row["query_video_path"]).name,
        "scenario_family": row["scenario_family"],
        "query": row["query"],
        "GT": row["GT"],
        "canonical_answer": row["canonical_answer"],
        "source_span": row["source_span"],
        "wrong_action_origin_qa_id": row.get("wrong_action_origin_qa_id"),
        "wrong_action_span": row.get("wrong_action_span"),
        "post_wrong_signal_origin_qa_id": row.get("post_wrong_signal_origin_qa_id"),
        "post_wrong_signal_span": row.get("post_wrong_signal_span"),
        "fix_action_origin_qa_ids": row.get("fix_action_origin_qa_ids"),
        "fix_action_spans": row.get("fix_action_spans"),
        "post_fix_signal_origin_qa_id": row.get("post_fix_signal_origin_qa_id"),
        "post_fix_signal_span": row.get("post_fix_signal_span"),
        "recovery_chain_audit_note": row.get("recovery_chain_audit_note"),
        "wrong_frame": wrong_frame,
        "post_wrong_frame": post_wrong_frame,
        "probe_frame": probe_frame,
        "fix_frame": fix_frame,
        "post_fix_frame": post_fix_frame,
        "clip_path": str(clip_path.relative_to(output_root)).replace("\\", "/"),
        "clip_start_frame": clip_meta["start_frame"],
        "clip_end_frame": clip_meta["end_frame"],
        "thumbs": [
            {
                "label": label,
                "path": (str(path.relative_to(output_root)).replace("\\", "/") if path is not None else None),
            }
            for label, path in thumbs
        ],
        "card_path": str(card_path.relative_to(output_root)).replace("\\", "/"),
    }
    render_sample_card(sample, thumbs, card_path)
    return sample


def build_markdown(samples: Sequence[Dict[str, Any]]) -> str:
    lines = ["# Recovery Post-Wrong Review", ""]
    for sample in samples:
        lines.append(f"## {sample['origin_qa_id']}")
        lines.append(f"- Scenario: `{sample['scenario_family']}`")
        lines.append(f"- Video: `{sample['video_name']}`")
        lines.append(f"- Query: {sample['query']}")
        lines.append(f"- GT: `{json.dumps(sample['GT'], ensure_ascii=False)}`")
        lines.append(f"- Wrong action span: `{sample.get('wrong_action_span')}`")
        lines.append(f"- Post-wrong signal span: `{sample.get('post_wrong_signal_span')}`")
        lines.append(f"- Fix spans: `{sample.get('fix_action_spans')}`")
        lines.append(f"- Post-fix signal span: `{sample.get('post_fix_signal_span')}`")
        if sample.get("recovery_chain_audit_note"):
            lines.append(f"- Audit note: {sample['recovery_chain_audit_note']}")
        lines.append(f"- Card: `{sample['card_path']}`")
        lines.append(f"- Clip: `{sample['clip_path']}`")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a small review bundle for 0424 recovery chains with post_wrong_signal.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Root of the generated multiform dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for the review bundle. Defaults to <dataset-root>/recovery_postwrong_review_v1.",
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root
    output_dir = args.output_dir or (dataset_root / "recovery_postwrong_review_v1")
    remove_tree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    recovery_rows = load_json(dataset_root / "recovery" / "video2txt" / "openqa.json")["data"]
    chains = [row for row in recovery_rows if row["qa_type"] == "recovery_chain"]
    chain_by_id = {row["origin_qa_id"]: row for row in chains}

    selected_rows: List[Dict[str, Any]] = []
    for qa_id in SELECTED_IDS:
        row = chain_by_id.get(qa_id)
        if row is not None:
            selected_rows.append(row)

    selected_samples: List[Dict[str, Any]] = []
    for row in selected_rows:
        video_name = Path(row["query_video_path"]).name
        video_path = dataset_root / "recovery" / "video2txt" / "videos" / video_name
        selected_samples.append(build_sample(row, video_path, output_dir))

    overview_path = output_dir / "recovery_chain_review_overview.png"
    render_overview([output_dir / sample["card_path"] for sample in selected_samples], overview_path)

    write_json(output_dir / "all_recovery_chains.json", {"data": chains})
    write_json(output_dir / "selected_samples.json", {"data": selected_samples})
    (output_dir / "selected_samples.md").write_text(build_markdown(selected_samples), encoding="utf-8")


if __name__ == "__main__":
    main()
