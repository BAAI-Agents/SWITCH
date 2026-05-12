from __future__ import annotations

import argparse
import csv
import json
import math
import re
import textwrap
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlparse

import cv2
from PIL import Image, ImageDraw, ImageFont


LIGHTING_FOLDER = "\u706f\u5149\u4e0e\u7167\u660e\u63a7\u5236-\u95ee\u5377"
DEFAULT_DATASET_DIR = (
    Path("annotations") / "latest" / LIGHTING_FOLDER / LIGHTING_FOLDER
)
DEFAULT_JSON = DEFAULT_DATASET_DIR / "SWITCHAction_3_Lighting.json"
DEFAULT_OUTPUT_DIR = DEFAULT_DATASET_DIR / "single_frame_results"


def safe_name(value: str, max_len: int = 96) -> str:
    value = re.sub(r"[^\w.-]+", "_", value, flags=re.UNICODE).strip("_")
    return value[:max_len] or "untitled"


def first_text(result: dict) -> str:
    text = result.get("meta", {}).get("text", [])
    if isinstance(text, list):
        return "; ".join(str(x) for x in text if x is not None)
    if text is None:
        return ""
    return str(text)


def detect_video_data_key(item: dict, requested_key: str | None = None) -> str:
    data = item.get("data", {})
    if requested_key and requested_key in data:
        return requested_key
    for key, value in data.items():
        if key == "meta":
            continue
        if isinstance(value, str) and value.lower().split("?")[0].endswith((".mp4", ".mov", ".avi", ".mkv")):
            return key
    for key in data:
        if key != "meta":
            return key
    return ""


def video_name_from_item(item: dict, data_key: str | None = None) -> str:
    key = detect_video_data_key(item, data_key)
    url = item.get("data", {}).get(key, "")
    parsed = urlparse(url)
    return Path(parsed.path).name or Path(url).name


def item_video_meta(item: dict, data_key: str | None = None) -> dict:
    key = detect_video_data_key(item, data_key)
    return item.get("data", {}).get("meta", {}).get(key, {})


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
    ]
    for path in candidates:
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size=size)
            except OSError:
                pass
    return ImageFont.load_default()


def frame_count_from_cap(cap: cv2.VideoCapture) -> int:
    value = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
    return max(0, value)


def fps_from_cap(cap: cv2.VideoCapture, fallback: float | None = None) -> float:
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    if fps > 0:
        return fps
    if fallback and fallback > 0:
        return float(fallback)
    return 30.0


class FrameReaderCache:
    def __init__(self) -> None:
        self._caps: dict[str, cv2.VideoCapture] = {}
        self._meta: dict[str, dict] = {}

    def _open(self, video_path: Path, fallback_fps: float | None = None) -> tuple[cv2.VideoCapture | None, dict]:
        key = str(video_path)
        if key in self._caps:
            return self._caps[key], self._meta[key]

        cap = cv2.VideoCapture(key)
        if not cap.isOpened():
            meta = {"ok": False, "error": "video_open_failed"}
            self._meta[key] = meta
            return None, meta

        meta = {
            "ok": True,
            "fps": fps_from_cap(cap, fallback_fps),
            "frame_count": frame_count_from_cap(cap),
        }
        self._caps[key] = cap
        self._meta[key] = meta
        return cap, meta

    def read(
        self,
        video_path: Path,
        one_based_frame: int,
        fallback_fps: float | None = None,
    ) -> tuple[Image.Image | None, dict]:
        cap, open_meta = self._open(video_path, fallback_fps)
        if cap is None:
            return None, dict(open_meta)

        count = int(open_meta.get("frame_count") or 0)
        fps = float(open_meta.get("fps") or fallback_fps or 30.0)
        if count <= 0:
            return None, {"ok": False, "error": "empty_video", "fps": fps, "frame_count": count}

        clamped_one_based = max(1, min(int(one_based_frame), count))
        zero_based = clamped_one_based - 1
        actual_zero_based = zero_based
        cap.set(cv2.CAP_PROP_POS_FRAMES, zero_based)
        ok, frame = cap.read()
        if not ok or frame is None:
            for fallback_zero_based in range(min(zero_based, count - 1), max(-1, zero_based - 30), -1):
                cap.set(cv2.CAP_PROP_POS_FRAMES, fallback_zero_based)
                ok, frame = cap.read()
                if ok and frame is not None:
                    actual_zero_based = fallback_zero_based
                    break
        if not ok or frame is None:
            return None, {
                "ok": False,
                "error": "frame_read_failed",
                "fps": fps,
                "frame_count": count,
                "requested_frame": one_based_frame,
                "opencv_frame_index": zero_based,
            }

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        return image, {
            "ok": True,
            "fps": fps,
            "frame_count": count,
            "requested_frame": one_based_frame,
            "clamped_frame": clamped_one_based,
            "actual_frame": actual_zero_based + 1,
            "opencv_frame_index": actual_zero_based,
        }

    def close(self) -> None:
        for cap in self._caps.values():
            cap.release()
        self._caps.clear()


def annotated_frame_to_30fps_frame(annotated_frame: int, source_fps: float) -> tuple[int, float]:
    time_sec = max(0.0, (int(annotated_frame) - 1) / source_fps)
    target_frame = int(round(time_sec * 30.0)) + 1
    return max(1, target_frame), time_sec


def target_30fps_frame_to_source_frame(target_frame: int, source_fps: float) -> tuple[int, float]:
    time_sec = max(0.0, (int(target_frame) - 1) / 30.0)
    source_frame = int(round(time_sec * source_fps)) + 1
    return max(1, source_frame), time_sec


def write_image(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, quality=92)


def wrap_lines(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, width: int) -> list[str]:
    if not text:
        return [""]
    words = text.split()
    if not words:
        return [text]
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        bbox = draw.textbbox((0, 0), candidate, font=font)
        if bbox[2] - bbox[0] <= width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def make_card(image: Image.Image, record: dict, mode: str, output_path: Path) -> None:
    card_w, card_h = 860, 720
    pad = 28
    title_h = 150
    footer_h = 145
    image_box = (pad, title_h, card_w - pad, card_h - footer_h)

    canvas = Image.new("RGB", (card_w, card_h), "#f8faf8")
    draw = ImageDraw.Draw(canvas)
    font_title = load_font(34)
    font_mid = load_font(24)
    font_small = load_font(21)

    color = "#1e8a78" if mode == "native_fps" else "#3466b3"
    draw.rectangle((0, 0, card_w, 14), fill=color)
    draw.text((pad, 28), f"{record['label']} | {mode}", fill="#1f2937", font=font_title)
    draw.text(
        (pad, 74),
        f"data_id={record['data_id']}  video={record['video_name']}",
        fill="#4b5563",
        font=font_small,
    )
    draw.text(
        (pad, 108),
        f"annotated_frame={record['annotated_frame']}  source_fps={record['source_fps']:.6g}",
        fill="#4b5563",
        font=font_small,
    )

    box_w = image_box[2] - image_box[0]
    box_h = image_box[3] - image_box[1]
    thumb = image.copy()
    thumb.thumbnail((box_w, box_h), Image.Resampling.LANCZOS)
    x = image_box[0] + (box_w - thumb.width) // 2
    y = image_box[1] + (box_h - thumb.height) // 2
    draw.rounded_rectangle(image_box, radius=18, outline="#d1d5db", width=3, fill="#eef2f7")
    canvas.paste(thumb, (x, y))

    footer_y = card_h - footer_h + 20
    if mode == "fps30_resampled":
        detail = (
            f"30fps_frame={record['frame_30fps']}  "
            f"time={record['time_sec_30fps']:.3f}s  "
            f"source_frame_for_30fps={record['source_frame_for_30fps']}"
        )
    else:
        detail = (
            f"native_frame={record['native_frame']}  "
            f"time={record['time_sec_native']:.3f}s  "
            f"opencv_index={record['native_opencv_frame_index']}"
        )
    draw.text((pad, footer_y), detail, fill="#374151", font=font_mid)

    text = record.get("text") or "(no text)"
    y2 = footer_y + 38
    for line in wrap_lines(draw, f"text: {text}", font_small, card_w - pad * 2)[:3]:
        draw.text((pad, y2), line, fill="#374151", font=font_small)
        y2 += 28

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92)


def make_overviews(records: list[dict], mode: str, output_dir: Path) -> None:
    cards_by_label: dict[str, list[Path]] = defaultdict(list)
    for record in records:
        value = record.get(f"{mode}_card") or ""
        if not value:
            continue
        path = Path(value)
        if path.exists() and path.is_file():
            cards_by_label[record["label"]].append(path)

    overview_dir = output_dir / "overviews" / mode
    overview_dir.mkdir(parents=True, exist_ok=True)
    for label, paths in sorted(cards_by_label.items()):
        cols = 4
        thumb_w, thumb_h = 360, 300
        rows = math.ceil(len(paths) / cols)
        canvas = Image.new("RGB", (cols * thumb_w, rows * thumb_h + 70), "#f5f7f4")
        draw = ImageDraw.Draw(canvas)
        font = load_font(28)
        draw.text((20, 20), f"{label} | {mode} | {len(paths)} frames", fill="#1f2937", font=font)
        for idx, path in enumerate(paths):
            img = Image.open(path).convert("RGB")
            img.thumbnail((thumb_w - 18, thumb_h - 18), Image.Resampling.LANCZOS)
            x0 = (idx % cols) * thumb_w + 9
            y0 = 70 + (idx // cols) * thumb_h + 9
            canvas.paste(img, (x0, y0))
        canvas.save(overview_dir / f"{safe_name(label)}.png", quality=92)


def collect_single_frame_records(json_path: Path, dataset_dir: Path, data_key: str | None = None) -> list[dict]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    records: list[dict] = []

    for item in data:
        item_id = item.get("id")
        current_data_key = detect_video_data_key(item, data_key)
        video_name = video_name_from_item(item, current_data_key)
        video_path = dataset_dir / video_name
        meta = item_video_meta(item, current_data_key)
        meta_fps = float(meta.get("fps") or 0)
        source_fps = meta_fps if meta_fps > 0 else 30.0
        meta_frame_count = int(round(meta.get("total_frames") or 0))

        annotation = (item.get("annotations") or [{}])[0]
        result_list = annotation.get("result") or []

        data_id = ""
        for result in result_list:
            labels = result.get("value", {}).get("timelinelabels", [])
            if "data_id" in labels:
                data_id = first_text(result)
                break
        if not data_id:
            data_id = Path(video_name).stem

        for result_index, result in enumerate(result_list):
            if result.get("type") != "timelinelabels":
                continue
            value = result.get("value", {})
            labels = value.get("timelinelabels", [])
            ranges = value.get("ranges", [])
            for range_index, range_value in enumerate(ranges):
                start = int(round(range_value.get("start", 0)))
                end = int(round(range_value.get("end", 0)))
                if start != end:
                    continue
                for label in labels:
                    records.append(
                        {
                            "item_id": item_id,
                            "video_data_key": current_data_key,
                            "annotation_id": annotation.get("id"),
                            "result_id": result.get("id", ""),
                            "result_index": result_index,
                            "range_index": range_index,
                            "data_id": data_id,
                            "video_name": video_name,
                            "video_path": str(video_path),
                            "label": label,
                            "text": first_text(result),
                            "annotated_frame": start,
                            "native_frame": start,
                            "source_fps": source_fps,
                            "meta_fps": meta_fps,
                            "meta_frame_count": meta_frame_count,
                        }
                    )

    return records


def export_records(records: list[dict], output_dir: Path) -> list[dict]:
    enriched: list[dict] = []
    reader = FrameReaderCache()
    try:
        for record in records:
            video_path = Path(record["video_path"])
            label_dir = safe_name(record["label"])
            prefix = (
                f"{safe_name(str(record['data_id']))}_"
                f"{safe_name(Path(record['video_name']).stem)}_"
                f"f{int(record['annotated_frame']):06d}_"
                f"{safe_name(record['label'])}_"
                f"r{record['result_index']:03d}"
            )

            native_image, native_meta = reader.read(
                video_path,
                int(record["native_frame"]),
                fallback_fps=float(record["source_fps"]),
            )
            record["native_read_ok"] = bool(native_meta.get("ok"))
            record["native_read_error"] = native_meta.get("error", "")
            record["native_frame_count"] = native_meta.get("frame_count", record.get("meta_frame_count", 0))
            record["native_clamped_frame"] = native_meta.get("clamped_frame", record["native_frame"])
            record["native_opencv_frame_index"] = native_meta.get("opencv_frame_index", max(0, record["native_frame"] - 1))
            record["time_sec_native"] = max(0.0, (int(record["native_frame"]) - 1) / float(record["source_fps"]))

            frame_30fps, original_time = annotated_frame_to_30fps_frame(
                int(record["annotated_frame"]),
                float(record["source_fps"]),
            )
            source_frame_for_30fps, time_sec_30fps = target_30fps_frame_to_source_frame(
                frame_30fps,
                float(record["source_fps"]),
            )
            record["frame_30fps"] = frame_30fps
            record["time_sec_original_label"] = original_time
            record["time_sec_30fps"] = time_sec_30fps
            record["source_frame_for_30fps"] = source_frame_for_30fps

            resampled_image, resampled_meta = reader.read(
                video_path,
                source_frame_for_30fps,
                fallback_fps=float(record["source_fps"]),
            )
            record["fps30_read_ok"] = bool(resampled_meta.get("ok"))
            record["fps30_read_error"] = resampled_meta.get("error", "")
            record["fps30_clamped_source_frame"] = resampled_meta.get("clamped_frame", source_frame_for_30fps)
            record["fps30_opencv_frame_index"] = resampled_meta.get("opencv_frame_index", max(0, source_frame_for_30fps - 1))

            if native_image is not None:
                native_frame_path = output_dir / "frames_native_fps" / label_dir / f"{prefix}.jpg"
                native_card_path = output_dir / "cards_native_fps" / label_dir / f"{prefix}.jpg"
                write_image(native_image, native_frame_path)
                make_card(native_image, record, "native_fps", native_card_path)
                record["native_fps_frame_image"] = str(native_frame_path)
                record["native_fps_card"] = str(native_card_path)
            else:
                record["native_fps_frame_image"] = ""
                record["native_fps_card"] = ""

            if resampled_image is not None:
                fps30_frame_path = output_dir / "frames_30fps_resampled" / label_dir / f"{prefix}_30fps_f{frame_30fps:06d}.jpg"
                fps30_card_path = output_dir / "cards_30fps_resampled" / label_dir / f"{prefix}_30fps_f{frame_30fps:06d}.jpg"
                write_image(resampled_image, fps30_frame_path)
                make_card(resampled_image, record, "fps30_resampled", fps30_card_path)
                record["fps30_resampled_frame_image"] = str(fps30_frame_path)
                record["fps30_resampled_card"] = str(fps30_card_path)
            else:
                record["fps30_resampled_frame_image"] = ""
                record["fps30_resampled_card"] = ""

            enriched.append(record)
    finally:
        reader.close()

    make_overviews(enriched, "native_fps", output_dir)
    make_overviews(enriched, "fps30_resampled", output_dir)
    return enriched


def write_csv(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "label",
        "data_id",
        "video_name",
        "text",
        "annotated_frame",
        "source_fps",
        "native_frame",
        "native_opencv_frame_index",
        "time_sec_native",
        "frame_30fps",
        "source_frame_for_30fps",
        "fps30_opencv_frame_index",
        "time_sec_30fps",
        "native_fps_frame_image",
        "fps30_resampled_frame_image",
        "native_fps_card",
        "fps30_resampled_card",
        "result_id",
        "item_id",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def rel(path: str | Path, base: Path) -> str:
    p = Path(path)
    try:
        return p.relative_to(base).as_posix()
    except ValueError:
        return p.as_posix()


def write_markdown(records: list[dict], output_dir: Path) -> None:
    counts = Counter(record["label"] for record in records)
    lines = [
        "# Lighting Single-Frame Annotation Export",
        "",
        "This bundle exports every `start == end` timeline label in two ways:",
        "",
        "- `native_fps`: read the annotated frame directly on the source-video frame axis.",
        "- `fps30_resampled`: map the annotation timestamp to a 30fps timeline, then read the corresponding source frame.",
        "",
        "Frame-number convention: Label Studio frame numbers are treated as 1-based; OpenCV indices are recorded as 0-based.",
        "",
        "## Counts",
        "",
        "| label | count | native overview | 30fps overview |",
        "|---|---:|---|---|",
    ]
    for label, count in sorted(counts.items()):
        label_file = f"{safe_name(label)}.png"
        lines.append(
            f"| `{label}` | {count} | "
            f"[native](overviews/native_fps/{label_file}) | "
            f"[30fps](overviews/fps30_resampled/{label_file}) |"
        )

    lines.extend(
        [
            "",
            "## Records",
            "",
            "| label | data_id | video | text | frame | 30fps frame | native card | 30fps card |",
            "|---|---|---|---|---:|---:|---|---|",
        ]
    )
    for record in records:
        text = (record.get("text") or "").replace("|", "\\|")
        if len(text) > 96:
            text = text[:93] + "..."
        native_card = rel(record.get("native_fps_card", ""), output_dir)
        fps30_card = rel(record.get("fps30_resampled_card", ""), output_dir)
        lines.append(
            f"| `{record['label']}` | `{record['data_id']}` | `{record['video_name']}` | "
            f"{text} | {record['annotated_frame']} | {record['frame_30fps']} | "
            f"[native]({native_card}) | [30fps]({fps30_card}) |"
        )

    (output_dir / "single_frame_annotations.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--data-key", default=None)
    args = parser.parse_args()

    records = collect_single_frame_records(args.json, args.dataset_dir, args.data_key)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    enriched = export_records(records, args.output_dir)

    json_path = args.output_dir / "single_frame_annotations.json"
    json_path.write_text(json.dumps(enriched, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(enriched, args.output_dir / "single_frame_annotations.csv")
    write_markdown(enriched, args.output_dir)

    counts = Counter(record["label"] for record in enriched)
    summary = {
        "json": str(args.json),
        "dataset_dir": str(args.dataset_dir),
        "output_dir": str(args.output_dir),
        "total_single_frame_records": len(enriched),
        "counts_by_label": dict(sorted(counts.items())),
        "native_read_failures": sum(not record["native_read_ok"] for record in enriched),
        "fps30_read_failures": sum(not record["fps30_read_ok"] for record in enriched),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
