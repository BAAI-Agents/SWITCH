#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter
from html import escape
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


VIDEO_SAMPLE_IDS = [
    "011_action_005",
    "015_action_005",
    "014_verify_006",
    "027_recovery_chain",
    "054_recovery_chain",
]

IMAGE_SAMPLE_IDS = [
    "011_state_007",
    "012_state_016",
    "013_final_002",
    "030_final_002",
    "015_state_008",
    "027_state_011",
    "035_final_002",
    "014_state_008",
    "024_final_002",
    "078_final_002",
]

SELECTED_SAMPLE_IDS = VIDEO_SAMPLE_IDS + IMAGE_SAMPLE_IDS

SELECTION_REASONS = {
    "011_action_005": "医疗自助机场景，动作跨度长，适合检查多页面切换动作是否标得过宽。",
    "015_action_005": "地铁售票机场景，包含明显的物理交互动作，适合看动作段与结果段是否混淆。",
    "014_verify_006": "电梯场景中的验证动作，适合检查 verification_action 是否和 verification_state 混在一起。",
    "027_recovery_chain": "地铁售票机场景的恢复链，包含错误价格和修正动作，适合看 recovery 闭环是否完整。",
    "054_recovery_chain": "医疗自助机场景的恢复链，适合和地铁 recovery 对比标注风格是否一致。",
    "011_state_007": "医疗自助机的 UI 成功信号，适合看单帧 state 是否足够明确。",
    "012_state_016": "医疗自助机的物理状态信号，适合看 physical_world_state 与 UI state 的区别。",
    "013_final_002": "医疗自助机 final_state，结果是 UI 文本，适合检查 final_state 是否过早。",
    "030_final_002": "医疗自助机 final_state，结果是取号成功，适合看物理终态是否清晰。",
    "015_state_008": "地铁售票机 UI state，适合看屏幕信息类单帧信号。",
    "027_state_011": "地铁售票机 physical state，适合看出票瞬间是否适合作为单帧证据。",
    "035_final_002": "另一条地铁售票机 final_state，便于比较同类场景的一致性。",
    "014_state_008": "电梯单帧状态“门已关闭”，适合检查中间态是否容易歧义。",
    "024_final_002": "电梯 final_state，适合检查到达楼层是否和 overall success 对齐。",
    "078_final_002": "标到 scenario_family=other 的样本，适合检查异常分组是否仍然合理。",
}

FONT = cv2.FONT_HERSHEY_SIMPLEX


def load_multiform_rows(json_path: Path, form_name: str, task_name: str) -> Dict[str, Dict[str, Any]]:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    rows = payload.get("data", [])
    indexed: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        indexed[row["origin_qa_id"]] = {
            **row,
            "_form_name": form_name,
            "_task_name": task_name,
        }
    return indexed


def load_multiform_index(multiform_root: Path) -> Dict[str, Dict[str, Dict[str, Any]]]:
    index: Dict[str, Dict[str, Dict[str, Any]]] = {}
    task_forms = [
        ("action", "video2txt", ("openqa", "vqa")),
        ("verification_action", "video2txt", ("openqa",)),
        ("verification_state", "video2txt", ("openqa", "vqa")),
        ("final_state", "video2txt", ("openqa", "vqa")),
        ("recovery", "video2txt", ("openqa",)),
    ]
    for task_name, subdir, forms in task_forms:
        for form_name in forms:
            path = multiform_root / task_name / subdir / f"{form_name}.json"
            rows = load_multiform_rows(path, form_name, task_name)
            for qa_id, row in rows.items():
                index.setdefault(qa_id, {})[form_name] = row
    return index


def load_video_metadata(qa_candidates_path: Path) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    payload = json.loads(qa_candidates_path.read_text(encoding="utf-8"))
    video_map: Dict[str, Dict[str, Any]] = {}
    qa_map: Dict[str, Dict[str, Any]] = {}
    for video in payload["videos"]:
        data_id = video["data_id"]
        video_map[data_id] = {
            "data_id": data_id,
            "video_name": video["video_name"],
            "scenario_family": video["scenario_family"],
            "main_task": video["main_task"],
            "main_verification": video["main_verification"],
        }
        for qa in video["qa_candidates"]:
            qa_map[qa["qa_id"]] = {
                "data_id": data_id,
                "scenario_family": video["scenario_family"],
                "main_task": video["main_task"],
                "main_verification": video["main_verification"],
                "task_family": qa["task_family"],
                "qa_type": qa["qa_type"],
                "source_label": qa["source_label"],
                "source_span": qa["source_span"],
                "answer": qa["answer"],
            }
    return video_map, qa_map


def parse_data_id(origin_qa_id: str) -> str:
    return origin_qa_id.split("_", 1)[0]


def safe_name(text: str) -> str:
    cleaned = []
    for char in text:
        if char.isalnum() or char in "._-":
            cleaned.append(char)
        else:
            cleaned.append("_")
    return "".join(cleaned).strip("._") or "item"


def choose_multiform_root(annotation_root: Path) -> Path:
    preferred = annotation_root / "hf_innovative_qa_v2_multiform_state_aligned_v1"
    if preferred.exists():
        return preferred
    return annotation_root / "hf_innovative_qa_v2_multiform"


def open_video(video_path: Path) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    return capture


def get_video_stats(video_path: Path) -> Dict[str, Any]:
    capture = open_video(video_path)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    capture.release()
    return {
        "total_frames": total_frames,
        "fps": fps,
        "width": width,
        "height": height,
    }


def clamp_frame(frame_index: int, total_frames: int) -> int:
    if total_frames <= 0:
        return max(0, frame_index)
    return max(0, min(frame_index, total_frames - 1))


def read_frame(video_path: Path, frame_index: int) -> np.ndarray:
    capture = open_video(video_path)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    target = clamp_frame(frame_index, total_frames)
    capture.set(cv2.CAP_PROP_POS_FRAMES, target)
    ok, frame = capture.read()
    if not ok and target > 0:
        capture.set(cv2.CAP_PROP_POS_FRAMES, target - 1)
        ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Unable to read frame {frame_index} from {video_path}")
    return frame


def resize_for_panel(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    src_h, src_w = frame.shape[:2]
    scale = min(width / src_w, height / src_h)
    new_w = max(1, int(src_w * scale))
    new_h = max(1, int(src_h * scale))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((height, width, 3), 247, dtype=np.uint8)
    y = (height - new_h) // 2
    x = (width - new_w) // 2
    canvas[y : y + new_h, x : x + new_w] = resized
    return canvas


def add_caption(frame: np.ndarray, title: str, subtitle: str) -> np.ndarray:
    h, w = frame.shape[:2]
    pad_top = 42
    pad_bottom = 34
    canvas = np.full((h + pad_top + pad_bottom, w, 3), 255, dtype=np.uint8)
    canvas[pad_top : pad_top + h, :] = frame
    cv2.putText(canvas, title, (10, 24), FONT, 0.62, (20, 40, 80), 2, cv2.LINE_AA)
    cv2.putText(canvas, subtitle, (10, h + pad_top + 22), FONT, 0.50, (90, 90, 90), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (0, 0), (w - 1, h + pad_top + pad_bottom - 1), (220, 224, 230), 2)
    return canvas


def write_image(output_path: Path, frame: np.ndarray) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), frame):
        raise RuntimeError(f"Unable to write image: {output_path}")


def extract_frame_image(video_path: Path, frame_index: int, output_path: Path) -> None:
    write_image(output_path, read_frame(video_path, frame_index))


def extract_clip(video_path: Path, start_frame: int, end_frame: int, output_path: Path) -> Tuple[int, int]:
    if end_frame < start_frame:
        raise ValueError(f"Invalid clip span: {start_frame}-{end_frame}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    capture = open_video(video_path)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    start_frame = clamp_frame(start_frame, total_frames)
    end_frame = clamp_frame(end_frame, total_frames)
    if end_frame < start_frame:
        end_frame = start_frame
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Unable to create video writer: {output_path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current = start_frame
    try:
        while current <= end_frame:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Unable to read frame {current} from {video_path}")
            writer.write(frame)
            current += 1
    finally:
        writer.release()
        capture.release()
    return start_frame, end_frame


def evenly_spaced_frames(start_frame: int, end_frame: int, count: int) -> List[int]:
    if count <= 1 or end_frame <= start_frame:
        return [start_frame]
    values = np.linspace(start_frame, end_frame, num=count)
    return [int(round(v)) for v in values]


def make_video_storyboard(
    video_path: Path,
    start_frame: int,
    end_frame: int,
    output_path: Path,
    num_frames: int = 6,
    thumb_size: Tuple[int, int] = (240, 160),
) -> List[int]:
    frame_indices = evenly_spaced_frames(start_frame, end_frame, num_frames)
    cells: List[np.ndarray] = []
    for index, frame_id in enumerate(frame_indices, start=1):
        thumb = resize_for_panel(read_frame(video_path, frame_id), thumb_size[0], thumb_size[1])
        annotated = add_caption(thumb, f"Frame {frame_id}", f"sample {index}/{len(frame_indices)}")
        cells.append(annotated)
    spacer = np.full((cells[0].shape[0], 12, 3), 255, dtype=np.uint8)
    strip = cells[0]
    for cell in cells[1:]:
        strip = np.hstack([strip, spacer, cell])
    write_image(output_path, strip)
    return frame_indices


def make_context_strip(
    video_path: Path,
    center_frame: int,
    output_path: Path,
    offsets: Sequence[int] = (-24, -12, 0, 12, 24),
    thumb_size: Tuple[int, int] = (220, 150),
) -> List[int]:
    stats = get_video_stats(video_path)
    frame_indices = [clamp_frame(center_frame + offset, stats["total_frames"]) for offset in offsets]
    cells: List[np.ndarray] = []
    for frame_id in frame_indices:
        thumb = resize_for_panel(read_frame(video_path, frame_id), thumb_size[0], thumb_size[1])
        label = "anchor" if frame_id == center_frame else "context"
        annotated = add_caption(thumb, f"Frame {frame_id}", label)
        cells.append(annotated)
    spacer = np.full((cells[0].shape[0], 12, 3), 255, dtype=np.uint8)
    strip = cells[0]
    for cell in cells[1:]:
        strip = np.hstack([strip, spacer, cell])
    write_image(output_path, strip)
    return frame_indices


def make_gallery(
    title: str,
    entries: Sequence[Tuple[str, Path]],
    output_path: Path,
    thumb_size: Tuple[int, int] = (240, 160),
    columns: int = 3,
) -> None:
    if not entries:
        return
    cards: List[np.ndarray] = []
    for label, image_path in entries:
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise RuntimeError(f"Unable to read gallery image: {image_path}")
        thumb = resize_for_panel(frame, thumb_size[0], thumb_size[1])
        card = add_caption(thumb, label, image_path.name)
        cards.append(card)

    rows: List[np.ndarray] = []
    spacer_x = np.full((cards[0].shape[0], 16, 3), 246, dtype=np.uint8)
    spacer_y = np.full((24, columns * cards[0].shape[1] + (columns - 1) * 16, 3), 246, dtype=np.uint8)
    for row_start in range(0, len(cards), columns):
        chunk = cards[row_start : row_start + columns]
        while len(chunk) < columns:
            chunk.append(np.full_like(cards[0], 246))
        row = chunk[0]
        for card in chunk[1:]:
            row = np.hstack([row, spacer_x, card])
        rows.append(row)

    canvas = rows[0]
    for row in rows[1:]:
        canvas = np.vstack([canvas, spacer_y, row])

    pad_top = 64
    full = np.full((canvas.shape[0] + pad_top + 18, canvas.shape[1], 3), 250, dtype=np.uint8)
    full[pad_top : pad_top + canvas.shape[0], :] = canvas
    cv2.putText(full, title, (10, 40), FONT, 1.0, (20, 40, 80), 2, cv2.LINE_AA)
    write_image(output_path, full)


def render_inline(value: Any) -> str:
    if value is None:
        return "`null`"
    if isinstance(value, list):
        if not value:
            return "`[]`"
        return "; ".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def html_badges(sample: Dict[str, Any]) -> str:
    items = [
        sample["scenario_family"],
        sample["task_family"],
        sample["qa_type"],
        sample.get("source_label") or "no_source_label",
        f"{sample['source_span']['start']}-{sample['source_span']['end']}",
    ]
    return "".join(f"<span class='badge'>{escape(str(item))}</span>" for item in items)


def html_text_block(title: str, value: Any, code_class: str = "") -> str:
    payload = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)
    return (
        f"<div class='text-block'>"
        f"<div class='text-title'>{escape(title)}</div>"
        f"<pre class='{code_class}'>{escape(payload or '')}</pre>"
        f"</div>"
    )


def write_json(output_path: Path, payload: List[Dict[str, Any]]) -> None:
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_markdown(output_path: Path, samples: Sequence[Dict[str, Any]]) -> None:
    scenario_counts = Counter(sample["scenario_family"] for sample in samples)
    group_counts = Counter(sample["selection_group"] for sample in samples)
    lines: List[str] = [
        "# 0424 SWITCHAction_2 Rich Review Bundle",
        "",
        "这个 bundle 用于人工检查 `annotations/0424/SWITCHAction_2` 的代表样本。",
        "视频样本保留真实 clip，并额外导出海报帧和分镜图；图片样本保留单帧图，并补充前后文拼图。",
        "",
        f"- 视频样本数: `{group_counts.get('video', 0)}`",
        f"- 图片样本数: `{group_counts.get('image', 0)}`",
        "",
        "## 场景分布",
        "",
    ]
    for scenario, count in sorted(scenario_counts.items()):
        lines.append(f"- `{scenario}`: `{count}`")

    lines.extend(
        [
            "",
            "## 总览资源",
            "",
            "- [视频样本总览](video_gallery.jpg)",
            "- [图片样本总览](image_gallery.jpg)",
            "- [交互式审查页](index.html)",
        ]
    )

    for group in ("video", "image"):
        title = "视频样本" if group == "video" else "图片样本"
        lines.extend(["", f"## {title}", ""])
        for sample in [item for item in samples if item["selection_group"] == group]:
            lines.extend(
                [
                    f"### {sample['sample_id']}",
                    "",
                    f"- 审查资源: `{sample['review_asset_path']}`",
                    f"- 上下文资源: `{sample['context_asset_path']}`",
                    f"- 海报/锚帧: `{sample['cover_asset_path']}`",
                    f"- 场景: `{sample['scenario_family']}`",
                    f"- 任务: `{sample['main_task']}`",
                    f"- `task_family`: `{sample['task_family']}`",
                    f"- `qa_type`: `{sample['qa_type']}`",
                    f"- `source_label`: {render_inline(sample['source_label'])}",
                    f"- `source_span`: {render_inline(sample['source_span'])}",
                    f"- `canonical_answer`: {render_inline(sample['canonical_answer'])}",
                    f"- `slice_tags`: {render_inline(sample['slice_tags'])}",
                    f"- 选择原因: {sample['selection_reason']}",
                    "",
                    "#### OpenQA Query",
                    "",
                    "```text",
                    sample["openqa_query"] or "",
                    "```",
                    "",
                    "#### OpenQA GT",
                    "",
                    "```json",
                    json.dumps(sample["openqa_gt"], ensure_ascii=False, indent=2),
                    "```",
                ]
            )
            if sample["vqa_query"]:
                lines.extend(
                    [
                        "",
                        "#### VQA Query",
                        "",
                        "```text",
                        sample["vqa_query"],
                        "```",
                        "",
                        "#### VQA GT",
                        "",
                        "```json",
                        json.dumps(sample["vqa_gt"], ensure_ascii=False, indent=2),
                        "```",
                    ]
                )
            lines.extend(
                [
                    "",
                    "#### 中文说明",
                    "",
                    f"- `question_zh`: {render_inline(sample['question_zh'])}",
                    f"- `answer_explanation_zh`: {render_inline(sample['answer_explanation_zh'])}",
                    f"- `semantic_anchor`: {render_inline(sample['semantic_anchor'])}",
                    f"- `prompt_variant`: {render_inline(sample['prompt_variant'])}",
                    "",
                ]
            )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_html(output_path: Path, samples: Sequence[Dict[str, Any]], multiform_root: Path) -> None:
    video_cards: List[str] = []
    image_cards: List[str] = []
    for sample in samples:
        metadata = (
            f"<div class='meta-row'><strong>Main task:</strong> {escape(sample['main_task'])}</div>"
            f"<div class='meta-row'><strong>Main verification:</strong> {escape(sample['main_verification'])}</div>"
            f"<div class='meta-row'><strong>Selection reason:</strong> {escape(sample['selection_reason'])}</div>"
        )
        qa_blocks = (
            html_text_block("OpenQA Query", sample["openqa_query"] or "")
            + html_text_block("OpenQA GT", sample["openqa_gt"], "json")
        )
        if sample.get("vqa_query"):
            qa_blocks += html_text_block("VQA Query", sample["vqa_query"])
            qa_blocks += html_text_block("VQA GT", sample["vqa_gt"], "json")

        if sample["selection_group"] == "video":
            media = (
                f"<div class='media-grid two-col'>"
                f"<div class='media-card'><div class='media-label'>Review clip</div>"
                f"<video controls preload='metadata' poster='{escape(sample['cover_asset_path'])}' src='{escape(sample['review_asset_path'])}'></video></div>"
                f"<div class='media-card'><div class='media-label'>Storyboard</div>"
                f"<img src='{escape(sample['context_asset_path'])}' alt='{escape(sample['sample_id'])} storyboard'></div>"
                f"</div>"
            )
        else:
            media = (
                f"<div class='media-grid two-col'>"
                f"<div class='media-card'><div class='media-label'>Anchor frame</div>"
                f"<img src='{escape(sample['review_asset_path'])}' alt='{escape(sample['sample_id'])} anchor'></div>"
                f"<div class='media-card'><div class='media-label'>Context strip</div>"
                f"<img src='{escape(sample['context_asset_path'])}' alt='{escape(sample['sample_id'])} context strip'></div>"
                f"</div>"
            )

        card = (
            f"<section class='sample-card'>"
            f"<div class='sample-header'>"
            f"<div><h2>{escape(sample['sample_id'])}</h2><div class='badge-row'>{html_badges(sample)}</div></div>"
            f"<div class='asset-kind'>{escape(sample['selection_group'].upper())}</div>"
            f"</div>"
            f"{metadata}"
            f"{media}"
            f"<div class='text-grid'>{qa_blocks}</div>"
            f"</section>"
        )
        if sample["selection_group"] == "video":
            video_cards.append(card)
        else:
            image_cards.append(card)

    html = f"""<!doctype html>
<html lang="zh">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>0424 SWITCHAction_2 Rich Review Bundle</title>
  <style>
    :root {{
      --bg: #f3f1ea;
      --panel: #ffffff;
      --line: #d9ddd7;
      --text: #1d2730;
      --muted: #5d6874;
      --accent: #1b8e85;
      --accent-2: #bf6b1c;
      --chip: #eef5f3;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: linear-gradient(180deg, #f4f2ea 0%, #eef5f3 100%);
      color: var(--text);
      font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
      line-height: 1.45;
    }}
    .page {{
      max-width: 1500px;
      margin: 0 auto;
      padding: 28px 24px 64px;
    }}
    .hero {{
      background: radial-gradient(circle at top right, #dff4ef, #ffffff 48%);
      border: 1px solid var(--line);
      border-radius: 24px;
      padding: 24px 28px;
      box-shadow: 0 18px 60px rgba(31, 58, 54, 0.06);
    }}
    .hero h1 {{
      margin: 0 0 8px;
      font-size: 34px;
    }}
    .hero p {{
      margin: 0;
      color: var(--muted);
      max-width: 980px;
    }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 14px;
      margin-top: 18px;
    }}
    .summary-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 16px 18px;
    }}
    .summary-card .label {{
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
    }}
    .summary-card .value {{
      margin-top: 8px;
      font-size: 28px;
      font-weight: 700;
    }}
    .section-title {{
      margin: 34px 0 14px;
      font-size: 24px;
    }}
    .gallery-row {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 18px;
      margin-bottom: 22px;
    }}
    .gallery-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 16px;
    }}
    .gallery-card h3 {{
      margin: 0 0 10px;
      font-size: 18px;
    }}
    .gallery-card img {{
      width: 100%;
      border-radius: 14px;
      border: 1px solid var(--line);
      display: block;
    }}
    .sample-list {{
      display: grid;
      gap: 18px;
    }}
    .sample-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 22px;
      padding: 18px;
      box-shadow: 0 10px 30px rgba(34, 44, 58, 0.05);
    }}
    .sample-header {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: flex-start;
      margin-bottom: 12px;
    }}
    .sample-header h2 {{
      margin: 0 0 8px;
      font-size: 22px;
    }}
    .asset-kind {{
      padding: 8px 12px;
      border-radius: 999px;
      background: #e7f5f2;
      color: var(--accent);
      font-weight: 700;
      font-size: 12px;
      letter-spacing: 0.08em;
    }}
    .badge-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }}
    .badge {{
      padding: 6px 10px;
      border-radius: 999px;
      background: var(--chip);
      color: #35534c;
      font-size: 12px;
      border: 1px solid #d4e6e1;
    }}
    .meta-row {{
      margin: 4px 0;
      color: var(--muted);
    }}
    .media-grid {{
      display: grid;
      gap: 14px;
      margin-top: 14px;
      margin-bottom: 14px;
    }}
    .two-col {{
      grid-template-columns: 1fr 1fr;
    }}
    .media-card {{
      background: #fbfcfc;
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 12px;
    }}
    .media-label {{
      font-size: 13px;
      font-weight: 700;
      color: var(--accent-2);
      margin-bottom: 8px;
    }}
    .media-card img, .media-card video {{
      width: 100%;
      display: block;
      border-radius: 12px;
      background: #eef1f4;
      border: 1px solid #dde4e8;
    }}
    .text-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 14px;
    }}
    .text-block {{
      background: #fafbfb;
      border: 1px solid var(--line);
      border-radius: 14px;
      overflow: hidden;
    }}
    .text-title {{
      padding: 10px 12px;
      background: #edf4f2;
      color: #294842;
      font-weight: 700;
      border-bottom: 1px solid var(--line);
    }}
    pre {{
      margin: 0;
      padding: 12px;
      white-space: pre-wrap;
      word-break: break-word;
      font-family: ui-monospace, "SFMono-Regular", Consolas, monospace;
      font-size: 12px;
      color: #24303a;
    }}
    @media (max-width: 1100px) {{
      .summary, .gallery-row, .two-col, .text-grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <section class="hero">
      <h1>0424 SWITCHAction_2 审查可视化包</h1>
      <p>视频类样本直接用可播放片段和分镜图审查；图片类样本用锚帧和前后文拼图审查。这个页面对应的数据根目录是 <code>{escape(str(multiform_root.relative_to(multiform_root.parents[2])) if len(multiform_root.parents) >= 3 else str(multiform_root))}</code>。</p>
      <div class="summary">
        <div class="summary-card"><div class="label">Video Samples</div><div class="value">{sum(1 for s in samples if s['selection_group'] == 'video')}</div></div>
        <div class="summary-card"><div class="label">Image Samples</div><div class="value">{sum(1 for s in samples if s['selection_group'] == 'image')}</div></div>
        <div class="summary-card"><div class="label">Scenarios</div><div class="value">{len(Counter(s['scenario_family'] for s in samples))}</div></div>
        <div class="summary-card"><div class="label">Task Families</div><div class="value">{len(Counter(s['task_family'] for s in samples))}</div></div>
      </div>
    </section>

    <h2 class="section-title">总览</h2>
    <div class="gallery-row">
      <div class="gallery-card">
        <h3>视频样本总览</h3>
        <img src="video_gallery.jpg" alt="video gallery">
      </div>
      <div class="gallery-card">
        <h3>图片样本总览</h3>
        <img src="image_gallery.jpg" alt="image gallery">
      </div>
    </div>

    <h2 class="section-title">视频样本</h2>
    <div class="sample-list">
      {''.join(video_cards)}
    </div>

    <h2 class="section-title">图片样本</h2>
    <div class="sample-list">
      {''.join(image_cards)}
    </div>
  </main>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")


def prune_assets(directory: Path, expected_names: Iterable[str]) -> None:
    if not directory.exists():
        return
    expected = set(expected_names)
    for path in directory.iterdir():
        if path.is_file() and path.name not in expected:
            path.unlink()


def collect_selected_samples(
    repo_root: Path,
    base_root: Path,
    multiform_index: Dict[str, Dict[str, Dict[str, Any]]],
    video_map: Dict[str, Dict[str, Any]],
    qa_map: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    selected_samples: List[Dict[str, Any]] = []
    images_dir = base_root / "images"
    image_context_dir = base_root / "image_context"
    videos_dir = base_root / "videos"
    video_posters_dir = base_root / "video_posters"
    video_storyboards_dir = base_root / "video_storyboards"

    for sample_id in SELECTED_SAMPLE_IDS:
        forms = multiform_index.get(sample_id)
        if not forms:
            raise RuntimeError(f"Missing multiform rows for {sample_id}")
        openqa = forms.get("openqa")
        vqa = forms.get("vqa")
        representative = openqa or vqa
        if representative is None:
            raise RuntimeError(f"No representative row for {sample_id}")

        data_id = parse_data_id(sample_id)
        video_info = video_map.get(data_id)
        if video_info is None:
            raise RuntimeError(f"Missing video metadata for {sample_id}")
        qa_info = qa_map.get(sample_id)

        source_span = representative.get("source_span") or (qa_info or {}).get("source_span")
        if not source_span:
            raise RuntimeError(f"Missing source span for {sample_id}")
        start = int(source_span["start"])
        end = int(source_span["end"])

        raw_video_path = repo_root / "annotations" / "0424" / "SWITCHAction_2" / f"{data_id}.mp4"
        stats = get_video_stats(raw_video_path)
        midpoint = clamp_frame((start + end) // 2, stats["total_frames"])

        if sample_id in IMAGE_SAMPLE_IDS:
            anchor_path = images_dir / f"{safe_name(sample_id)}.jpg"
            context_path = image_context_dir / f"{safe_name(sample_id)}_context.jpg"
            extract_frame_image(raw_video_path, midpoint, anchor_path)
            context_frames = make_context_strip(raw_video_path, midpoint, context_path)
            review_asset_rel = anchor_path.relative_to(base_root).as_posix()
            cover_asset_rel = review_asset_rel
            context_asset_rel = context_path.relative_to(base_root).as_posix()
            asset_kind = "image"
            visual_debug = {
                "anchor_frame": midpoint,
                "context_frames": context_frames,
            }
        else:
            clip_path = videos_dir / f"{safe_name(sample_id)}.mp4"
            poster_path = video_posters_dir / f"{safe_name(sample_id)}_poster.jpg"
            storyboard_path = video_storyboards_dir / f"{safe_name(sample_id)}_storyboard.jpg"
            clipped_start, clipped_end = extract_clip(raw_video_path, start, end, clip_path)
            extract_frame_image(raw_video_path, midpoint, poster_path)
            storyboard_frames = make_video_storyboard(raw_video_path, clipped_start, clipped_end, storyboard_path)
            review_asset_rel = clip_path.relative_to(base_root).as_posix()
            cover_asset_rel = poster_path.relative_to(base_root).as_posix()
            context_asset_rel = storyboard_path.relative_to(base_root).as_posix()
            asset_kind = "video"
            visual_debug = {
                "poster_frame": midpoint,
                "storyboard_frames": storyboard_frames,
            }

        selected_samples.append(
            {
                "sample_id": sample_id,
                "selection_group": "video" if sample_id in VIDEO_SAMPLE_IDS else "image",
                "review_asset_kind": asset_kind,
                "review_asset_path": review_asset_rel,
                "cover_asset_path": cover_asset_rel,
                "context_asset_path": context_asset_rel,
                "raw_video_path": raw_video_path.relative_to(repo_root).as_posix(),
                "data_id": data_id,
                "video_name": video_info["video_name"],
                "scenario_family": representative.get("scenario_family") or video_info["scenario_family"],
                "main_task": video_info["main_task"],
                "main_verification": video_info["main_verification"],
                "task_family": representative.get("task_family") or (qa_info or {}).get("task_family"),
                "qa_type": representative.get("qa_type") or (qa_info or {}).get("qa_type"),
                "source_label": (qa_info or {}).get("source_label"),
                "source_span": source_span,
                "source_duration_frames": end - start + 1,
                "canonical_answer": representative.get("canonical_answer"),
                "openqa_query": (openqa or {}).get("query"),
                "openqa_gt": (openqa or {}).get("GT"),
                "vqa_query": (vqa or {}).get("query"),
                "vqa_gt": (vqa or {}).get("GT"),
                "question_zh": representative.get("question_zh"),
                "answer_explanation_zh": representative.get("answer_explanation_zh"),
                "slice_tags": representative.get("slice_tags"),
                "semantic_anchor": representative.get("semantic_anchor"),
                "prompt_variant": representative.get("prompt_variant"),
                "rewrite_type": representative.get("rewrite_type"),
                "output_schema": representative.get("output_schema"),
                "selection_reason": SELECTION_REASONS[sample_id],
                "visual_debug": visual_debug,
            }
        )

    return selected_samples


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    annotation_root = repo_root / "annotations" / "0424" / "SWITCHAction_2"
    multiform_root = choose_multiform_root(annotation_root)
    output_root = annotation_root / "rich_review_bundle_5video_10image"
    output_root.mkdir(parents=True, exist_ok=True)
    for subdir in ["images", "image_context", "videos", "video_posters", "video_storyboards"]:
        (output_root / subdir).mkdir(parents=True, exist_ok=True)

    multiform_index = load_multiform_index(multiform_root)
    qa_candidates_path = annotation_root / "SWITCHAction_2_all.qa_candidates.json"
    if not qa_candidates_path.exists():
        qa_candidates_path = annotation_root / "SWITCHAction_2.qa_candidates.json"
    video_map, qa_map = load_video_metadata(qa_candidates_path)
    samples = collect_selected_samples(repo_root, output_root, multiform_index, video_map, qa_map)

    video_gallery_entries = [
        (sample["sample_id"], output_root / sample["cover_asset_path"])
        for sample in samples
        if sample["selection_group"] == "video"
    ]
    image_gallery_entries = [
        (sample["sample_id"], output_root / sample["review_asset_path"])
        for sample in samples
        if sample["selection_group"] == "image"
    ]
    make_gallery("Video Sample Gallery", video_gallery_entries, output_root / "video_gallery.jpg", columns=2)
    make_gallery("Image Sample Gallery", image_gallery_entries, output_root / "image_gallery.jpg", columns=2)

    write_json(output_root / "selected_samples.json", samples)
    write_markdown(output_root / "selected_samples.md", samples)
    write_html(output_root / "index.html", samples, multiform_root)

    prune_assets(output_root / "images", [f"{safe_name(sample_id)}.jpg" for sample_id in IMAGE_SAMPLE_IDS])
    prune_assets(output_root / "image_context", [f"{safe_name(sample_id)}_context.jpg" for sample_id in IMAGE_SAMPLE_IDS])
    prune_assets(output_root / "videos", [f"{safe_name(sample_id)}.mp4" for sample_id in VIDEO_SAMPLE_IDS])
    prune_assets(output_root / "video_posters", [f"{safe_name(sample_id)}_poster.jpg" for sample_id in VIDEO_SAMPLE_IDS])
    prune_assets(output_root / "video_storyboards", [f"{safe_name(sample_id)}_storyboard.jpg" for sample_id in VIDEO_SAMPLE_IDS])

    print(f"Wrote bundle: {output_root}")
    print(f"Multiform root: {multiform_root}")
    print(f"Video samples: {len(VIDEO_SAMPLE_IDS)}")
    print(f"Image samples: {len(IMAGE_SAMPLE_IDS)}")


if __name__ == "__main__":
    main()
