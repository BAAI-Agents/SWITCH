#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2


SOURCE_PROMPT_VERSION = "single_image_gen_prompt_v2_full"
IMAGE_PROMPT_VERSION = "single_image_gen_prompt_v2_image_only"
VIDEO_PROMPT_VERSION = "video_input_output_prompt_v2_segment"

SCENE_INVARIANTS = [
    "keep the same first-person viewpoint",
    "preserve the same device structure and button layout",
    "preserve the same object and hand identity",
]

SCENE_INVARIANTS_ZH = [
    "保持相同的第一人称视角",
    "保持相同的设备结构和按钮布局",
    "保持相同的物体与手部身份",
]

COMMON_FORBIDDEN_VIDEO = [
    "do not redesign the UI layout",
    "do not invent new buttons or displays",
    "do not add unrelated objects",
    "do not jump to a different camera viewpoint",
]

COMMON_FORBIDDEN_VIDEO_ZH = [
    "不要重新设计界面布局",
    "不要虚构新的按钮或显示内容",
    "不要添加无关物体",
    "不要切换到不同的摄像机视角",
]

DEVICE_LABEL_ZH = {
    "elevator_system": "电梯系统",
    "subway_ticket_machine": "地铁售票机",
    "hospital_registration_machine": "医院自助机",
}

TASK_TYPES = (
    "state_transition_video",
    "final_state_video",
    "verification_state_video",
    "recovery_video",
)


def normalize_spaces(text: str) -> str:
    return " ".join((text or "").split())


def clean_text(text: Optional[str]) -> str:
    return normalize_spaces(text or "")


def humanize_identifier(text: str) -> str:
    return normalize_spaces((text or "").replace("_", " "))


def with_indefinite_article(text: str) -> str:
    cleaned = humanize_identifier(text)
    if not cleaned:
        return cleaned
    article = "an" if cleaned[0].lower() in {"a", "e", "i", "o", "u"} else "a"
    return f"{article} {cleaned}"


def render_inline_list(values: Optional[Sequence[str]]) -> str:
    if not values:
        return "null"
    return "; ".join(clean_text(value) for value in values if clean_text(value))


def relative_repo_path(path: Path, repo_root: Path) -> str:
    return path.relative_to(repo_root).as_posix()


def safe_name(text: str) -> str:
    cleaned = []
    for char in text:
        if char.isalnum() or char in "._-":
            cleaned.append(char)
        else:
            cleaned.append("_")
    return "".join(cleaned).strip("._") or "item"


def device_label_zh(device_family: str) -> str:
    return DEVICE_LABEL_ZH.get(device_family, device_family)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            row["_line"] = line_number
            rows.append(row)
    return rows


def span_kind(sample: Dict[str, Any]) -> str:
    span = sample.get("anchor_source_span")
    if not isinstance(span, dict):
        return "frame"
    start = span.get("start")
    end = span.get("end")
    if start is None or end is None:
        return "frame"
    return "segment" if int(end) > int(start) else "frame"


def infer_sample_fps(sample: Dict[str, Any]) -> float:
    anchor_frame = sample.get("anchor_frame")
    anchor_time = sample.get("anchor_frame_time")
    try:
        anchor_frame_value = float(anchor_frame)
        anchor_time_value = float(anchor_time)
    except (TypeError, ValueError):
        return 30.0
    if anchor_frame_value > 0 and anchor_time_value > 0:
        return anchor_frame_value / anchor_time_value
    return 30.0


class MediaExporter:
    def copy_frame(self, source_path: Path, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, output_path)

    def extract_clip(self, video_path: Path, start_frame: int, end_frame: int, output_path: Path) -> None:
        if end_frame < start_frame:
            raise ValueError(f"Invalid clip span for {output_path.name}: {start_frame}-{end_frame}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            return
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open video: {video_path}")
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if width <= 0 or height <= 0:
            capture.release()
            raise RuntimeError(f"Unable to determine frame size for {video_path}")
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
                    raise RuntimeError(
                        f"Unable to read frame {current} while extracting {start_frame}-{end_frame} from {video_path}"
                    )
                writer.write(frame)
                current += 1
        finally:
            writer.release()
            capture.release()


def lines_to_text(lines: Iterable[str]) -> str:
    return "\n".join(lines)


def filtered_temporal_stages(sample: Dict[str, Any]) -> Optional[List[str]]:
    stages = sample.get("temporal_stages")
    if not isinstance(stages, list):
        return None
    anchor_text = clean_text(((sample.get("anchor_source_span") or {}).get("text")))
    filtered: List[str] = []
    for stage in stages:
        cleaned = clean_text(stage)
        if not cleaned:
            continue
        if anchor_text and cleaned == anchor_text:
            continue
        filtered.append(cleaned)
    return filtered or None


def build_video_prompt_en(sample: Dict[str, Any]) -> str:
    device_family = sample["device_family"]
    device_text = with_indefinite_article(device_family)
    goal_text = sample["goal_text"]
    task_type = sample["task_type"]
    input_text = clean_text(((sample.get("anchor_source_span") or {}).get("text"))) or "the referenced input segment"
    required_ui = sample.get("required_evidence_ui")
    required_physical = sample.get("required_evidence_physical")
    stop_condition = clean_text(sample.get("stop_condition")) or "null"
    verification_hint = clean_text(sample.get("verification_action_hint"))
    correction_actions = sample.get("correction_actions")
    correction_text = render_inline_list(correction_actions)
    continuation_stages = filtered_temporal_stages(sample)

    if task_type == "state_transition_video":
        lines = [
            f"You are given an egocentric input video clip of {device_text}.",
            "Input-output mode: video_to_video",
            "Task type: state_transition_video",
            f"Goal: {goal_text}",
            f"The input clip already shows this action segment: {input_text}",
            "Generate the next short output video that starts immediately after the input clip ends.",
            "The output video must:",
            "- preserve the same device layout, viewpoint, and object identities;",
            "- continue naturally from the end of the input clip;",
            "- show the direct visible result of the action already shown in the input clip;",
            f"- stop once this stop condition is satisfied: {stop_condition}",
            "Required visible evidence in the output video:",
            f"- UI: {render_inline_list(required_ui)}",
            f"- Physical: {render_inline_list(required_physical)}",
            "Please:",
        ]
        lines.extend(f"- {item}" for item in SCENE_INVARIANTS)
        lines.append("Do not:")
        lines.extend(f"- {item}" for item in COMMON_FORBIDDEN_VIDEO)
        lines.append("- do not replay the input action from the beginning")
        lines.append("- do not jump directly to the overall success state")
        if continuation_stages:
            lines.append("Expected continuation stages:")
            lines.extend(f"- {item}" for item in continuation_stages)
        return lines_to_text(lines)

    if task_type == "verification_state_video":
        lines = [
            f"You are given an egocentric input video clip of {device_text}.",
            "Input-output mode: video_to_video",
            "Task type: verification_state_video",
            f"Goal being verified: {goal_text}",
            f"The input clip provides the lead-in context: {input_text}",
            "Generate the next short output video that continues immediately after the input clip and reveals the visible verification evidence.",
            "Do not add unrelated new actions.",
            "Focus on visible evidence only.",
            "Required visible evidence in the output video:",
            f"- UI: {render_inline_list(required_ui)}",
            f"- Physical: {render_inline_list(required_physical)}",
        ]
        if verification_hint:
            lines.append(f"Optional verification hint: {verification_hint}")
        lines.append("Please:")
        lines.extend(f"- {item}" for item in SCENE_INVARIANTS)
        lines.append("Do not:")
        lines.extend(f"- {item}" for item in COMMON_FORBIDDEN_VIDEO)
        lines.append("- do not invent extra actions that are not implied by the input clip")
        lines.append("- do not finish before the required visible evidence becomes observable")
        return lines_to_text(lines)

    if task_type == "final_state_video":
        lines = [
            f"You are given an egocentric input video clip of {device_text}.",
            "Input-output mode: video_to_video",
            "Task type: final_state_video",
            f"Goal: {goal_text}",
            f"The input clip provides this earlier task segment: {input_text}",
            "Generate the next short output video that continues from the input clip and reaches successful completion.",
            "The output video must:",
            "- preserve the same device layout, viewpoint, and object identities;",
            "- remain consistent with the progress already shown in the input clip;",
            f"- stop once this stop condition is satisfied: {stop_condition}",
            "Required visible success evidence in the output video:",
            f"- UI: {render_inline_list(required_ui)}",
            f"- Physical: {render_inline_list(required_physical)}",
            "Please:",
        ]
        lines.extend(f"- {item}" for item in SCENE_INVARIANTS)
        lines.append("Do not:")
        lines.extend(f"- {item}" for item in COMMON_FORBIDDEN_VIDEO)
        lines.append("- do not contradict the progression already shown in the input clip")
        lines.append("- do not stop before the success evidence becomes visible")
        return lines_to_text(lines)

    if task_type == "recovery_video":
        lines = [
            f"You are given an egocentric input video clip of {device_text}.",
            "Input-output mode: video_to_video",
            "Task type: recovery_video",
            f"Goal: {goal_text}",
            f"The input clip shows this wrong-action segment: {input_text}",
            f"Observed error state after the input clip: {clean_text(sample.get('error_state')) or 'null'}",
            f"Correction actions in sequence: {correction_text}",
            f"Post-fix state to reach first: {clean_text(sample.get('post_fix_state')) or 'null'}",
            "Generate the next short output video that starts immediately after the input clip, performs the correction actions in order, reaches the post-fix state, and then continues toward task success.",
            "Required visible success evidence in the output video:",
            f"- UI: {render_inline_list(required_ui)}",
            f"- Physical: {render_inline_list(required_physical)}",
            "Please:",
        ]
        lines.extend(f"- {item}" for item in SCENE_INVARIANTS)
        lines.append("Do not:")
        lines.extend(f"- {item}" for item in COMMON_FORBIDDEN_VIDEO)
        lines.append("- do not skip the correction stage")
        lines.append("- do not jump directly to the success state")
        lines.append("- do not repeat the wrong action as if it were correct")
        return lines_to_text(lines)

    raise ValueError(f"Unsupported task type: {task_type}")


def build_video_prompt_zh(sample: Dict[str, Any]) -> str:
    device_family = sample["device_family"]
    goal_text = sample["goal_text"]
    task_type = sample["task_type"]
    input_text = clean_text(((sample.get("anchor_source_span") or {}).get("text"))) or "该输入片段"
    required_ui = sample.get("required_evidence_ui")
    required_physical = sample.get("required_evidence_physical")
    stop_condition = clean_text(sample.get("stop_condition")) or "null"
    verification_hint = clean_text(sample.get("verification_action_hint"))
    correction_actions = sample.get("correction_actions")
    correction_text = render_inline_list(correction_actions)
    continuation_stages = filtered_temporal_stages(sample)

    if task_type == "state_transition_video":
        lines = [
            f"你将看到一段来自{device_label_zh(device_family)}的第一人称输入视频片段。",
            "输入输出模式：video_to_video",
            "任务类型：state_transition_video",
            f"任务目标：{goal_text}",
            f"输入片段已经展示了这段动作过程：{input_text}",
            "请生成一段紧接在输入片段之后的短输出视频。",
            "输出视频必须：",
            "- 保持相同的设备布局、观察视角和对象身份；",
            "- 与输入片段的结尾自然衔接；",
            "- 展示输入片段中该动作带来的直接可见结果；",
            f"- 当满足以下停止条件时结束：{stop_condition}",
            "输出视频中必须出现的可见证据：",
            f"- 界面证据：{render_inline_list(required_ui)}",
            f"- 物理证据：{render_inline_list(required_physical)}",
            "请：",
        ]
        lines.extend(f"- {item}" for item in SCENE_INVARIANTS_ZH)
        lines.append("不要：")
        lines.extend(f"- {item}" for item in COMMON_FORBIDDEN_VIDEO_ZH)
        lines.append("- 不要从头重放输入片段中已经出现的动作")
        lines.append("- 不要直接跳到整个任务的全局成功状态")
        if continuation_stages:
            lines.append("建议遵循的后续阶段：")
            lines.extend(f"- {item}" for item in continuation_stages)
        return lines_to_text(lines)

    if task_type == "verification_state_video":
        lines = [
            f"你将看到一段来自{device_label_zh(device_family)}的第一人称输入视频片段。",
            "输入输出模式：video_to_video",
            "任务类型：verification_state_video",
            f"正在验证的目标：{goal_text}",
            f"输入片段提供了验证前的上下文：{input_text}",
            "请生成一段紧接在输入片段之后的短输出视频，展示接下来应当出现的可见验证证据。",
            "不要额外添加与输入无关的新动作。",
            "只关注可见证据。",
            "输出视频中必须出现的可见证据：",
            f"- 界面证据：{render_inline_list(required_ui)}",
            f"- 物理证据：{render_inline_list(required_physical)}",
        ]
        if verification_hint:
            lines.append(f"可选验证提示：{verification_hint}")
        lines.append("请：")
        lines.extend(f"- {item}" for item in SCENE_INVARIANTS_ZH)
        lines.append("不要：")
        lines.extend(f"- {item}" for item in COMMON_FORBIDDEN_VIDEO_ZH)
        lines.append("- 不要编造输入片段没有暗示的额外动作")
        lines.append("- 不要在所需可见证据出现之前提前结束")
        return lines_to_text(lines)

    if task_type == "final_state_video":
        lines = [
            f"你将看到一段来自{device_label_zh(device_family)}的第一人称输入视频片段。",
            "输入输出模式：video_to_video",
            "任务类型：final_state_video",
            f"任务目标：{goal_text}",
            f"输入片段提供了任务较早阶段的过程：{input_text}",
            "请生成一段紧接在输入片段之后的短输出视频，继续推进并到达成功完成状态。",
            "输出视频必须：",
            "- 保持相同的设备布局、观察视角和对象身份；",
            "- 与输入片段已经展示的任务进度保持一致；",
            f"- 当满足以下停止条件时结束：{stop_condition}",
            "输出视频中必须出现的成功证据：",
            f"- 界面证据：{render_inline_list(required_ui)}",
            f"- 物理证据：{render_inline_list(required_physical)}",
            "请：",
        ]
        lines.extend(f"- {item}" for item in SCENE_INVARIANTS_ZH)
        lines.append("不要：")
        lines.extend(f"- {item}" for item in COMMON_FORBIDDEN_VIDEO_ZH)
        lines.append("- 不要违背输入片段已经展示出的任务推进方向")
        lines.append("- 不要在成功证据出现之前提前结束")
        return lines_to_text(lines)

    if task_type == "recovery_video":
        lines = [
            f"你将看到一段来自{device_label_zh(device_family)}的第一人称输入视频片段。",
            "输入输出模式：video_to_video",
            "任务类型：recovery_video",
            f"任务目标：{goal_text}",
            f"输入片段展示了这段错误动作过程：{input_text}",
            f"输入片段之后的错误状态：{clean_text(sample.get('error_state')) or 'null'}",
            f"需要按顺序执行的修正动作：{correction_text}",
            f"首先需要回到的修正后状态：{clean_text(sample.get('post_fix_state')) or 'null'}",
            "请生成一段紧接在输入片段之后的短输出视频，按顺序完成修正动作，回到修正后状态，再继续推进到任务成功。",
            "输出视频中必须出现的成功证据：",
            f"- 界面证据：{render_inline_list(required_ui)}",
            f"- 物理证据：{render_inline_list(required_physical)}",
            "请：",
        ]
        lines.extend(f"- {item}" for item in SCENE_INVARIANTS_ZH)
        lines.append("不要：")
        lines.extend(f"- {item}" for item in COMMON_FORBIDDEN_VIDEO_ZH)
        lines.append("- 不要跳过修正阶段")
        lines.append("- 不要直接跳到成功状态")
        lines.append("- 不要把错误动作当成正确流程再次重复")
        return lines_to_text(lines)

    raise ValueError(f"Unsupported task type: {task_type}")


def build_image_sample(
    sample: Dict[str, Any],
    repo_root: Path,
    image_root: Path,
    exporter: MediaExporter,
) -> Dict[str, Any]:
    source_frame_path = repo_root / sample["anchor_frame_path"]
    output_frame_path = image_root / "frames" / f"{safe_name(sample['sample_id'])}.jpg"
    exporter.copy_frame(source_frame_path, output_frame_path)
    updated = dict(sample)
    updated.pop("_line", None)
    updated["prompt_version"] = IMAGE_PROMPT_VERSION
    updated["source_prompt_version"] = sample.get("prompt_version")
    updated["split_bucket"] = "single_image"
    updated["classification_reason"] = "anchor_source_span_is_single_frame_or_missing"
    updated["input_modality"] = "image"
    updated["output_modality"] = "video"
    updated["input_frame_path"] = relative_repo_path(output_frame_path, repo_root)
    updated["anchor_frame_path"] = updated["input_frame_path"]
    return updated


def build_video_sample(
    sample: Dict[str, Any],
    repo_root: Path,
    video_root: Path,
    exporter: MediaExporter,
) -> Dict[str, Any]:
    span = sample["anchor_source_span"]
    start = int(span["start"])
    end = int(span["end"])
    source_video_path = repo_root / sample["source_video"]
    output_clip_path = video_root / "clips" / f"{safe_name(sample['sample_id'])}.mp4"
    exporter.extract_clip(source_video_path, start, end, output_clip_path)
    fps = infer_sample_fps(sample)

    updated = dict(sample)
    updated.pop("_line", None)
    updated["prompt_version"] = VIDEO_PROMPT_VERSION
    updated["source_prompt_version"] = sample.get("prompt_version")
    updated["source_prompt_en"] = sample.get("prompt_en")
    updated["source_prompt_zh"] = sample.get("prompt_zh")
    updated["prompt_en"] = build_video_prompt_en(sample)
    updated["prompt_zh"] = build_video_prompt_zh(sample)
    updated["split_bucket"] = "video_to_video"
    updated["classification_reason"] = "anchor_source_span_is_segment"
    updated["input_modality"] = "video"
    updated["output_modality"] = "video"
    updated["input_clip_path"] = relative_repo_path(output_clip_path, repo_root)
    updated["input_clip_start_frame"] = start
    updated["input_clip_end_frame"] = end
    updated["input_clip_frame_count"] = end - start + 1
    updated["input_clip_time_start"] = round(start / fps, 3)
    updated["input_clip_time_end"] = round(end / fps, 3)
    updated["input_reference_text"] = clean_text(span.get("text"))
    updated["legacy_anchor_frame_path"] = sample.get("anchor_frame_path")
    updated["anchor_frame_path"] = None
    return updated


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def prune_assets(directory: Path, expected_names: Sequence[str]) -> None:
    if not directory.exists():
        return
    expected = set(expected_names)
    for path in directory.iterdir():
        if path.is_file() and path.name not in expected:
            path.unlink()


def write_bucket_summary(path: Path, bucket_name: str, rows: Sequence[Dict[str, Any]], asset_label: str) -> None:
    task_counts = Counter(row["task_type"] for row in rows)
    lines = [
        f"# {bucket_name} Summary",
        "",
        f"- Prompt version: `{rows[0]['prompt_version']}`" if rows else "- Prompt version: `n/a`",
        f"- Total samples: `{len(rows)}`",
        "",
        "## Task Counts",
        "",
    ]
    for task_type in TASK_TYPES:
        lines.append(f"- `{task_type}`: `{task_counts.get(task_type, 0)}`")
    lines.extend(
        [
            "",
            "## Assets",
            "",
            f"- `{asset_label}`",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_bucket_readme(path: Path, bucket_name: str, rows: Sequence[Dict[str, Any]], asset_field: str) -> None:
    task_counts = Counter(row["task_type"] for row in rows)
    lines = [
        f"# {bucket_name}",
        "",
        f"- Source dataset: `annotations/0421/switch/{SOURCE_PROMPT_VERSION}/dataset.jsonl`",
        f"- Samples: `{len(rows)}`",
        "",
        "## Task Counts",
        "",
    ]
    for task_type in TASK_TYPES:
        lines.append(f"- `{task_type}`: `{task_counts.get(task_type, 0)}`")
    lines.extend(
        [
            "",
            "## Key Fields",
            "",
            f"- `{asset_field}`: input asset path inside this bucket",
            "- `input_modality` / `output_modality`: explicit I/O mode for generation",
            "- `classification_reason`: why the sample was routed to this bucket",
            "- `source_prompt_version`: original prompt version before the split",
        ]
    )
    if rows and rows[0]["split_bucket"] == "video_to_video":
        lines.extend(
            [
                "- `input_clip_start_frame` / `input_clip_end_frame`: inclusive source clip span",
                "- `source_prompt_en` / `source_prompt_zh`: legacy single-frame prompt kept for auditing",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_root_summary(path: Path, image_rows: Sequence[Dict[str, Any]], video_rows: Sequence[Dict[str, Any]]) -> None:
    total_rows = len(image_rows) + len(video_rows)
    lines = [
        "# Prompt Input-Mode Split Summary",
        "",
        f"- Source prompt version: `{SOURCE_PROMPT_VERSION}`",
        f"- Total samples scanned: `{total_rows}`",
        f"- `single_image`: `{len(image_rows)}`",
        f"- `video_to_video`: `{len(video_rows)}`",
        "",
        "## Task Breakdown",
        "",
    ]
    for task_type in TASK_TYPES:
        image_count = sum(1 for row in image_rows if row["task_type"] == task_type)
        video_count = sum(1 for row in video_rows if row["task_type"] == task_type)
        lines.append(f"- `{task_type}`: image=`{image_count}`, video=`{video_count}`")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_root_readme(path: Path) -> None:
    lines = [
        "# Prompt Input-Mode Split v2",
        "",
        "This directory splits the original `single_image_gen_prompt_v2_full` dataset by the type of input evidence.",
        "",
        "## Routing Rule",
        "",
        "- If `anchor_source_span.end > anchor_source_span.start`, the sample is treated as `video_to_video`.",
        "- Otherwise, the sample is kept as `single_image`.",
        "",
        "## Buckets",
        "",
        "- `single_image/`: image-input, video-output samples with copied frame assets.",
        "- `video_to_video/`: segment-input, video-output samples with extracted clip assets and rewritten prompts.",
        "",
        "## Notes",
        "",
        "- The original `single_image_gen_prompt_v2_full/` directory is kept unchanged for auditability.",
        "- The `video_to_video` prompts explicitly continue from the end of the input clip instead of replaying a compressed single-frame anchor.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def validate_image_rows(rows: Sequence[Dict[str, Any]], repo_root: Path, image_root: Path) -> None:
    for row in rows:
        input_path = repo_root / row["input_frame_path"]
        if not input_path.exists():
            raise RuntimeError(f"Missing input frame: {input_path}")
        if not input_path.is_relative_to(image_root):
            raise RuntimeError(f"Image input escapes image bucket: {row['sample_id']}")


def validate_video_rows(rows: Sequence[Dict[str, Any]], repo_root: Path, video_root: Path) -> None:
    for row in rows:
        input_path = repo_root / row["input_clip_path"]
        if not input_path.exists():
            raise RuntimeError(f"Missing input clip: {input_path}")
        if not input_path.is_relative_to(video_root):
            raise RuntimeError(f"Video input escapes video bucket: {row['sample_id']}")
        if "Input-output mode: video_to_video" not in row["prompt_en"]:
            raise RuntimeError(f"Video prompt missing mode header: {row['sample_id']}")
        if "anchor frame" in row["prompt_en"].lower():
            raise RuntimeError(f"Video prompt still references anchor frame: {row['sample_id']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Split prompt v2 dataset into single-image and video-to-video buckets.")
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("annotations") / "0421" / "switch" / SOURCE_PROMPT_VERSION,
        help="Source directory that contains the original dataset.jsonl and frames/.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("annotations") / "0421" / "switch" / SOURCE_PROMPT_VERSION / "by_input_mode",
        help="Output directory for the split buckets.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    source_root = (repo_root / args.source_root).resolve()
    output_root = (repo_root / args.output_root).resolve()
    image_root = output_root / "single_image"
    video_root = output_root / "video_to_video"

    rows = load_jsonl(source_root / "dataset.jsonl")
    exporter = MediaExporter()
    image_rows: List[Dict[str, Any]] = []
    video_rows: List[Dict[str, Any]] = []

    for sample in rows:
        kind = span_kind(sample)
        if kind == "segment":
            video_rows.append(build_video_sample(sample, repo_root, video_root, exporter))
        else:
            image_rows.append(build_image_sample(sample, repo_root, image_root, exporter))

    image_rows.sort(key=lambda row: row["sample_id"])
    video_rows.sort(key=lambda row: row["sample_id"])

    write_jsonl(image_root / "dataset.jsonl", image_rows)
    write_jsonl(video_root / "dataset.jsonl", video_rows)
    write_bucket_summary(image_root / "summary.md", "Single Image Bucket", image_rows, "frames/{sample_id}.jpg")
    write_bucket_summary(video_root / "summary.md", "Video-to-Video Bucket", video_rows, "clips/{sample_id}.mp4")
    write_bucket_readme(image_root / "README.md", "Single Image Bucket", image_rows, "input_frame_path")
    write_bucket_readme(video_root / "README.md", "Video-to-Video Bucket", video_rows, "input_clip_path")
    write_root_summary(output_root / "summary.md", image_rows, video_rows)
    write_root_readme(output_root / "README.md")

    prune_assets(image_root / "frames", [f"{safe_name(row['sample_id'])}.jpg" for row in image_rows])
    prune_assets(video_root / "clips", [f"{safe_name(row['sample_id'])}.mp4" for row in video_rows])

    validate_image_rows(image_rows, repo_root, image_root)
    validate_video_rows(video_rows, repo_root, video_root)

    print(f"Source rows: {len(rows)}")
    print(f"Single-image rows: {len(image_rows)}")
    print(f"Video-to-video rows: {len(video_rows)}")
    print(f"Wrote: {output_root}")


if __name__ == "__main__":
    main()
