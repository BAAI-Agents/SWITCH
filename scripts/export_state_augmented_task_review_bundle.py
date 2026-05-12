#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional


SELECTED_SAMPLE_IDS = [
    "003_state_transition_video_003",
    "003_state_transition_video_005",
    "003_state_transition_video_004",
    "004_state_transition_video_005",
    "003_final_state_video_002",
    "006_final_state_video_002",
    "011_final_state_video_002",
    "004_final_state_video_002",
    "003_verification_state_video_008",
    "004_verification_state_video_007",
    "006_verification_state_video_009",
    "004_verification_state_video_010",
    "010_recovery_video_003",
    "022_recovery_video_005",
    "023_recovery_video_005",
]

REVIEW_NOTES = {
    "003_state_transition_video_003": {
        "selection_reason": "Single-image baseline with clean UI evidence, clean physical evidence, and no quality flags.",
        "watch_for": [
            "Check whether the prompt still reads naturally when the input is a single action frame.",
            "Confirm the local stop condition is neither too short nor leaking the full task completion target.",
        ],
    },
    "003_state_transition_video_005": {
        "selection_reason": "Single-image baseline with a longer evidence chain and more temporal stages.",
        "watch_for": [
            "Look for noisy text like repeated or misspelled physical evidence strings.",
            "Check whether the stop condition feels too global for a state-transition sample.",
        ],
    },
    "003_state_transition_video_004": {
        "selection_reason": "Video-to-video state-transition sample with missing UI evidence, physical evidence, and stop condition.",
        "watch_for": [
            "Decide whether this sample should exist at all when the visible continuation signal is null.",
            "Check whether the prompt becomes too underspecified after converting the action span into an input clip.",
        ],
    },
    "004_state_transition_video_005": {
        "selection_reason": "Video-to-video state-transition sample with UI-only evidence and a long input clip.",
        "watch_for": [
            "Check whether the long clip already contains too much of the target transition.",
            "Verify that a UI-only stop condition is sufficient here.",
        ],
    },
    "003_final_state_video_002": {
        "selection_reason": "State-derived single-image final-state sample with physical-only evidence.",
        "watch_for": [
            "Check whether using a late success-state frame as image input makes the prompt degenerate.",
            "Verify whether the prompt wording still makes sense when the input is already near the final state.",
        ],
    },
    "006_final_state_video_002": {
        "selection_reason": "State-derived final-state sample with UI-only evidence and no physical evidence.",
        "watch_for": [
            "Check whether UI-only final-state supervision is sufficient for this task.",
            "See whether the lack of verification hint makes the goal ambiguous.",
        ],
    },
    "011_final_state_video_002": {
        "selection_reason": "Another UI-only final-state sample from a different scenario family.",
        "watch_for": [
            "Compare whether hospital-machine final-state prompts are more or less legible than elevator cases.",
            "Check whether the chosen state frame is too close to the stop condition.",
        ],
    },
    "004_final_state_video_002": {
        "selection_reason": "The only final-state sample that remains video-to-video after state augmentation.",
        "watch_for": [
            "Check whether this residual video sample should stay as clip input or be re-routed another way.",
            "Verify whether the clip continuation prompt is specific enough without physical evidence.",
        ],
    },
    "003_verification_state_video_008": {
        "selection_reason": "Balanced single-image verification sample with both UI and physical evidence.",
        "watch_for": [
            "Use this as the healthiest verification reference.",
            "Check whether the new state-frame input is clearly better than the legacy action-anchor input.",
        ],
    },
    "004_verification_state_video_007": {
        "selection_reason": "UI-only state-derived verification sample with action-fallback history.",
        "watch_for": [
            "Check whether the prompt over-relies on a single UI state cue.",
            "Verify whether the fallback history is still understandable after moving to a state-frame input.",
        ],
    },
    "006_verification_state_video_009": {
        "selection_reason": "State-derived verification sample that also lacks a verification-action hint.",
        "watch_for": [
            "Check whether missing verification hint causes prompt ambiguity.",
            "Verify whether this sample should be filtered or down-weighted during training.",
        ],
    },
    "004_verification_state_video_010": {
        "selection_reason": "The only verification sample that remains video-to-video and still uses a verification-action clip.",
        "watch_for": [
            "Check whether verification-action clip input is more faithful than state-frame input for this case.",
            "Verify whether the prompt wording still distinguishes verification from plain continuation.",
        ],
    },
    "010_recovery_video_003": {
        "selection_reason": "Recovery reference sample with explicit error state, correction action, and post-fix state.",
        "watch_for": [
            "Use this as the strongest recovery baseline.",
            "Check whether the output prompt clearly prevents skipping the correction stage.",
        ],
    },
    "022_recovery_video_005": {
        "selection_reason": "Recovery sample with a two-step correction chain and missing error state.",
        "watch_for": [
            "Check whether multi-step correction ordering is clear enough in the prompt.",
            "Decide whether missing error state should disqualify the sample.",
        ],
    },
    "023_recovery_video_005": {
        "selection_reason": "Another two-step recovery sample with missing error state, useful for consistency checks.",
        "watch_for": [
            "Compare with 022 to see whether the same structural issue repeats consistently.",
            "Check whether the chosen wrong-action clip is too short to support recovery generation.",
        ],
    },
}


def load_dataset_rows(dataset_path: Path, bucket_name: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            row["_line"] = line_number
            row["_bucket"] = bucket_name
            rows.append(row)
    return rows


def ensure_selected_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_id = {row["sample_id"]: row for row in rows}
    missing = [sample_id for sample_id in SELECTED_SAMPLE_IDS if sample_id not in by_id]
    if missing:
        raise RuntimeError(f"Missing selected samples: {missing}")
    return [by_id[sample_id] for sample_id in SELECTED_SAMPLE_IDS]


def copy_if_exists(source_path: Optional[Path], target_path: Path) -> Optional[str]:
    if source_path is None or not source_path.exists():
        return None
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target_path)
    return target_path.name


def inline_value(value: Any) -> str:
    if value is None:
        return "`null`"
    if isinstance(value, list):
        if not value:
            return "`[]`"
        return "; ".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def trim_sample(
    repo_root: Path,
    sample: Dict[str, Any],
    frames_dir: Path,
    clips_dir: Path,
    legacy_frames_dir: Path,
) -> Dict[str, Any]:
    note = REVIEW_NOTES[sample["sample_id"]]
    review_asset_path: Optional[str] = None
    review_asset_type: Optional[str] = None

    if sample.get("input_modality") == "image":
        source_asset = repo_root / sample["input_frame_path"]
        copied_name = copy_if_exists(source_asset, frames_dir / f"{sample['sample_id']}.jpg")
        review_asset_path = f"frames/{copied_name}" if copied_name else None
        review_asset_type = "image"
    elif sample.get("input_modality") == "video":
        source_asset = repo_root / sample["input_clip_path"]
        copied_name = copy_if_exists(source_asset, clips_dir / f"{sample['sample_id']}.mp4")
        review_asset_path = f"clips/{copied_name}" if copied_name else None
        review_asset_type = "video"

    legacy_anchor_review_path = None
    legacy_anchor_path_value = sample.get("legacy_anchor_frame_path") or sample.get("anchor_frame_path")
    if isinstance(legacy_anchor_path_value, str):
        legacy_source = repo_root / legacy_anchor_path_value
        copied_legacy = copy_if_exists(legacy_source, legacy_frames_dir / f"{sample['sample_id']}.jpg")
        if copied_legacy:
            legacy_anchor_review_path = f"legacy_frames/{copied_legacy}"

    return {
        "sample_id": sample["sample_id"],
        "dataset_bucket": sample["_bucket"],
        "dataset_line": sample["_line"],
        "task_type": sample["task_type"],
        "split_bucket": sample["split_bucket"],
        "input_modality": sample["input_modality"],
        "output_modality": sample["output_modality"],
        "input_variant": sample.get("input_variant"),
        "classification_reason": sample.get("classification_reason"),
        "review_asset_type": review_asset_type,
        "review_asset_path": review_asset_path,
        "legacy_anchor_review_path": legacy_anchor_review_path,
        "source_video": sample["source_video"],
        "goal_text": sample["goal_text"],
        "anchor_frame": sample.get("anchor_frame"),
        "anchor_frame_time": sample.get("anchor_frame_time"),
        "anchor_source_type": sample.get("anchor_source_type"),
        "anchor_source_span": sample.get("anchor_source_span"),
        "legacy_anchor_source_type": sample.get("legacy_anchor_source_type"),
        "legacy_anchor_source_span": sample.get("legacy_anchor_source_span"),
        "state_input_source_label": sample.get("state_input_source_label"),
        "state_input_source_span": sample.get("state_input_source_span"),
        "input_state_text": sample.get("input_state_text"),
        "input_clip_path": sample.get("input_clip_path"),
        "input_clip_start_frame": sample.get("input_clip_start_frame"),
        "input_clip_end_frame": sample.get("input_clip_end_frame"),
        "input_clip_frame_count": sample.get("input_clip_frame_count"),
        "input_clip_time_start": sample.get("input_clip_time_start"),
        "input_clip_time_end": sample.get("input_clip_time_end"),
        "next_action": sample.get("next_action"),
        "required_evidence_ui": sample.get("required_evidence_ui"),
        "required_evidence_physical": sample.get("required_evidence_physical"),
        "temporal_stages": sample.get("temporal_stages"),
        "stop_condition": sample.get("stop_condition"),
        "overall_success_condition": sample.get("overall_success_condition"),
        "verification_action_hint": sample.get("verification_action_hint"),
        "error_action": sample.get("error_action"),
        "error_state": sample.get("error_state"),
        "correction_actions": sample.get("correction_actions"),
        "post_fix_state": sample.get("post_fix_state"),
        "quality_flags": sample.get("quality_flags"),
        "source_spans": sample.get("source_spans"),
        "prompt_en": sample.get("prompt_en"),
        "prompt_zh": sample.get("prompt_zh"),
        "review_focus": note,
    }


def write_json(output_path: Path, payload: List[Dict[str, Any]]) -> None:
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def render_list(values: List[str]) -> List[str]:
    return [f"- {value}" for value in values]


def write_markdown(output_path: Path, samples: List[Dict[str, Any]]) -> None:
    lines: List[str] = [
        "# State-Augmented Task Review Bundle",
        "",
        "These selected samples come from `by_input_mode_with_state_single_image/` and are grouped to expose potential issues across the four task types.",
        "",
    ]
    current_task = None
    task_titles = {
        "state_transition_video": "State Transition",
        "final_state_video": "Final State",
        "verification_state_video": "Verification State",
        "recovery_video": "Recovery",
    }

    for sample in samples:
        if sample["task_type"] != current_task:
            current_task = sample["task_type"]
            lines.extend([f"## {task_titles.get(current_task, current_task)}", ""])

        focus = sample["review_focus"]
        lines.extend(
            [
                f"### {sample['sample_id']}",
                "",
                f"- Dataset bucket: `{sample['dataset_bucket']}`",
                f"- Dataset line: `{sample['dataset_line']}`",
                f"- Split bucket: `{sample['split_bucket']}`",
                f"- Input modality: `{sample['input_modality']}`",
                f"- Input variant: {inline_value(sample['input_variant'])}",
                f"- Review asset: {inline_value(sample['review_asset_path'])}",
                f"- Legacy anchor asset: {inline_value(sample['legacy_anchor_review_path'])}",
                f"- Goal: {sample['goal_text']}",
                f"- Classification reason: {inline_value(sample['classification_reason'])}",
                f"- Anchor source type: {inline_value(sample['anchor_source_type'])}",
                f"- Anchor source span: {inline_value(sample['anchor_source_span'])}",
                f"- Legacy anchor source span: {inline_value(sample['legacy_anchor_source_span'])}",
                f"- State input source span: {inline_value(sample['state_input_source_span'])}",
                f"- Next action: {inline_value(sample['next_action'])}",
                f"- Required UI evidence: {inline_value(sample['required_evidence_ui'])}",
                f"- Required physical evidence: {inline_value(sample['required_evidence_physical'])}",
                f"- Stop condition: {inline_value(sample['stop_condition'])}",
                f"- Overall success condition: {inline_value(sample['overall_success_condition'])}",
                f"- Verification action hint: {inline_value(sample['verification_action_hint'])}",
                f"- Quality flags: {inline_value(sample['quality_flags'])}",
                f"- Correction actions: {inline_value(sample['correction_actions'])}",
                f"- Error state: {inline_value(sample['error_state'])}",
                f"- Post-fix state: {inline_value(sample['post_fix_state'])}",
                "",
                "#### Selection Reason",
                "",
                focus["selection_reason"],
                "",
                "#### What To Check",
                "",
            ]
        )
        lines.extend(render_list(focus["watch_for"]))
        lines.extend(
            [
                "",
                "#### Prompt EN",
                "",
                "```text",
                sample["prompt_en"] or "",
                "```",
                "",
                "#### Prompt ZH",
                "",
                "```text",
                sample["prompt_zh"] or "",
                "```",
                "",
                "#### Source Spans",
                "",
                "```json",
                json.dumps(sample["source_spans"], ensure_ascii=False, indent=2),
                "```",
                "",
            ]
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    source_root = repo_root / "annotations" / "0421" / "switch" / "single_image_gen_prompt_v2_full" / "by_input_mode_with_state_single_image"
    output_root = source_root / "task_issue_review_bundle"
    frames_dir = output_root / "frames"
    clips_dir = output_root / "clips"
    legacy_frames_dir = output_root / "legacy_frames"

    rows: List[Dict[str, Any]] = []
    rows.extend(load_dataset_rows(source_root / "single_image" / "dataset.jsonl", "single_image"))
    rows.extend(load_dataset_rows(source_root / "video_to_video" / "dataset.jsonl", "video_to_video"))

    selected_rows = ensure_selected_rows(rows)
    trimmed_samples = [
        trim_sample(repo_root, sample, frames_dir, clips_dir, legacy_frames_dir)
        for sample in selected_rows
    ]

    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "selected_samples.json", trimmed_samples)
    write_markdown(output_root / "selected_samples.md", trimmed_samples)

    print(f"Wrote bundle: {output_root}")
    print(f"Selected samples: {len(trimmed_samples)}")


if __name__ == "__main__":
    main()
