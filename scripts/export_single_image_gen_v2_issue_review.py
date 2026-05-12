#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List


SELECTED_SAMPLE_IDS = [
    "003_state_transition_video_003",
    "003_final_state_video_002",
    "011_verification_state_video_007",
    "010_recovery_video_003",
]

ISSUE_NOTES = {
    "003_state_transition_video_003": {
        "summary": "A clean state-transition baseline with aligned action, UI cue, physical cue, and local stop condition.",
        "strengths": [
            "The next action, visible evidence, and temporal stages are tightly aligned.",
            "The stop condition is local and does not leak the full-task completion target.",
            "Source spans map cleanly to one action, one UI change, and one physical change.",
        ],
        "issues": [
            "The English prompt uses 'a elevator system' instead of 'an elevator system'.",
            "The Chinese prompt is structurally Chinese but still keeps English action and evidence strings.",
            "No verification hint is attached, so downstream review cannot compare generation against a later check step.",
        ],
        "suggested_followups": [
            "Fix article agreement in English device phrasing.",
            "Decide whether `prompt_zh` should remain mixed-language or be fully localized.",
            "Consider adding optional verification context in metadata for stronger auditing.",
        ],
    },
    "003_final_state_video_002": {
        "summary": "A final-state sample that correctly focuses on task completion, but currently relies on physical evidence only.",
        "strengths": [
            "The prompt clearly frames this as a full completion trajectory rather than a local next-step transition.",
            "The stop condition is tied to the overall success condition.",
            "The evidence set includes both arrival and door-opening cues from the late stage of the video.",
        ],
        "issues": [
            "The sample is flagged with `missing_required_evidence_ui`, so the success cue is single-modality.",
            "The prompt does not use the available `verification_action_hint` even though it exists in metadata.",
            "The anchor is very early, which can make final-state generation hard for long-horizon models.",
        ],
        "suggested_followups": [
            "Make it explicit in the prompt when UI evidence is unavailable and physical evidence is sufficient.",
            "Optionally expose the verification hint as auxiliary context for final-state generation.",
            "Review whether long-horizon final-state samples need a later anchor variant.",
        ],
    },
    "011_verification_state_video_007": {
        "summary": "A representative verification-state sample that behaves more like visible-state supervision than full verification playback.",
        "strengths": [
            "The prompt correctly emphasizes visible evidence only and avoids adding extra actions.",
            "The single UI state cue is precise and easy to audit.",
            "The metadata preserves a useful `verification_action_hint` even though it is not injected into the prompt.",
        ],
        "issues": [
            "The sample falls back to an action-derived anchor, not a verification-action-derived anchor.",
            "It is flagged with `missing_required_evidence_physical`, so it lacks cross-modality support.",
            "The prompt body does not explain that this is a partial observable success signal rather than a full verification routine.",
        ],
        "suggested_followups": [
            "Differentiate 'single visible success signal' from 'full verification evidence' in prompt wording or metadata.",
            "Use the verification hint when present to make the evaluation target less ambiguous.",
            "Consider a separate subtype for UI-only verification-state samples.",
        ],
    },
    "010_recovery_video_003": {
        "summary": "A strong recovery sample with an explicit wrong action, correction action, post-fix state, and final success evidence.",
        "strengths": [
            "The error, correction, and post-fix state are all explicit and source-traceable.",
            "The prompt correctly forbids skipping the correction stage and forbids jumping straight to success.",
            "The metadata includes both the correction chain and a downstream verification hint.",
        ],
        "issues": [
            "The sample is flagged with `missing_required_evidence_ui`, so final success evidence is physical-only.",
            "The temporal stages stop at the post-fix state and do not include the final success cue.",
            "The Chinese prompt still contains English task and evidence strings, which reduces readability for pure Chinese review.",
        ],
        "suggested_followups": [
            "Append final success evidence to `temporal_stages` for recovery samples.",
            "Clarify in prompt text whether UI evidence is optional or absent in this scenario.",
            "If needed, fully localize Chinese prompts for manual review workflows.",
        ],
    },
}


def load_dataset_rows(dataset_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            row["_line"] = line_number
            rows.append(row)
    return rows


def ensure_selected_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_id = {row["sample_id"]: row for row in rows}
    missing = [sample_id for sample_id in SELECTED_SAMPLE_IDS if sample_id not in by_id]
    if missing:
        raise RuntimeError(f"Missing selected samples: {missing}")
    return [by_id[sample_id] for sample_id in SELECTED_SAMPLE_IDS]


def copy_anchor_frame(repo_root: Path, sample: Dict[str, Any], output_frames_dir: Path) -> str:
    source_path = repo_root / sample["anchor_frame_path"]
    target_path = output_frames_dir / f"{sample['sample_id']}.jpg"
    output_frames_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target_path)
    return target_path.name


def trim_sample(sample: Dict[str, Any], copied_frame_name: str) -> Dict[str, Any]:
    note = ISSUE_NOTES[sample["sample_id"]]
    return {
        "sample_id": sample["sample_id"],
        "dataset_line": sample["_line"],
        "task_type": sample["task_type"],
        "source_video": sample["source_video"],
        "goal_text": sample["goal_text"],
        "anchor_frame_path": sample["anchor_frame_path"],
        "review_frame_path": f"frames/{copied_frame_name}",
        "anchor_source_type": sample["anchor_source_type"],
        "anchor_source_span": sample["anchor_source_span"],
        "next_action": sample["next_action"],
        "required_evidence_ui": sample["required_evidence_ui"],
        "required_evidence_physical": sample["required_evidence_physical"],
        "temporal_stages": sample["temporal_stages"],
        "stop_condition": sample["stop_condition"],
        "overall_success_condition": sample["overall_success_condition"],
        "verification_action_hint": sample["verification_action_hint"],
        "error_action": sample["error_action"],
        "error_state": sample["error_state"],
        "correction_actions": sample["correction_actions"],
        "post_fix_state": sample["post_fix_state"],
        "quality_flags": sample["quality_flags"],
        "source_spans": sample["source_spans"],
        "prompt_en": sample["prompt_en"],
        "prompt_zh": sample["prompt_zh"],
        "issue_review": note,
    }


def write_json(output_path: Path, payload: List[Dict[str, Any]]) -> None:
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def render_list(values: List[str]) -> List[str]:
    return [f"- {value}" for value in values]


def inline_value(value: Any) -> str:
    if value is None:
        return "`null`"
    if isinstance(value, list):
        if not value:
            return "`[]`"
        return "; ".join(str(item) for item in value)
    return str(value)


def write_markdown(output_path: Path, samples: List[Dict[str, Any]]) -> None:
    lines: List[str] = [
        "# Selected Issue Review Samples",
        "",
        "These four samples are extracted from `single_image_gen_prompt_v2_full/dataset.jsonl` for quick manual issue review.",
        "",
    ]
    for sample in samples:
        note = sample["issue_review"]
        lines.extend(
            [
                f"## {sample['sample_id']}",
                "",
                f"- Task type: `{sample['task_type']}`",
                f"- Dataset line: `{sample['dataset_line']}`",
                f"- Goal: `{sample['goal_text']}`",
                f"- Review frame: `frames/{sample['sample_id']}.jpg`",
                f"- Anchor source type: `{sample['anchor_source_type']}`",
                f"- Next action: {inline_value(sample['next_action'])}",
                f"- Required UI evidence: {inline_value(sample['required_evidence_ui'])}",
                f"- Required physical evidence: {inline_value(sample['required_evidence_physical'])}",
                f"- Stop condition: {inline_value(sample['stop_condition'])}",
                f"- Overall success condition: {inline_value(sample['overall_success_condition'])}",
                f"- Verification action hint: {inline_value(sample['verification_action_hint'])}",
                f"- Quality flags: {inline_value(sample['quality_flags'])}",
                "",
                "### Summary",
                "",
                note["summary"],
                "",
                "### Strengths",
                "",
            ]
        )
        lines.extend(render_list(note["strengths"]))
        lines.extend(["", "### Issues", ""])
        lines.extend(render_list(note["issues"]))
        lines.extend(["", "### Suggested Follow-ups", ""])
        lines.extend(render_list(note["suggested_followups"]))
        lines.extend(
            [
                "",
                "### Prompt EN",
                "",
                "```text",
                sample["prompt_en"],
                "```",
                "",
                "### Prompt ZH",
                "",
                "```text",
                sample["prompt_zh"],
                "```",
                "",
                "### Source Spans",
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
    dataset_root = repo_root / "annotations" / "0421" / "switch" / "single_image_gen_prompt_v2_full"
    dataset_path = dataset_root / "dataset.jsonl"
    output_root = dataset_root / "issue_review_bundle"
    output_frames_dir = output_root / "frames"
    output_root.mkdir(parents=True, exist_ok=True)

    rows = load_dataset_rows(dataset_path)
    selected = ensure_selected_rows(rows)

    reviewed_samples: List[Dict[str, Any]] = []
    for sample in selected:
        copied_frame_name = copy_anchor_frame(repo_root, sample, output_frames_dir)
        reviewed_samples.append(trim_sample(sample, copied_frame_name))

    write_json(output_root / "selected_samples.json", reviewed_samples)
    write_markdown(output_root / "selected_samples.md", reviewed_samples)

    print(f"Wrote review bundle: {output_root}")
    print(f"Selected samples: {len(reviewed_samples)}")


if __name__ == "__main__":
    main()
