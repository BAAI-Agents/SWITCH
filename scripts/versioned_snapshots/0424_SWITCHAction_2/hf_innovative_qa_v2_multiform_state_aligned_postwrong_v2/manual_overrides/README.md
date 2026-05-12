# Manual Overrides

This directory stores data-only review fixes for `hf_innovative_qa_v2_multiform_state_aligned_postwrong_v2`.

The refinement script reads these files:

- `query_overrides.json`
- `gt_overrides.json`
- `option_replacements.json`
- `delete_samples.json`

## Match Fields

Every rule must identify the target rows with either a top-level match field or a nested `match` object.

Common match fields:

- `relpath`: for example `final_state/img2img/vqa.json`
- `task_family`: for example `final_state`
- `form`: for example `img2img`
- `json_type`: `vqa` or `openqa`
- `id`
- `origin_qa_id`
- `qa_type`
- `scenario_family`

## Query Override Example

```json
[
  {
    "match": {
      "origin_qa_id": "012_final_002",
      "task_family": "final_state",
      "form": "img2img"
    },
    "query": "This frame is before the final state is visible. Which option shows the final state after the appointment form is collected?",
    "reason": "Reviewer requested clearer wording."
  }
]
```

## GT Override Example

```json
[
  {
    "match": {
      "origin_qa_id": "027_recovery_chain",
      "task_family": "recovery",
      "json_type": "openqa"
    },
    "GT": {
      "wrong_action": "Click the touchscreen button 3 yuan",
      "post_wrong_signal": "The screen shows the transaction was cancelled and asks the user to retrieve the cash.",
      "fix_steps": ["Click the 2 yuan button on the screen"],
      "post_fix_signal": "The screen shows a price of 2 yuan and a quantity of 1 ticket."
    },
    "canonical_answer": "Wrong action: Click the touchscreen button 3 yuan; Post-wrong signal: The screen shows the transaction was cancelled and asks the user to retrieve the cash.; Fix: Click the 2 yuan button on the screen; Post-fix signal: The screen shows a price of 2 yuan and a quantity of 1 ticket.",
    "reason": "Manual recovery correction after visual review."
  }
]
```

## Text Option Replacement Example

```json
[
  {
    "match": {
      "origin_qa_id": "011_action_003",
      "task_family": "action",
      "form": "img2txt"
    },
    "option_label": "B",
    "option_text": "Confirm the selected project information.",
    "source_type": "manual_text_override",
    "reason": "Option wording was too vague."
  }
]
```

## Image Option Replacement Example

Use an existing image or extract a replacement frame from a video.

```json
[
  {
    "match": {
      "origin_qa_id": "058_final_001",
      "task_family": "final_state",
      "form": "img2img"
    },
    "option_label": "B",
    "source_video_path": "D:/Search/BAAI/SWITCH/annotations/0424/SWITCHAction_2/videos/058.mp4",
    "frame_index": 640,
    "source_type": "manual_physical_world_state",
    "origin_override_qa_id": "058_physical_world_state_640",
    "reason": "Use state after the change rather than a visually invisible change point."
  }
]
```

## Video Option Replacement Example

```json
[
  {
    "match": {
      "origin_qa_id": "011_action_003",
      "task_family": "action",
      "form": "img2video"
    },
    "option_label": "D",
    "source_video_path": "D:/Search/BAAI/SWITCH/annotations/0424/SWITCHAction_2/videos/011.mp4",
    "start_frame": 62,
    "end_frame": 120,
    "source_type": "manual_action_clip",
    "origin_override_qa_id": "011_action_003",
    "reason": "Reviewer selected a cleaner action clip."
  }
]
```

## Delete Sample Example

```json
[
  {
    "match": {
      "origin_qa_id": "bad_sample_id",
      "task_family": "final_state",
      "form": "video2img"
    },
    "reason": "The query already contains the answer state."
  }
]
```

## Command

```powershell
python scripts/apply_versioned_qa_refinements.py `
  --dataset-root annotations/0424/SWITCHAction_2/hf_innovative_qa_v2_multiform_state_aligned_postwrong_v2 `
  --manual-overrides-dir scripts/versioned_snapshots/0424_SWITCHAction_2/hf_innovative_qa_v2_multiform_state_aligned_postwrong_v2/manual_overrides `
  --output-root annotations/0424/SWITCHAction_2/hf_innovative_qa_v2_multiform_state_aligned_postwrong_v2_refined_v1
```
