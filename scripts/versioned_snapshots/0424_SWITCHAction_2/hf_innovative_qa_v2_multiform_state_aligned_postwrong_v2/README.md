# `hf_innovative_qa_v2_multiform_state_aligned_postwrong_v2`

## Output

- [hf_innovative_qa_v2_multiform_state_aligned_postwrong_v2](/d:/Search/BAAI/SWITCH/annotations/0424/SWITCHAction_2/hf_innovative_qa_v2_multiform_state_aligned_postwrong_v2)

## Snapshot Scripts

- [optimize_multiform_state_logic_postwrong_v2.py](/d:/Search/BAAI/SWITCH/scripts/versioned_snapshots/0424_SWITCHAction_2/hf_innovative_qa_v2_multiform_state_aligned_postwrong_v2/optimize_multiform_state_logic_postwrong_v2.py)
- [export_0424_recovery_postwrong_review_v1.py](/d:/Search/BAAI/SWITCH/scripts/versioned_snapshots/0424_SWITCHAction_2/hf_innovative_qa_v2_multiform_state_aligned_postwrong_v2/export_0424_recovery_postwrong_review_v1.py)
- [render_0424_state_aligned_review_visuals_v2.py](/d:/Search/BAAI/SWITCH/scripts/versioned_snapshots/0424_SWITCHAction_2/hf_innovative_qa_v2_multiform_state_aligned_postwrong_v2/render_0424_state_aligned_review_visuals_v2.py)
- [apply_versioned_qa_refinements_postwrong_v2.py](/d:/Search/BAAI/SWITCH/scripts/versioned_snapshots/0424_SWITCHAction_2/hf_innovative_qa_v2_multiform_state_aligned_postwrong_v2/apply_versioned_qa_refinements_postwrong_v2.py)
- [manual_overrides](/d:/Search/BAAI/SWITCH/scripts/versioned_snapshots/0424_SWITCHAction_2/hf_innovative_qa_v2_multiform_state_aligned_postwrong_v2/manual_overrides)

## Commands Used

### State Alignment

```powershell
python -u scripts/optimize_multiform_state_logic.py `
  --annotation-root annotations/0424/SWITCHAction_2 `
  --source-dirname hf_innovative_qa_v2_multiform_postwrong_v2 `
  --target-dirname hf_innovative_qa_v2_multiform_state_aligned_postwrong_v2
```

### Recovery Review Export

```powershell
python -u scripts/export_0424_recovery_postwrong_review.py `
  --dataset-root annotations/0424/SWITCHAction_2/hf_innovative_qa_v2_multiform_state_aligned_postwrong_v2
```

### Versioned Manual Refinements

```powershell
python -u scripts/apply_versioned_qa_refinements.py `
  --dataset-root annotations/0424/SWITCHAction_2/hf_innovative_qa_v2_multiform_state_aligned_postwrong_v2 `
  --manual-overrides-dir scripts/versioned_snapshots/0424_SWITCHAction_2/hf_innovative_qa_v2_multiform_state_aligned_postwrong_v2/manual_overrides `
  --output-root annotations/0424/SWITCHAction_2/hf_innovative_qa_v2_multiform_state_aligned_postwrong_v2_refined_v1
```

## Related Outputs

- Recovery review:
  - [recovery_postwrong_review_v1](/d:/Search/BAAI/SWITCH/annotations/0424/SWITCHAction_2/hf_innovative_qa_v2_multiform_state_aligned_postwrong_v2/recovery_postwrong_review_v1)
- State-alignment report:
  - [state_alignment_report.json](/d:/Search/BAAI/SWITCH/annotations/0424/SWITCHAction_2/hf_innovative_qa_v2_multiform_state_aligned_postwrong_v2/state_alignment_report.json)

## Notes

- This snapshot corresponds to the first `postwrong_v2` state-aligned dataset.
- The review-visual renderer is snapshotted here as the matching visual inspection helper for this version.
- Manual QA refinements should be recorded in `manual_overrides/*.json` and applied with `apply_versioned_qa_refinements.py` so query fixes, GT fixes, option replacements, and sample deletions remain data-driven.
