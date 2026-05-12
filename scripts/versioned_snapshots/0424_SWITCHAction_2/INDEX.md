# 0424 SWITCHAction_2 Script Snapshots

This folder stores version-named script snapshots for the `annotations/0424/SWITCHAction_2` data products so later iterations can be traced cleanly.

## Exact Snapshots

### `hf_innovative_qa_v2_multiform_postwrong_v2`

- Dataset output:
  - [hf_innovative_qa_v2_multiform_postwrong_v2](/d:/Search/BAAI/SWITCH/annotations/0424/SWITCHAction_2/hf_innovative_qa_v2_multiform_postwrong_v2)
- Snapshot script:
  - [build_switch_hf_multiform_postwrong_v2.py](/d:/Search/BAAI/SWITCH/scripts/versioned_snapshots/0424_SWITCHAction_2/hf_innovative_qa_v2_multiform_postwrong_v2/build_switch_hf_multiform_postwrong_v2.py)

### `hf_innovative_qa_v2_multiform_state_aligned_postwrong_v2`

- Dataset output:
  - [hf_innovative_qa_v2_multiform_state_aligned_postwrong_v2](/d:/Search/BAAI/SWITCH/annotations/0424/SWITCHAction_2/hf_innovative_qa_v2_multiform_state_aligned_postwrong_v2)
- Snapshot scripts:
  - [optimize_multiform_state_logic_postwrong_v2.py](/d:/Search/BAAI/SWITCH/scripts/versioned_snapshots/0424_SWITCHAction_2/hf_innovative_qa_v2_multiform_state_aligned_postwrong_v2/optimize_multiform_state_logic_postwrong_v2.py)
  - [export_0424_recovery_postwrong_review_v1.py](/d:/Search/BAAI/SWITCH/scripts/versioned_snapshots/0424_SWITCHAction_2/hf_innovative_qa_v2_multiform_state_aligned_postwrong_v2/export_0424_recovery_postwrong_review_v1.py)
  - [render_0424_state_aligned_review_visuals_v2.py](/d:/Search/BAAI/SWITCH/scripts/versioned_snapshots/0424_SWITCHAction_2/hf_innovative_qa_v2_multiform_state_aligned_postwrong_v2/render_0424_state_aligned_review_visuals_v2.py)
  - [apply_versioned_qa_refinements_postwrong_v2.py](/d:/Search/BAAI/SWITCH/scripts/versioned_snapshots/0424_SWITCHAction_2/hf_innovative_qa_v2_multiform_state_aligned_postwrong_v2/apply_versioned_qa_refinements_postwrong_v2.py)
  - [manual_overrides](/d:/Search/BAAI/SWITCH/scripts/versioned_snapshots/0424_SWITCHAction_2/hf_innovative_qa_v2_multiform_state_aligned_postwrong_v2/manual_overrides)

## Historical Note

### `hf_innovative_qa_v2_multiform_state_aligned_postwrong_v1`

- Dataset output:
  - [hf_innovative_qa_v2_multiform_state_aligned_postwrong_v1](/d:/Search/BAAI/SWITCH/annotations/0424/SWITCHAction_2/hf_innovative_qa_v2_multiform_state_aligned_postwrong_v1)
- Status:
  - The exact generating script revision for `v1` was not snapshotted at the time it was produced.
  - The closest current descendants are the `postwrong_v2` snapshots above.

## Suggested Rule Going Forward

- Whenever a new dataset version is written, create a same-named folder under `scripts/versioned_snapshots/0424_SWITCHAction_2/`.
- Copy the exact script files used for that run into the folder before the next iteration changes them.
- Add the command line used for that version in the folder README.
