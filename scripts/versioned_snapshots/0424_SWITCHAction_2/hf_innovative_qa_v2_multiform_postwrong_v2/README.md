# `hf_innovative_qa_v2_multiform_postwrong_v2`

## Output

- [hf_innovative_qa_v2_multiform_postwrong_v2](/d:/Search/BAAI/SWITCH/annotations/0424/SWITCHAction_2/hf_innovative_qa_v2_multiform_postwrong_v2)

## Snapshot Script

- [build_switch_hf_multiform_postwrong_v2.py](/d:/Search/BAAI/SWITCH/scripts/versioned_snapshots/0424_SWITCHAction_2/hf_innovative_qa_v2_multiform_postwrong_v2/build_switch_hf_multiform_postwrong_v2.py)

## Command Used

```powershell
python -u scripts/build_switch_hf_multiform_v2.py `
  --annotation-root annotations/0424/SWITCHAction_2 `
  --output-dirname hf_innovative_qa_v2_multiform_postwrong_v2
```

## Notes

- This version adds:
  - `vqa_state/img2txt`
  - `action/img2video`
  - `verification_action/img2video`
  - `verification_action/video2video`
  - `verification_state/video2img`
- The output root was completed by writing `form_matrix.json`, `dataset_manifest.json`, and `README.md` after the long build finished generating assets.
