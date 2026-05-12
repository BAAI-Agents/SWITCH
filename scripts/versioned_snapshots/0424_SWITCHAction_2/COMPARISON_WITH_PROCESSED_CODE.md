# QA Generation Script Comparison

This note compares three QA generation code lines:

- `processed_code_v1`
- `processed_code_v2`
- `scripts/versioned_snapshots/0424_SWITCHAction_2/hf_innovative_qa_v2_multiform_state_aligned_postwrong_v2`

## High-Level Difference

| Code line | Main role | Strength | Main weakness |
| --- | --- | --- | --- |
| `processed_code_v1` | Early task-specific dataset builders from raw annotations | Clear per-task extraction logic; useful task-specific frame/span semantics | Hard-coded paths, duplicated logic, mostly MCQ, weak versioning and auditing |
| `processed_code_v2` | Annotation result aggregation, filtering, asset copying, instruction cleanup | Declarative task/form registry; usable filtering; manual correction dictionaries; option replacements | Depends on already-produced assets; less direct control over raw frame/span semantics |
| `state_aligned_postwrong_v2` snapshot | Current integrated, versioned multiform builder and optimizer | Versioned output, OpenQA + MCQ, strict future checks, state alignment, recovery post-wrong signal, review visuals | Still needs better correction/override layer, UI grounding, stronger hybrid distractor selection, and cleaner candidate intermediate files |

## `processed_code_v1`

`processed_code_v1` is closest to a raw annotation to dataset builder. It has separate scripts for:

- `vqa_task`
- `vqa_state`
- `action`
- `final_state`
- `verification_action`
- `verification_state`
- `ui_grounding`

Important ideas worth keeping:

- Per-family extraction logic is explicit. For example, `final_state_new.py` extracts `previous_state_caption`, `action_text`, and `current_state_caption` before assembling QA.
- `verification_state.py` builds state verification from frames after a `Verification` marker, which is useful for understanding old task intent.
- `action_process_json2dict.py` groups action candidates by `scenario + verb`, which is a good debugging/intermediate-analysis view.
- Most option generation uses `SentenceTransformer` similarity with same-scenario preference. This is useful as a semantic hard-negative layer.
- `ui_grounding_*` scripts contain a complete route for UI interaction grounding from action points to framed/boxed images.

Limitations:

- The scripts are path-hardcoded to old local directories.
- Most logic is duplicated per family.
- Query templates are relatively mechanical.
- Option selection is primarily text-similarity based and does not enforce the visual-distance constraints we needed later.
- There is little manifest, audit trail, or versioned review output.

## `processed_code_v2`

`processed_code_v2` is more like a post-processing and release assembly layer.

Important ideas worth keeping:

- `TASK_INFO_dict` provides a declarative task/form registry with expected input and output paths.
- `form2output_dict` cleanly maps `img2txt`, `video2img`, `img2video`, etc. to required assets.
- `step1_5_keep_no_overlap` prevents reusing examples from prior releases by checking `origin_key`.
- `option_replacements` supports reviewer/manual replacement of bad visual options.
- `global_replacements_dict.json` and `specific_corrections_dict.json` provide a concrete correction layer.
- `apply_instruction_refinements.py` has a useful priority order: specific override, global replacement, then formatting.
- `qwen_extend_video_0104.py` handles too-short video clips by extending duration, which is useful for review and model input stability.

Limitations:

- It mostly consumes previously generated/judged data rather than raw annotations.
- The correction system rewrites instruction text inside existing queries, but does not understand our newer metadata such as `semantic_anchor`, `rewrite_type`, `source_span`, or `post_wrong_signal`.
- It is not fully deterministic if LLM-generated assets or external review replacements are introduced without frozen intermediate files.

## Current Snapshot

The current `state_aligned_postwrong_v2` line has two layers:

- `hf_innovative_qa_v2_multiform_postwrong_v2/build_switch_hf_multiform_postwrong_v2.py` builds the multiform package.
- `hf_innovative_qa_v2_multiform_state_aligned_postwrong_v2/optimize_multiform_state_logic_postwrong_v2.py` applies final-state and verification-state alignment.

Current advantages:

- One versioned output folder per dataset version.
- `form_matrix.json`, `dataset_manifest.json`, README, review batches, and visual review scripts are produced together.
- It supports both MCQ and OpenQA, with `prompt_variant`, `rewrite_type`, `semantic_anchor`, `output_schema`, and structured recovery answers.
- It adds strict future filtering for `final_state/img2img` and `final_state/video2img`.
- It overrides visual `final_state` GT to annotated `is_final_state`.
- It treats single-image `verification_state` as post-change state rather than the change point.
- It upgrades `recovery_chain` with `post_wrong_signal`.

Current gaps:

- `ui_grounding` is not yet included.
- Manual override/correction logic is still scattered; it should become a first-class stage.
- Some candidate construction is still too implicit inside the builder. We should persist intermediate candidate files.
- Hard negatives should combine semantic similarity, visual distance, same-device/cross-scene structure, and reviewer overrides.
- Recovery can miss unlabeled visual events, so it needs an override/audit path keyed by `origin_qa_id`.

## Best Ideas To Borrow

1. Borrow `processed_code_v2`'s declarative registry.

   Move task/form definitions toward a table like `TASK_INFO_dict` plus `form2output_dict`. This will make missing forms obvious before generation.

2. Borrow `processed_code_v2`'s correction layer.

   Add a versioned correction stage with:

   - `global_replacements.json`
   - `specific_corrections.json`
   - `manual_overrides.json`
   - `option_replacements.json`

   This is the right place for reviewer fixes such as a specific recovery `post_wrong_signal`, replacing bad visual options, or deleting a poor sample.

3. Borrow `processed_code_v1`'s semantic candidate intermediates.

   Write intermediate files before final materialization:

   - `candidate_intermediates/action_candidates.json`
   - `candidate_intermediates/state_candidates.json`
   - `candidate_intermediates/verification_candidates.json`
   - `candidate_intermediates/recovery_candidates.json`

   Each item should keep source frame/span, label type, caption/action text, scenario, semantic group, and source video path.

4. Borrow `processed_code_v1`'s embedding-based option ranking, but gate it with current visual quality checks.

   Good next selector:

   - semantic similarity from text/caption embeddings
   - same scenario or same device preference
   - minimum visual distance from GT and query
   - maximum visual distance to avoid weak unrelated distractors
   - pairwise option diversity so A/B/C/D do not collapse into near-duplicates

5. Bring back `ui_grounding`.

   Use the old `ui_grounding` route as a new task family or optional slice:

   - query image with UI point/box
   - output text or point-choice
   - source from UI interaction actions and annotated points

6. Keep the current strict/state-aligned logic as the base.

   The older code should not replace the current builder. It should feed improvements into it: cleaner candidates, stronger corrections, better option choice, and more complete task coverage.

## Recommended Next Refactor

1. Add `qa_generation_registry.py` or `qa_generation_registry.json`.
2. Add `apply_versioned_qa_refinements.py` modeled after `processed_code_v2/apply_instruction_refinements.py`, but aware of current fields.
3. Add `manual_overrides/*.json` for sample-level fixes and option replacements.
4. Add persisted `candidate_intermediates/` before asset writing.
5. Add hybrid hard-negative selection using both old text embeddings and current visual-distance checks.
6. Add `ui_grounding` as a new family after the above scaffolding is stable.

## Practical Priority

The most valuable short-term change is not adding more forms. It is adding a versioned correction and override layer. That will let reviewer feedback become data, not one-off code changes, and it will make issues like bad recovery signals, visually duplicated options, and awkward query wording much easier to fix without rewriting the whole generator each time.
