#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


QA_FILE_NAMES = {"vqa.json", "openqa.json"}
MATCH_KEYS = {
    "relpath",
    "task_family",
    "form",
    "json_type",
    "id",
    "origin_qa_id",
    "qa_type",
    "scenario_family",
    "source_file",
}
ACTION_KEYS = {
    "match",
    "query",
    "find",
    "replace",
    "GT",
    "canonical_answer",
    "alias_list",
    "numeric_slots",
    "answer_type",
    "output_schema",
    "option_index",
    "option_label",
    "option_text",
    "source_path",
    "source_video_path",
    "frame_index",
    "start_frame",
    "end_frame",
    "source_type",
    "origin_override_qa_id",
    "reason",
    "note",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_rule_file(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    payload = load_json(path)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("rules"), list):
        return payload["rules"]
    raise ValueError(f"Override file must be a list or an object with a rules list: {path}")


def relpath(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def infer_context(json_path: Path, dataset_root: Path) -> Optional[Dict[str, str]]:
    relative = json_path.relative_to(dataset_root)
    parts = relative.parts
    if len(parts) < 3:
        return None
    if json_path.name not in QA_FILE_NAMES:
        return None
    form = parts[-2]
    task_family = parts[-3]
    if "2" not in form:
        return None
    return {
        "task_family": task_family,
        "form": form,
        "json_type": "openqa" if json_path.name == "openqa.json" else "vqa",
        "relpath": relative.as_posix(),
    }


def iter_qa_files(dataset_root: Path) -> Iterable[Tuple[Path, Dict[str, str]]]:
    for json_path in sorted(dataset_root.rglob("*.json")):
        context = infer_context(json_path, dataset_root)
        if context is None:
            continue
        yield json_path, context


def normalize_matcher(rule: Dict[str, Any]) -> Dict[str, Any]:
    matcher = rule.get("match")
    if matcher is not None:
        if not isinstance(matcher, dict):
            raise ValueError(f"Rule match must be an object: {rule}")
        return matcher
    return {key: value for key, value in rule.items() if key in MATCH_KEYS}


def row_matches(rule: Dict[str, Any], row: Dict[str, Any], context: Dict[str, str]) -> bool:
    matcher = normalize_matcher(rule)
    if not matcher:
        raise ValueError(f"Rule has no match fields: {rule}")
    for key, expected in matcher.items():
        actual = context.get(key, row.get(key))
        if isinstance(expected, list):
            if str(actual) not in {str(item) for item in expected}:
                return False
        elif str(actual) != str(expected):
            return False
    return True


def rule_reason(rule: Dict[str, Any]) -> Optional[str]:
    value = rule.get("reason") or rule.get("note")
    return str(value) if value else None


def add_history(row: Dict[str, Any], action: str, rule: Dict[str, Any]) -> None:
    history = row.setdefault("refinement_history", [])
    entry: Dict[str, Any] = {
        "action": action,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    reason = rule_reason(rule)
    if reason:
        entry["reason"] = reason
    matcher = normalize_matcher(rule)
    if matcher:
        entry["match"] = matcher
    history.append(entry)


def option_index_from_rule(rule: Dict[str, Any]) -> int:
    if "option_index" in rule:
        idx = int(rule["option_index"])
        if idx < 0 or idx > 3:
            raise ValueError(f"option_index must be 0..3: {rule}")
        return idx
    label = str(rule.get("option_label", "")).strip().upper()
    if label not in {"A", "B", "C", "D"}:
        raise ValueError(f"Option replacement needs option_label A-D or option_index 0..3: {rule}")
    return ord(label) - ord("A")


def label_for_index(index: int) -> str:
    return chr(ord("A") + index)


def rebuild_options_string(row: Dict[str, Any]) -> Optional[str]:
    option_fields = [f"option_{label.lower()}" for label in "ABCD"]
    if not all(field in row for field in option_fields):
        return None
    lines = []
    for label, field in zip("ABCD", option_fields):
        text = str(row[field]).strip()
        if text and not text.endswith("."):
            text += "."
        lines.append(f"{label}. {text}")
    return "\n".join(lines) + "\n"


def replace_query_options_block(query: str, options: str) -> str:
    markers = [f"\n{label}. " for label in "ABCD"]
    positions = [query.find(marker) for marker in markers if query.find(marker) != -1]
    if not positions:
        return query
    start = min(positions) + 1
    return query[:start] + options.rstrip()


def apply_query_override(row: Dict[str, Any], rule: Dict[str, Any]) -> bool:
    original = row.get("query", "")
    changed = False
    if "query" in rule:
        row["query"] = str(rule["query"])
        changed = row["query"] != original
    elif "find" in rule and "replace" in rule:
        find_text = str(rule["find"])
        replace_text = str(rule["replace"])
        row["query"] = str(original).replace(find_text, replace_text)
        changed = row["query"] != original
    if changed:
        add_history(row, "query_override", rule)
    return changed


def apply_gt_override(row: Dict[str, Any], rule: Dict[str, Any]) -> bool:
    changed = False
    for key in ["GT", "canonical_answer", "alias_list", "numeric_slots", "answer_type", "output_schema"]:
        if key in rule and row.get(key) != rule[key]:
            row[key] = deepcopy(rule[key])
            changed = True
    if changed:
        add_history(row, "gt_override", rule)
    return changed


def ensure_cv2() -> Any:
    try:
        import cv2  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "opencv-python-headless is required for extracting override frames/clips. "
            "Install it with: python -m pip install opencv-python-headless"
        ) from exc
    return cv2


def resolve_source_path(path_text: str, dataset_root: Path, override_dir: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    candidates = [override_dir / path, dataset_root / path, Path.cwd() / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def extract_frame(source_video: Path, frame_index: int, destination: Path) -> None:
    cv2 = ensure_cv2()
    cap = cv2.VideoCapture(str(source_video))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open source video: {source_video}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Unable to read frame {frame_index} from {source_video}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), frame):
        raise RuntimeError(f"Unable to write replacement frame: {destination}")


def extract_clip(source_video: Path, start_frame: int, end_frame: int, destination: Path) -> None:
    cv2 = ensure_cv2()
    cap = cv2.VideoCapture(str(source_video))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open source video: {source_video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))
    current = int(start_frame)
    while current <= int(end_frame):
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        current += 1
    cap.release()
    writer.release()


def replace_asset_option(
    row: Dict[str, Any],
    rule: Dict[str, Any],
    dataset_root: Path,
    json_path: Path,
    override_dir: Path,
    *,
    dry_run: bool,
) -> bool:
    idx = option_index_from_rule(rule)
    is_image = "option_imgs_path" in row
    is_video = "option_videos_path" in row
    if not is_image and not is_video:
        return False
    key = "option_imgs_path" if is_image else "option_videos_path"
    option_paths = row.get(key) or []
    if idx >= len(option_paths):
        raise ValueError(f"Option index {idx} is out of range for row {row.get('origin_qa_id')}")
    destination = json_path.parent / option_paths[idx]

    if "source_path" in rule:
        source = resolve_source_path(str(rule["source_path"]), dataset_root, override_dir)
        if not source.exists():
            raise FileNotFoundError(f"Replacement asset does not exist: {source}")
        if not dry_run:
            shutil.copy2(source, destination)
    elif "source_video_path" in rule and is_image and "frame_index" in rule:
        source_video = resolve_source_path(str(rule["source_video_path"]), dataset_root, override_dir)
        if not source_video.exists():
            raise FileNotFoundError(f"Replacement source video does not exist: {source_video}")
        if not dry_run:
            extract_frame(source_video, int(rule["frame_index"]), destination)
    elif "source_video_path" in rule and is_video and "start_frame" in rule and "end_frame" in rule:
        source_video = resolve_source_path(str(rule["source_video_path"]), dataset_root, override_dir)
        if not source_video.exists():
            raise FileNotFoundError(f"Replacement source video does not exist: {source_video}")
        if not dry_run:
            extract_clip(source_video, int(rule["start_frame"]), int(rule["end_frame"]), destination)
    else:
        raise ValueError(
            "Asset option replacement needs source_path, source_video_path+frame_index, "
            "or source_video_path+start_frame+end_frame"
        )

    update_option_metadata(row, rule, idx, is_image=is_image)
    add_history(row, "option_asset_replacement", rule)
    return True


def update_option_metadata(row: Dict[str, Any], rule: Dict[str, Any], idx: int, *, is_image: bool) -> None:
    if is_image and "frame_index" in rule:
        ensure_list_slot(row, "option_source_frames", idx, None)
        row["option_source_frames"][idx] = int(rule["frame_index"])
    if (not is_image) and "start_frame" in rule and "end_frame" in rule:
        ensure_list_slot(row, "option_source_ranges", idx, None)
        row["option_source_ranges"][idx] = {
            "start_frame": int(rule["start_frame"]),
            "end_frame": int(rule["end_frame"]),
        }
    if "source_type" in rule:
        ensure_list_slot(row, "option_source_types", idx, None)
        row["option_source_types"][idx] = rule["source_type"]
    if "origin_override_qa_id" in rule:
        ensure_list_slot(row, "option_origin_qa_ids", idx, None)
        row["option_origin_qa_ids"][idx] = rule["origin_override_qa_id"]


def ensure_list_slot(row: Dict[str, Any], key: str, idx: int, default: Any) -> None:
    values = row.setdefault(key, [])
    while len(values) <= idx:
        values.append(deepcopy(default))


def replace_text_option(row: Dict[str, Any], rule: Dict[str, Any]) -> bool:
    if "option_text" not in rule:
        return False
    if "options" not in row and not any(f"option_{label.lower()}" in row for label in "ABCD"):
        return False
    idx = option_index_from_rule(rule)
    label = label_for_index(idx)
    field = f"option_{label.lower()}"
    row[field] = str(rule["option_text"])
    options = rebuild_options_string(row)
    if options is not None:
        row["options"] = options
        if "query" in row:
            row["query"] = replace_query_options_block(str(row["query"]), options)
    if "source_type" in rule:
        ensure_list_slot(row, "option_source_types", idx, None)
        row["option_source_types"][idx] = rule["source_type"]
    if "origin_override_qa_id" in rule:
        ensure_list_slot(row, "option_origin_qa_ids", idx, None)
        row["option_origin_qa_ids"][idx] = rule["origin_override_qa_id"]
    add_history(row, "option_text_replacement", rule)
    return True


def apply_option_replacement(
    row: Dict[str, Any],
    rule: Dict[str, Any],
    dataset_root: Path,
    json_path: Path,
    override_dir: Path,
    *,
    dry_run: bool,
) -> bool:
    changed = False
    if "option_text" in rule:
        changed = replace_text_option(row, rule) or changed
    if "source_path" in rule or "source_video_path" in rule:
        changed = replace_asset_option(row, rule, dataset_root, json_path, override_dir, dry_run=dry_run) or changed
    return changed


def apply_rules_to_payload(
    payload: Dict[str, Any],
    context: Dict[str, str],
    json_path: Path,
    dataset_root: Path,
    override_dir: Path,
    rules: Dict[str, List[Dict[str, Any]]],
    report: Dict[str, Any],
    *,
    dry_run: bool,
) -> bool:
    rows = payload.get("data")
    if not isinstance(rows, list):
        return False

    changed = False
    kept_rows: List[Dict[str, Any]] = []
    for row in rows:
        delete_rule = next((rule for rule in rules["delete_samples"] if row_matches(rule, row, context)), None)
        if delete_rule is not None:
            report["deleted_samples"].append(
                {
                    "relpath": context["relpath"],
                    "id": row.get("id"),
                    "origin_qa_id": row.get("origin_qa_id"),
                    "reason": rule_reason(delete_rule),
                }
            )
            changed = True
            continue

        row_changed = False
        for rule in rules["query_overrides"]:
            if row_matches(rule, row, context):
                row_changed = apply_query_override(row, rule) or row_changed
        for rule in rules["gt_overrides"]:
            if row_matches(rule, row, context):
                row_changed = apply_gt_override(row, rule) or row_changed
        for rule in rules["option_replacements"]:
            if row_matches(rule, row, context):
                row_changed = apply_option_replacement(
                    row,
                    rule,
                    dataset_root,
                    json_path,
                    override_dir,
                    dry_run=dry_run,
                ) or row_changed

        if row_changed:
            report["changed_rows"].append(
                {
                    "relpath": context["relpath"],
                    "id": row.get("id"),
                    "origin_qa_id": row.get("origin_qa_id"),
                }
            )
            changed = True
        kept_rows.append(row)

    if changed:
        payload["data"] = kept_rows
    return changed


def update_manifest(dataset_root: Path, report: Dict[str, Any]) -> None:
    manifest_path = dataset_root / "dataset_manifest.json"
    if not manifest_path.exists():
        return
    manifest = load_json(manifest_path)
    refinements = manifest.setdefault("refinement_history", [])
    refinements.append(
        {
            "timestamp": report["timestamp"],
            "source_dataset_root": report["source_dataset_root"],
            "manual_overrides_dir": report["manual_overrides_dir"],
            "changed_row_count": len(report["changed_rows"]),
            "deleted_sample_count": len(report["deleted_samples"]),
            "report_path": "refinement_report.json",
        }
    )
    write_json(manifest_path, manifest)


def update_readme(dataset_root: Path, report: Dict[str, Any]) -> None:
    readme_path = dataset_root / "README.md"
    if not readme_path.exists():
        return
    readme = readme_path.read_text(encoding="utf-8")
    section = (
        "\n\n## Versioned Refinements\n\n"
        f"- Applied at: `{report['timestamp']}`\n"
        f"- Manual overrides: `{report['manual_overrides_dir']}`\n"
        f"- Changed rows: `{len(report['changed_rows'])}`\n"
        f"- Deleted samples: `{len(report['deleted_samples'])}`\n"
        "- Detailed log: `refinement_report.json`\n"
    )
    if "## Versioned Refinements" in readme:
        return
    readme_path.write_text(readme.rstrip() + section, encoding="utf-8")


def load_all_rules(override_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    return {
        "query_overrides": load_rule_file(override_dir / "query_overrides.json"),
        "gt_overrides": load_rule_file(override_dir / "gt_overrides.json"),
        "option_replacements": load_rule_file(override_dir / "option_replacements.json"),
        "delete_samples": load_rule_file(override_dir / "delete_samples.json"),
    }


def count_rules(rules: Dict[str, List[Dict[str, Any]]]) -> Dict[str, int]:
    return {name: len(items) for name, items in rules.items()}


def prepare_working_root(source_root: Path, output_root: Optional[Path], in_place: bool, force: bool) -> Path:
    if in_place:
        return source_root
    if output_root is None:
        output_root = source_root.with_name(source_root.name + "_refined_v1")
    if output_root.resolve() == source_root.resolve():
        raise ValueError("output root must differ from source root unless --in-place is used")
    if output_root.exists():
        if not force:
            raise FileExistsError(f"Output root already exists. Use --force to replace it: {output_root}")
        shutil.rmtree(output_root)
    shutil.copytree(source_root, output_root)
    return output_root


def apply_refinements(
    source_root: Path,
    override_dir: Path,
    output_root: Optional[Path],
    *,
    in_place: bool,
    force: bool,
    dry_run: bool,
) -> Dict[str, Any]:
    if not source_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {source_root}")
    if not override_dir.exists():
        raise FileNotFoundError(f"Manual overrides dir does not exist: {override_dir}")

    rules = load_all_rules(override_dir)
    working_root = source_root if dry_run else prepare_working_root(source_root, output_root, in_place, force)

    report: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source_dataset_root": str(source_root.resolve()),
        "output_dataset_root": str(working_root.resolve()),
        "manual_overrides_dir": str(override_dir.resolve()),
        "dry_run": dry_run,
        "rule_counts": count_rules(rules),
        "changed_files": [],
        "changed_rows": [],
        "deleted_samples": [],
    }

    for json_path, context in iter_qa_files(working_root):
        payload = load_json(json_path)
        original_payload = deepcopy(payload) if dry_run else None
        changed = apply_rules_to_payload(
            payload,
            context,
            json_path,
            working_root,
            override_dir,
            rules,
            report,
            dry_run=dry_run,
        )
        if changed:
            report["changed_files"].append(context["relpath"])
            if dry_run:
                payload = original_payload
                del payload
            else:
                write_json(json_path, payload)

    if not dry_run:
        write_json(working_root / "refinement_report.json", report)
        update_manifest(working_root, report)
        update_readme(working_root, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply versioned manual QA refinements to a SWITCH HF-style dataset.",
    )
    parser.add_argument("--dataset-root", required=True, type=Path, help="Source dataset root.")
    parser.add_argument("--manual-overrides-dir", required=True, type=Path, help="Directory containing override JSON files.")
    parser.add_argument("--output-root", type=Path, help="Output dataset root. Defaults to <dataset-root>_refined_v1.")
    parser.add_argument("--in-place", action="store_true", help="Modify dataset-root directly.")
    parser.add_argument("--force", action="store_true", help="Replace output-root if it already exists.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and report changes without writing files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = apply_refinements(
        args.dataset_root,
        args.manual_overrides_dir,
        args.output_root,
        in_place=args.in_place,
        force=args.force,
        dry_run=args.dry_run,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
