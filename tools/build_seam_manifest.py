from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

SEAMS_ROOT = Path("output") / "_debug" / "_seams"
MANIFEST_NAME = "_seam_manifest.csv"
STAGE_ORDER = ["split_h", "split_v_top", "split_v_bottom"]


def warn(message: str) -> None:
    print(f"Warning: {message}", file=sys.stderr)


def load_json(path: Path, *, required: bool) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        if required:
            warn(f"missing {path}")
        return None
    except (json.JSONDecodeError, OSError) as exc:
        if required:
            warn(f"failed to read {path}: {exc}")
        return None


def format_box(box: Any) -> str:
    if isinstance(box, (list, tuple)) and len(box) == 4:
        return ",".join(str(x) for x in box)
    if box is None:
        return ""
    return str(box)


def rel_if_exists(path: Path) -> str:
    try:
        if path.exists():
            return path.relative_to(SEAMS_ROOT).as_posix()
    except ValueError:
        return ""
    return ""


def numeric_from_meta(meta: Dict[str, Any], primary: str, fallback: str) -> Any:
    if primary in meta:
        return meta.get(primary)
    return meta.get(fallback, "")


def count_valid_runs(valid: Sequence[Any]) -> Tuple[int, int]:
    total = 0
    best_run = 0
    current_run = 0
    for value in valid:
        if bool(value):
            total += 1
            current_run += 1
            best_run = max(best_run, current_run)
        else:
            current_run = 0
    return total, best_run


def clamp_index(index: int, length: Optional[int]) -> int:
    if length is None:
        return index
    return max(0, min(index, max(length - 1, 0)))


def parse_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def determine_chosen_idx(meta: Dict[str, Any], profiles: Dict[str, Any]) -> Optional[int]:
    chosen_split = parse_int(profiles.get("chosen_split"))
    if chosen_split is not None:
        return chosen_split

    split_px = parse_float(meta.get("split_px"))
    scale_used = parse_float(profiles.get("scale_used"))
    if split_px is not None and scale_used not in (None, 0):
        return int(round(split_px / scale_used))

    return None


def get_length_for_ws(ws: Any) -> Optional[int]:
    if isinstance(ws, Iterable) and not isinstance(ws, (str, bytes)):
        try:
            return len(ws)  # type: ignore[arg-type]
        except TypeError:
            return None
    return parse_int(ws)


def value_from_array(arr: Any, index: int) -> Any:
    if not isinstance(arr, Sequence):
        return ""
    if index < 0 or index >= len(arr):
        return ""
    return arr[index]


def metric_at_index(arr: Any, index: int) -> str:
    value = value_from_array(arr, index)
    numeric = parse_float(value)
    if numeric is None:
        return ""
    return str(numeric)


def build_row(
    scan_dir: Path,
    stage_dir: Path,
    stage_name: str,
    meta: Dict[str, Any],
    profiles: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "scan_stem": scan_dir.name,
        "stage": stage_name,
        "axis": meta.get("axis", ""),
        "width": numeric_from_meta(meta, "width", "W"),
        "height": numeric_from_meta(meta, "height", "H"),
        "split_px": meta.get("split_px", ""),
        "overlap_px": meta.get("overlap_px", ""),
        "stage_used": meta.get("stage_used", ""),
        "reason": meta.get("reason", ""),
        "confidence": meta.get("confidence", ""),
        "box_a": format_box(meta.get("box_a")),
        "box_b": format_box(meta.get("box_b")),
        "scan_dir_rel": rel_if_exists(scan_dir),
        "debug_dir_rel": rel_if_exists(stage_dir),
        "stage_scan_rel": rel_if_exists(scan_dir / "stage_scan.png"),
        "seam_overlay_rel": rel_if_exists(stage_dir / "seam_overlay.png"),
        "out_a_rel": rel_if_exists(stage_dir / "out_a.png"),
        "out_b_rel": rel_if_exists(stage_dir / "out_b.png"),
        "seam_meta_rel": rel_if_exists(stage_dir / "seam_meta.json"),
        "seam_profiles_rel": rel_if_exists(stage_dir / "seam_profiles.json"),
        "Ws": "",
        "anchor": "",
        "num_valid": "",
        "best_run_len": "",
        "chosen_idx": "",
        "bg_ratio_at_split": "",
        "ink_ratio_at_split": "",
        "seam_align_at_split": "",
        "strong_align_at_split": "",
        "geom_ok_at_split": "",
        "dt_best": "",
        "dt_score_at_best": "",
        "dt_score_at_split": "",
        "cv2_gap_error": "",
    }

    if not profiles:
        return row

    ws_value = profiles.get("Ws")
    ws_len = get_length_for_ws(ws_value)
    if ws_value is not None:
        row["Ws"] = str(ws_value)

    if "anchor" in profiles and profiles.get("anchor") is not None:
        row["anchor"] = str(profiles.get("anchor"))

    valid = profiles.get("valid")
    if isinstance(valid, Sequence) and not isinstance(valid, (str, bytes)):
        num_valid, best_run = count_valid_runs(valid)
        row["num_valid"] = str(num_valid)
        row["best_run_len"] = str(best_run)

    chosen_idx = determine_chosen_idx(meta, profiles)
    if chosen_idx is not None:
        clamped_idx = clamp_index(chosen_idx, ws_len)
        row["chosen_idx"] = str(clamped_idx)
    else:
        clamped_idx = None

    if clamped_idx is not None and clamped_idx >= 0:
        row["bg_ratio_at_split"] = metric_at_index(profiles.get("bg_ratio"), clamped_idx)
        row["ink_ratio_at_split"] = metric_at_index(profiles.get("ink_ratio"), clamped_idx)
        row["seam_align_at_split"] = metric_at_index(profiles.get("seam_align"), clamped_idx)
        row["strong_align_at_split"] = metric_at_index(profiles.get("strong_align"), clamped_idx)
        geom_ok = value_from_array(profiles.get("geom_ok"), clamped_idx)
        if geom_ok != "":
            row["geom_ok_at_split"] = str(bool(geom_ok))

    dt_best = parse_int(profiles.get("dt_best"))
    if dt_best is not None:
        row["dt_best"] = str(dt_best)

    dt_score = profiles.get("dt_score")
    if dt_best is not None and isinstance(dt_score, Sequence):
        row["dt_score_at_best"] = metric_at_index(dt_score, dt_best)

    if clamped_idx is not None and isinstance(dt_score, Sequence):
        row["dt_score_at_split"] = metric_at_index(dt_score, clamped_idx)

    if "cv2_gap_error" in profiles and profiles.get("cv2_gap_error") is not None:
        row["cv2_gap_error"] = str(profiles.get("cv2_gap_error"))

    return row


def collect_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not SEAMS_ROOT.exists():
        return rows

    scan_dirs = [
        path for path in SEAMS_ROOT.iterdir() if path.is_dir() and not path.name.startswith("_")
    ]
    for scan_dir in sorted(scan_dirs, key=lambda p: p.name):
        for stage_name in STAGE_ORDER:
            stage_dir = scan_dir / stage_name
            if not stage_dir.is_dir():
                continue

            meta_path = stage_dir / "seam_meta.json"
            meta = load_json(meta_path, required=True)
            if not meta:
                continue

            profiles_path = stage_dir / "seam_profiles.json"
            profiles: Optional[Dict[str, Any]] = None
            if profiles_path.exists():
                profiles = load_json(profiles_path, required=False)
                if profiles is None:
                    warn(f"corrupt {profiles_path}")

            row = build_row(scan_dir, stage_dir, stage_name, meta, profiles)
            rows.append(row)

    return rows


def write_manifest(rows: List[Dict[str, Any]]) -> Path:
    manifest_path = SEAMS_ROOT / MANIFEST_NAME
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "scan_stem",
        "stage",
        "axis",
        "width",
        "height",
        "split_px",
        "overlap_px",
        "stage_used",
        "reason",
        "confidence",
        "box_a",
        "box_b",
        "scan_dir_rel",
        "debug_dir_rel",
        "stage_scan_rel",
        "seam_overlay_rel",
        "out_a_rel",
        "out_b_rel",
        "seam_meta_rel",
        "seam_profiles_rel",
        "Ws",
        "anchor",
        "num_valid",
        "best_run_len",
        "chosen_idx",
        "bg_ratio_at_split",
        "ink_ratio_at_split",
        "seam_align_at_split",
        "strong_align_at_split",
        "geom_ok_at_split",
        "dt_best",
        "dt_score_at_best",
        "dt_score_at_split",
        "cv2_gap_error",
    ]

    with manifest_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (r["scan_stem"], STAGE_ORDER.index(r["stage"]))):
            writer.writerow(row)

    return manifest_path


def main() -> None:
    rows = collect_rows()
    manifest_path = write_manifest(rows)
    print("Usage:")
    print("- python tools/build_seam_manifest.py")
    print("- then open output/_debug/_seams/_seam_manifest.csv in Excel (or Sheets)")
    print(f"Wrote {len(rows)} row(s) to {manifest_path}")


if __name__ == "__main__":
    main()
