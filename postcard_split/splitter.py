import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import scipy.ndimage as ndi
from PIL import Image

# === Postcard geometry (physical, scaled by DPI) ===
BASE_DPI = 200
LONG_SIDE_RANGE_IN = (5.0, 6.5)
SHORT_SIDE_RANGE_IN = (3.0, 4.5)

# Retry search
RETRY_RADIUS_PX = 150
RETRY_STEP_PX = 10

# Ink threshold (light cards safe)
INK_THRESHOLD = 190

@dataclass
class SplitContext:
    dpi: float
    debug: bool

    @property
    def dpi_scale(self):
        return self.dpi / BASE_DPI


DEBUG_SEAMS = False  # set by CLI


@dataclass
class SeamResult:
    position: int
    confidence: float
    stage: str


# ============================================================
# DPI helpers
# ============================================================

def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if v]
    return float(sum(vals) / len(vals)) if vals else 0.0


def get_effective_dpi(img: Image.Image, grid: tuple[int, int] = (2, 2)) -> float:
    """Prefer embedded DPI; otherwise, estimate from a 2x2 postcard grid."""
    dpi_info = img.info.get("dpi")
    if isinstance(dpi_info, tuple) and len(dpi_info) >= 2:
        valid = [d for d in dpi_info[:2] if isinstance(d, (int, float)) and d > 10]
        if valid:
            return float(_mean(valid))

    # Estimate: assume a grid of postcards (default 2x2) roughly matching the
    # physical ranges below.
    long_mid = _mean(LONG_SIDE_RANGE_IN)
    short_mid = _mean(SHORT_SIDE_RANGE_IN)
    exp_w_in = long_mid * grid[0]
    exp_h_in = short_mid * grid[1]

    est_w_dpi = img.width / exp_w_in
    est_h_dpi = img.height / exp_h_in
    estimate = _mean([est_w_dpi, est_h_dpi]) or BASE_DPI

    # Clamp to a realistic range for flatbed scans.
    return float(np.clip(estimate, 180, 800))


# ============================================================
# Background classification
# ============================================================

def classify_background(gray_s):
    Hs, Ws = gray_s.shape
    t = max(2, int(0.06 * min(Hs, Ws)))

    border = np.concatenate([
        gray_s[:t, :].ravel(),
        gray_s[-t:, :].ravel(),
        gray_s[:, :t].ravel(),
        gray_s[:, -t:].ravel(),
    ])

    lo, hi = np.percentile(border, [1, 99])
    border = border[(border >= lo) & (border <= hi)]

    med = float(np.median(border))
    frac_white = float(np.mean(border >= 230))
    frac_black = float(np.mean(border <= 60))

    if med >= 170 and frac_white >= 0.20:
        return "white", border
    if med <= 115 and frac_black >= 0.20:
        return "black", border
    return "unknown", border


def thresholds_from_border(bg_mode, border):
    if bg_mode == "white":
        return {"WHITE_T": max(200, int(np.percentile(border, 25) - 5))}
    if bg_mode == "black":
        t = int(np.clip(np.percentile(border, 60) + 10, 25, 150))
        return {"BLACK_T": t}
    return {
        "WHITE_T": max(200, int(np.percentile(border, 35) - 5)),
        "BLACK_T": int(np.clip(np.percentile(border, 50) + 10, 25, 150)),
    }


def plausible_postcard_dims(card_w, card_h, dpi):
    long_min = LONG_SIDE_RANGE_IN[0] * dpi
    long_max = LONG_SIDE_RANGE_IN[1] * dpi
    short_min = SHORT_SIDE_RANGE_IN[0] * dpi
    short_max = SHORT_SIDE_RANGE_IN[1] * dpi

    long_side = max(card_w, card_h)
    short_side = min(card_w, card_h)

    return (
        short_min <= short_side <= short_max and
        long_min <= long_side <= long_max
    )


def log_debug(filename, axis, reason):
    if DEBUG_SEAMS:
        name = filename or "unknown"
        print(f"[DEBUG seam] file={name} axis={axis} reason={reason}")


# ============================================================
# Background mask
# ============================================================

def connected_bg_mask(gray_s, mode, thr):
    if mode == "white":
        cand = gray_s >= thr
    else:
        cand = gray_s <= thr

    cand = ndi.binary_closing(cand, structure=np.ones((5, 5)), border_value=True)
    labels, _ = ndi.label(cand)

    border_labels = np.unique(np.concatenate([
        labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]
    ]))
    border_labels = border_labels[border_labels != 0]

    return np.isin(labels, border_labels)


def build_bgmask(gray_s):
    bg_mode, border = classify_background(gray_s)
    thr_map = thresholds_from_border(bg_mode, border)

    if bg_mode in ("white", "black"):
        mode = bg_mode
    else:
        frac_white = float(np.mean(border >= 230))
        frac_black = float(np.mean(border <= 60))
        mode = "black" if frac_black >= frac_white else "white"

    thr_key = "WHITE_T" if mode == "white" else "BLACK_T"
    thr = thr_map.get(thr_key, 230 if mode == "white" else 80)

    return connected_bg_mask(gray_s, mode, thr)


# ============================================================
# Seam detection
# ============================================================

def vertical_edge_map(gray_s):
    g = ndi.gaussian_filter(gray_s.astype(np.float32), 1.0)
    gx = ndi.sobel(g, axis=1)
    return ndi.gaussian_filter(np.abs(gx), 1.0)


def detect_card_boxes(gray_s, bgmask, axis, dpi, filename=None):
    lenient_long_min = LONG_SIDE_RANGE_IN[0] * dpi * 0.85
    lenient_long_max = LONG_SIDE_RANGE_IN[1] * dpi * 1.15
    lenient_short_min = SHORT_SIDE_RANGE_IN[0] * dpi * 0.85
    lenient_short_max = SHORT_SIDE_RANGE_IN[1] * dpi * 1.15

    fg = ~bgmask
    labels, n = ndi.label(fg)
    if n < 2:
        log_debug(filename, axis, f"no fg components (n={n})")
        return None

    slices = ndi.find_objects(labels)
    candidates = []

    for sl in slices:
        h = sl[0].stop - sl[0].start
        w = sl[1].stop - sl[1].start
        long_side = max(w * 2, h * 2)
        short_side = min(w * 2, h * 2)
        plausible = (
            lenient_short_min <= short_side <= lenient_short_max and
            lenient_long_min <= long_side <= lenient_long_max
        )

        if plausible:
            cx = (sl[1].start + sl[1].stop) / 2.0
            cy = (sl[0].start + sl[0].stop) / 2.0
            candidates.append({
                "slice": sl,
                "area": w * h,
                "cx": cx,
                "cy": cy,
            })

    if len(candidates) < 2:
        log_debug(filename, axis, f"not enough postcard-like blobs ({len(candidates)})")
        return None

    best_pair: Optional[Tuple[dict, dict]] = None
    best_sep = -1.0

    for i, c1 in enumerate(candidates):
        for c2 in candidates[i + 1:]:
            if axis == "vertical":
                sep = abs(c1["cx"] - c2["cx"])
            else:
                sep = abs(c1["cy"] - c2["cy"])
            if sep > best_sep:
                best_sep = sep
                best_pair = (c1, c2)

    if best_pair is None:
        log_debug(filename, axis, "unable to pair postcard blobs")
        return None

    sl1 = best_pair[0]["slice"]
    sl2 = best_pair[1]["slice"]

    if axis == "vertical":
        left, right = (sl1, sl2) if best_pair[0]["cx"] <= best_pair[1]["cx"] else (sl2, sl1)
        gap_start = left[1].stop
        gap_end = right[1].start
    else:
        top, bottom = (sl1, sl2) if best_pair[0]["cy"] <= best_pair[1]["cy"] else (sl2, sl1)
        gap_start = top[0].stop
        gap_end = bottom[0].start

    if gap_end <= gap_start:
        seam = int(round((best_pair[0]["cx"] + best_pair[1]["cx"]) / 2.0)) if axis == "vertical" else \
            int(round((best_pair[0]["cy"] + best_pair[1]["cy"]) / 2.0))
        log_debug(filename, axis, "structural gap collapsed; using centroid midpoint")
        return np.clip(seam, 1, gray_s.shape[1] - 2 if axis == "vertical" else gray_s.shape[0] - 2)

    gap_span = gap_end - gap_start
    if gap_span < 0.03 * gray_s.shape[1]:
        seam = (gap_start + gap_end) // 2
        log_debug(filename, axis, f"gap narrow ({gap_span}px); using midpoint")
        return seam

    return (gap_start + gap_end) // 2


def find_boundary(gray_s, color_s, bgmask, axis, dpi, filename=None, orig_shape=None):
    Hs, Ws = gray_s.shape
    if Hs < 40 or Ws < 40:
        return None, "geometry-too-small"

    log_debug(
        filename,
        axis,
        f"searching boundary: sampled_shape=({Ws}x{Hs}), dpi={dpi:.1f}"
    )

    scale = orig_shape[1] / gray_s.shape[1] if orig_shape else 2.0
    full_w = Ws * scale
    full_h = Hs * scale

    def halves_plausible_at(s):
        lw, rw = s, Ws - s
        if axis == "vertical":
            return (
                plausible_postcard_dims(lw * scale, full_h, dpi) and
                plausible_postcard_dims(rw * scale, full_h, dpi)
            )
        return (
            plausible_postcard_dims(full_w, lw * scale, dpi) and
            plausible_postcard_dims(full_w, rw * scale, dpi)
        )

    structural = detect_card_boxes(gray_s, bgmask, axis, dpi, filename)
    if structural is not None:
        seam = int(np.clip(structural, 1, Ws - 2))
        log_debug(filename, axis, "stage=STRUCTURAL")
        return SeamResult(seam, 1.0, "STRUCTURAL"), "STRUCTURAL"

    band = slice(int(0.15 * Hs), int(0.85 * Hs))
    band_gray = gray_s[band]
    band_color = color_s[band]

    margin = max(4, int(0.07 * Ws))
    idx = np.arange(margin, Ws - margin)
    if idx.size == 0:
        return None, "geometry-too-narrow"

    edge = vertical_edge_map(gray_s)[band]
    seam_align = ((edge >= np.percentile(edge, 60)) &
                  (edge <= np.percentile(edge, 90))).mean(axis=0)
    strong_align = (edge >= np.percentile(edge, 90)).mean(axis=0)

    bg_ratio = bgmask[band].mean(axis=0)
    ink_ratio = (gray_s[band] < INK_THRESHOLD).mean(axis=0)

    col_mean = band_color.mean(axis=0)
    col_sq = (band_gray ** 2).mean(axis=0)

    cumsum = np.vstack([np.zeros((1, 3)), np.cumsum(col_mean, axis=0)])
    cumsum_sq = np.concatenate([[0.0], np.cumsum(col_sq)])

    bw = int(np.clip(0.04 * Ws, 6, 40))
    l = np.clip(idx - bw, 0, Ws)
    r = np.clip(idx + bw, 0, Ws)

    mean_l = (cumsum[idx] - cumsum[l]) / np.maximum(1, idx - l)[:, None]
    mean_r = (cumsum[r] - cumsum[idx]) / np.maximum(1, r - idx)[:, None]

    dissim = np.mean(np.abs(mean_l - mean_r), axis=1)
    anchor = int(idx[np.argmax(dissim)])

    geom_ok = np.array([halves_plausible_at(int(s)) for s in idx], dtype=bool)

    valid = (
        (bg_ratio[idx] >= 0.60) &
        (ink_ratio[idx] <= 0.25) &
        (seam_align[idx] >= 0.35) &
        (strong_align[idx] <= 0.22) &
        geom_ok
    )

    if np.any(valid):
        score = bg_ratio[idx] * valid
        best = np.argmax(score)
        split = int(idx[best])
        confidence = float(score[best])
        log_debug(
            filename,
            axis,
            f"stage=STRICT_HEURISTIC seam candidate count={valid.sum()} best_split={split} confidence={confidence:.3f}"
        )

        best_retry = None
        best_score = -1
        for dx in range(-RETRY_RADIUS_PX, RETRY_RADIUS_PX + 1, RETRY_STEP_PX):
            s = split + dx
            if s <= 0 or s >= Ws:
                continue
            if bg_ratio[s] >= 0.60 and ink_ratio[s] <= 0.25:
                score_retry = bg_ratio[s] - abs(dx) / RETRY_RADIUS_PX
                if score_retry > best_score:
                    best_score = score_retry
                    best_retry = s

        final_split = best_retry if best_retry is not None else split
        final_confidence = float(best_score) if best_retry is not None else confidence
        if best_retry is None:
            log_debug(filename, axis, "retry sweep did not improve seam; using strict heuristic result")
        else:
            log_debug(filename, axis, f"retry sweep seam={best_retry} score={best_score:.3f}")

        return SeamResult(int(final_split), final_confidence, "STRICT_HEURISTIC"), "STRICT_HEURISTIC"

    # Stage C — Geometry anchored fallback
    if halves_plausible_at(anchor) and bg_ratio[anchor] >= 0.40:
        log_debug(
            filename,
            axis,
            f"stage=GEOMETRY_FALLBACK anchor={anchor} bg_ratio={bg_ratio[anchor]:.3f}"
        )
        return SeamResult(int(anchor), 0.20, "GEOMETRY_FALLBACK"), "GEOMETRY_FALLBACK"

    # Stage D — Forced geometry split
    center = Ws // 2
    geometry_center_ok = halves_plausible_at(center)
    margin_allowance = 0.08
    grid_card_w = max(full_w / 2 * (1 - margin_allowance), full_w / 2 - margin_allowance * full_w)
    grid_card_h = max(full_h / 2 * (1 - margin_allowance), full_h / 2 - margin_allowance * full_h)
    grid_plausible = plausible_postcard_dims(grid_card_w, grid_card_h, dpi)

    if geometry_center_ok or grid_plausible:
        log_debug(
            filename,
            axis,
            "stage=FORCED_GRID_SPLIT (geometry_center_ok=%s, grid_plausible=%s)" %
            (geometry_center_ok, grid_plausible)
        )
        return SeamResult(int(center), 0.10, "FORCED_GRID_SPLIT"), "FORCED_GRID_SPLIT"

    log_debug(filename, axis, "no valid seam candidates after staged search (geometry rejected)")
    return None, "geometry-invalid"


# ============================================================
# Axis-aware split
# ============================================================

def split_once(img, axis, filename=None, context: Optional[SplitContext] = None):
    ctx = context or SplitContext(dpi=get_effective_dpi(img), debug=DEBUG_SEAMS)

    if ctx.debug:
        log_debug(
            filename,
            axis,
            f"starting split: size={img.size}, dpi={ctx.dpi:.1f}, scale={ctx.dpi_scale:.2f}"
        )

    color = np.array(img.convert("RGB"))
    gray = np.array(img.convert("L"))

    if axis == "horizontal":
        color = color.transpose(1, 0, 2)
        gray = gray.T

    gray_s = gray[::2, ::2]
    color_s = color[::2, ::2]

    scale = gray.shape[1] / gray_s.shape[1]
    bgmask = build_bgmask(gray_s)
    seam_result, reason = find_boundary(
        gray_s,
        color_s,
        bgmask,
        axis,
        ctx.dpi,
        filename,
        gray.shape
    )

    if seam_result is None:
        log_debug(filename, axis, f"no split found (reason={reason}, dpi={ctx.dpi:.1f})")
        if reason == "geometry-invalid":
            return [img.copy()]

        fallback_split = int(round((gray_s.shape[1] // 2) * scale))
        log_debug(
            filename,
            axis,
            f"forcing fallback seam at {fallback_split}px after staged search failure"
        )
        split = fallback_split
        stage_used = "FORCED_FALLBACK"
    else:
        split_s = seam_result.position
        split = int(round(split_s * scale))
        stage_used = seam_result.stage
    overlap = max(16, int(0.01 * gray.shape[1]))

    if ctx.debug:
        log_debug(
            filename,
            axis,
            f"split chosen at {split}px (stage={stage_used}, overlap={overlap}px)"
        )

    if axis == "horizontal":
        return [
            img.crop((0, 0, img.width, split + overlap)),
            img.crop((0, max(0, split - overlap), img.width, img.height))
        ]

    return [
        img.crop((0, 0, split + overlap, img.height)),
        img.crop((max(0, split - overlap), 0, img.width, img.height))
    ]


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Split scanned postcard grids into individual cards."
    )
    parser.add_argument(
        "-d",
        "--debug-seams",
        action="store_true",
        help="Enable verbose seam detection logging.",
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Image files or directories containing scans.",
    )
    args = parser.parse_args()

    global DEBUG_SEAMS
    DEBUG_SEAMS = args.debug_seams

    inputs = [Path(p) for p in args.inputs]
    scans = []

    for p in inputs:
        if p.is_dir():
            scans.extend(sorted(
                f for f in p.iterdir()
                if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff")
            ))
        elif p.is_file():
            scans.append(p)

    if not scans:
        print("No image files found.")
        sys.exit(1)

    out_root = Path("output")
    out_root.mkdir(exist_ok=True)

    BACK_MIRROR_MAP = {1: 2, 2: 1, 3: 4, 4: 3}

    for scan_idx, scan_path in enumerate(scans):
        img = Image.open(scan_path)
        context = SplitContext(
            dpi=get_effective_dpi(img),
            debug=DEBUG_SEAMS,
        )
        if DEBUG_SEAMS:
            log_debug(
                scan_path.name,
                "both",
                f"effective_dpi={context.dpi:.1f}, scale={context.dpi_scale:.2f}"
            )

        is_front = (scan_idx % 2 == 0)
        pair_id = scan_idx // 2 + 1

        out_dir = out_root / f"pair_{pair_id:03d}"
        out_dir.mkdir(exist_ok=True)

        parts = []
        for half in split_once(img, "horizontal", scan_path.name, context):
            parts.extend(split_once(half, "vertical", scan_path.name, context))

        if len(parts) != 4:
            print(f"WARNING: {scan_path.name} produced {len(parts)} parts")

        for idx, p in enumerate(parts, 1):
            if is_front:
                fname = f"postcard_{idx}_front.jpg"
            else:
                fname = f"postcard_{BACK_MIRROR_MAP[idx]}_back.jpg"
            p.save(out_dir / fname, quality=95)

        print(f"[OK] {scan_path.name}")

    print("All scans processed.")


if __name__ == "__main__":
    main()
