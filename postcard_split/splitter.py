import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import scipy.ndimage as ndi
from PIL import Image

import postcard_split.cv2_bridge as cv2_bridge

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


@dataclass
class DeskewResult:
    image: Image.Image
    angle_applied: float
    reason: str


@dataclass
class AngleEstimate:
    angle: float
    method: str
    confidence: float
    reason: str


def _normalize_card_angle(angle: float) -> float:
    """Map any angle to the nearest upright-aligned equivalent in [-45, 45]."""
    return ((angle + 45) % 90) - 45


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
        p70 = np.percentile(border, 70)
        p90 = np.percentile(border, 90)
        seed = int(np.clip(p90 - 2, 205, 252))
        grow = int(np.clip(p70 - 10, 190, seed))
        return {"WHITE_SEED_T": seed, "WHITE_GROW_T": grow}
    if bg_mode == "black":
        t = int(np.clip(np.percentile(border, 60) + 10, 25, 150))
        return {"BLACK_T": t}
    return {
        "WHITE_SEED_T": int(np.clip(np.percentile(border, 70) - 2, 205, 245)),
        "WHITE_GROW_T": int(np.clip(np.percentile(border, 40) - 8, 190, 235)),
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
    return reason


def pick_fill_color(gray_s):
    mode, border = classify_background(gray_s)
    if border.size == 0:
        return 240

    median_bg = float(np.median(border))
    if mode == "black":
        return int(np.clip(median_bg, 0, 50))
    if mode == "white":
        return int(np.clip(median_bg, 230, 255))
    return int(np.clip(median_bg, 200, 245))


def _largest_foreground_component(mask):
    labels, n = ndi.label(mask)
    if n == 0:
        return None, 0

    counts = np.bincount(labels.ravel())
    counts[0] = 0
    largest_idx = int(np.argmax(counts))
    if counts[largest_idx] < 30:
        return None, 0
    return labels == largest_idx, int(counts[largest_idx])


def _mask_bbox(mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return x0, y0, x1, y1


def _border_touch_fraction(mask: np.ndarray) -> float:
    top = mask[0, :].mean()
    bot = mask[-1, :].mean()
    left = mask[:, 0].mean()
    right = mask[:, -1].mean()
    return float(max(top, bot, left, right))


def _pick_card_component(fg: np.ndarray, dpi: float, scale_to_full: float = 2.0):
    """
    Choose a card-like component from a foreground mask.
    Border-touching components are heavily downweighted (often sleeve/scanner junk).
    DPI + plausible_postcard_dims is used as a strong prior when available.
    """
    labels, n = ndi.label(fg)
    if n == 0:
        return None

    H, W = fg.shape
    cx0, cy0 = (W - 1) / 2.0, (H - 1) / 2.0
    counts = np.bincount(labels.ravel())
    counts[0] = 0

    best = None
    best_score = -1.0
    for lab in range(1, n + 1):
        area = float(counts[lab])
        if area < 0.01 * H * W:
            continue
        comp = (labels == lab)
        bb = _mask_bbox(comp)
        if bb is None:
            continue
        x0, y0, x1, y1 = bb
        bw, bh = (x1 - x0), (y1 - y0)
        plausible = plausible_postcard_dims(bw * scale_to_full, bh * scale_to_full, dpi)

        ccx, ccy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        dist = np.hypot((ccx - cx0) / max(1.0, W), (ccy - cy0) / max(1.0, H))

        score = area
        if plausible:
            score *= 3.0
        touch = _border_touch_fraction(comp)
        if touch > 0 and not (plausible and area > 0.20 * H * W):
            penalty = np.interp(touch, [0.02, 0.20, 0.60], [0.95, 0.65, 0.25])
            score *= penalty
        score *= 1.0 / (1.0 + 3.0 * dist)

        if score > best_score:
            best_score = score
            best = comp
    return best


def _edge_hist_angle(gray_s: np.ndarray, ring: np.ndarray):
    """
    Angle from boundary-only gradients (avoids interior texture/text).
    Returns (angle_deg, confidence) or None.
    """
    g = ndi.gaussian_filter(gray_s.astype(np.float32), 1.0)
    gx = ndi.sobel(g, axis=1)
    gy = ndi.sobel(g, axis=0)
    mag = np.hypot(gx, gy)
    if ring.sum() < 50:
        return None

    thr = np.percentile(mag[ring], 75)
    m = ring & (mag >= thr)
    if m.sum() < 50:
        return None

    line_theta = np.degrees(np.arctan2(gy[m], gx[m]) + np.pi / 2.0)
    ang = ((line_theta + 45.0) % 90.0) - 45.0
    w = mag[m]

    bins = np.arange(-45.0, 45.5, 0.5)
    hist, edges = np.histogram(ang, bins=bins, weights=w)
    if hist.sum() <= 0:
        return None
    centers = (edges[:-1] + edges[1:]) / 2.0
    peak_idx = int(np.argmax(hist))
    peak = float(centers[peak_idx])

    sel = (ang >= peak - 2.0) & (ang <= peak + 2.0)
    if sel.sum() >= 30:
        peak = float(np.average(ang[sel], weights=w[sel]))

    conf = float(hist[peak_idx] / (hist.sum() + 1e-9))
    return peak, conf


# ============================================================
# Background mask
# ============================================================

def connected_bg_mask(gray_s, mode, thr):
    if mode == "white":
        seed_thr, grow_thr = thr
        seeds = gray_s >= seed_thr
        seeds = ndi.binary_closing(seeds, structure=np.ones((5, 5)), border_value=True)

        labels, _ = ndi.label(seeds)
        border_labels = np.unique(np.concatenate([
            labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]
        ]))
        border_labels = border_labels[border_labels != 0]
        border_seed = np.isin(labels, border_labels)

        grow_mask = gray_s >= grow_thr
        return ndi.binary_propagation(border_seed, mask=grow_mask)

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

    if mode == "white":
        thr_seed = thr_map.get("WHITE_SEED_T", 238)
        thr_grow = thr_map.get("WHITE_GROW_T", max(200, thr_seed - 12))
        return connected_bg_mask(gray_s, mode, (thr_seed, thr_grow))

    thr = thr_map.get("BLACK_T", 80)
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
# De-skew
# ============================================================

def estimate_card_angle(gray_s: np.ndarray, dpi: Optional[float] = None) -> AngleEstimate:
    """
    Return a correction angle in degrees (feed directly to PIL.Image.rotate)
    plus metadata for logging/decisioning.
    """
    dpi = dpi or BASE_DPI
    if min(gray_s.shape) < 40:
        return AngleEstimate(0.0, "pca", 0.0, "too-small")

    bgmask = build_bgmask(gray_s)
    fg = ndi.binary_fill_holes(~bgmask)
    fg = ndi.binary_opening(fg, structure=np.ones((3, 3)))

    card = _pick_card_component(fg, dpi=dpi, scale_to_full=2.0)
    if card is None:
        return AngleEstimate(0.0, "pca", 0.0, "no-card-component")

    ring = card & ~ndi.binary_erosion(card, structure=np.ones((5, 5)))

    coords = np.column_stack(np.nonzero(ring))
    if coords.shape[0] < 80:
        return AngleEstimate(0.0, "pca", 0.0, "insufficient-edge")

    coords_centered = coords - coords.mean(axis=0)
    cov = np.cov(coords_centered, rowvar=False)
    evals, evecs = np.linalg.eigh(cov)
    principal = evecs[:, np.argmax(evals)]
    pca_angle = _normalize_card_angle(float(np.degrees(np.arctan2(principal[0], principal[1]))))

    eh = _edge_hist_angle(gray_s, ring)
    if eh is not None:
        ang, conf = eh
        if conf >= 0.25:
            return AngleEstimate(float(ang), "edge-hist", float(conf), "edge-hist")
        if conf >= 0.18:
            blend = min(0.5, max(0.0, (conf - 0.18) / 0.07) * 0.5)
            blended = _normalize_card_angle(float((1 - blend) * pca_angle + blend * ang))
            return AngleEstimate(blended, "edge-blend", float(conf), "edge-blend")
        return AngleEstimate(float(pca_angle), "pca", float(conf), "edge-lowconf")

    return AngleEstimate(float(pca_angle), "pca", 0.0, "pca-fallback")


def deskew_postcard(img: Image.Image, filename=None, context: Optional[SplitContext] = None) -> DeskewResult:
    ctx = context or SplitContext(dpi=get_effective_dpi(img), debug=False)

    gray = np.array(img.convert("L"))
    gray_s = gray[::2, ::2]
    est = estimate_card_angle(gray_s, dpi=ctx.dpi)

    base_thresh = 0.35
    if est.method in ("edge-hist", "edge-blend"):
        if est.confidence >= 0.25:
            thresh = 0.20
        elif est.confidence >= 0.18:
            thresh = 0.28
        else:
            thresh = base_thresh
    else:
        thresh = base_thresh

    if abs(est.angle) < thresh:
        return DeskewResult(img, 0.0, est.reason)

    angle = float(np.clip(est.angle, -20.0, 20.0))

    fill = pick_fill_color(gray_s)
    fill_rgb = (fill, fill, fill) if img.mode != "L" else fill
    rotated = img.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=fill_rgb)

    rot_gray_s = np.array(rotated.convert("L"))[::2, ::2]
    est2 = estimate_card_angle(rot_gray_s, dpi=ctx.dpi)
    valid2 = (
        est2.reason not in {"too-small", "no-card-component", "insufficient-edge"} and
        ((est2.method in {"edge-hist", "edge-blend"} and est2.confidence >= 0.18))
    )
    before = abs(angle)
    after = abs(est2.angle)
    improved = after <= 0.65 * before and (before - after) >= 0.15
    if not valid2 or not improved:
        return DeskewResult(img, 0.0, est.reason + " | reverted: sanity-check")

    if ctx.debug:
        log_debug(filename, "deskew", f"method={est.method} conf={est.confidence:.2f} angle={angle:.2f}")

    return DeskewResult(rotated, angle, est.reason)


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
        "--use-cv2",
        action="store_true",
        help="Run a no-op OpenCV roundtrip before saving (requires opencv-python-headless).",
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
            deskewed = deskew_postcard(p, scan_path.name, context).image
            if args.use_cv2:
                cv2 = cv2_bridge.require_cv2()
                bgr = cv2_bridge.pil_to_bgr(deskewed)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                bgr2 = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                deskewed = cv2_bridge.bgr_to_pil(bgr2)
            if is_front:
                fname = f"postcard_{idx}_front.jpg"
            else:
                fname = f"postcard_{BACK_MIRROR_MAP[idx]}_back.jpg"
            deskewed.save(out_dir / fname, quality=95)

        print(f"[OK] {scan_path.name}")

    print("All scans processed.")


if __name__ == "__main__":
    main()
