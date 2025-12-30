import sys
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi
from PIL import Image

# === Postcard geometry at 200 DPI ===
POSTCARD_MIN_W = 600
POSTCARD_MAX_W = 1100
POSTCARD_MIN_H = 900
POSTCARD_MAX_H = 1500

# Retry search
RETRY_RADIUS_PX = 150
RETRY_STEP_PX = 10

# Ink threshold (light cards safe)
INK_THRESHOLD = 190

# Debug flag (silent by default)
DEBUG_SEAMS = False

# Soft confidence threshold to trigger retry
CONFIDENCE_SOFT_THRESHOLD = 0.25


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


def plausible_postcard_dims(card_w, card_h):
    return (
        POSTCARD_MIN_W <= card_w <= POSTCARD_MAX_W and
        POSTCARD_MIN_H <= card_h <= POSTCARD_MAX_H
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

    cand = ndi.binary_closing(cand, structure=np.ones((5, 5)), border_value=False)
    labels, _ = ndi.label(cand)

    border_labels = np.unique(np.concatenate([
        labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]
    ]))
    border_labels = border_labels[border_labels != 0]

    return np.isin(labels, border_labels)


def build_bgmask(gray_s):
    bg_mode, border = classify_background(gray_s)
    thr_map = thresholds_from_border(bg_mode, border)

    mode = bg_mode if bg_mode in ("white", "black") else "white"
    thr = thr_map.get("WHITE_T", 230)

    return connected_bg_mask(gray_s, mode, thr)


# ============================================================
# Seam detection (axis-agnostic)
# ============================================================

def vertical_edge_map(gray_s):
    g = ndi.gaussian_filter(gray_s.astype(np.float32), 1.0)
    gx = ndi.sobel(g, axis=1)
    return ndi.gaussian_filter(np.abs(gx), 1.0)


def detect_card_boxes(gray_s, bgmask, axis):
    # foreground = non-background
    fg = ~bgmask

    labels, n = ndi.label(fg)
    if n < 2:
        return None

    slices = ndi.find_objects(labels)
    candidates = []

    H, W = gray_s.shape

    for sl in slices:
        h = sl[0].stop - sl[0].start
        w = sl[1].stop - sl[1].start

        if plausible_postcard_dims(w * 2, h * 2):  # account for downsample
            candidates.append((sl, w * h))

    if len(candidates) < 2:
        return None

    # Keep two largest
    candidates.sort(key=lambda x: x[1], reverse=True)
    (sl1, _), (sl2, _) = candidates[:2]

    # Compute gap between boxes (axis-explicit)
    if axis == "vertical":
        gap_start = min(sl1[1].stop, sl2[1].stop)
        gap_end = max(sl1[1].start, sl2[1].start)
    else:
        gap_start = min(sl1[0].stop, sl2[0].stop)
        gap_end = max(sl1[0].start, sl2[0].start)

    if gap_end <= gap_start:
        return None

    gap_width = gap_end - gap_start
    if gap_width < 0.03 * W:
        return None

    return (gap_start + gap_end) // 2


def find_boundary(gray_s, color_s, bgmask, axis, filename=None, orig_shape=None):
    Hs, Ws = gray_s.shape
    if Hs < 40 or Ws < 40:
        return None

    # Structural detection shortcut:
    # If two postcard-sized regions are detected, split between them
    # instead of using edge-based seam detection.
    structural_split = detect_card_boxes(gray_s, bgmask, axis)
    if structural_split is not None:
        log_debug(filename, axis, "structural detection used")
        return structural_split, 1.0

    band_top = int(0.15 * Hs)
    band_bot = int(0.85 * Hs)
    band = slice(band_top, band_bot)

    band_gray = gray_s[band].astype(np.float32)
    band_color = color_s[band].astype(np.float32)

    margin = max(4, int(0.07 * Ws))
    idx = np.arange(margin, Ws - margin)
    if idx.size == 0:
        return None

    edge = vertical_edge_map(gray_s)[band]
    weak_thr = np.percentile(edge, 60)
    strong_thr = np.percentile(edge, 90)

    seam_mask = (edge >= weak_thr) & (edge <= strong_thr)
    strong_mask = edge >= strong_thr

    seam_align = seam_mask.mean(axis=0)
    strong_align = strong_mask.mean(axis=0)
    bg_ratio = bgmask[band].mean(axis=0)
    ink_ratio = (gray_s[band] < INK_THRESHOLD).mean(axis=0)
    # NOTE:
    # bg_ratio enforces that seams pass through background (gaps),
    # not internal printed lines. ink_ratio suppresses text, dividers,
    # and structural features (e.g., bridges).
    # Geometry is a hard gate:
    # seams that cannot produce two plausible postcards
    # must not participate in scoring.

    col_mean = band_color.mean(axis=0)
    col_gray = band_gray.mean(axis=0)
    col_sq = (band_gray ** 2).mean(axis=0)

    cumsum = np.vstack([np.zeros((1, 3)), np.cumsum(col_mean, axis=0)])
    cumsum_g = np.concatenate([[0.0], np.cumsum(col_gray)])
    cumsum_sq = np.concatenate([[0.0], np.cumsum(col_sq)])

    band_w = int(np.clip(0.04 * Ws, 6, 40))

    def band_stats(start, end):
        w = np.maximum(1, end - start)
        mean = (cumsum[end] - cumsum[start]) / w[:, None]
        g = (cumsum_g[end] - cumsum_g[start]) / w
        v = (cumsum_sq[end] - cumsum_sq[start]) / w - g ** 2
        return mean, g, np.clip(v, 0, None)

    left_s = np.clip(idx - band_w, 0, Ws)
    right_e = np.clip(idx + band_w, 0, Ws)

    l_mean, l_g, l_v = band_stats(left_s, idx)
    r_mean, r_g, r_v = band_stats(idx, right_e)

    dissim = (
        np.mean(np.abs(l_mean - r_mean), axis=1)
        + 0.6 * np.sqrt(np.abs(l_v - r_v))
    )
    dis_thr = max(18.0, np.percentile(dissim, 78))
    anchor = int(idx[np.argmax(dissim)])

    # align all signals
    d = dissim
    c = bg_ratio[idx]
    s = seam_align[idx]
    st = strong_align[idx]

    scale = (orig_shape[1] / gray_s.shape[1]) if orig_shape else 2.0

    geom_valid = np.zeros_like(idx, dtype=bool)
    for i, s_idx in enumerate(idx):
        lw = s_idx
        rw = Ws - s_idx
        if axis == "vertical":
            left_dims = (lw * scale, Hs * scale)
            right_dims = (rw * scale, Hs * scale)
        else:
            left_dims = (Ws * scale, lw * scale)
            right_dims = (Ws * scale, rw * scale)

        geom_ok = (
            plausible_postcard_dims(*left_dims) and
            plausible_postcard_dims(*right_dims)
        )
        if geom_ok:
            geom_valid[i] = True

    valid = (
        (d >= dis_thr) &
        (c >= 0.60) &
        (ink_ratio[idx] <= 0.25) &
        (s >= 0.35) &
        (st <= 0.22)
    )
    valid = valid & geom_valid

    if not np.any(valid):
        log_debug(filename, axis, "no geometry-valid candidates; entering retry")
        best = None
    else:
        score = (
            0.5 * np.clip((d - dis_thr) / (dis_thr + 25), 0, 1) +
            0.3 * np.clip((c - 0.6) / 0.4, 0, 1) +
            0.2 * np.clip((s - 0.35) / 0.3, 0, 1)
        )
        score *= valid

        best = np.argmax(score)
        split = int(idx[best])
        confidence = float(score[best])

    if best is None:
        split = anchor
        confidence = 0.0

    left_w = split
    right_w = Ws - split

    needs_retry = (
        best is None or
        confidence < CONFIDENCE_SOFT_THRESHOLD or
        not (
            plausible_postcard_dims(*( (left_w * scale, Hs * scale) if axis == "vertical" else (Ws * scale, left_w * scale) )) and
            plausible_postcard_dims(*( (right_w * scale, Hs * scale) if axis == "vertical" else (Ws * scale, right_w * scale) ))
        )
    )

    retry_candidates = []
    if needs_retry:
        if best is None:
            log_debug(filename, axis, "retry due to missing initial candidate")
        elif confidence < CONFIDENCE_SOFT_THRESHOLD:
            log_debug(filename, axis, "retry due to low confidence")
        else:
            log_debug(filename, axis, "retry due to geometry failure")

        for dx in range(-RETRY_RADIUS_PX, RETRY_RADIUS_PX + 1, RETRY_STEP_PX):
            s_local = split + dx
            if s_local <= 0 or s_local >= Ws:
                continue

            lw = s_local
            rw = Ws - s_local

            if axis == "vertical":
                left_dims = (lw * scale, Hs * scale)
                right_dims = (rw * scale, Hs * scale)
            else:
                left_dims = (Ws * scale, lw * scale)
                right_dims = (Ws * scale, rw * scale)

            if not (
                plausible_postcard_dims(*left_dims) and
                plausible_postcard_dims(*right_dims)
            ):
                continue

            if bg_ratio[s_local] >= 0.60 and ink_ratio[s_local] <= 0.25:
                retry_score = (
                    0.6 * bg_ratio[s_local] +
                    0.3 * (1.0 - ink_ratio[s_local]) +
                    0.1 * (1.0 - abs(s_local - anchor) / RETRY_RADIUS_PX)
                )
                retry_candidates.append((retry_score, s_local))

        if not retry_candidates:
            log_debug(filename, axis, "retry exhausted without candidates")
            return None

        retry_candidates.sort(key=lambda x: x[0], reverse=True)
        split = retry_candidates[0][1]

    # Physical plausibility (geometry + background)
    # overrides statistical confidence.
    return split, confidence


# ============================================================
# Axis-aware split
# ============================================================

def split_once(img, axis, filename=None):
    color = np.array(img.convert("RGB"))
    gray = np.array(img.convert("L"))

    if axis == "horizontal":
        color = color.transpose(1, 0, 2)
        gray = gray.T

    color_s = color[::2, ::2]
    gray_s = gray[::2, ::2]

    bgmask = build_bgmask(gray_s)
    result = find_boundary(
        gray_s,
        color_s,
        bgmask,
        axis,
        filename=filename,
        orig_shape=gray.shape,
    )

    if not result:
        return [img.copy()]

    split_s, conf = result
    scale = gray.shape[1] / gray_s.shape[1]
    split = int(round(split_s * scale))

    H, W = gray.shape
    overlap = max(16, int(0.01 * W))

    if axis == "horizontal":
        top = img.crop((0, 0, img.width, split + overlap))
        bot = img.crop((0, max(0, split - overlap), img.width, img.height))
        return [top, bot]

    left = img.crop((0, 0, split + overlap, img.height))
    right = img.crop((max(0, split - overlap), 0, img.width, img.height))
    return [left, right]


# ============================================================
# Main
# ============================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python postcard_split.py <scan1> <scan2> ...")
        sys.exit(1)

    out_root = Path("output")
    out_root.mkdir(exist_ok=True)

    scans = [Path(p) for p in sys.argv[1:]]

    BACK_MIRROR_MAP = {1: 2, 2: 1, 3: 4, 4: 3}

    for scan_idx, scan_path in enumerate(scans):
        img = Image.open(scan_path)

        # Determine role purely by sequence
        is_front = (scan_idx % 2 == 0)
        pair_id = scan_idx // 2 + 1

        out_dir = out_root / f"pair_{pair_id:03d}"
        out_dir.mkdir(exist_ok=True)

        # ---- split (unchanged logic) ----
        parts = []
        for half in split_once(img, "horizontal", filename=scan_path.name):
            parts.extend(split_once(half, "vertical", filename=scan_path.name))

        if len(parts) != 4:
            print(f"WARNING: {scan_path.name} produced {len(parts)} parts")

        # ---- naming logic ----
        for idx, p in enumerate(parts, 1):
            if is_front:
                fname = f"postcard_{idx}_front.jpg"
            else:
                paired_idx = BACK_MIRROR_MAP[idx]
                fname = f"postcard_{paired_idx}_back.jpg"

            p.save(out_dir / fname, quality=95)

        role = "front" if is_front else "back"
        print(f"[OK] {scan_path.name} → {role} (pair {pair_id})")

    print("All scans processed.")


if __name__ == "__main__":
    main()
