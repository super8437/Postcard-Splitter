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
# Seam detection
# ============================================================

def vertical_edge_map(gray_s):
    g = ndi.gaussian_filter(gray_s.astype(np.float32), 1.0)
    gx = ndi.sobel(g, axis=1)
    return ndi.gaussian_filter(np.abs(gx), 1.0)


def detect_card_boxes(gray_s, bgmask, axis):
    fg = ~bgmask
    labels, n = ndi.label(fg)
    if n < 2:
        return None

    slices = ndi.find_objects(labels)
    candidates = []

    for sl in slices:
        h = sl[0].stop - sl[0].start
        w = sl[1].stop - sl[1].start
        if plausible_postcard_dims(w * 2, h * 2):
            candidates.append((sl, w * h))

    if len(candidates) < 2:
        return None

    candidates.sort(key=lambda x: x[1], reverse=True)
    (sl1, _), (sl2, _) = candidates[:2]

    if axis == "vertical":
        gap_start = min(sl1[1].stop, sl2[1].stop)
        gap_end = max(sl1[1].start, sl2[1].start)
    else:
        gap_start = min(sl1[0].stop, sl2[0].stop)
        gap_end = max(sl1[0].start, sl2[0].start)

    if gap_end <= gap_start:
        return None

    if (gap_end - gap_start) < 0.03 * gray_s.shape[1]:
        return None

    return (gap_start + gap_end) // 2


def find_boundary(gray_s, color_s, bgmask, axis, filename=None, orig_shape=None):
    Hs, Ws = gray_s.shape
    if Hs < 40 or Ws < 40:
        return None

    structural = detect_card_boxes(gray_s, bgmask, axis)
    if structural is not None:
        log_debug(filename, axis, "structural detection used")
        return structural, 1.0

    band = slice(int(0.15 * Hs), int(0.85 * Hs))
    band_gray = gray_s[band]
    band_color = color_s[band]

    margin = max(4, int(0.07 * Ws))
    idx = np.arange(margin, Ws - margin)
    if idx.size == 0:
        return None

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

    scale = orig_shape[1] / gray_s.shape[1] if orig_shape else 2.0

    geom_ok = []
    for s in idx:
        lw, rw = s, Ws - s
        if axis == "vertical":
            geom_ok.append(
                plausible_postcard_dims(lw * scale, Hs * scale) and
                plausible_postcard_dims(rw * scale, Hs * scale)
            )
        else:
            geom_ok.append(
                plausible_postcard_dims(Ws * scale, lw * scale) and
                plausible_postcard_dims(Ws * scale, rw * scale)
            )

    geom_ok = np.array(geom_ok, dtype=bool)

    valid = (
        (bg_ratio[idx] >= 0.60) &
        (ink_ratio[idx] <= 0.25) &
        (seam_align[idx] >= 0.35) &
        (strong_align[idx] <= 0.22) &
        geom_ok
    )

    if not np.any(valid):
        # Fallback to dissimilarity anchor if geometry is valid
        lw = anchor
        rw = Ws - anchor

        if axis == "vertical":
            geom_ok = (
                plausible_postcard_dims(lw * scale, Hs * scale) and
                plausible_postcard_dims(rw * scale, Hs * scale)
            )
        else:
            geom_ok = (
                plausible_postcard_dims(Ws * scale, lw * scale) and
                plausible_postcard_dims(Ws * scale, rw * scale)
            )

        if geom_ok:
            log_debug(filename, axis, "fallback to dissimilarity anchor")
            return anchor, 0.20  # low but non-zero confidence

        return None

    score = bg_ratio[idx] * valid
    best = np.argmax(score)
    split = int(idx[best])
    confidence = float(score[best])

    # Retry sweep
    best = None
    best_score = -1
    for dx in range(-RETRY_RADIUS_PX, RETRY_RADIUS_PX + 1, RETRY_STEP_PX):
        s = split + dx
        if s <= 0 or s >= Ws:
            continue
        if bg_ratio[s] >= 0.60 and ink_ratio[s] <= 0.25:
            score = bg_ratio[s] - abs(dx) / RETRY_RADIUS_PX
            if score > best_score:
                best_score = score
                best = s

    if best is None:
        return None

    return best, best_score


# ============================================================
# Axis-aware split
# ============================================================

def split_once(img, axis, filename=None):
    color = np.array(img.convert("RGB"))
    gray = np.array(img.convert("L"))

    if axis == "horizontal":
        color = color.transpose(1, 0, 2)
        gray = gray.T

    gray_s = gray[::2, ::2]
    color_s = color[::2, ::2]

    bgmask = build_bgmask(gray_s)
    result = find_boundary(gray_s, color_s, bgmask, axis, filename, gray.shape)

    if not result:
        return [img.copy()]

    split_s, _ = result
    scale = gray.shape[1] / gray_s.shape[1]
    split = int(round(split_s * scale))
    overlap = max(16, int(0.01 * gray.shape[1]))

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
    if len(sys.argv) < 2:
        print("Usage: python postcard_split.py <image_or_directory> ...")
        sys.exit(1)

    inputs = [Path(p) for p in sys.argv[1:]]
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

        is_front = (scan_idx % 2 == 0)
        pair_id = scan_idx // 2 + 1

        out_dir = out_root / f"pair_{pair_id:03d}"
        out_dir.mkdir(exist_ok=True)

        parts = []
        for half in split_once(img, "horizontal", scan_path.name):
            parts.extend(split_once(half, "vertical", scan_path.name))

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
