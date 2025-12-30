import sys
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi
from PIL import Image


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


def find_boundary(gray_s, color_s, bgmask):
    Hs, Ws = gray_s.shape
    if Hs < 40 or Ws < 40:
        return None

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
    continuity = np.mean(bgmask[band] | seam_mask, axis=0)

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

    # align all signals
    d = dissim
    c = continuity[idx]
    s = seam_align[idx]
    st = strong_align[idx]

    valid = (
        (d >= dis_thr) &
        (c >= 0.60) &
        (s >= 0.35) &
        (st <= 0.22)
    )

    if not np.any(valid):
        return None

    score = (
        0.5 * np.clip((d - dis_thr) / (dis_thr + 25), 0, 1) +
        0.3 * np.clip((c - 0.6) / 0.4, 0, 1) +
        0.2 * np.clip((s - 0.35) / 0.3, 0, 1)
    )
    score *= valid

    best = np.argmax(score)
    split = int(idx[best])
    confidence = float(score[best])

    if confidence < 0.15:
        return None

    return split, confidence


# ============================================================
# Axis-aware split
# ============================================================

def split_once(img, axis):
    color = np.array(img.convert("RGB"))
    gray = np.array(img.convert("L"))

    if axis == "horizontal":
        color = color.transpose(1, 0, 2)
        gray = gray.T

    color_s = color[::2, ::2]
    gray_s = gray[::2, ::2]

    bgmask = build_bgmask(gray_s)
    result = find_boundary(gray_s, color_s, bgmask)

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
        for half in split_once(img, "horizontal"):
            parts.extend(split_once(half, "vertical"))

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
