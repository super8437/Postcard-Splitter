import sys
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi
from PIL import Image


# ----------------------------
# background classification
# ----------------------------

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


# ----------------------------
# background mask + depth
# ----------------------------

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


def build_bg_and_depth(gray_s, bg_mode, border):
    thr_map = thresholds_from_border(bg_mode, border)
    modes = [bg_mode] if bg_mode in ("white", "black") else ["white", "black"]

    best = None
    for mode in modes:
        key = "WHITE_T" if mode == "white" else "BLACK_T"
        if key not in thr_map:
            continue

        thr = thr_map[key]
        bg = connected_bg_mask(gray_s, mode, thr)
        depth = ndi.distance_transform_edt(bg.astype(np.uint8))
        score = float(np.percentile(depth, 90)) / max(gray_s.shape)

        if best is None or score > best[0]:
            best = (score, mode, thr, bg, depth)

    if best is None:
        empty = np.zeros_like(gray_s, bool)
        return "unknown", {"WHITE_T": 230}, empty, np.zeros_like(gray_s, float)

    _, mode, thr, bg, depth = best
    return mode, {("WHITE_T" if mode == "white" else "BLACK_T"): thr}, bg, depth


# ----------------------------
# edge-based white vertical logic
# ----------------------------

def vertical_edge_map(gray_s):
    g = ndi.gaussian_filter(gray_s.astype(np.float32), 1.0)
    gx = ndi.sobel(g, axis=1)
    return ndi.gaussian_filter(np.abs(gx), 1.0)


def find_vertical_boundary(gray_s, color_s, bgmask):
    Hs, Ws = gray_s.shape
    band_top = max(0, int(0.15 * Hs))
    band_bot = min(Hs, int(0.85 * Hs))
    if band_bot - band_top < 10:
        return None

    band_slice = slice(band_top, band_bot)
    band_gray = gray_s[band_slice].astype(np.float32)
    band_color = color_s[band_slice].astype(np.float32)

    edge = vertical_edge_map(gray_s)
    edge_band = edge[band_slice]

    weak_thr = float(np.percentile(edge_band, 55))
    strong_thr = float(np.percentile(edge_band, 92))

    seam_mask = (edge_band >= weak_thr) & (edge_band <= strong_thr)
    strong_mask = edge_band >= strong_thr

    seam_alignment = seam_mask.mean(axis=0)
    strong_alignment = strong_mask.mean(axis=0)

    bg_band = bgmask[band_slice]
    continuity = np.mean(bg_band | seam_mask, axis=0)

    # left-right dissimilarity using RGB means + gray variance
    col_mean = band_color.mean(axis=0)  # (W, 3)
    col_mean_gray = band_gray.mean(axis=0)
    col_sq_mean = (band_gray ** 2).mean(axis=0)

    cumsum = np.concatenate([np.zeros((1, 3)), np.cumsum(col_mean, axis=0)], axis=0)
    cumsum_gray = np.concatenate([[0.0], np.cumsum(col_mean_gray)])
    cumsum_sq = np.concatenate([[0.0], np.cumsum(col_sq_mean)])

    band_w = max(4, int(0.03 * Ws))
    idx = np.arange(Ws)

    left_start = np.clip(idx - band_w, 0, Ws)
    left_end = idx
    left_width = np.maximum(1, left_end - left_start)

    right_start = np.clip(idx + 1, 0, Ws)
    right_end = np.clip(idx + band_w + 1, 0, Ws)
    right_width = np.maximum(1, right_end - right_start)

    left_sum = cumsum[left_end] - cumsum[left_start]
    right_sum = cumsum[right_end] - cumsum[right_start]
    left_mean = left_sum / left_width[:, None]
    right_mean = right_sum / right_width[:, None]

    left_mean_gray = (cumsum_gray[left_end] - cumsum_gray[left_start]) / left_width
    right_mean_gray = (cumsum_gray[right_end] - cumsum_gray[right_start]) / right_width

    left_sq_mean = (cumsum_sq[left_end] - cumsum_sq[left_start]) / left_width
    right_sq_mean = (cumsum_sq[right_end] - cumsum_sq[right_start]) / right_width

    left_var = np.clip(left_sq_mean - left_mean_gray ** 2, 0.0, None)
    right_var = np.clip(right_sq_mean - right_mean_gray ** 2, 0.0, None)

    mean_diff = np.mean(np.abs(left_mean - right_mean), axis=1)
    var_diff = np.abs(left_var - right_var)
    dissimilarity = mean_diff + 0.5 * np.sqrt(var_diff)

    dissimilarity_thr = max(12.0, float(np.percentile(dissimilarity, 70)))
    dissimilarity_gate = dissimilarity >= dissimilarity_thr

    continuity_gate = continuity >= 0.70
    alignment_gate = seam_alignment >= 0.45
    strong_gate = strong_alignment <= 0.20

    candidate = dissimilarity_gate & continuity_gate & alignment_gate & strong_gate
    if not np.any(candidate):
        return None

    dissimilarity_strength = np.clip((dissimilarity - dissimilarity_thr) / (dissimilarity_thr + 20.0), 0.0, 1.0)
    continuity_strength = np.clip((continuity - 0.70) / 0.30, 0.0, 1.0)
    alignment_strength = np.clip((seam_alignment - 0.45) / 0.35, 0.0, 1.0)
    strong_penalty = np.clip(1.0 - strong_alignment / 0.25, 0.0, 1.0)

    score = (
        0.45 * dissimilarity_strength +
        0.30 * continuity_strength +
        0.25 * alignment_strength
    ) * strong_penalty

    score *= candidate
    split = int(np.argmax(score))
    confidence = float(score[split])

    if confidence <= 0.0:
        return None

    return split, confidence


# ----------------------------
# splitting
# ----------------------------

def horizontal_split(img):
    gray = np.array(img.convert("L"))
    H, W = gray.shape
    gray_s = gray[::2, ::2]

    bg_mode, _ = classify_background(gray_s)

    y = H // 2
    print(f"[Step 1] bg={bg_mode}, y_split={y}")
    return img.crop((0, 0, W, y)), img.crop((0, y, W, H))


def vertical_split(half_img, tag):
    color = np.array(half_img.convert("RGB"))
    gray = np.array(half_img.convert("L"))
    H, W = gray.shape
    color_s = color[::2, ::2]
    gray_s = gray[::2, ::2]

    bg_mode, border = classify_background(gray_s)
    _, _, bgmask, _ = build_bg_and_depth(gray_s, bg_mode, border)

    result = find_vertical_boundary(gray_s, color_s, bgmask)
    if not result:
        print(f"[Step 2:{tag}] no-split (no proven boundary)")
        return [half_img.copy()]

    x_s, conf = result
    x = int(x_s * 2)
    print(f"[Step 2:{tag}] split x={x} conf={conf:.3f}")

    overlap = max(20, int(0.01 * W))
    left = half_img.crop((0, 0, min(W, x + overlap), H))
    right = half_img.crop((max(0, x - overlap), 0, W, H))
    return [left.copy(), right.copy()]


# ----------------------------
# main
# ----------------------------

def main():
    if len(sys.argv) != 2:
        print("Usage: python postcard_split.py <image>")
        sys.exit(1)

    img = Image.open(sys.argv[1])
    out = Path("output")
    out.mkdir(exist_ok=True)

    top, bottom = horizontal_split(img)
    parts = []
    parts.extend(vertical_split(top, "top"))
    parts.extend(vertical_split(bottom, "bottom"))

    for idx, postcard in enumerate(parts, start=1):
        postcard.save(out / f"postcard_{idx}.jpg", quality=95)

    print("Saved outputs.")


if __name__ == "__main__":
    main()
