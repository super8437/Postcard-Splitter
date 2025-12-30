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


def find_vertical_split_white(gray_s, bgmask, white_t, depth):
    Hs, Ws = gray_s.shape

    edge = vertical_edge_map(gray_s)
    col_edge = edge.mean(axis=0)
    col_mean = gray_s.mean(axis=0)

    norm = max(1e-3, np.percentile(col_edge, 90))
    edge_density = np.clip(col_edge / norm, 0, 1)
    edge_absence = 1.0 - edge_density

    fg_density = np.mean(gray_s < white_t, axis=0)
    bg_clear = 1.0 - fg_density

    low_edge = edge <= np.percentile(edge, 35)
    stable_gap = np.mean(bgmask & low_edge, axis=0)

    band = max(3, int(0.01 * Ws))
    idx = np.arange(Ws)
    col_cumsum = np.concatenate([[0.0], np.cumsum(col_mean)])

    left_start = np.maximum(0, idx - band)
    left_end = idx
    left_sum = col_cumsum[left_end + 1] - col_cumsum[left_start]
    left_width = np.maximum(1, left_end - left_start)
    left_mean = left_sum / left_width

    right_start = idx
    right_end = np.minimum(Ws, idx + band)
    right_sum = col_cumsum[right_end] - col_cumsum[right_start]
    right_width = np.maximum(1, right_end - right_start)
    right_mean = right_sum / right_width

    contrast = np.abs(left_mean - right_mean)
    contrast_thr = 10.0
    contrast_gate = contrast >= contrast_thr
    contrast_norm = np.clip((contrast - contrast_thr) / max(1.0, contrast_thr), 0, 1)

    continuity = np.mean(bgmask, axis=0)
    continuity_gate = continuity >= 0.25

    center = np.linspace(-1, 1, Ws)
    center_bias = 1.0 - np.abs(center)

    score = (
        0.45 * bg_clear +
        0.45 * edge_absence +
        0.10 * center_bias
    )
    score *= (0.5 + 0.5 * contrast_norm)
    score *= (0.5 + 0.5 * stable_gap)
    score *= contrast_gate & continuity_gate

    strong_edge = edge_density >= 0.6
    dilate = max(2, int(0.006 * Ws))
    no_cut = ndi.binary_dilation(strong_edge, iterations=dilate)
    score[no_cut] = 0.0

    idx = np.where(score >= 0.15)[0]
    if idx.size == 0:
        return None

    split = int(idx[np.argmax(score[idx])])
    confidence = (
        float(score[split]) *
        (1.0 - edge_density[split]) *
        (0.5 + 0.5 * stable_gap[split])
    )

    return split, confidence


# ----------------------------
# splitting
# ----------------------------

def horizontal_split(img):
    gray = np.array(img.convert("L"))
    H, W = gray.shape
    gray_s = gray[::2, ::2]

    bg_mode, border = classify_background(gray_s)
    mode, thr, bgmask, depth = build_bg_and_depth(gray_s, bg_mode, border)

    y = H // 2
    print(f"[Step 1] bg={bg_mode}, y_split={y}")
    return img.crop((0, 0, W, y)), img.crop((0, y, W, H))


def vertical_split(half_img, tag):
    gray = np.array(half_img.convert("L"))
    H, W = gray.shape
    gray_s = gray[::2, ::2]

    bg_mode, border = classify_background(gray_s)
    mode, thr, bgmask, depth = build_bg_and_depth(gray_s, bg_mode, border)

    if mode == "white":
        result = find_vertical_split_white(gray_s, bgmask, thr["WHITE_T"], depth)
        if not result:
            print(f"[Step 2:{tag}] no-split (low confidence)")
            return half_img.copy(), half_img.copy()

        x_s, conf = result
        if conf < 0.07:
            print(f"[Step 2:{tag}] no-split (conf={conf:.3f})")
            return half_img.copy(), half_img.copy()

        x = x_s * 2
        print(f"[Step 2:{tag}] white split x={x} conf={conf:.3f}")
    else:
        x = W // 2
        print(f"[Step 2:{tag}] black fallback split x={x}")

    overlap = max(20, int(0.01 * W))
    left = half_img.crop((0, 0, min(W, x + overlap), H))
    right = half_img.crop((max(0, x - overlap), 0, W, H))
    return left, right


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
    tl, tr = vertical_split(top, "top")
    bl, br = vertical_split(bottom, "bottom")

    tl.save(out / "postcard_1.jpg", quality=95)
    tr.save(out / "postcard_2.jpg", quality=95)
    bl.save(out / "postcard_3.jpg", quality=95)
    br.save(out / "postcard_4.jpg", quality=95)

    print("Saved outputs.")


if __name__ == "__main__":
    main()
