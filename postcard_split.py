import sys
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi
from PIL import Image


def classify_background(gray_s):
    """Return bg_mode ('white'/'black'/'unknown') and border pixels."""
    Hs, Ws = gray_s.shape
    t = max(2, int(0.06 * min(Hs, Ws)))  # 6% border thickness

    top = gray_s[:t, :]
    bottom = gray_s[-t:, :]
    left = gray_s[:, :t]
    right = gray_s[:, -t:]

    border = np.concatenate([top.ravel(), bottom.ravel(), left.ravel(), right.ravel()])

    # trim outliers (reflections, dust)
    lo = np.percentile(border, 1)
    hi = np.percentile(border, 99)
    border_t = border[(border >= lo) & (border <= hi)]
    if border_t.size < 100:
        border_t = border

    med = float(np.median(border_t))
    frac_white = float(np.mean(border_t >= 230))
    frac_black = float(np.mean(border_t <= 60))

    # slightly permissive—your black sleeve has glare and perforation highlights
    if med >= 170 and frac_white >= 0.20:
        return "white", border_t
    if med <= 115 and frac_black >= 0.20:
        return "black", border_t
    return "unknown", border_t


def thresholds_from_border(bg_mode, border_pixels):
    if bg_mode == "white":
        # border in white sleeves is very bright; use a lower quantile to tolerate texture
        white_t = max(200, int(np.percentile(border_pixels, 25) - 5))
        return {"WHITE_T": white_t}
    if bg_mode == "black":
        # border in black sleeves is dark but can have glare; use mid quantile and clamp
        black_t = int(np.percentile(border_pixels, 60) + 10)
        black_t = int(np.clip(black_t, 25, 150))
        return {"BLACK_T": black_t}
    # unknown: provide both, derived from border anyway
    white_t = max(200, int(np.percentile(border_pixels, 35) - 5))
    black_t = int(np.clip(int(np.percentile(border_pixels, 50) + 10), 25, 150))
    return {"WHITE_T": white_t, "BLACK_T": black_t}


def connected_bg_mask(gray_s, mode, thr_value):
    if mode == "white":
        cand = gray_s >= thr_value
    else:
        cand = gray_s <= thr_value

    # Avoid injecting artificial background at the borders during morphology.
    cand = ndi.binary_closing(cand, structure=np.ones((5, 5), dtype=bool), border_value=False)
    labels, _ = ndi.label(cand)
    border_labels = np.unique(
        np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]])
    )
    border_labels = border_labels[border_labels != 0]
    bg = np.isin(labels, border_labels)
    return bg


def build_bg_and_depth(gray_s, bg_mode, border_pixels):
    thr_map = thresholds_from_border(bg_mode, border_pixels)
    modes = []
    if bg_mode == "white":
        modes = ["white"]
    elif bg_mode == "black":
        modes = ["black"]
    else:
        modes = ["white", "black"]

    best = None
    for mode in modes:
        key = "WHITE_T" if mode == "white" else "BLACK_T"
        if key not in thr_map:
            continue
        thr_val = thr_map[key]

        def attempt(thr):
            bg = connected_bg_mask(gray_s, mode, thr)
            cover = float(bg.mean())
            # if coverage is tiny, relax threshold (white: lower; black: higher)
            if cover < 0.01:
                adj = -10 if mode == "white" else 10
                thr2 = thr + adj
                bg2 = connected_bg_mask(gray_s, mode, thr2)
                if float(bg2.mean()) > cover:
                    return bg2, thr2
            # if coverage is huge, tighten threshold
            if cover > 0.90:
                adj = 10 if mode == "white" else -10
                thr2 = thr + adj
                bg2 = connected_bg_mask(gray_s, mode, thr2)
                if 0.05 < float(bg2.mean()) < cover:
                    return bg2, thr2
            return bg, thr

        bgmask, thr_used = attempt(thr_val)
        depth = ndi.distance_transform_edt(bgmask.astype(np.uint8))
        norm = max(gray_s.shape)
        score = float(np.percentile(depth, 90)) / max(1.0, norm)
        if best is None or score > best[0]:
            best = (score, mode, thr_used, bgmask, depth)

    if best is None:
        empty = np.zeros_like(gray_s, dtype=bool)
        return "unknown", {"WHITE_T": 230}, empty, np.zeros_like(gray_s, dtype=np.float32)

    _, mode, thr_used, bgmask, depth = best
    thr_return = {"WHITE_T": thr_used} if mode == "white" else {"BLACK_T": thr_used}
    return mode, thr_return, bgmask, depth


def best_scored_run(scores, start_idx, min_len, center_penalty=0.5, threshold=0.20, axis_length=None):
    scores = np.asarray(scores, dtype=np.float32)
    thr = threshold
    ok = scores >= thr
    if not np.any(ok):
        thr = max(0.10, threshold - 0.05)
        ok = scores >= thr
    if not np.any(ok):
        return None

    idx = np.flatnonzero(ok)
    runs = []
    a = b = idx[0]
    for i in idx[1:]:
        if i == b + 1:
            b = i
        else:
            runs.append((a, b))
            a = b = i
    runs.append((a, b))

    best = None
    for a, b in runs:
        length = b - a + 1
        if length < min_len:
            continue
        mean = float(scores[a : b + 1].mean())
        mid = (a + b) / 2.0
        quality = length * mean
        axis_len = axis_length if axis_length is not None else len(scores)
        split_pos = start_idx + mid
        center = axis_len / 2.0
        center_dist = abs(split_pos - center) / max(1.0, axis_len)
        final = quality / (1.0 + center_penalty * center_dist)
        cand = (final, mean, length, mid)
        if best is None or cand > best:
            best = cand
    if best is None:
        return None

    _, mean, length, mid = best
    split = int(round(start_idx + mid))
    return {"split": split, "mean": mean, "length": length}


def find_split_from_depth(depth, axis):
    Hs, Ws = depth.shape
    D = 2  # depth floor on downsampled grid
    if axis == "horizontal":
        # y-search band and x-exclusion per instructions
        y0, y1 = int(0.30 * Hs), int(0.70 * Hs)
        x0, x1 = int(0.10 * Ws), int(0.90 * Ws)
        cx0, cx1 = int(0.45 * Ws), int(0.55 * Ws)
        xs = np.r_[np.arange(x0, cx0), np.arange(cx1, x1)]
        band = depth[y0:y1, :][:, xs]
        row_score = np.mean(band >= D, axis=1)
        run = best_scored_run(
            row_score,
            start_idx=y0,
            min_len=max(3, int(0.01 * Hs)),
            axis_length=Hs,
        )
        return run
    else:
        # vertical split: focus near center width but allow moderate drift
        x0, x1 = int(0.20 * Ws), int(0.80 * Ws)
        y0, y1 = int(0.10 * Hs), int(0.90 * Hs)
        band = depth[y0:y1, x0:x1]
        col_score = np.mean(band >= D, axis=0)
        run = best_scored_run(
            col_score,
            start_idx=x0,
            min_len=max(3, int(0.01 * Ws)),
            axis_length=Ws,
        )
        return run


# ----------------------------
# sanity checks
# ----------------------------

def sanity_check_horizontal(gray_s, mode, thr, y_split_s):
    Hs, Ws = gray_s.shape
    cx0, cx1 = int(0.20 * Ws), int(0.80 * Ws)

    if mode == "white":
        white_t = thr["WHITE_T"]
        bg = gray_s[:, cx0:cx1] >= white_t
    else:
        black_t = thr["BLACK_T"]
        bg = gray_s[:, cx0:cx1] <= black_t

    top_fg = 1.0 - float(np.mean(bg[:y_split_s, :]))
    bot_fg = 1.0 - float(np.mean(bg[y_split_s:, :]))
    return (top_fg >= 0.10) and (bot_fg >= 0.10)


def sanity_check_vertical(gray_s, mode, thr, x_split_s):
    Hs, Ws = gray_s.shape
    ry0, ry1 = int(0.20 * Hs), int(0.80 * Hs)

    if mode == "white":
        white_t = thr["WHITE_T"]
        bg = gray_s[ry0:ry1, :] >= white_t
    else:
        black_t = thr["BLACK_T"]
        bg = gray_s[ry0:ry1, :] <= black_t

    left_fg = 1.0 - float(np.mean(bg[:, :x_split_s]))
    right_fg = 1.0 - float(np.mean(bg[:, x_split_s:]))
    return (left_fg >= 0.10) and (right_fg >= 0.10)


# ----------------------------
# Step 1 + Step 2
# ----------------------------

def horizontal_split(img):
    gray = np.array(img.convert("L"))
    H, W = gray.shape

    s = 2  # integer stride for downsample
    gray_s = gray[::s, ::s]

    bg_mode, border = classify_background(gray_s)
    mode, thr_map, _, depth = build_bg_and_depth(gray_s, bg_mode, border)
    run = find_split_from_depth(depth, axis="horizontal")

    if run:
        y_split_s = run["split"]
        ok = sanity_check_horizontal(gray_s, mode, thr_map, y_split_s)
        if ok:
            y_split = y_split_s * s
            print(
                f"[Step 1] bg={bg_mode}, mode={mode}, score={run['mean']:.3f}, "
                f"y_split={y_split}, thr={thr_map}"
            )
        else:
            y_split = H // 2
            print(f"[Step 1] bg={bg_mode}, mode=fallback(sanity), y_split={y_split}")
    else:
        y_split = H // 2
        print(f"[Step 1] bg={bg_mode}, mode=fallback(none), y_split={y_split}")

    overlap = max(20, int(0.01 * H))
    top = img.crop((0, 0, W, min(H, y_split + overlap)))
    bottom = img.crop((0, max(0, y_split - overlap), W, H))
    return top, bottom


def vertical_split(half_img, tag):
    gray = np.array(half_img.convert("L"))
    H, W = gray.shape

    s = 2  # integer stride for downsample
    gray_s = gray[::s, ::s]

    bg_mode, border = classify_background(gray_s)
    mode, thr_map, _, depth = build_bg_and_depth(gray_s, bg_mode, border)
    run = find_split_from_depth(depth, axis="vertical")

    if run:
        x_split_s = run["split"]
        ok = sanity_check_vertical(gray_s, mode, thr_map, x_split_s)

        if ok:
            x_split = x_split_s * s
            print(
                f"[Step 2:{tag}] bg={bg_mode}, mode={mode}, score={run['mean']:.3f}, "
                f"x_split={x_split}, thr={thr_map}"
            )
        else:
            x_split = W // 2
            print(f"[Step 2:{tag}] bg={bg_mode}, mode=fallback(sanity), x_split={x_split}")
    else:
        x_split = W // 2
        print(f"[Step 2:{tag}] bg={bg_mode}, mode=fallback(none), x_split={x_split}")

    overlap = max(20, int(0.01 * W))
    left = half_img.crop((0, 0, min(W, x_split + overlap), H))
    right = half_img.crop((max(0, x_split - overlap), 0, W, H))
    return left, right


# ----------------------------
# main
# ----------------------------

def main():
    if len(sys.argv) != 2:
        print("Usage: python postcard_split.py <input_image>")
        sys.exit(1)

    img = Image.open(sys.argv[1])

    top, bottom = horizontal_split(img)

    out = Path("output")
    out.mkdir(exist_ok=True)

    tl, tr = vertical_split(top, "top")
    bl, br = vertical_split(bottom, "bottom")

    tl.save(out / "postcard_1.jpg", quality=95)
    tr.save(out / "postcard_2.jpg", quality=95)
    bl.save(out / "postcard_3.jpg", quality=95)
    br.save(out / "postcard_4.jpg", quality=95)

    print("Saved 4 postcard images.")


if __name__ == "__main__":
    main()
