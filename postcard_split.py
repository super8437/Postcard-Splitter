import sys
from pathlib import Path

import numpy as np
from PIL import Image


# ----------------------------
# helpers
# ----------------------------

def best_run(mask, min_len):
    runs = []
    start = None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_len:
                runs.append((start, i - 1))
            start = None
    if start is not None and len(mask) - start >= min_len:
        runs.append((start, len(mask) - 1))
    if not runs:
        return None

    center = len(mask) / 2
    # prefer longer runs; tie-break toward center
    runs.sort(key=lambda r: ((r[1] - r[0]), -abs(((r[0] + r[1]) / 2) - center)), reverse=True)
    return runs[0]


def smooth_1d(x, k=7):
    k = max(3, int(k))
    if k % 2 == 0:
        k += 1
    w = np.ones(k, dtype=np.float32) / k
    return np.convolve(x, w, mode="same")


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
        WHITE_T = max(200, int(np.percentile(border_pixels, 25) - 5))
        return {"WHITE_T": WHITE_T}
    if bg_mode == "black":
        # border in black sleeves is dark but can have glare; use mid quantile and clamp
        BLACK_T = int(np.percentile(border_pixels, 60) + 10)
        BLACK_T = int(np.clip(BLACK_T, 25, 150))
        return {"BLACK_T": BLACK_T}
    # unknown: provide both, derived from border anyway
    WHITE_T = max(200, int(np.percentile(border_pixels, 35) - 5))
    BLACK_T = int(np.clip(int(np.percentile(border_pixels, 50) + 10), 25, 150))
    return {"WHITE_T": WHITE_T, "BLACK_T": BLACK_T}


# ----------------------------
# stripe detectors
# ----------------------------

def find_horizontal_split(gray_s, bg_mode, border_pixels):
    Hs, Ws = gray_s.shape
    thr = thresholds_from_border(bg_mode, border_pixels)

    # use central width so left/right borders don't dominate
    cx0, cx1 = int(0.20 * Ws), int(0.80 * Ws)

    # only search for a split near the middle of the image
    sy0, sy1 = int(0.25 * Hs), int(0.75 * Hs)

    candidates = []

    if bg_mode in ("white", "unknown") and "WHITE_T" in thr:
        WHITE_T = thr["WHITE_T"]
        row_bg = np.mean(gray_s[:, cx0:cx1] >= WHITE_T, axis=1)
        row_bg_s = smooth_1d(row_bg, k=max(7, int(0.02 * Hs)))

        # white sleeves: background stripe should be very strong
        good = row_bg_s >= 0.92
        run = best_run(good[sy0:sy1], min_len=max(3, int(0.01 * Hs)))
        if run:
            r0, r1 = run
            y_split_s = sy0 + (r0 + r1) // 2
            score = float(np.mean(row_bg_s[y_split_s - 2:y_split_s + 3]))
            candidates.append(("white", score, y_split_s, thr))

    if bg_mode in ("black", "unknown") and "BLACK_T" in thr:
        BLACK_T = thr["BLACK_T"]
        row_bg = np.mean(gray_s[:, cx0:cx1] <= BLACK_T, axis=1)
        row_bg_s = smooth_1d(row_bg, k=max(7, int(0.02 * Hs)))

        # black sleeves: allow more texture/glare
        good = row_bg_s >= 0.80
        run = best_run(good[sy0:sy1], min_len=max(3, int(0.01 * Hs)))
        if run:
            r0, r1 = run
            y_split_s = sy0 + (r0 + r1) // 2
            score = float(np.mean(row_bg_s[y_split_s - 2:y_split_s + 3]))
            candidates.append(("black", score, y_split_s, thr))

    if not candidates:
        return None

    # choose best score
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0]  # (mode, score, y_split_s, thr)


def find_vertical_split(gray_s, bg_mode, border_pixels):
    Hs, Ws = gray_s.shape
    thr = thresholds_from_border(bg_mode, border_pixels)

    # use central height so top/bottom margins don't dominate
    ry0, ry1 = int(0.20 * Hs), int(0.80 * Hs)

    # only search near center width
    sx0, sx1 = int(0.25 * Ws), int(0.75 * Ws)

    candidates = []

    if bg_mode in ("white", "unknown") and "WHITE_T" in thr:
        WHITE_T = thr["WHITE_T"]
        col_bg = np.mean(gray_s[ry0:ry1, :] >= WHITE_T, axis=0)
        col_bg_s = smooth_1d(col_bg, k=max(7, int(0.02 * Ws)))

        good = col_bg_s >= 0.92
        run = best_run(good[sx0:sx1], min_len=max(3, int(0.01 * Ws)))
        if run:
            r0, r1 = run
            x_split_s = sx0 + (r0 + r1) // 2
            score = float(np.mean(col_bg_s[x_split_s - 2:x_split_s + 3]))
            candidates.append(("white", score, x_split_s, thr))

    if bg_mode in ("black", "unknown") and "BLACK_T" in thr:
        BLACK_T = thr["BLACK_T"]
        col_bg = np.mean(gray_s[ry0:ry1, :] <= BLACK_T, axis=0)
        col_bg_s = smooth_1d(col_bg, k=max(7, int(0.02 * Ws)))

        good = col_bg_s >= 0.80
        run = best_run(good[sx0:sx1], min_len=max(3, int(0.01 * Ws)))
        if run:
            r0, r1 = run
            x_split_s = sx0 + (r0 + r1) // 2
            score = float(np.mean(col_bg_s[x_split_s - 2:x_split_s + 3]))
            candidates.append(("black", score, x_split_s, thr))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0]


def sanity_check_horizontal(gray_s, mode, thr, y_split_s):
    Hs, Ws = gray_s.shape
    cx0, cx1 = int(0.20 * Ws), int(0.80 * Ws)

    if mode == "white":
        WHITE_T = thr["WHITE_T"]
        bg = gray_s[:, cx0:cx1] >= WHITE_T
    else:
        BLACK_T = thr["BLACK_T"]
        bg = gray_s[:, cx0:cx1] <= BLACK_T

    top_fg = 1.0 - float(np.mean(bg[:y_split_s, :]))
    bot_fg = 1.0 - float(np.mean(bg[y_split_s:, :]))
    return (top_fg >= 0.10) and (bot_fg >= 0.10)


def sanity_check_vertical(gray_s, mode, thr, x_split_s):
    Hs, Ws = gray_s.shape
    ry0, ry1 = int(0.20 * Hs), int(0.80 * Hs)

    if mode == "white":
        WHITE_T = thr["WHITE_T"]
        bg = gray_s[ry0:ry1, :] >= WHITE_T
    else:
        BLACK_T = thr["BLACK_T"]
        bg = gray_s[ry0:ry1, :] <= BLACK_T

    left_fg = 1.0 - float(np.mean(bg[:, :x_split_s]))
    right_fg = 1.0 - float(np.mean(bg[:, x_split_s:]))
    return (left_fg >= 0.10) and (right_fg >= 0.10)


# ----------------------------
# Step 1 + Step 2
# ----------------------------

def horizontal_split(img):
    gray = np.array(img.convert("L"))
    H, W = gray.shape

    s = 2
    gray_s = gray[::s, ::s]

    bg_mode, border = classify_background(gray_s)
    candidate = find_horizontal_split(gray_s, bg_mode, border)

    if candidate:
        mode, score, y_split_s, thr = candidate
        ok = sanity_check_horizontal(gray_s, mode, thr, y_split_s)

        if ok:
            y_split = y_split_s * s
            print(f"[Step 1] bg={bg_mode}, mode={mode}, score={score:.3f}, y_split={y_split}")
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

    s = 2
    gray_s = gray[::s, ::s]

    bg_mode, border = classify_background(gray_s)
    candidate = find_vertical_split(gray_s, bg_mode, border)

    if candidate:
        mode, score, x_split_s, thr = candidate
        ok = sanity_check_vertical(gray_s, mode, thr, x_split_s)

        if ok:
            x_split = x_split_s * s
            print(f"[Step 2:{tag}] bg={bg_mode}, mode={mode}, score={score:.3f}, x_split={x_split}")
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
