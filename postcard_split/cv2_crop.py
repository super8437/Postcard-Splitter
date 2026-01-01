import numpy as np

from postcard_split.cv2_bridge import require_cv2

__all__ = ["require_cv2", "bgmask_cv2", "debug_save_bgmask", "classify_background", "thresholds_from_border"]


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


def _binary_close_border_true(mask_u8: np.ndarray, ksize: int = 5) -> np.ndarray:
    cv2 = require_cv2()
    if mask_u8.dtype != np.uint8:
        raise ValueError("mask_u8 must be uint8")
    if ksize <= 1:
        return np.ascontiguousarray(mask_u8)
    if ksize % 2 == 0:
        ksize += 1  # auto-adjust to nearest odd size
    kernel = np.ones((ksize, ksize), dtype=np.uint8)
    r = ksize // 2
    padded = cv2.copyMakeBorder(
        np.ascontiguousarray(mask_u8),
        r,
        r,
        r,
        r,
        borderType=cv2.BORDER_CONSTANT,
        value=255,
    )
    closed = cv2.morphologyEx(padded, cv2.MORPH_CLOSE, kernel)
    return closed[r:-r, r:-r]


def _border_seeds(seed_mask: np.ndarray) -> list[tuple[int, int]]:
    H, W = seed_mask.shape
    seed_border = np.zeros_like(seed_mask, dtype=bool)
    seed_border[0, :] = seed_mask[0, :]
    seed_border[-1, :] = seed_mask[-1, :]
    seed_border[:, 0] |= seed_mask[:, 0]
    seed_border[:, -1] |= seed_mask[:, -1]

    coords = np.argwhere(seed_border)
    if coords.size == 0:
        return []

    max_seeds = 2048
    if len(coords) > max_seeds:
        idx = np.linspace(0, len(coords) - 1, num=max_seeds, dtype=int)
        coords = coords[idx]

    return [(int(xy[1]), int(xy[0])) for xy in coords]


def bgmask_cv2(gray: np.ndarray) -> np.ndarray:
    if gray.dtype != np.uint8:
        raise ValueError("bgmask_cv2 expects a uint8 grayscale image.")
    if gray.ndim != 2:
        raise ValueError("bgmask_cv2 expects a 2D grayscale image.")

    cv2 = require_cv2()

    bg_mode, border = classify_background(gray)
    thr_map = thresholds_from_border(bg_mode, border)

    if bg_mode in ("white", "black"):
        mode = bg_mode
    else:
        frac_white = float(np.mean(border >= 230))
        frac_black = float(np.mean(border <= 60))
        mode = "black" if frac_black >= frac_white else "white"

    if mode == "white":
        seed_thr = thr_map.get("WHITE_SEED_T", 238)
        grow_thr = thr_map.get("WHITE_GROW_T", max(200, seed_thr - 12))
        seed_mask_u8 = np.where(gray >= seed_thr, 255, 0).astype(np.uint8, copy=False)
        seed_mask_u8 = _binary_close_border_true(seed_mask_u8, 5)
        seed_mask = seed_mask_u8 > 0
        candidate_u8 = np.where(gray >= grow_thr, 255, 0).astype(np.uint8, copy=False)
    else:
        thr = thr_map.get("BLACK_T", 80)
        cand_u8 = np.where(gray <= thr, 255, 0).astype(np.uint8, copy=False)
        cand_u8 = _binary_close_border_true(cand_u8, 5)
        seed_mask = cand_u8 > 0
        candidate_u8 = cand_u8

    seeds = _border_seeds(seed_mask)
    if not seeds:
        return np.zeros_like(gray, dtype=bool)

    fill_img = np.ascontiguousarray(candidate_u8)
    mask = np.zeros((gray.shape[0] + 2, gray.shape[1] + 2), dtype=np.uint8)

    for sx, sy in seeds:
        if fill_img[sy, sx] != 255:
            continue
        cv2.floodFill(fill_img, mask, (int(sx), int(sy)), 128, loDiff=0, upDiff=0, flags=4)

    return fill_img == 128


def debug_save_bgmask(gray: np.ndarray, bgmask: np.ndarray, out_path: str) -> None:
    cv2 = require_cv2()
    overlay = np.where(bgmask, 255, 0).astype(np.uint8, copy=False)
    cv2.imwrite(out_path, overlay)


if __name__ == "__main__":
    # Minimal smoke test for manual inspection.
    import argparse

    parser = argparse.ArgumentParser(description="Generate a background mask preview using cv2 flood fill.")
    parser.add_argument("input", help="Path to a grayscale image")
    parser.add_argument("output", help="Path to save the background mask PNG")
    args = parser.parse_args()

    cv2 = require_cv2()
    img = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise SystemExit(f"Failed to read image: {args.input}")
    mask = bgmask_cv2(img)
    debug_save_bgmask(img, mask, args.output)
