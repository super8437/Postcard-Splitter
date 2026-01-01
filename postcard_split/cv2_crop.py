from pathlib import Path

import numpy as np
from PIL import Image

from postcard_split.cv2_bridge import require_cv2

# === Postcard geometry (physical, scaled by DPI) ===
BASE_DPI = 200
LONG_SIDE_RANGE_IN = (5.0, 6.5)
SHORT_SIDE_RANGE_IN = (3.0, 4.5)

__all__ = [
    "require_cv2",
    "bgmask_cv2",
    "clean_fgmask_cv2",
    "debug_save_bgmask",
    "classify_background",
    "thresholds_from_border",
    "fill_holes_cv2",
    "debug_save_mask",
    "plausible_postcard_dims",
    "find_postcard_rect_cv2",
    "rect_to_safe_aabb",
    "tight_crop_postcard_cv2",
]


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


def clean_fgmask_cv2(fgmask_bool: np.ndarray) -> np.ndarray:
    cv2 = require_cv2()
    if fgmask_bool.dtype != bool or fgmask_bool.ndim != 2:
        raise ValueError("clean_fgmask_cv2 expects a 2D boolean mask.")

    fg_u8 = np.ascontiguousarray(np.where(fgmask_bool, 255, 0).astype(np.uint8, copy=False))

    close_kernel = np.ones((7, 7), dtype=np.uint8)
    open_kernel = np.ones((5, 5), dtype=np.uint8)

    closed = cv2.morphologyEx(fg_u8, cv2.MORPH_CLOSE, close_kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, open_kernel)
    return opened > 0


def fill_holes_cv2(fgmask_bool: np.ndarray) -> np.ndarray:
    cv2 = require_cv2()
    if fgmask_bool.dtype != bool or fgmask_bool.ndim != 2:
        raise ValueError("fill_holes_cv2 expects a 2D boolean mask.")

    inv = np.logical_not(fgmask_bool)
    inv_u8 = np.ascontiguousarray(np.where(inv, 255, 0).astype(np.uint8, copy=False))

    seeds = _border_seeds(inv)
    if not seeds:
        return fgmask_bool.copy()

    fill_img = np.ascontiguousarray(inv_u8)
    mask = np.zeros((inv_u8.shape[0] + 2, inv_u8.shape[1] + 2), dtype=np.uint8)

    for sx, sy in seeds:
        if fill_img[sy, sx] != 255:
            continue
        cv2.floodFill(fill_img, mask, (int(sx), int(sy)), 128, loDiff=0, upDiff=0, flags=4)

    holes = (fill_img == 255) & inv
    if not np.any(holes):
        return fgmask_bool

    return np.logical_or(fgmask_bool, holes)


def debug_save_mask(mask_bool: np.ndarray, out_path: str) -> None:
    cv2 = require_cv2()
    mask_u8 = np.ascontiguousarray(np.where(mask_bool, 255, 0).astype(np.uint8, copy=False))
    cv2.imwrite(out_path, mask_u8)


def find_postcard_rect_cv2(gray: np.ndarray, dpi: float) -> dict | None:
    """
    Locate a robust postcard-like rectangle using contour detection and cv2.minAreaRect.
    Returns a dictionary with rect info or None if no plausible contour is found.
    """
    cv2 = require_cv2()
    if gray.dtype != np.uint8 or gray.ndim != 2:
        raise ValueError("find_postcard_rect_cv2 expects a uint8 grayscale image.")

    bg = bgmask_cv2(gray)
    fg = np.logical_not(bg)
    fg = clean_fgmask_cv2(fg)

    fg_u8 = np.ascontiguousarray(np.where(fg, 255, 0).astype(np.uint8, copy=False))
    contours_info = cv2.findContours(fg_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours_info) == 3:
        _, contours, _ = contours_info
    else:
        contours, _ = contours_info

    if not contours:
        return None

    H, W = gray.shape
    min_area = max(50.0, 0.002 * H * W)

    best = None
    for idx, contour in enumerate(contours):
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue

        rect = cv2.minAreaRect(contour)
        (_, _), (rw, rh), angle = rect
        plausible = bool(plausible_postcard_dims(rw, rh, dpi) or plausible_postcard_dims(rh, rw, dpi))

        x, y, bw, bh = cv2.boundingRect(contour)
        # Use inclusive low edges and an exclusive-ish high edge to flag AABBs that graze the border.
        touches_border = (x <= 1) or (y <= 1) or (x + bw >= W - 1) or (y + bh >= H - 1)

        score = area
        if plausible:
            score *= 1.4
        if touches_border:
            score *= 0.6

        if best is None or score > best["score"]:
            box_points = cv2.boxPoints(rect)
            best = {
                "rect": rect,
                "contour_area": area,
                "contour_idx": idx,
                "plausible": plausible,
                "touches_border": touches_border,
                "score": score,
                "box_points": box_points,
                "angle": angle,
                "bounding_rect": (x, y, bw, bh),
            }

    return best


def rect_to_safe_aabb(rect, W: int, H: int, pad_px: float = 0.0) -> tuple[int, int, int, int]:
    """Return a padded, clamped axis-aligned bounding box around a rotated rect."""
    cv2 = require_cv2()
    pts = cv2.boxPoints(rect)
    xs = pts[:, 0]
    ys = pts[:, 1]

    x0 = float(np.min(xs)) - pad_px
    y0 = float(np.min(ys)) - pad_px
    x1 = float(np.max(xs)) + pad_px
    y1 = float(np.max(ys)) + pad_px

    x0_i = int(np.clip(np.floor(x0), 0, max(0, W - 1)))
    y0_i = int(np.clip(np.floor(y0), 0, max(0, H - 1)))
    x1_i = int(np.clip(np.ceil(x1), 0, W))
    y1_i = int(np.clip(np.ceil(y1), 0, H))

    if x1_i <= x0_i:
        x1_i = min(W, x0_i + 1)
    if y1_i <= y0_i:
        y1_i = min(H, y0_i + 1)
    return x0_i, y0_i, x1_i, y1_i


def _debug_save_masks_and_overlay(
    debug_dir: Path,
    gray: np.ndarray,
    bgmask: np.ndarray,
    fgmask: np.ndarray,
    rect_info: dict,
    crop_box: tuple[int, int, int, int],
) -> None:
    cv2 = require_cv2()
    debug_dir.mkdir(parents=True, exist_ok=True)

    bgmask_u8 = np.where(bgmask, 255, 0).astype(np.uint8, copy=False)
    fgmask_u8 = np.where(fgmask, 255, 0).astype(np.uint8, copy=False)
    cv2.imwrite(str(debug_dir / "tight_crop_bgmask.png"), bgmask_u8)
    cv2.imwrite(str(debug_dir / "tight_crop_fgmask.png"), fgmask_u8)

    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    box_pts = rect_info.get("box_points")
    if box_pts is None:
        box_pts = cv2.boxPoints(rect_info["rect"])
    box_pts_i = np.intp(box_pts)
    cv2.drawContours(overlay, [box_pts_i], -1, (0, 255, 0), 2)
    x0, y0, x1, y1 = crop_box
    cv2.rectangle(overlay, (x0, y0), (x1 - 1, y1 - 1), (0, 0, 255), 2)
    cv2.imwrite(str(debug_dir / "tight_crop_overlay.png"), overlay)


def tight_crop_postcard_cv2(
    img_pil: Image.Image,
    dpi: float,
    debug: bool = False,
    debug_dir: Path | None = None,
) -> Image.Image:
    """
    Attempt a conservative tight crop around a detected postcard rectangle.

    If detection fails or the crop is implausibly small, the original image is returned.
    """
    try:
        gray = np.array(img_pil.convert("L"), dtype=np.uint8, copy=False)
        H, W = gray.shape

        rect_info = find_postcard_rect_cv2(gray, dpi)
        if rect_info is None:
            return img_pil

        pad_px = max(12, int(0.01 * min(W, H)))
        crop_box = rect_to_safe_aabb(rect_info["rect"], W, H, pad_px)
        x0, y0, x1, y1 = crop_box

        crop_area = (x1 - x0) * (y1 - y0)
        if crop_area < 0.5 * W * H:
            return img_pil

        if debug or debug_dir:
            bgmask = bgmask_cv2(gray)
            fgmask = clean_fgmask_cv2(np.logical_not(bgmask))
            if debug_dir is not None:
                _debug_save_masks_and_overlay(debug_dir, gray, bgmask, fgmask, rect_info, crop_box)

        return img_pil.crop((x0, y0, x1, y1))
    except Exception:
        return img_pil


if __name__ == "__main__":
    # Minimal smoke test for manual inspection.
    import argparse

    parser = argparse.ArgumentParser(description="Preview background mask and contour-based rectangle utilities.")
    parser.add_argument("input", help="Path to a grayscale image")
    parser.add_argument("output", help="Path to save the background mask PNG")
    args = parser.parse_args()

    cv2 = require_cv2()
    img = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise SystemExit(f"Failed to read image: {args.input}")
    mask = bgmask_cv2(img)
    debug_save_bgmask(img, mask, args.output)
