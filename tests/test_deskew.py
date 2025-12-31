import numpy as np
from PIL import Image, ImageDraw

from postcard_split import SplitContext, deskew_postcard
from postcard_split.splitter import estimate_card_angle


def build_rotated_postcard(angle):
    canvas = Image.new("RGB", (800, 600), (246, 246, 246))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([120, 80, 680, 520], fill=(50, 50, 60))
    return canvas.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=(246, 246, 246))


def measure_angle(img):
    gray = np.array(img.convert("L"))
    gray_s = gray[::2, ::2]
    angle, _ = estimate_card_angle(gray_s)
    return angle


def test_deskew_corrects_rotation():
    rotated = build_rotated_postcard(8.0)
    ctx = SplitContext(dpi=220, debug=False)

    result = deskew_postcard(rotated, filename="rotated", context=ctx)

    assert abs(abs(result.angle_applied) - 8.0) < 1.0
    residual = measure_angle(result.image)
    assert abs(residual) < 0.75


def test_deskew_skips_already_upright():
    upright = build_rotated_postcard(0.0)
    ctx = SplitContext(dpi=220, debug=False)

    result = deskew_postcard(upright, filename="upright", context=ctx)

    assert abs(result.angle_applied) < 0.05
    residual = measure_angle(result.image)
    assert abs(residual) < 0.5
