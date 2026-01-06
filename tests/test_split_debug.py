from PIL import Image, ImageDraw

from postcard_split import SplitContext, split_once


def _build_simple_grid(dpi: int = 200):
    card_w, card_h = 900, 1300
    gap = 60
    margin = 40
    canvas_w = margin * 2 + card_w * 2 + gap
    canvas_h = margin * 2 + card_h * 2 + gap

    img = Image.new("RGB", (canvas_w, canvas_h), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    colors = [
        (210, 210, 210),
        (215, 215, 215),
        (212, 220, 210),
        (220, 212, 210),
    ]
    positions = [
        (margin, margin),
        (margin + card_w + gap, margin),
        (margin, margin + card_h + gap),
        (margin + card_w + gap, margin + card_h + gap),
    ]
    for color, (x0, y0) in zip(colors, positions):
        draw.rectangle([x0, y0, x0 + card_w, y0 + card_h], fill=color)

    img.info["dpi"] = (dpi, dpi)
    return img


def test_debug_dir_writes_do_not_crash(tmp_path):
    img = _build_simple_grid()
    ctx = SplitContext(dpi=200, debug=False)

    halves = split_once(img, "horizontal", "debug_grid", ctx, debug_dir=tmp_path / "h")
    assert len(halves) == 2
    parts = []
    for idx, half in enumerate(halves):
        parts.extend(split_once(half, "vertical", "debug_grid", ctx, debug_dir=tmp_path / f"v{idx}"))
    assert len(parts) == 4

    seam_meta = tmp_path.joinpath("h", "seam_meta.json")
    assert seam_meta.exists()
