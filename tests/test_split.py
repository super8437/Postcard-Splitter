import unittest

from PIL import Image, ImageDraw

from postcard_split import split_once, SplitContext


def build_grid(card_size, gap, margin, dpi, bg_color=(0, 0, 0)):
    card_w, card_h = card_size
    canvas_w = margin * 2 + card_w * 2 + gap
    canvas_h = margin * 2 + card_h * 2 + gap

    bg = Image.new("RGB", (canvas_w, canvas_h), bg_color)
    draw = ImageDraw.Draw(bg)

    colors = [
        (220, 210, 210),
        (210, 220, 210),
        (210, 210, 220),
        (220, 220, 210),
    ]
    positions = [
        (margin, margin),
        (margin + card_w + gap, margin),
        (margin, margin + card_h + gap),
        (margin + card_w + gap, margin + card_h + gap),
    ]

    for color, (x0, y0) in zip(colors, positions):
        draw.rectangle([x0, y0, x0 + card_w, y0 + card_h], fill=color)

    bg.info["dpi"] = (dpi, dpi)
    return bg


class SplitSmokeTests(unittest.TestCase):
    def _assert_split(self, dpi, card_size, bg_color=(0, 0, 0)):
        img = build_grid(card_size, gap=60, margin=40, dpi=dpi, bg_color=bg_color)
        ctx = SplitContext(dpi=dpi, debug=False)

        parts = []
        for half in split_once(img, "horizontal", "test_grid", ctx):
            parts.extend(split_once(half, "vertical", "test_grid", ctx))

        self.assertEqual(4, len(parts))
        for p in parts:
            self.assertGreater(p.width, card_size[0] * 0.7)
            self.assertGreater(p.height, card_size[1] * 0.7)

    def test_split_at_200_dpi(self):
        self._assert_split(200, (900, 1300))

    def test_split_at_300_dpi(self):
        self._assert_split(300, (1350, 1950))

    def test_split_on_white_background(self):
        self._assert_split(220, (900, 1300), bg_color=(246, 246, 246))


if __name__ == "__main__":
    unittest.main()
