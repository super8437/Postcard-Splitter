from PIL import Image
import numpy as np

from postcard_split import split_once, SplitContext


def make_fake_scan(width=2400, height=1600):
    """
    Create a simple synthetic 2x1 postcard scan with a clear vertical seam.
    White background, dark cards.
    """
    img = np.ones((height, width), dtype=np.uint8) * 240

    # left postcard
    img[100:1500, 100:1100] = 60

    # right postcard
    img[100:1500, 1300:2300] = 60

    return Image.fromarray(img, mode="L").convert("RGB")


def test_vertical_split_exists():
    img = make_fake_scan()
    parts = split_once(img, "vertical", filename="synthetic.png")

    # This is the regression guard:
    assert len(parts) == 2, "Expected a vertical split but none occurred"
