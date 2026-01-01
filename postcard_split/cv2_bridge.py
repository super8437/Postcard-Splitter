import numpy as np
from PIL import Image

__all__ = ["pil_to_bgr", "pil_to_gray", "bgr_to_pil", "require_cv2"]


def _require_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV support requires opencv-python-headless; install it to enable cv2 features."
        ) from exc
    return cv2


def require_cv2():
    return _require_cv2()


def pil_to_bgr(img: Image.Image) -> np.ndarray:
    rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
    return np.ascontiguousarray(rgb[..., ::-1])


def pil_to_gray(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("L"), dtype=np.uint8)


def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    cv2 = _require_cv2()
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)
