## Postcard Splitter

Split a flatbed scan that contains a 2x2 grid of postcards into four cropped images.

### Usage

```bash
python postcard_split.py [-d|--debug-seams] <image_or_directory> ...
```

* Provide one or more image paths or directories (JPEG/PNG/TIFF files are discovered in directories).
* Use `-d` / `--debug-seams` to print seam-finding diagnostics, including the computed DPI and why a seam was accepted or rejected.

### DPI handling

The splitter reads embedded DPI metadata when available; otherwise it estimates a DPI from the overall scan size assuming a 2x2 postcard grid. Postcard geometry thresholds scale with this effective DPI, so higher-resolution scans (e.g., 300 DPI) still pass the plausibility checks used by the seam detector.
