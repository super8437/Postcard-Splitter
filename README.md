POSTCARD SPLITTER — README (plain text)

Overview
--------
This project splits flatbed scans of postcards laid out in a grid into individual postcard images.

The command-line tool is optimized for a common “2x2 grid per scan” workflow and will:
  1) split the scan into top/bottom halves,
  2) split each half left/right to yield 4 cards,
  3) deskew each card,
  4) optionally apply a conservative OpenCV-based tight crop,
  5) save outputs as JPEGs.

There is also a small library API if you want to call the splitter from your own code.


Requirements
------------
- Python 3.10+
- Dependencies installed via pip (see requirements.txt / pyproject.toml):
  - Pillow
  - NumPy
  - SciPy
  - OpenCV (opencv-python-headless) is included as a dependency, but most OpenCV behavior is behind flags.


Install
-------
From the repo root:

  python -m pip install -r requirements.txt

Or install the package:

  python -m pip install .


Quick start (CLI)
-----------------
Run the module entrypoint:

  python -m postcard_split.splitter <image_or_directory> [more_inputs...]

Inputs can be:
- image files (.jpg, .jpeg, .png, .tif, .tiff)
- directories (the CLI scans the immediate directory and processes supported image files in sorted filename order)


Output layout
-------------
Outputs are written under an “output/” folder in the current working directory.

The CLI assumes scans come in front/back pairs in order. For each pair it creates:

  output/
    pair_001/
      postcard_1_front.jpg
      postcard_2_front.jpg
      postcard_3_front.jpg
      postcard_4_front.jpg
      postcard_1_back.jpg
      postcard_2_back.jpg
      postcard_3_back.jpg
      postcard_4_back.jpg
    pair_002/
      ...

Important labeling behavior:
- Scan index 0 is treated as the “front”, scan index 1 as the “back”, etc.
- Back images are mapped with a left/right mirror inside each row: 1<->2 and 3<->4.
  This matches the common “flip the sheet over” scanning workflow where left/right positions swap.

If your physical workflow does NOT mirror like that, adjust BACK_MIRROR_MAP in postcard_split/splitter.py.


Recommended scanning setup
--------------------------
This tool works best when:
- The background is a uniform solid color (either black or white) and visible at the borders.
- There is a clear gap between postcards (even a thin one helps).
- The scan is not heavily skewed/rotated.
- Your scanner software is not auto-cropping too aggressively (leave a border).

The splitter classifies the background from the border and builds a border-connected background mask.
If the border isn’t representative of the true background, seam detection gets much harder.


CLI options (high-level)
------------------------
Debugging and seam introspection
- -d / --debug-seams
  Prints verbose seam-finding logs (stage selection, confidence, DPI estimates, rejection reasons).

- --debug-seam-images
  Writes seam debug artifacts (images + metadata). Current layout is:

    output/pair_###/_debug/_seams/<scan_stem>/
      stage_scan.png
      split_h/
        stage_input.png
        seam_overlay.png
        out_a.png
        out_b.png
        seam_meta.json
        seam_profiles.json
        seam_profiles_plot.png
      split_v_top/
        ...
      split_v_bottom/
        ...


Conservative “don’t shave the postcard” guardrails
- --safer-seams
  Adds conservative guardrails after a seam is selected:
    - measures “ink ratio” in a thin band around the seam and attempts a small local shift to reduce risk,
    - increases overlap when the seam looks risky, so each side keeps extra pixels near the boundary.

  This is intended to reduce the chance of cutting into postcard pixels at the cost of including a bit more background
  or neighbor area.


OpenCV-related features (optional behaviors)
These require OpenCV (already included in dependencies) but are only used when enabled:

- --tight-crop
  After deskewing each postcard, attempt a conservative OpenCV contour/min-area-rect crop to remove excess background.
  If detection fails or the crop looks implausible, it returns the original postcard image unchanged.

  When enabled, per-card debug is written to:

    output/pair_###/_debug/<postcard_stem>/
      stage2_tightcrop.png
      ...mask/overlay artifacts...
      rect.json

- --seam-cv2-mask
  Uses OpenCV morphology to clean the seam mask before searching for the best seam.

- --seam-cv2-gap
  Uses an OpenCV distance transform–based “gap score” to help pick seam positions.

- --use-cv2
  Performs a no-op OpenCV roundtrip (PIL -> cv2 -> PIL) before saving outputs.
  This is mainly a plumbing/sanity check.


Seam manifest tool (experimental)
---------------------------------
There is a helper at tools/build_seam_manifest.py intended to aggregate seam debug metadata into a CSV.

Run:

  python tools/build_seam_manifest.py

NOTE (important): As currently committed, the manifest tool looks for seam debug artifacts at:

  output/_debug/_seams/...

…but the CLI currently writes them under:

  output/pair_###/_debug/_seams/...

So, out of the box, the manifest tool will not find anything unless you:
- adjust the seam root in tools/build_seam_manifest.py, or
- change the CLI seam output location to a shared output/_debug/_seams root, or
- add recursive discovery in the manifest builder.


Library usage (example)
-----------------------
Example of importing and calling the splitter directly:

  from PIL import Image
  from postcard_split import SplitContext, split_once, deskew_postcard

  img = Image.open("scan.jpg").convert("RGB")
  ctx = SplitContext(dpi=200, debug=False, safer_seams=True)

  # Split a 2x2 scan (top/bottom, then left/right):
  halves = split_once(img, "horizontal", filename="scan.jpg", context=ctx)
  parts = []
  for half in halves:
      parts.extend(split_once(half, "vertical", filename="scan.jpg", context=ctx))

  # Deskew each part:
  parts = [deskew_postcard(p, filename="scan.jpg", context=ctx).image for p in parts]

split_once() returns a list of two images when it finds a valid seam; otherwise it returns a single image unchanged.


Development
-----------
Run tests:

  pytest

The test suite includes basic synthetic-image regression checks and debug-directory smoke tests.
If you’re improving seam logic, consider adding a “2x2 quadrant color” correctness test so regressions show up immediately.


Repo notes
----------
This repo includes vendored/experimental folders (e.g., OpenCV-Document-Scanner-master/, unpaper-main/) that are not wired
into the main CLI flow by default. The supported splitter implementation lives in postcard_split/.
