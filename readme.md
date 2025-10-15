# Vectorization Engine CLI Options (English Documentation)

## I/O Group & Modes
- **--input:** Path to the source image (PNG/JPEG).
- **--output:** Output SVG file path.
- **--mode {auto, color, mono}**
  - **auto:** Automatically determines the optimal number of color layers (heuristic-based).
  - **color:** Forces exactly *k* color clusters. Ideal for logos or flat-style images.
  - **mono:** Black–white threshold mode → produces the sharpest contours, best for anti-alias refinement.

- **--k:** Number of color clusters in the source image
  - **Increase:** Preserves more tonal variation/details → more paths, larger file, slight aliasing at color transitions.
  - **Decrease:** Simplifies colors → smoother look, fewer paths, ideal for logos.
  - **Recommendation:** Logos 3–12; flat/icons 4–8; posterized photos 8–16; 50+ only for high-detail use cases.

---

## Image Region / Pre-processing Group
- **--min-area INT (default 80 px²):** Ignores regions smaller than this threshold.
  - **Increase:** Removes dust/specks → cleaner, smoother SVG.
  - **Decrease:** Keeps tiny details → may look noisy or jagged.
  - **Typical:** Logo = 50–300; noisy scans = 150–800.

- **--area-max-ratio FLOAT (default 0.92):** Limits a region’s area relative to the total image.
  - **Increase (→ 1.0):** Keeps large areas (backgrounds) but may merge background into one block → harder to smooth.
  - **Decrease:** Treats overly large regions as background → tighter focus on the main object.

- **--super-sample INT (SSAA, default 4):** Supersampling factor — upscales internally before edge detection (anti-alias).
  - **Increase:** Smoother, more accurate edges (especially curves) but higher CPU/RAM usage.
  - **Decrease:** Faster but more “stair-stepped” edges.
  - **Typical:** Logo = 3–6; large images = 2–4.

- **--decimate INT (default 1):** Downsamples input contour points.
  - **Increase:** Fewer points → smoother path, but may lose detail or break corners.
  - **Decrease (→ 1):** Retains all points → maximum fidelity; combine with smoothing to avoid jaggedness.

- **--no-auto-crop (flag):** Disable automatic object cropping. Auto-crop improves smoothness by removing noisy borders.
- **--crop x,y,w,h:** Manual crop. Useful to focus on key regions for better vectorization.
- **--subpixel-sigma FLOAT (default 0.8):** Light Gaussian blur before edge detection (subpixel smoothing).
  - **Increase:** Reduces aliasing but rounds corners.
  - **Decrease:** Sharper corners, more aliasing; 0 = disabled.
  - **Typical:** Logo = 0.5–1.0; mono/noisy scans = 0.8–1.5.

---

## Edge Repair & Morphology Group
- **--no-repair (flag):** Disable automatic edge welding / small-hole filling after segmentation. Disabling preserves raw shapes but may leave pinholes or jagged edges.

- **--repair-clean FLOAT (default 0.01, relative to image size):** Merge/fill threshold for minor gaps.
  - **Increase:** Aggressively smooths edges, may merge thin areas.
  - **Decrease:** Minimal cleanup, more small gaps remain.

- **--morph-close INT (default 4 px):** Closing operation (dilate → erode) to fill small gaps.
  - **Increase:** Stronger edge joining, smoother result but can inflate fine details.
  - **Decrease:** Preserves sharp edges, possible small breaks.

- **--morph-open INT (default 1 px):** Opening operation (erode → dilate) to remove small noise specks.
  - **Increase:** Cleaner image, may erase thin features.
  - **Decrease:** Keeps details but may appear grainy.

---

## Corner, Line, and Fit-Error Group (Critical for Visual Smoothness)
- **--corner-eps FLOAT (default 0.9):** Corner detection sensitivity.
  - **Increase:** Fewer detected corners → more curves → smoother but may round sharp corners.
  - **Decrease:** More corners → sharper accuracy, but may look segmented.

- **--line-tol FLOAT (default 0.1):** Tolerance to classify segments as straight lines.
  - **Increase:** Easier snapping to straight lines → cleaner edges in logos.
  - **Decrease:** Fewer straight lines, more curves → smoother curves but edges may bend slightly.

- **--max-err FLOAT (default 0.26):** Maximum fitting error when approximating with line/curve segments.
  - **Increase:** Fewer control points → compact, smooth overall path, possible slight shape deviation.
  - **Decrease:** Closer fit → more control points → higher fidelity but risk of visible aliasing.

---

## Curve Smoothing (Mono-only)
- **--curve-sigma FLOAT (default 1.4; 0 = disabled):** Gaussian smoothing applied to curved segments only.
  - **Increase:** Noticeably smoother curves (especially in mono mode), minimizes stair-stepping. Too high may blur details.
  - **Decrease:** Preserves sharp curvature; lower visual smoothness.


=====================================================================
# How to Tune Parameters for “Smooth Yet Accurate Shapes”

### 1. Flat logos with many straight edges
```
--mode color --k 4..8
--super-sample 4..6
--subpixel-sigma 0.6..0.9
--line-tol 0.15..0.3
--corner-eps 0.8..1.0
--max-err 0.2..0.35
--morph-close 2..5 --morph-open 1..2
--min-area 80..200
```
→ Goal: ensure sharp alignment for straight edges while keeping curves smooth and minimizing jagged edges.

### 2. Icons with many curves (rounded shapes, thick outlines)
```
--mode color --k 3..6
--super-sample 4..6
--subpixel-sigma 0.8..1.2
--line-tol 0.08..0.15
--corner-eps 0.9..1.2
--max-err 0.25..0.4
--morph-close 3..6 --morph-open 1
--min-area 50..120
```
→ Prioritize smooth curvature and reduce the number of nodes.

### 3. Black–white / monochrome scans
```
--mode mono
--super-sample 3..5
--curve-sigma 1.2..2.0
--subpixel-sigma 0.6..1.0
--line-tol 0.12..0.25
--corner-eps 0.8..1.0
--max-err 0.22..0.35
--morph-close 3..7 
--morph-open 1..3
--min-area 120..400
```
→ Focus on denoising and smoothing curved regions while keeping straight edges intact.

# “Cheat Sheet” for Quick Smoothing
- Still jagged? → Increase `--super-sample`, raise `--subpixel-sigma` (by 0.2–0.4 each time), or increase `--curve-sigma` (for mono mode).
- Straight edges become wavy? → Increase `--line-tol`, decrease `--curve-sigma`, lower `--subpixel-sigma`.
- Corners too rounded? → Decrease `--corner-eps`, lower `--max-err`.
- Too many tiny or rough details? → Increase `--min-area`, raise `--morph-open`, raise `--repair-clean`.
- Paths too heavy? → Increase `--max-err`, increase `--decimate`, reduce `--k`.
