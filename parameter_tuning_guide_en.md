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
