---
status: open        # open | in-progress | done
requested-by: ibrahimaasim
assigned-to: evan
date: 2026-08-21
---

## What

`lego_locator_xyz.py --osc` now sends a second OSC message per confirmed
piece, alongside the existing `/obj`:

```
/outline -> [name, n_points, x1_mm, y1_mm, x2_mm, y2_mm, ..., xn_mm, yn_mm, height_cm]
```

- `name` - same piece name as `/obj` (e.g. `yel1`, `red`)
- `n_points` - how many (x,y) pairs follow (piece contour simplified via
  `cv2.approxPolyDP`, capped at 16 points)
- the `x,y` pairs are the piece's REAL outline vertices, in mm, already in
  floor-frame world space (same frame as `/obj`'s x/y) - they already carry
  the piece's true rotation, so no separate angle needs to be applied
- `height_cm` - a fixed flat extrusion height (currently 2.0cm, see
  `--outline-height` in `lego_locator_xyz.py`), NOT a real scanned height -
  just enough to give the mesh some thickness

Build a Blueprint (or extend the existing OSC receiver) that, on `/outline`,
uses UE5's **Geometry Script** plugin to extrude that 2D polygon into real
mesh (Geometry Script has a built-in "append extrude polygon"-style function
for exactly this - 2D points + a height -> solid mesh at runtime). This
replaces the placeholder cube/block currently spawned per piece with
something that actually looks like the built Lego shape's footprint.

## Why

The team wants a piece placed in front of the camera to show up in Unreal as
its actual outline instead of a generic block, without needing a real 3D
scan/import pipeline (no time for photogrammetry mid-hackathon). A flat
extrusion of the real detected 2D silhouette is a good-enough visual match
for a demo. Full context/back-and-forth on why this approach (vs. a real
scan) is in the chat that spawned this task; short version: `/obj`'s cube
placeholder was always meant to be swappable (see `docs/ROADMAP.md`'s
"Option 1 vs Option 2" note) and this is the cheap version of Option 2.

## Notes

- `/obj` is unchanged - keep using it for position/rotation/size of whatever
  actor represents the piece. `/outline` is purely for building/updating that
  actor's mesh.
- Only confirmed (settled 3s) pieces send `/outline`, same gating as `/obj`.
- If a piece moves, a NEW `/outline` arrives each frame with (very likely)
  the same point count - simplest approach is probably to rebuild the
  Geometry Script mesh each time rather than trying to diff/patch it,
  unless that's visibly too expensive.
- I (Claude) can't test the Unreal side myself - no Unreal Editor access in
  this environment, and this needs a live Editor session. Flagging that up
  front so it's clear this task needs manual verification once picked up.
