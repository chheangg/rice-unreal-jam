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
- **`/outline` always has >= 3 points now** (as of this session) - a
  degenerate simplification (a sliver contour collapsing to a line/point)
  falls back to the min-area-rect corners instead, and the script skips
  sending `/outline` entirely if it still can't produce a real polygon. You
  should never see `n_points` < 3 on this address.
- **Units - read carefully, this is an easy mistake:** the `x,y` pairs are
  in **millimeters**, but `height_cm` is exactly what its name says -
  **centimeters**. They are NOT the same unit within one message. Unreal's
  default world unit is 1 unit = 1cm, so: divide every `x_mm`/`y_mm` by 10
  to get the actor-space value in Unreal units, then use `height_cm`
  directly (no further conversion) for the extrusion depth. **Do NOT reuse
  the "÷50" scale divisor mentioned elsewhere in `docs/ROADMAP.md`'s
  "Carried-over knowledge" section** - that number is from the OLDER,
  unrelated `detect.py` pipeline, which sends raw PIXEL coordinates (not
  metric), and ÷50 was an arbitrary fit for pixel-space, not a unit
  conversion. `/obj` and `/outline` from `lego_locator_xyz.py` are real
  metric millimeters/centimeters - the correct, exact conversion is ÷10 for
  mm, not ÷50 or any other pixel-era constant. If `/obj`'s receiver
  Blueprint is currently using ÷50 on `lego_locator_xyz.py`'s `/obj`
  output, that's very likely ALSO wrong for the same reason and worth
  checking while you're in there.
- **Floor-frame axis orientation is fixed within a run, not globally
  meaningful:** `/obj`'s and `/outline`'s X/Y live in a floor-relative frame
  whose in-plane axes are derived from the fitted floor plane's normal, not
  from any real-world compass direction - so "floor X" doesn't point any
  particular way in the room, and can differ between separate runs of the
  tracker (though it's now held continuous frame-to-frame WITHIN a run - a
  bug where it could flip 180 degrees mid-run was fixed this session, see
  `docs/PRODUCTION_READINESS.md`). Practical effect: don't hardcode an
  assumption about which screen direction "positive X" corresponds to when
  wiring up the receiver/camera framing - eyeball it once per session
  instead, since it can differ each time the tracker restarts.
- I (Claude) can't test the Unreal side myself - no Unreal Editor access in
  this environment, and this needs a live Editor session. Flagging that up
  front so it's clear this task needs manual verification once picked up.
