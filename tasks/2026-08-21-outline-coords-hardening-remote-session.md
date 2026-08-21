---
status: in-progress # open | in-progress | done
requested-by: ibrahimaasim
assigned-to: claude (autonomous session)
date: 2026-08-21
---

## What

A single autonomous session's scope of work, tracked here so the pieces
land somewhere durable instead of only existing in a chat transcript:

1. Audit and pressure-test the X/Y/Z and size math in `lego_locator_xyz.py`
   with synthetic unit tests (`tests/test_coords.py`) — units, axis
   convention, floor-frame transform, `/outline` winding order.
2. Piece identification for one specific built assembly
   (`/Users/ibrahimaasim/Downloads/IMG_3010.HEIC`) without training a model
   on a single photo: feature-matching (`shape_recognizer.py`, ORB +
   homography, tested against synthetic transformed copies of the cropped
   reference) presented alongside the already-decided ArUco tag approach —
   a recommendation, not a silent pick.
3. Compatibility pass: confirm `detect.py`/`detect_xyz.py` and
   `lego_locator_xyz.py` still import/compile cleanly, and that any
   `/outline` schema changes stay in sync with
   `tasks/2026-08-21-unreal-outline-extrude.md` and `README.md` §2c.
4. `docs/PRODUCTION_READINESS.md` — concrete gaps between this hackathon
   script and a reliable unattended live pipeline (camera reconnect, OSC
   resync, logging, config persistence, multi-piece disambiguation,
   unused calibration tooling).

## Why

The team wants the coordinate math trusted (not just "looks right on
screen") before Evan builds the Unreal-side outline extrusion against it,
and wants an honest choice between feature-matching and ArUco for
recognizing the specific lime-green assembly, instead of one getting
silently implemented.

## Notes

- Hard constraint for this session: **no real camera use** — `cv2.VideoCapture`
  against a live device, and any `cv2.imshow`/`waitKey` loop, are off-limits.
  All validation is synthetic data or math-only unit tests. Anything that
  genuinely needs a camera (verifying `resolve_intrinsics()`'s FOV-fallback
  path against reality, tape-measuring an actual piece, the Unreal Editor
  receiver) is written up here or in a follow-up task instead of attempted.
- No Unreal Editor GUI interaction either — Unreal-side work stays limited to
  docs/task specs for `evan` (see `tasks/2026-08-21-unreal-outline-extrude.md`).
- Working branch: `outline-and-coords-hardening`.
