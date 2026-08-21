---
status: done # open | in-progress | done
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

## Result

All 4 scope items done, no camera or Unreal Editor touched:

1. `tests/test_coords.py` — 35 synthetic tests: `resolve_intrinsics` FOV
   math, `backproject` scaling, `FloorFrame` RANSAC fit (normal direction,
   orthonormal basis, height sign, degenerate-input rejection, the
   floor.ok==False fallback path), `build_outline_mm` units/winding-order/
   point-cap, `Tracks` settle-gating/size-lock/angle-wrap/same-colour
   matching (including a **confirmed** identity-swap failure mode), and
   `classify_shape` edge cases (degenerate/zero-area/huge/sliver contours,
   frame-edge translation invariance). Required extracting `backproject()`,
   `to_world_xy()`, and `build_outline_mm()` out of `main()` in
   `lego_locator_xyz.py` as pure functions (same math, now testable) — see
   commit `3adc365`.
2. `shape_recognizer.py` (ORB + homography) + `tests/test_shape_recognizer.py`
   (11 tests against synthetic transforms of the real reference photo, plus
   a real distractor). **Recommendation: keep ArUco as the production
   method** — feature-matching works on moderate transforms but LEGO's
   repetitive stud texture makes it fragile under motion blur/oblique
   angle, and it gives no ID/rotation guarantee the way a tag does.
   `lego_locator_xyz.py` is untouched by this — presented for review, not
   wired in.
3. Confirmed clean: all trackers `py_compile` clean, `/outline` schema in
   `lego_locator_xyz.py` matches `tasks/2026-08-21-unreal-outline-extrude.md`
   and `README.md` §2c exactly (no drift).
4. `docs/PRODUCTION_READINESS.md` — 8 concrete findings from reading the
   code (not speculation): no camera reconnect, no OSC resync, print-only
   logging, no config persistence, confirmed same-colour identity swap,
   unused calibration JSON files, self-bounding-but-costly transient
   Tracks slots, and a DepthEstimator thread-safety review (clean — no
   unlocked access anywhere, verified by grep and by
   `tests/test_depth_estimator_threading.py`'s concurrent execution).

Backlog also covered: more edge-case tests (huge/sliver contours,
floor-fit degenerate-data paths) and dedicated `DepthEstimator` concurrency
tests (rapid submit() drops frames correctly, no torn state across
concurrent readers, `stop()` unblocks promptly) — driven via a bare
instance with `_load()` skipped, so no torch/DA3 network call was made.

**Test count: 50/50 passing** across `tests/test_coords.py` (35),
`tests/test_shape_recognizer.py` (11), `tests/test_depth_estimator_threading.py` (4).

### What was found but NOT fixed (and why)

- The same-colour identity-swap bug (#5 in `docs/PRODUCTION_READINESS.md`)
  is real and demonstrated, but fixing it (feeding ArUco ID into `Tracks`
  matching ahead of nearest-centroid) is a design change to a core class
  that's out of this session's scope (audit + document, not rewrite) and
  deserves its own review rather than a same-session patch.
- Config persistence (item 4 in the readiness doc) is flagged as
  "warranted" but not built — same reasoning: a new on-disk format touching
  every script's CLI is a scoped feature, not a hardening fix.
- `resolve_intrinsics()` not reading `iphone_intrinsics.json` — flagged,
  not wired up, since it's genuinely ambiguous whether the team wants DA3's
  own intrinsics kept as primary or the chessboard calibration preferred;
  that's a product decision, not a bug.

### Single most important next human action

**Run `python -m pytest tests/ -v` once, then read
`docs/PRODUCTION_READINESS.md` section 5** (same-colour piece identity
swap) — it's the one finding here that can visibly break a live demo
(two same-color pieces crossing paths swaps which is which on screen), and
the cheapest fix (feed ArUco ID into `Tracks` matching) is already scoped
in that section. Everything else in the doc is real but lower urgency for
a demo-length session.
