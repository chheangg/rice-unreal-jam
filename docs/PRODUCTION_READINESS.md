# Production readiness — gaps between this hackathon script and a reliable, unattended live pipeline

This is a concrete list of things that would bite a team running
`lego_locator_xyz.py` (or `detect_xyz.py`) unattended for hours, found by
reading the actual code (`lego_locator_xyz.py`, `depth_estimator.py`,
`calibrate_iphone.py`, `calibrate_table_pose.py`), not speculative
boilerplate advice. Written as part of the coords-hardening session
(`tasks/2026-08-21-outline-coords-hardening-remote-session.md`); no camera
was used to produce it.

## 1. No camera reconnect

`lego_locator_xyz.py`'s main loop does exactly this on a failed frame read:

```python
ok, frame = cap.read()
if not ok:
    print("no more frames"); break
```

Any transient camera hiccup (a phone-stream Wi-Fi drop, a USB webcam
re-enumerating, a Continuity Camera handoff) ends the whole process, not
just that frame. For a video file this is correct (end of file); for a live
`--source` (camera index or stream URL) it isn't. There's no retry/backoff,
no distinguishing "this was a video file, stop" from "this was a live
source, try to reopen." Same pattern exists in `detect.py`/`detect_xyz.py`.

**Gap:** a live-source run should retry `cv2.VideoCapture(...)` a few times
with backoff before giving up, and only treat immediate `not ok` as fatal
for a file source.

## 2. No OSC resync / delivery guarantee

OSC here is fire-and-forget UDP (`pythonosc.udp_client.SimpleUDPClient`).
If Unreal's OSC listener isn't up yet when the tracker starts, or restarts
mid-session (editor recompile, crash-recover), messages sent during that
window are silently dropped — no ack, no retry, no "Unreal came back, catch
it up." Since `/obj` and `/outline` are per-frame snapshots (not deltas),
this self-heals the moment Unreal reconnects and the next frame's messages
arrive — but there's no way for the Python side to know Unreal missed
anything, and no periodic "resend current state" independent of new frames
arriving (a piece that stops moving still needs its `/obj`/`/outline`
re-sent if Unreal reconnects, and today it only gets re-sent on the next
frame where that piece is detected as changed — actually every confirmed
piece is re-sent every frame regardless of movement, so this specific case
is fine; the real gap is purely "no delivery confirmation," which matters
if you ever move off localhost UDP).

**Gap:** low risk on localhost (`127.0.0.1:7000`, same machine, as
configured here) where UDP loss is rare; would matter if `--osc-host` ever
points across a real network.

## 3. No logging — `print()` only, unbounded stdout

Every diagnostic (`[depth]`, `[osc]`, per-frame `--debug-size` lines) goes
to stdout via bare `print()`. Run this for hours in a terminal with
scrollback limits (or piped to a file with no rotation) and:
- there's no severity level (can't filter noise from real errors)
- there's no timestamp on most lines (only the HUD has fps/frame-relative
  state, not wall-clock)
- `--debug-size` prints one line per confirmed piece per frame — at 30fps
  with several pieces that's hundreds of lines/second with no way to sample
  or rate-limit it

**Gap:** swap to `logging` with a rotating file handler for any unattended
run; keep `--debug-size` but rate-limit it (e.g. print at most once/sec per
piece).

## 4. Config persistence — FIXED (`locator_config.py`)

Values a team tunes once and wants to keep (per `lego_locator_xyz.py`
`argparse` args): `--fov`, `--outline-height`, `--osc-host`, `--osc-port`,
`--settle`, plus the **live GUI sliders** `S floor` / `V floor` / `Min
area`. Previously none of this persisted — every run started from the same
hardcoded defaults.

**Status: implemented this session.** `locator_config.py` provides
`load()`/`save()` against `lego_locator_config.json` (not committed — a
per-rig/per-lighting snapshot, same reasoning as why the file isn't a repo
default). `lego_locator_xyz.py` now seeds its `argparse` defaults from the
saved config (an explicit CLI flag still wins), seeds the GUI sliders'
initial positions from it, and a new `s` key saves the CURRENT sliders +
flags back to the file. A missing or corrupt config file falls back to the
original hardcoded defaults rather than crashing — see
`tests/test_locator_config.py` (9 tests: round-trip, missing/corrupt/
non-dict file, partial file, atomic save, unknown-key handling).

## 5. Multi-piece disambiguation: same-colour pieces can swap identity — PARTIALLY FIXED

Original finding: `Tracks.update()` matched an incoming detection to the
nearest existing slot **of the same colour** within `match_dist` (90px),
with no other identity signal — and worse, a separate bug meant the ArUco
tag's own encoded ID was never even available to use: `detect_markers()`
called `cv2.aruco.detectMarkers()`, which returns `(corners, ids, rejected)`,
but zipped `corners` alone in its output loop — `ids` was checked for
`None` and then silently discarded. Confirmed two same-colour pieces
crossing paths would swap slot identity
(`tests/test_coords.py::test_tracks_crossing_same_color_pieces_can_swap_identity`).

**Status: fixed for TAGGED pieces this session.**
1. `detect_markers()` now zips `corners` against `ids.ravel()` and returns
   each tag's real `marker_id` alongside its center/angle/corners
   (`tests/test_coords.py::test_detect_markers_pairs_correct_id_with_each_tag`
   is a regression test for the original silent-discard bug).
2. `Tracks.update()` takes a `tag_id` parameter. When present, it's a HARD
   identity key: an existing slot already carrying that exact `tag_id` wins
   the match outright (no distance limit — a tagged piece is recognized
   even after a big frame-to-frame jump), and when no slot owns that tag
   yet, nearest-centroid matching runs only over slots that are untagged or
   share the tag (a slot already locked to a *different* tag is excluded,
   so it can never be stolen). An untagged detection (no tag on the piece,
   or the tag briefly not visible) still falls back to plain
   nearest-centroid over every slot — unchanged, since position is the only
   signal available for it.
3. `main()` now looks up `tag_id` alongside `angle` when a marker's centre
   falls inside a piece's contour, and passes it through to
   `tracks.update()`.

Verified with 4 new tests
(`test_tracks_tagged_pieces_do_not_swap_identity_when_crossing`,
`test_tracks_tag_id_matches_regardless_of_distance`,
`test_tracks_untagged_detection_still_uses_nearest_centroid_fallback`,
`test_tracks_second_tag_of_different_id_does_not_steal_untagged_slot_owner`).

**Still a real gap: UNTAGGED same-colour pieces.** The fix only helps
pieces that actually carry a visible ArUco tag. Two same-colour pieces
with no tag (or a currently-occluded tag) crossing paths will still swap,
by design — position is genuinely the only signal for them. Remaining
options if that matters: (a) require every tracked piece to carry a tag
(a process/demo-prep constraint, not a code fix), (b) a proper tracker with
velocity prediction (Kalman-ish) so "nearest" accounts for expected motion
instead of just last-known position, for the untagged case specifically.

## 6. Calibration tooling exists but the active tracker doesn't use it

`calibrate_iphone.py` and `calibrate_table_pose.py` are real, working
scripts — the former does an OpenCV chessboard intrinsics calibration and
writes `iphone_intrinsics.json` (camera matrix, distortion coefficients);
the latter reads that file, solves for the camera's pose relative to a
table-plane marker, and writes `table_pose.json`. Both are consumed by
`side_height.py` and `detect_stereo.py`.

**Neither file is read anywhere in `lego_locator_xyz.py` or
`detect_xyz.py`.** `resolve_intrinsics()` (`lego_locator_xyz.py:269`) only
ever considers two sources:

```python
def resolve_intrinsics(depth, frame_w, frame_h, assumed_fov_deg):
    if depth.available:
        intr = depth.intrinsics_for_frame()
        if intr is not None:
            return intr, "DA3"
    f = (frame_w / 2.0) / math.tan(math.radians(assumed_fov_deg / 2.0))
    return (f, f, frame_w / 2.0, frame_h / 2.0), f"FOV{assumed_fov_deg:g}"
```

DA3's own intrinsics estimate, or a guessed-FOV pinhole model. A team that
ran the real chessboard calibration for their exact rig gets no benefit
from it in the tracker that's actually used live — that measured, accurate
intrinsics data just sits in a JSON file nothing reads.

**Gap:** either (a) add a third, preferred branch to `resolve_intrinsics()`
that loads `iphone_intrinsics.json` if present (most accurate: measured,
not assumed/estimated), or (b) if the team has decided DA3's own intrinsics
are good enough in practice and the calibration scripts are for a
different, unused code path (`side_height.py`/`detect_stereo.py`), say so
explicitly somewhere (this doc, or a note in `docs/ROADMAP.md`) so the next
person doesn't assume they're wired up. Flagging as a real gap either way —
found by reading the code, not guessed.

## 7. Tracks: self-bounding but has a real transient-blob cost

`Tracks.slots[color]` is a plain list, pruned each frame by `age()`
(`lego_locator_xyz.py:261`) — any slot unmatched for more than `max_miss`
(15) frames is dropped. Per-slot memory is bounded (`w_samps`/`h_samps` are
`deque(maxlen=90)`, `shapes` is `deque(maxlen=9)`), so there's no unbounded
leak. The real cost: color-mask noise (a flickering false-positive blob
right at `min_area`) creates a brand-new slot every time it appears more
than `match_dist` from any existing slot, and that slot lingers for up to
15 frames before being pruned even though it never confirms. Under noisy
lighting with several colours flickering, `Tracks.update()`'s O(slots) scan
per detection could see meaningfully more slots than real pieces on the
table. Not a leak, just wasted work and slightly stale HUD/`age()` cost —
worth a note, not urgent.

## 8. `FloorFrame`'s in-plane axes could flip 180° mid-run — FIXED

`FloorFrame.fit()` runs every single frame (not just once) — the camera and
floor are physically fixed, but depth noise still perturbs the RANSAC-fitted
normal slightly frame to frame. The in-plane basis (u/v — what `/obj` and
`/outline` actually report as X/Y) used to be picked from a hardcoded
threshold on the normal's alignment with a fixed world axis:

```python
seed = np.array([1.0, 0.0, 0.0])
if abs(nrm.dot(seed)) > 0.9:
    seed = np.array([0.0, 1.0, 0.0])
```

**Confirmed bug:** two normal values on either side of that `0.9` threshold
produce a basis whose u/v axes point in **opposite (180°) directions** —
verified directly:

```
dot=0.899: u=[ 0.  0. -1.]
dot=0.901: u=[0. 0. 1.]
```

Since `fit()` reruns every frame from noisy real depth data, a floor
normal that happens to sit near that threshold would flip the ENTIRE
floor-frame coordinate system every time noise nudged it across — meaning
every confirmed piece's `/obj` X/Y and every `/outline` vertex would
suddenly negate/mirror, for a piece that never moved. This would show up
in Unreal as pieces randomly teleporting/mirroring mid-demo with no
apparent trigger.

**Fixed this session:** `FloorFrame._basis_for_normal()` now reuses the
*previous* fit's u axis (projected onto the new plane) as this fit's seed,
so a small normal change only rotates u/v slightly — never flips them. The
fixed-seed method is now only a first-fit bootstrap (no previous `self.u`
yet). See `tests/test_coords.py::test_floorframe_basis_does_not_flip_across_the_old_seed_threshold`
(reproduces the exact 0.899/0.901 case above and confirms `dot(u1, u2) >
0.9` after the fix) and
`tests/test_coords.py::test_floorframe_basis_first_fit_has_no_previous_state_to_flip`.
Also documented for whoever builds the Unreal side in
`tasks/2026-08-21-unreal-outline-extrude.md`: the floor frame's absolute
orientation is still arbitrary (not tied to any real-world compass
direction, and can differ between separate tracker runs) — only the
frame-to-frame flip within a single run was the bug.

## 9. `DepthEstimator` thread safety — reviewed, looks correct

Checked specifically because the task asked for it: `_depth`, `_intr`,
`_depth_src_shape` are always read/written under `_depth_lock`
(`depth_at`, `depth_in_mask`, `has_depth`, `latest_depth_map`,
`intrinsics_for_frame`, and the worker's write in `_run`) — no unlocked
access remains anywhere in the repo (verified with a repo-wide grep for
`._depth` / `._intr` / `._depth_lock` outside the class itself; this
session's `latest_depth_map()` addition replaced the one place
`lego_locator_xyz.py` used to reach into `depth._depth_lock` /
`depth._depth` directly). `_latest_frame` is similarly always guarded by
`_frame_lock` in both `submit()` and the worker's `_run()`. `submit()`
intentionally overwrites `_latest_frame` under rapid calls — that's the
correct "drop stale frames, keep only the newest" behavior described in
the module's own docstring, not a race. No changes needed here; this
section exists to record that the check was actually done.

## 10. `detect.py`/`detect_xyz.py` opened a real camera (and loaded DA3) as an import side effect — FIXED

Both scripts had `cap = cv2.VideoCapture(0)` — and `detect_xyz.py` also had
`depth = DepthEstimator()` (which loads/downloads a DA3 checkpoint) —
sitting at **module level**, with no `if __name__ == "__main__":` guard.
Every other camera-touching script in the repo (`lego_locator.py`,
`lego_locator_xyz.py`, `camera_probe.py`, `detect_stereo.py`,
`detect_platform.py`, `detect_platform_aruco.py`, `calibrate_iphone.py`,
`calibrate_table_pose.py` — checked all of them) already guards its
`VideoCapture` call inside `main()`/behind the `__name__` check; these two
were the only outliers.

**Why this matters:** a bare `import detect` or `import detect_xyz` from
*anywhere* — an IDE's autocomplete indexer, a linter, a future test suite,
another script importing `find_blobs` for reuse — would silently open a
real camera device (and, for `detect_xyz`, start a model download) with no
explicit call, violating this session's own "never open a real camera"
constraint just by being imported. This session confirmed it never
triggered that (only used `py_compile`, which doesn't execute module code,
until after the fix was applied and verified safe).

**Fixed this session:** both files now wrap their capture-open/loop/cleanup
in `main()`, guarded by `if __name__ == "__main__":`, matching every other
script in the repo. `detect_xyz.py`'s `send()` now takes `depth` as a
parameter instead of reading a module-level global, since `depth` moved
inside `main()`. Confirmed both modules now import in well under 5 seconds
with no camera/model side effects — see
`tests/test_detect_import_safety.py` (6 tests: import-safety timing for
both files, plus `find_blobs()` correctness on synthetic HSV data,
including the same red-hue-wrap edge case already covered for
`lego_locator.py`). Behavior when actually run as a script
(`python detect.py`) is unchanged — this was a pure refactor.

## 11. `side_height.py` / `platform_height.py` math — audited, no bugs found

These two are the unused-by-the-live-pipeline calibration/height modules
flagged in #6 — worth checking independently since they contain real
geometry (similar-triangles depth-from-known-size, ray/plane intersection,
camera↔world point transforms via `solvePnP`'s rvec/tvec) that had never
been exercised by a test. Added `tests/test_side_height.py` (8 tests) and
`tests/test_platform_height.py` (11 tests), all synthetic/no camera:

- `side_height.py`: the depth-from-known-width formula
  (`depth_mm = fx * real_width / pixel_width`) checked against hand-derived
  values, inverse scaling with apparent width, simple-mode fallback
  formula, unknown-color/degenerate-width → `None`. The camera↔world
  rotation inversion (`p_world = R⁻¹ @ (p_cam - t)`) is tested with
  `R = identity` specifically to avoid asserting anything about the real
  calibration's Z-axis sign convention, which depends on chessboard corner
  ordering at calibration time and isn't independently re-derivable from a
  synthetic test — `calibrate_table_pose.py`'s own printed sanity check
  (camera height vs. a tape measure) is the right place to verify that,
  not a unit test.
- `platform_height.py`: `_classify_corners()`'s quadrant sorting (axis-
  aligned, order-independent, rotated, degenerate-input → `None`);
  `ground_xy_mm()`'s ray/plane intersection against hand-derived expected
  values (principal-point ray hits directly below the camera; an off-
  center pixel's lateral offset scales exactly with depth; parallel-ray
  and behind-the-ray degenerate cases both return `None` instead of
  garbage); `camera_point_to_world()` round-trips the forward
  `p_cam = R @ p_world + t` transform exactly, for a real (non-identity)
  rotation; `solve_marker_camera_point()` recovers a known synthetic
  ArUco pose's translation via `cv2.projectPoints` → `solvePnP` round trip.

**Result: no bugs found in either file.** All 19 tests pass — the math in
both modules is correct as written. Recorded here because "audited and
found clean" is itself useful information for whoever eventually wires
these into a live pipeline (see #6's still-open question of whether that
should happen).

## Not investigated (would need a live camera or Unreal Editor — out of
scope for this session per its hard constraints)

- Whether `resolve_intrinsics()`'s FOV-fallback actually tracks real-world
  scale well enough on hardware without DA3 intrinsics — needs a tape
  measure and a live rig.
- Whether the Unreal-side OSC receiver actually handles a `/outline`
  rebuild every frame at acceptable cost — needs the Editor
  (`tasks/2026-08-21-unreal-outline-extrude.md`, assigned to evan).
