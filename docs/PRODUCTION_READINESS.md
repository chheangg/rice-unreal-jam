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

## 4. No config persistence — every tuned value resets on each run

Values a team tunes once and wants to keep (per `lego_locator_xyz.py`
`argparse` args): `--fov`, `--outline-height`, `--osc-host`, `--osc-port`,
`--settle`, plus the **live GUI sliders** `S floor` / `V floor` / `Min
area` (`DEFAULT_S_FLOOR`, `DEFAULT_V_FLOOR`, `DEFAULT_MIN_AREA_100` in
`lego_locator.py`, currently hardcoded module constants, not even CLI
flags). None of this persists — every run starts from the same defaults,
so the exact same color-tuning ritual from `docs/ROADMAP.md`'s "Step 1"
has to be repeated by hand, or flags have to be retyped, each session.

**Gap:** a small JSON/YAML config file (read on startup if present, CLI
flags override it, an optional `--save-config` writes the current tuned
values back out) is warranted here — this is a repo where the team
explicitly documents re-tuning per lighting setup (`README.md` "Notes"),
which is exactly the workflow config persistence is for. Not a big lift:
one `json.load`/`json.dump` pair and a slider-change callback that writes
back, since the sliders already read/write via `cv2.getTrackbarPos`.

## 5. Multi-piece disambiguation: same-colour pieces can swap identity

`Tracks.update()` (`lego_locator_xyz.py:207`) matches an incoming
detection to the nearest existing slot **of the same colour**, within
`match_dist` (90px) pixels, with no other identity signal (no ArUco ID
feeding into the match — the tag's `angle` is attached to whichever slot
wins the nearest-centroid match, not the other way around):

```python
for s in lst:
    d = math.hypot(s["cx"] - cx, s["cy"] - cy)
    if d < best_d:
        best, best_d = s, d
```

**Actual failure mode** (confirmed with a synthetic test,
`tests/test_coords.py::test_tracks_crossing_same_color_pieces_can_swap_identity`):
two pieces of the same colour, tracked in separate slots, that pass within
90px of each other's position on a frame will have their slots'
identities swap — slot A silently starts tracking piece B's position/size
history, and vice versa doesn't even need to happen for the effect to
show: as demonstrated, B moving *near* A's slot (not even to the exact same
point) is enough to be matched to A's slot instead of staying attached to
its own. Since size is locked at confirmation
(`best["size_locked"]`), an already-confirmed piece's reported *size*
won't visibly change after a swap (it's frozen), but its **position**
and **angle** on-screen would suddenly reflect the other piece's, and any
still-unlocked (`size_locked == False`) size sampling gets contaminated
with a sample from a physically different piece (also confirmed in the
same test).

**Gap:** the tracker has no per-piece appearance/ID signal beyond position
and colour. Options, roughly in order of effort: (a) lower `match_dist` at
the cost of losing fast-moving pieces between frames, (b) feed the ArUco
tag ID (when present) into matching as a hard identity key instead of a
side-channel angle, ahead of pure nearest-centroid, (c) a proper tracker
with velocity prediction (Kalman-ish) so "nearest" accounts for expected
motion, not just last-known position. (b) is the cheapest given ArUco is
already the recommended ID mechanism (see `shape_recognizer.py`'s
recommendation) — worth doing before (c).

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

## 8. `DepthEstimator` thread safety — reviewed, looks correct

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

## Not investigated (would need a live camera or Unreal Editor — out of
scope for this session per its hard constraints)

- Whether `resolve_intrinsics()`'s FOV-fallback actually tracks real-world
  scale well enough on hardware without DA3 intrinsics — needs a tape
  measure and a live rig.
- Whether the Unreal-side OSC receiver actually handles a `/outline`
  rebuild every frame at acceptable cost — needs the Editor
  (`tasks/2026-08-21-unreal-outline-extrude.md`, assigned to evan).
