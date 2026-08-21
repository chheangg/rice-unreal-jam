# rice-unreal-jam — MechTwin03

Real-time physical/digital twin: a webcam tracks colored blocks on a surface
and mirrors their position, rotation, and size live inside an Unreal Engine
scene over OSC.

See `AGENTS.md` for collaboration conventions and `docs/ROADMAP.md` for
current direction.

## Repo layout

```
MT03_RealTimeLayout/   Unreal Engine project (engine 5.8)
detect.py              webcam block tracker -> sends OSC to Unreal
send_test.py           fake tracker - replays the real OSC message, no camera
camera_probe.py        checks what OpenCV can see (indices or a phone URL)
inspect_bp.py          debug script, run inside Unreal's embedded Python
Video/, mt03 outputs/  recordings + Premiere project files (Git LFS)
tasks/                 task requests — see tasks/README.md
docs/                  roadmap and other project docs
```

## Prerequisites

- **Unreal Engine 5.8**
- **Python 3** with:
  ```
  pip install opencv-python numpy python-osc
  ```
- **(Optional) Depth Anything 3** — only needed for the xyz tracker
  (`detect_xyz.py`), which adds a real `z` (depth) per block. DA3 is
  ByteDance's monocular depth model and wants a CUDA GPU for interactive
  rates:
  ```
  pip install "torch>=2" torchvision xformers
  git clone https://github.com/ByteDance-Seed/Depth-Anything-3
  pip install -e Depth-Anything-3
  ```
  The DA3 checkpoint (`depth-anything/DA3METRIC-LARGE`) downloads from
  HuggingFace on first run. Without these, `detect_xyz.py` still runs but
  falls back to 2D (sends `z = -1.0`).

  **On a MacBook (Apple Silicon M1/M2/M3):** it works, with two changes.
  `xformers` does **not** support Apple Silicon — **skip it** (installing it
  causes runtime "operator not supported" errors), and let DA3 use the Mac
  GPU via Metal (MPS):
  ```
  pip install "torch>=2" torchvision            # NO xformers on Mac
  git clone https://github.com/ByteDance-Seed/Depth-Anything-3
  pip install -e Depth-Anything-3
  ```
  `depth_estimator.py` auto-detects the M-series GPU (`mps`) and sets
  `PYTORCH_ENABLE_MPS_FALLBACK=1` so any Metal-unsupported ops fall back to
  CPU instead of crashing. Expect it to be usable but not fast: the M2 GPU is
  far slower than a discrete NVIDIA card. For a smoother demo on a MacBook,
  switch `DEFAULT_MODEL` in `depth_estimator.py` to `depth-anything/DA3-SMALL`
  (faster, but relative depth rather than metric).
- **Git LFS** — required to pull the video files and large Unreal cache
  files correctly:
  ```
  git lfs install
  ```
  (run once per machine, before or after cloning)
- A webcam, if you want live tracking (not required to just open the Unreal
  project)

## Running it

### 1. Open the Unreal project

Open `MT03_RealTimeLayout/MT03_RealTimeLayout.uproject` in Unreal Engine
5.8. The **OSC** plugin is already enabled in the project and listens on
`127.0.0.1:7000` for incoming block data.

### 2. Start the tracker

With the Unreal editor running (so it's listening on port 7000):

```
python detect.py
```

This opens your default camera, detects:
- 3 yellow blocks → `yel1`, `yel2`, `yel3`
- 1 red block → `red`

Only these two colors are detected (no direction marker) so nothing else in
frame gets mistaken for a tracked block. `angle` is always `0` in the OSC
message below (kept only for index compatibility with Unreal).

and streams `[name, x, y, angle, sizeX, sizeY]` to Unreal over OSC on the
`/obj` address. A preview window shows what's being detected; press `Esc`
to quit.

### 2b. Want depth (x, y, **z**)?

Run the depth-augmented tracker instead:

```
python detect_xyz.py
```

Same tracking as `detect.py`, but each block also gets a `z` (metric depth
in meters, smaller = closer) from Depth Anything 3, computed on a background
thread so the preview stays smooth. It streams
`[name, x, y, angle, sizeX, sizeY, z]` on `/obj` — note `z` is **appended at
the end**, so every existing "Get at Index 0..5" node in the Unreal Blueprint
keeps working; just add a "Get at Index 6" for `z`. Absolute depth scale can
need per-rig calibration — if `z` looks off versus a tape measure, tune
`metric_scale` in `depth_estimator.py`. Requires the optional DA3 install
above; without it, it prints a warning and sends `z = -1.0`.

### 2c. `lego_locator_xyz.py` — colour + shape + XYZ, and a real outline

```
python lego_locator_xyz.py 1 --osc
```

The richer tracker: per confirmed piece it reports shape (square/circle/
cross), rotation (from an ArUco tag on the piece, if present), and metric
X/Y/Z + real-world size in cm, using Depth Anything 3 same as `detect_xyz.py`
(see `depth_estimator.py`). Pass a camera index as the first arg (run
`camera_probe.py` first if unsure which index is your actual webcam, not a
phone/Continuity Camera pretending to be index 0).

With `--osc`, it sends **two** messages per confirmed piece:

- `/obj` — same layout as `detect_xyz.py`: `[name, x_mm, y_mm, angle, w_cm, h_cm, z_mm]`
- `/outline` — `[name, n_points, x1_mm, y1_mm, ..., xn_mm, yn_mm, height_cm]`,
  the piece's real 2D silhouette (simplified contour, back-projected into the
  same world space as `/obj`'s x/y — vertices already carry the piece's true
  rotation) plus a fixed extrusion height (`--outline-height`, default 2cm).
  Lets Unreal show the piece's actual outline, flat-extruded, instead of a
  generic placeholder block. See `docs/ROADMAP.md` and
  `tasks/2026-08-21-unreal-outline-extrude.md` for the Unreal-side plan
  (Geometry Script polygon extrude).

### 3. No camera handy?

Run `send_test.py` instead — it fakes the tracker. It sends the *same*
`/obj` message the real trackers send (`[name, x, y, angle, sizeX, sizeY, z]`),
at a realistic frame rate, for several blocks circling around the frame — so
the Unreal receiver can be built and debugged with no camera, no lighting rig
and no blocks on the table.

```
python send_test.py                  # 4 blocks, 30 Hz
python send_test.py --ids 6          # more blocks (extras named tag4, tag5, ...)
python send_test.py --churn 3        # a block appears/disappears every 3s
python send_test.py --no-z           # 6-element message, matching detect.py
python send_test.py --rate 5         # slow it down to read UE logs
```

Two things it does on purpose that the current colour tracker doesn't:
`angle` really sweeps 0–359 (ArUco will send a true angle, so the receiver's
rotation path needs testing now), and `--churn` makes names come and go, which
is what the ID-based receiver has to survive — spawn on a name it hasn't seen,
cope with one going quiet. Ctrl-C to stop.

### 4. Using a phone as the camera

`camera_probe.py` answers "can OpenCV see this camera, and how fast?" before
you touch the tracker:

```
python camera_probe.py                                  # scan indices 0..5
python camera_probe.py 0                                # preview an index
python camera_probe.py http://172.20.10.1:8080/video    # preview a phone stream
```

It reports resolution and, more importantly, **measured** fps rather than the
fps the source claims - an IP camera app will happily advertise 30 and deliver
6, which looks like a tracking bug but is a network problem.

On **iPhone + Windows**, Continuity Camera is not an option (macOS only). Two
routes work:

- **IP camera app** (e.g. IP Camera Lite) - serves MJPEG over HTTP, no driver
  on the laptop. Point OpenCV straight at the URL.
- **Virtual webcam driver** (iVCam, EpocCam) - needs a laptop-side driver, but
  the phone then shows up as an ordinary camera index and no code changes.

For a demo, connect the laptop to the **iPhone's Personal Hotspot** rather than
venue wifi. It's a direct private link, so it survives networks that isolate
clients from each other, and needs no internet at all. On that hotspot the
iPhone is always **172.20.10.1** (Apple uses a fixed 172.20.10.0/28 subnet).

Don't route this through a relay like ngrok: it sends video to a cloud server
and back to a laptop three feet away, adding latency to the one thing that has
to feel instant, and puts your camera feed on a public URL.

## Notes

- `inspect_bp.py` isn't run from a normal terminal — it's meant to be
  executed inside Unreal's own Python environment (Editor Utility / Python
  console) to inspect Blueprint graphs.
- Camera color thresholds (`YELLOW`, `RED` ranges in `detect.py`) are tuned
  for a specific lighting setup — if detection is flaky, that's the first
  place to look. `RED` wraps around the HSV hue circle (0 and 180 are both
  "red"), so it's defined as two merged ranges; if it picks up skin tone or
  other reddish/pinkish objects in frame, narrow the saturation/value floor
  first before touching the hue bounds.
