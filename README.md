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
send_test.py           sends fake OSC messages, for testing without a camera
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
- 1 silver block → `slv`
- a pink marker inside each block, used to compute its facing angle

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

### 2c. Lego shapes: colour + ArUco + snapping (`detect_lego.py`)

The current direction (see `docs/ROADMAP.md`): each Lego colour = one shape,
one ArUco tag per shape for a stable id + true rotation, and everything snaps
to a grid so shapes combine cleanly.

```
python detect_lego.py
```

- Print ArUco tags from **`DICT_4X4_50`**. **Ids 0–3 are the board corners**
  (0=TL, 1=TR, 2=BR, 3=BL) — tape them at the table corners. Give every shape
  its own tag with **id ≥ 4**.
- The 4 corner tags let it snap in a rectified **board frame** instead of
  camera-pixel space, so snapping works even though the camera is slightly
  angled. Set `BOARD_COLS`/`BOARD_ROWS` to how many grid cells span the taped
  rectangle; the on-screen grid shows the lattice everything snaps to.
- Tune the `COLORS` HSV ranges with the tuner in `docs/ROADMAP.md`.
- Streams `/shape [id, x, y, z, angle, shape, color]` (one message per shape)
  and `/shape_gone [id]` when a shape leaves. **x,y are board grid cells** —
  in Unreal place at `x,y * cellSize` (no ÷50 in this path). `z` is metric
  depth in meters (needs the optional DA3 install above; set `USE_DEPTH=False`
  to skip). Build the ID-based receiver against `/shape`.

### 3. No camera handy?

Run `send_test.py` instead — it sends a fixed test message on a loop so you
can confirm the Unreal side is receiving OSC without needing the camera rig
set up.

## Notes

- `inspect_bp.py` isn't run from a normal terminal — it's meant to be
  executed inside Unreal's own Python environment (Editor Utility / Python
  console) to inspect Blueprint graphs.
- Camera color thresholds (`YELLOW`, `SILVER`, `MARKER` ranges in
  `detect.py`) are tuned for a specific lighting setup — if detection is
  flaky, that's the first place to look.
