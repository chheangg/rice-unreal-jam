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
