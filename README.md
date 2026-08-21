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
lego_locator_xyz.py    current tracker: all colours, shape, size, XYZ
unreal_bridge.py       turns detections into the /obj message BP_OSCreciver parses
ue_multi_receiver.py   runs inside Unreal: spawns one actor per tracked piece
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

### 2c. Driving Unreal from the Lego locator

`lego_locator_xyz.py` is the current tracker (all colours, shape, real size in
cm, metric XYZ). To make it drive the Unreal scene:

```
python lego_locator_xyz.py 0 --osc --osc-verbose
```

`--osc` does **not** send the metric numbers the locator prints on screen. It
sends the message `BP_OSCreciver` actually parses, via `unreal_bridge.py`:

```
/obj  [name:str, cx:int, cy:int, angle:int, sizeX:int, sizeY:int]
```

Two details in there are the whole reason a "working" tracker can show up as an
empty Unreal scene, and both live in the blueprint, not the tracker:

- **The blueprint reads integers.** Every field is pulled with *Get OSC Message
  Integer at Index*. python-osc tags a Python float as OSC type `f`, and UE's
  `GetInt32` does not coerce it - it fails and the graph gets `0`. A stream of
  correct floats therefore parks every cube at the receiver's origin, which
  looks exactly like "nothing arrived".
- **The blueprint only knows four names.** Index 0 feeds a *Switch on String*
  with cases `yel1`, `yel2`, `yel3`, `red`, each wired to one pre-placed cube
  component. There is no spawn path, so `yellow`, `blue`, `red/cross` and
  friends fall out the default pin and vanish. The bridge maps detections onto
  those four names (yellows small->large, biggest red) and prints a one-time
  warning for any colour the blueprint has no cube for.

#### Several pieces at once

The tracker follows every piece the camera can see, of any colour, including
several of the **same** colour, and gives each a stable name that follows that
brick: `yellow1`, `yellow2`, `red1`, ... The name is what decides which Unreal
cube a brick drives, so a given brick stays on a given cube instead of the
whole scene reshuffling whenever two similar bricks swap size order.

`BP_OSCreciver` itself can only ever show **four** pieces — it has four
pre-placed cubes (three yellow, one red) and no spawn path — so when you drive
the blueprint, the bridge leases those four cubes out and prints what it had to
drop:

```
[bridge] red2: all 1 red cube(s) in the blueprint are already leased, so this
         piece is tracked but not shown. The receiver needs to spawn per name
         to go past 1.
[bridge] 'blue' tracked but BP_OSCreciver has no cube for it
[bridge] 6 tracked, 4 shown (2 over capacity)
```

Press `p` in the preview window for a numbered list of everything being
tracked; the HUD shows a live `pieces:` count. To get past four pieces in
Unreal, use the spawning receiver below instead of the blueprint.

Sizes are per brick, not per colour: `sizes.json` holds a list of known sizes
for each colour+shape and matches a piece to the nearest one, so two yellow
rectangles of different sizes keep their own centimetres. Old single-size
`sizes.json` files still load.

`--osc-metric` sends the rich float message
(`[name, x_mm, y_mm, angle, long_cm, short_cm, z_mm]`) instead. The current
blueprint **cannot** read it - it's there for when the receiver is rebuilt.

### 2d. Showing more than four pieces in Unreal (`ue_multi_receiver.py`)

`BP_OSCreciver` is a fixed set of cubes, so the ceiling is four. This script
replaces it for the multi-object case: it runs **inside Unreal's Python**,
listens for the OSC stream itself, and spawns/moves/destroys **one actor per
tracked piece** — any count, any colour. The blueprint is left untouched and
still works if you prefer it.

In the Unreal editor, open the Output Log, set the Cmd dropdown to **Python**,
and run:

```
exec(open(r"D:\Projects\rice-unreal-jam\ue_multi_receiver.py").read())
```

Then start the tracker pointed at the receiver's port:

```
python lego_locator_xyz.py 0 --osc-metric --osc-port 7001
```

`stop_receiver()` stops it and cleans up its actors; re-running the `exec` line
restarts it cleanly (it removes its own leftovers first). Actors are labelled
`LegoTwin_<piece>`, e.g. `LegoTwin_yellow2`.

Notable differences from the blueprint path:

- **It runs without pressing Play.** It ticks off Slate, not BeginPlay, so the
  twin updates in the editor viewport with the level stopped — the blueprint's
  OSC server only exists during Play, which is most of why "nothing shows up"
  was so easy to hit.
- **It uses `--osc-metric`.** Real centimetres and a real metric position, so
  size and layout are physically faithful rather than the blueprint's
  pixels-÷-50. It also accepts the legacy int message, if that's what's
  running.
- **Port 7001, not 7000**, so it doesn't fight the blueprint's server for the
  bind when the level *is* playing.
- **Knobs at the top of the file:** `DEMO_SCALE` (life-size bricks are a few cm
  and read tiny next to UE's 100 uu cube — 5–10 makes a better demo, and scales
  positions with sizes so the layout stays faithful), `WORLD_ORIGIN`, `FLIP_Y`,
  `DROP_AFTER_S`, and the per-colour `MATERIALS` map.

Its OSC parsing and unit maths are covered by a run of the real tracker against
the real datagrams (6 pieces → 6 distinct actors, correct sizes and
materials). The `unreal` API calls themselves — spawn, destroy, set transform —
have not been exercised in-engine yet, so expect to shake those out on first
run.

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

## Nothing appears in Unreal

The tracker printing detections proves the camera half works and nothing else.
Walk the link from the Unreal end inwards - the first three checks are on the
Unreal side, because that is where this usually fails:

1. **Is `BP_OSCreciver` in the level?** It is an actor with the five cubes as
   *components*, moved with *Set Relative Location*. If it was never dragged
   into `NewMap`, there is nothing in the world to move. Check the Outliner.
2. **Is the level playing?** The OSC server is created on *Event BeginPlay*.
   In the editor with PIE stopped, nothing is listening on port 7000 at all.
3. **Can you see the cubes at rest?** They start stacked at the actor's origin.
   If the actor is behind the camera or a long way off, they move correctly and
   you never see it. Point the viewport at the actor before blaming the stream.
4. **Is anything arriving?** Run `python send_test.py --no-z --rate 2`. That
   sends the exact legacy message with the four names the blueprint knows, no
   camera involved. If the cubes don't move for that, the problem is entirely
   in Unreal (steps 1-3, or the OSC plugin/port), not in the tracker.
5. **Is the tracker sending?** `--osc-verbose` prints each outgoing `/obj`
   line. No lines means no piece was detected in a colour the blueprint has a
   cube for - the bridge says so once per colour.
6. **Firewall.** OSC is UDP. Loopback is normally fine; sending from another
   machine (`--osc-host`) needs UDP 7000 open on the Unreal box.
7. **Only four pieces ever appear.** That's the blueprint's ceiling, not a
   tracking bug — the bridge prints `N tracked, 4 shown`. Use
   `ue_multi_receiver.py` (2d) for more.

Do not "fix" this by sending floats or richer names until the blueprint is
rebuilt to match - see 2c above for why those silently produce nothing.

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
