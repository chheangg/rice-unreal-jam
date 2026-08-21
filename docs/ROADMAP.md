# Hackathon Roadmap — Lego Shape Tracking → Unreal

> Pivot from paper markers to Lego pieces. One color = one shape. One ArUco tag = that shape's stable ID + rotation. Unreal Engine 5.8.

---

## Core concept

- **5 Lego colors**, each color = a specific pre-built shape (e.g. one color always forms an L, another a U).
- **Color + contour** → identifies *which* shape it is, and its outline.
- **One ArUco tag per shape** → gives that shape's stable ID, rotation (true 0–360°), and anchor position — no flicker, no size-sort reshuffling.
- Combine both: color tells you *what* it is, tag tells you *where/how it's turned*, reliably.

This replaces the earlier pink-sticky-note + size-sorting approach, which broke down under jitter, race conditions, and same-color duplicates.

---

## Open decision (answer before building the Unreal side)

**How should each shape be represented in Unreal?**
1. **Pre-made meshes (recommended)** — model an L mesh, U mesh, etc. ahead of time. Color/tag ID picks which mesh to spawn; tag moves/rotates it. Fast, reliable, demo-safe.
2. **Live-generated mesh** — send the detected outline polygon, build the mesh at runtime. Impressive but real engineering risk under a deadline — not recommended for today.

→ Go with **Option 1** unless there's a strong reason otherwise.

---

## 8-Hour Hackathon Plan

| Time | Task |
|---|---|
| **Hour 0–1** | Print ArUco tags (one per shape). Tape flat on top of each joined-Lego shape. Confirm camera + lighting (flashlight trick helps a lot). |
| **Hour 1–3** | Python: color detection (tuned per shape) + ArUco detection combined. Send OSC `[id, x, y, angle]` (+ color if needed) per shape. |
| **Hour 3–5** | Unreal: ID-based receiver (spawn cube/mesh per new tag ID, update on move) — replaces the old fixed Switch-on-color setup. |
| **Hour 5–6.5** | "Wow" layer — pick ONE: correct-colored meshes, snap-to-grid assembly, or per-ID distinct meshes. Don't attempt all three. |
| **Hour 6.5–7.5** | Harden for demo: lock lighting, tape camera down, add smoothing (Lerp/RInterp), rehearse the exact demo motion 10×. Prepare a fallback recorded clip. |
| **Hour 7.5–8** | Buffer — do not schedule work here. |

**Demo pitch:** "Build anything from these Lego pieces, and watch it appear in the virtual world in real time."

---

## Step 1 — Color tuning (in progress)

Use the HSV slider tuner below to find exact ranges for each of the 5 Lego colors. Tune under actual demo lighting.

```python
import cv2
import numpy as np

# HSV slider tuner — find exact ranges for each Lego color.
# Hold one color block in view, drag sliders until ONLY it shows white
# in the mask window, then write down the 6 numbers.

def nothing(x): pass

cap = cv2.VideoCapture(0)          # try 1 / 2 for DJI
cv2.namedWindow("Tuner")
for name, val in [("H low",0),("H high",179),("S low",0),
                  ("S high",255),("V low",0),("V high",255)]:
    cv2.createTrackbar(name, "Tuner", val, 179 if "H" in name else 255, nothing)

while True:
    ok, frame = cap.read()
    if not ok: break
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    hl = cv2.getTrackbarPos("H low","Tuner");  hh = cv2.getTrackbarPos("H high","Tuner")
    sl = cv2.getTrackbarPos("S low","Tuner");  sh = cv2.getTrackbarPos("S high","Tuner")
    vl = cv2.getTrackbarPos("V low","Tuner");  vh = cv2.getTrackbarPos("V high","Tuner")

    mask = cv2.inRange(hsv, (hl,sl,vl), (hh,sh,vh))
    result = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow("Camera", frame)
    cv2.imshow("Mask (white = detected)", mask)
    cv2.imshow("Result", result)
    print(f"HSV low=({hl},{sl},{vl})  high=({hh},{sh},{vh})")

    if cv2.waitKey(1) == 27: break

cap.release()
cv2.destroyAllWindows()
```

**Usage:** run per color (~2 min each) → drag sliders until only that shape is white in the mask → record the printed low/high → repeat for all 5 → paste results back for the next step (combined color+ArUco detection script).

**Status:** tuning ranges not yet collected — do this first.

---

## Depth / z (new — added alongside the 2D tracker)

`detect_xyz.py` + `depth_estimator.py` add a real `z` per block using
**Depth Anything 3** (ByteDance's monocular depth model — *not* Facebook's,
despite how it's sometimes referred to). Design notes:

- Heavy transformer, so depth runs on a **background thread** on the newest
  frame; the detection loop just reads the latest depth map. Preview stays
  smooth, z refreshes as fast as the GPU allows.
- `z` is **appended** to the OSC message (`[name,x,y,angle,sx,sy,z]`) so the
  existing Blueprint index reads don't shift — UE just needs a new
  "Get at Index 6".
- Metric depth in meters (smaller = closer). Absolute scale may need per-rig
  calibration (`metric_scale` knob).
- **Fails soft:** no torch / no DA3 / no GPU → falls back to 2D, sends
  `z = -1.0`. The original `detect.py` is untouched as the safe fallback.
- Wants a CUDA GPU for interactive rates; on the CPU it will crawl (consider
  `DA3-SMALL` for speed at the cost of metric accuracy).

## Real outline instead of a placeholder block (new — 2026-08-21)

Decision, resolving the "Option 1 vs Option 2" mesh question above in a
cheaper direction than either: don't do a full 3D scan/import, and don't
live-build a full 3D mesh — instead **flat-extrude the real detected 2D
outline** by a fixed height. `lego_locator_xyz.py --osc` now sends this as a
second message per confirmed piece:

```
/outline -> [name, n_points, x1_mm, y1_mm, ..., xn_mm, yn_mm, height_cm]
```

Vertices are the piece's simplified contour (`cv2.approxPolyDP`), already
back-projected into the same floor-frame world space as `/obj`'s x/y — so
they already carry the piece's true rotation, no separate angle math needed
on the Unreal side. `height_cm` is a **fixed** constant (`--outline-height`,
default 2.0cm) — not measured, just enough thickness to read as a solid.
Tradeoff: correct top-down footprint, uniformly flat on top (no real
knob/step detail). Unreal-side build (Geometry Script polygon extrude) is
tracked in `tasks/2026-08-21-unreal-outline-extrude.md`.

## Engine / stack

- **Unreal Engine 5.8** (MechTwin03 project baseline; OSC receiver Blueprint built for this version)
- If the hackathon PC has a different UE version: opening a 5.8 project on an *older* version can fail — install 5.8 to match rather than downgrade. Same/newer 5.8 is fine (just lets shaders recompile once).
- **OSC plugin** ("OSC (Open Sound Control)", not "Remote Control Protocol OSC"), port **7000**
- **Python:** OpenCV + `cv2.aruco` + `python-osc`

---

## Carried-over knowledge (from prior sessions — still applies)

- Code goes in the **file**, never pasted into the terminal; run with `python detect.py`.
- Any OSC "Get ... at Index" node in Blueprint has an exec pin — **must be threaded into the execution chain** or it's pruned and reads 0.
- Scale divisor that worked previously: **÷50** (not ÷100).
- Marker/tag color (or ArUco tag itself) must not collide with a tracked shape's own color.
- Lighting has an outsized effect on stability — bring the flashlight.
- Plain `minAreaRect` angle is 180°-ambiguous; ArUco avoids this entirely (true 0–360°).
- Previous race condition risk: multiple OSC messages per frame can overlap in a long Blueprint graph. Consider per-ID/per-address messages if this reappears.

---

## Next actions

1. [ ] Run the HSV tuner on all 5 Lego colors under demo lighting; record ranges.
2. [ ] Decide: pre-made meshes vs. live-generated (→ recommend pre-made).
3. [ ] Build combined color + ArUco detection script (position, rotation, ID, shape/color label).
4. [ ] Build Unreal ID-based spawn/update receiver.
5. [ ] Add one "wow" feature only.
6. [ ] Harden + rehearse.
