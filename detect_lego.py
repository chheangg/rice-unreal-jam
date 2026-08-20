"""
detect_lego.py — combined COLOR + ArUco Lego shape tracker with SNAPPING.

This is the "combined color + ArUco detection" step from docs/ROADMAP.md,
plus the snap-to-grid "wow" layer so shapes combine cleanly.

The doc's core idea:
  - 5 Lego colours, each colour = one pre-built shape (L, U, T, I, Z ...).
    Colour tells you WHAT the shape is (+ gives its outline/centre).
  - One ArUco tag per shape gives a stable ID and TRUE 0-360 rotation
    (no minAreaRect 180 ambiguity, no size-sort reshuffling, no flicker).
  - Pair them: the tag sitting inside a colour blob => that blob's shape,
    its stable id, and its rotation.

Why snapping:
  Lego lives on a grid at right angles. We quantise each shape's position to
  a grid and its rotation to 90 deg, with hysteresis (a deadband) so small
  jitter doesn't make it twitch between cells. Result: place two shapes near
  each other and they land on the SAME lattice -> in Unreal they line up and
  "snap" together instead of floating at slightly-off angles.

OSC (address /shape, one message per shape per frame — per the doc's note
about avoiding multi-message race conditions, each shape is its own message):
    [id, x, y, angle, shape, color]
    id     : ArUco tag id (stable per shape)
    x, y   : SNAPPED pixel centre (Unreal divides by the scale divisor, ÷50)
    angle  : SNAPPED rotation, 0/90/180/270
    shape  : shape label for this colour ("L", "U", ...)
    color  : colour name ("red", ...)
When a shape that was being tracked disappears, one /shape_gone [id] is sent
so the Unreal side can despawn it.

Run:  python detect_lego.py      (Esc to quit)
Tune: adjust COLORS ranges with the HSV tuner in docs/ROADMAP.md, then set
      GRID / ANGLE_STEP to match your Lego stud size on screen.
"""

import cv2
import numpy as np
import math
from pythonosc.udp_client import SimpleUDPClient

# ======================================================================
# CONFIG — tune this block for your lighting / Lego set / grid.
# ======================================================================
OSC_IP, OSC_PORT = "127.0.0.1", 7000
CAM_INDEX = 0                 # try 1 / 2 for an external / DJI cam

# One entry per Lego colour. hsv is a LIST of (low, high) ranges so colours
# that wrap around the hue circle (red) can use two ranges. `shape` is the
# pre-built shape that colour always forms; `draw` is just the overlay colour.
# ---> Replace these ranges with numbers from the ROADMAP HSV tuner. <---
COLORS = [
    {"name": "red",    "shape": "L", "draw": (0, 0, 255),
     "hsv": [((0, 120, 90), (8, 255, 255)), ((170, 120, 90), (179, 255, 255))]},
    {"name": "orange", "shape": "T", "draw": (0, 140, 255),
     "hsv": [((9, 130, 120), (19, 255, 255))]},
    {"name": "yellow", "shape": "U", "draw": (0, 255, 255),
     "hsv": [((20, 90, 130), (35, 255, 255))]},
    {"name": "green",  "shape": "I", "draw": (0, 200, 0),
     "hsv": [((40, 70, 70), (85, 255, 255))]},
    {"name": "blue",   "shape": "Z", "draw": (255, 120, 0),
     "hsv": [((95, 90, 70), (130, 255, 255))]},
]

MIN_AREA = 800               # ignore colour blobs smaller than this (px^2)
BLUR = 5                     # gaussian blur kernel to calm colour noise (odd)

GRID = 40                    # px per grid cell — set to your Lego stud size
SNAP_MARGIN = 0.35           # 0..0.5 : extra distance past a cell before it jumps
ANGLE_STEP = 90              # Lego snaps to right angles
ANGLE_MARGIN = 15            # deg past the 45 boundary before angle flips

GONE_FRAMES = 12             # frames a shape can be unseen before /shape_gone

ARUCO_DICT = "DICT_4X4_50"   # print tags from this dictionary
# ======================================================================

osc = SimpleUDPClient(OSC_IP, OSC_PORT)


# ---- ArUco: build a detector that works on old AND new OpenCV -----------
def make_aruco():
    a = cv2.aruco
    dic = a.getPredefinedDictionary(getattr(a, ARUCO_DICT))
    try:
        params = a.DetectorParameters()          # OpenCV >= 4.7
    except AttributeError:
        params = a.DetectorParameters_create()   # OpenCV < 4.7
    try:
        return ("new", a.ArucoDetector(dic, params))
    except AttributeError:
        return ("old", (dic, params))


def detect_markers(gray, aruco_state):
    mode, obj = aruco_state
    if mode == "new":
        corners, ids, _ = obj.detectMarkers(gray)
    else:
        dic, params = obj
        corners, ids, _ = cv2.aruco.detectMarkers(gray, dic, parameters=params)
    out = []
    if ids is None:
        return out
    for c, i in zip(corners, ids.flatten()):
        pts = c.reshape(4, 2)                      # TL, TR, BR, BL
        cx, cy = pts.mean(axis=0)
        tl, tr = pts[0], pts[1]
        angle = math.degrees(math.atan2(tr[1] - tl[1], tr[0] - tl[0])) % 360
        out.append({"id": int(i), "cx": float(cx), "cy": float(cy),
                    "angle": angle, "pts": pts})
    return out


# ---- colour blobs -------------------------------------------------------
def color_blobs(hsv):
    """Return {color_name: [contour, ...]} for blobs above MIN_AREA."""
    kernel = np.ones((5, 5), np.uint8)
    found = {}
    for c in COLORS:
        mask = None
        for lo, hi in c["hsv"]:
            m = cv2.inRange(hsv, np.array(lo), np.array(hi))
            mask = m if mask is None else (mask | m)
        # clean the mask: close small holes, drop specks
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        found[c["name"]] = [ct for ct in contours
                            if cv2.contourArea(ct) >= MIN_AREA]
    return found


def blob_for_point(blobs, px, py):
    """Which colour/contour contains point (px,py)? -> (color_dict, contour)."""
    for c in COLORS:
        for ct in blobs.get(c["name"], []):
            if cv2.pointPolygonTest(ct, (float(px), float(py)), False) >= 0:
                return c, ct
    return None, None


# ---- snapping (per shape id), with hysteresis so it holds steady --------
def _snap_pos(v, cur):
    cell = v / GRID
    if cur is None:
        return round(cell) * GRID
    if abs(cell - cur / GRID) > (0.5 + SNAP_MARGIN):   # past deadband -> jump
        return round(cell) * GRID
    return cur


def _snap_angle(a, cur):
    nearest = (round(a / ANGLE_STEP) * ANGLE_STEP) % 360
    if cur is None:
        return nearest
    d = ((a - cur + 180) % 360) - 180                  # signed diff to current
    if abs(d) > (ANGLE_STEP / 2 + ANGLE_MARGIN):
        return nearest
    return cur


class ShapeState:
    def __init__(self):
        self.x = self.y = self.a = None
        self.unseen = 0

    def snap(self, x, y, a):
        self.x = _snap_pos(x, self.x)
        self.y = _snap_pos(y, self.y)
        self.a = _snap_angle(a, self.a)
        self.unseen = 0
        return int(self.x), int(self.y), int(self.a)


def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("Camera not found — try CAM_INDEX 1 or 2")
        return
    aruco_state = make_aruco()
    states = {}                                        # id -> ShapeState

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if BLUR >= 3:
            frame = cv2.GaussianBlur(frame, (BLUR, BLUR), 0)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        blobs = color_blobs(hsv)
        markers = detect_markers(gray, aruco_state)

        seen_ids = set()
        for mk in markers:
            color, ct = blob_for_point(blobs, mk["cx"], mk["cy"])
            if color is None:
                continue                               # tag not on any shape
            M = cv2.moments(ct)                         # shape centroid = anchor
            if M["m00"] == 0:
                continue
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]

            st = states.setdefault(mk["id"], ShapeState())
            sx, sy, sa = st.snap(cx, cy, mk["angle"])
            seen_ids.add(mk["id"])

            osc.send_message("/shape", [mk["id"], sx, sy, sa,
                                        color["shape"], color["name"]])

            # overlay
            cv2.drawContours(frame, [ct], -1, color["draw"], 2)
            cv2.circle(frame, (sx, sy), 4, color["draw"], -1)
            end = (int(sx + 30 * math.cos(math.radians(sa))),
                   int(sy + 30 * math.sin(math.radians(sa))))
            cv2.arrowedLine(frame, (sx, sy), end, color["draw"], 2)
            cv2.putText(frame, f"#{mk['id']} {color['shape']}/{color['name']} {sa}deg",
                        (sx + 8, sy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        color["draw"], 2)

        # despawn shapes that have been gone too long
        for sid in list(states):
            if sid in seen_ids:
                continue
            states[sid].unseen += 1
            if states[sid].unseen > GONE_FRAMES:
                osc.send_message("/shape_gone", [sid])
                del states[sid]

        # faint grid so you can see the lattice everything snaps to
        h, w = frame.shape[:2]
        for gx in range(0, w, GRID):
            cv2.line(frame, (gx, 0), (gx, h), (60, 60, 60), 1)
        for gy in range(0, h, GRID):
            cv2.line(frame, (0, gy), (w, gy), (60, 60, 60), 1)

        cv2.imshow("Lego shapes (color + ArUco, snapped)", frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
