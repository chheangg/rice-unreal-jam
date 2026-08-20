"""
detect_lego.py — COLOR + ArUco Lego tracker, snapping in a rectified BOARD
frame, with depth (z).

Why a board frame?
------------------
A slightly-angled top-down camera does NOT see the Lego lattice aligned to
its pixel grid: the whole board is rotated (and a bit perspective-warped)
relative to the screen. Snapping in pixel space therefore snaps to the wrong
lattice — pieces that are physically adjacent land in non-adjacent screen
cells, and "straight" pieces read as odd angles.

Fix: tape 4 ArUco tags at the table corners. We compute a homography that
maps camera pixels onto a flat, axis-aligned board grid measured in *cells*.
Everything — positions and angles — is transformed into that board frame and
snapped there, so snapping lines up with the real Lego grid regardless of how
the camera is tilted/rotated. The camera is fixed, so the homography is
essentially constant: we compute it whenever all 4 corners are visible and
cache it, staying robust if a corner tag is briefly occluded.

Per shape we output: stable id, board cell (x,y), metric depth (z), snapped
angle (board-relative), shape label, colour.

OSC (one message per shape — avoids the multi-message race noted in the doc):
    /shape       [id, x, y, z, angle, shape, color]
    /shape_gone  [id]                      (fired when a tracked shape leaves)
  x, y   : SNAPPED board cell coordinates (integers). In Unreal, place at
           x,y * your cell size — no ÷50 needed, this is already grid space.
  z      : metric depth in meters from Depth Anything 3 (smaller = closer to
           the camera). -1.0 if depth is unavailable.
  angle  : SNAPPED board-relative rotation, 0/90/180/270.

Setup:
  - Print ArUco tags from DICT_4X4_50.
  - Tag ids 0,1,2,3 are the board corners: 0=top-left, 1=top-right,
    2=bottom-right, 3=bottom-left. Tape them flat at the table corners.
  - Give every shape its own tag with id >= 4.
  - Set BOARD_COLS / BOARD_ROWS to how many grid cells span the taped
    rectangle (this defines the snap lattice).
  - Tune the COLORS HSV ranges with the tuner in docs/ROADMAP.md.

Run:  python detect_lego.py      (Esc to quit)
"""

import cv2
import numpy as np
import math
from pythonosc.udp_client import SimpleUDPClient

from depth_estimator import DepthEstimator

# ======================================================================
# CONFIG
# ======================================================================
OSC_IP, OSC_PORT = "127.0.0.1", 7000
CAM_INDEX = 0
CAP_W, CAP_H = 1280, 720        # cap capture res for real-time margin (0,0 = default)

# Board corner tags (reserved ids) and how many grid cells span the board.
CORNER_IDS = {0: "TL", 1: "TR", 2: "BR", 3: "BL"}   # tag id -> corner role
BOARD_COLS = 16                 # cells across (width)  -> snap lattice
BOARD_ROWS = 12                 # cells down   (height)
FIRST_SHAPE_ID = 4              # shape tags must use ids >= this

USE_DEPTH = True                # set False to skip Depth Anything (faster)

# One entry per Lego colour: colour -> its pre-built shape. hsv is a LIST of
# (low, high) ranges (red wraps the hue circle, so it uses two).
# ---> Replace ranges with numbers from the ROADMAP HSV tuner. <---
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

MIN_AREA = 800                  # ignore colour blobs smaller than this (px^2)
BLUR = 5                        # gaussian blur to calm colour noise (odd)

SNAP_MARGIN = 0.35              # 0..0.5 : extra distance past a cell before it jumps
ANGLE_STEP = 90                 # Lego snaps to right angles
ANGLE_MARGIN = 15              # deg past the 45 boundary before angle flips
GONE_FRAMES = 12               # frames unseen before /shape_gone

ARUCO_DICT = "DICT_4X4_50"
Z_MISSING = -1.0
# ======================================================================

osc = SimpleUDPClient(OSC_IP, OSC_PORT)


# ---- ArUco: detector that works on old AND new OpenCV --------------------
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
        pts = c.reshape(4, 2)                      # TL, TR, BR, BL of the tag
        cx, cy = pts.mean(axis=0)
        out.append({"id": int(i), "cx": float(cx), "cy": float(cy), "pts": pts})
    return out


# ---- homography (camera px -> board cells) ------------------------------
BOARD_DST = {
    "TL": (0, 0), "TR": (BOARD_COLS, 0),
    "BR": (BOARD_COLS, BOARD_ROWS), "BL": (0, BOARD_ROWS),
}


def compute_homography(markers):
    """If all 4 corner tags are visible, return H (px->cells), else None."""
    src, dst = [], []
    seen = {}
    for m in markers:
        role = CORNER_IDS.get(m["id"])
        if role:
            seen[role] = (m["cx"], m["cy"])
    if len(seen) < 4:
        return None
    for role in ("TL", "TR", "BR", "BL"):
        src.append(seen[role])
        dst.append(BOARD_DST[role])
    return cv2.getPerspectiveTransform(np.array(src, np.float32),
                                       np.array(dst, np.float32))


def warp(H, x, y):
    p = cv2.perspectiveTransform(np.array([[[x, y]]], np.float32), H)
    return float(p[0, 0, 0]), float(p[0, 0, 1])


def board_angle(H, m):
    """Marker orientation expressed in the board frame, 0..360."""
    tl, tr = m["pts"][0], m["pts"][1]              # top edge direction in px
    c = warp(H, m["cx"], m["cy"])
    tip = warp(H, m["cx"] + (tr[0] - tl[0]), m["cy"] + (tr[1] - tl[1]))
    return math.degrees(math.atan2(tip[1] - c[1], tip[0] - c[0])) % 360


# ---- colour blobs -------------------------------------------------------
def color_blobs(hsv):
    kernel = np.ones((5, 5), np.uint8)
    found = {}
    for c in COLORS:
        mask = None
        for lo, hi in c["hsv"]:
            m = cv2.inRange(hsv, np.array(lo), np.array(hi))
            mask = m if mask is None else (mask | m)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        found[c["name"]] = [ct for ct in contours
                            if cv2.contourArea(ct) >= MIN_AREA]
    return found


def blob_for_point(blobs, px, py):
    for c in COLORS:
        for ct in blobs.get(c["name"], []):
            if cv2.pointPolygonTest(ct, (float(px), float(py)), False) >= 0:
                return c, ct
    return None, None


# ---- snapping (in board-cell units), with hysteresis --------------------
def _snap_cell(v, cur):
    if cur is None:
        return round(v)
    if abs(v - cur) > (0.5 + SNAP_MARGIN):         # past deadband -> jump
        return round(v)
    return cur


def _snap_angle(a, cur):
    nearest = (round(a / ANGLE_STEP) * ANGLE_STEP) % 360
    if cur is None:
        return nearest
    d = ((a - cur + 180) % 360) - 180
    if abs(d) > (ANGLE_STEP / 2 + ANGLE_MARGIN):
        return nearest
    return cur


class ShapeState:
    def __init__(self):
        self.x = self.y = self.a = None
        self.unseen = 0

    def snap(self, x, y, a):
        self.x = _snap_cell(x, self.x)
        self.y = _snap_cell(y, self.y)
        self.a = _snap_angle(a, self.a)
        self.unseen = 0
        return int(self.x), int(self.y), int(self.a)


def draw_board_grid(frame, H):
    """Draw the rectified cell grid back onto the frame (visual confirmation)."""
    Hinv = np.linalg.inv(H)
    col = (70, 70, 70)
    for i in range(BOARD_COLS + 1):
        a = warp(Hinv, i, 0); b = warp(Hinv, i, BOARD_ROWS)
        cv2.line(frame, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), col, 1)
    for j in range(BOARD_ROWS + 1):
        a = warp(Hinv, 0, j); b = warp(Hinv, BOARD_COLS, j)
        cv2.line(frame, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), col, 1)


def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("Camera not found — try CAM_INDEX 1 or 2")
        return
    if CAP_W and CAP_H:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_H)

    aruco_state = make_aruco()
    depth = DepthEstimator() if USE_DEPTH else None
    if depth:
        depth.start()
    states = {}
    H_cached = None                                # last good homography

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if depth:
                depth.submit(frame)               # feed the depth worker (raw frame)
            work = cv2.GaussianBlur(frame, (BLUR, BLUR), 0) if BLUR >= 3 else frame
            hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)

            markers = detect_markers(gray, aruco_state)
            H = compute_homography(markers)
            if H is not None:
                H_cached = H
            H = H_cached                          # use cached if corners hidden

            blobs = color_blobs(hsv)
            seen_ids = set()

            if H is None:
                cv2.putText(frame, "Show all 4 corner tags (ids 0-3) to lock the board",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                draw_board_grid(frame, H)
                for mk in markers:
                    if mk["id"] < FIRST_SHAPE_ID:     # skip corner tags
                        continue
                    color, ct = blob_for_point(blobs, mk["cx"], mk["cy"])
                    if color is None:
                        continue
                    M = cv2.moments(ct)
                    if M["m00"] == 0:
                        continue
                    cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]   # centroid (px)

                    bx, by = warp(H, cx, cy)          # -> board cell space
                    ang = board_angle(H, mk)
                    st = states.setdefault(mk["id"], ShapeState())
                    sx, sy, sa = st.snap(bx, by, ang)
                    seen_ids.add(mk["id"])

                    z = depth.depth_at(cv2.boundingRect(ct)) if depth else None
                    z_val = Z_MISSING if z is None else round(z, 3)

                    osc.send_message("/shape", [mk["id"], sx, sy, z_val, sa,
                                                color["shape"], color["name"]])

                    cv2.drawContours(frame, [ct], -1, color["draw"], 2)
                    cv2.circle(frame, (int(cx), int(cy)), 4, color["draw"], -1)
                    cv2.putText(frame,
                                f"#{mk['id']} {color['shape']}/{color['name']} "
                                f"({sx},{sy}) {sa}d z={z_val}",
                                (int(cx) + 8, int(cy) - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color["draw"], 2)

            # despawn shapes gone too long
            for sid in list(states):
                if sid in seen_ids:
                    continue
                states[sid].unseen += 1
                if states[sid].unseen > GONE_FRAMES:
                    osc.send_message("/shape_gone", [sid])
                    del states[sid]

            if depth:
                s = "DEPTH on" if depth.available and depth.has_depth() else \
                    ("DEPTH loading" if depth.available else "DEPTH off")
                cv2.putText(frame, s, (10, frame.shape[0] - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            cv2.imshow("Lego shapes (board-frame snap + depth)", frame)
            if cv2.waitKey(1) == 27:
                break
    finally:
        if depth:
            depth.stop()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
