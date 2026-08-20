"""
detect_lego.py — COLOR + ArUco Lego tracker that snaps in the BOARD frame,
with depth (z).

The problem with a slightly-angled top-down camera
--------------------------------------------------
The camera does NOT see the Lego lattice aligned to its pixel grid: the board
is rotated on screen, so snapping in pixel space snaps to the wrong lattice —
adjacent pieces land in non-adjacent cells and "straight" pieces read as odd
angles.

Two ways to fix it (set BOARD_MODE):

  "selfcal"  (default, ZERO extra setup) — every shape already carries an
             ArUco tag, and Lego pieces are grid-aligned, so all their tag
             angles agree modulo 90°. We read the board's rotation from the
             pieces themselves (a robust circular estimate, smoothed over
             time since the camera is fixed) and snap in that rotated frame.
             Corrects rotation, not perspective — fine for a *slightly*
             angled cam. No corner tags to tape.

  "corners"  (higher accuracy) — tape 4 ArUco tags at the table corners
             (ids 0=TL,1=TR,2=BR,3=BL). A homography maps pixels onto a flat
             board grid, correcting rotation AND perspective/tilt. More setup.

Per shape we output stable id, board cell (x,y), metric depth (z), snapped
board-relative angle, shape label, colour.

OSC (one message per shape — avoids the multi-message race noted in the doc):
    /shape       [id, x, y, z, angle, shape, color]
    /shape_gone  [id]
  x,y   : SNAPPED board cell coords (ints). In Unreal place at x,y * cellSize.
  z     : metric depth in meters (Depth Anything 3), -1.0 if unavailable.
  angle : SNAPPED board-relative rotation, 0/90/180/270.

Setup:
  - Print ArUco tags from DICT_4X4_50; give every shape its own tag.
    In "corners" mode reserve ids 0-3 for the corners and use ids >= 4 for
    shapes. In "selfcal" mode any ids are fine.
  - Tune the COLORS HSV ranges with the tuner in docs/ROADMAP.md.
  - Set GRID (selfcal) or BOARD_COLS/ROWS (corners) to your Lego stud size.

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

BOARD_MODE = "selfcal"          # "selfcal" (no extra tags) or "corners" (homography)

# --- selfcal mode ---
GRID = 48                       # px per grid cell — set near your Lego stud size
ROT_SMOOTH = 0.15               # 0..1 : how fast the board-rotation estimate adapts

# --- corners mode ---
CORNER_IDS = {0: "TL", 1: "TR", 2: "BR", 3: "BL"}
BOARD_COLS, BOARD_ROWS = 16, 12
FIRST_SHAPE_ID = 4              # in corners mode, shape tags must be >= this

USE_DEPTH = True                # False = skip Depth Anything for max FPS

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

SNAP_MARGIN = 0.35              # 0..0.5 : distance past a cell before it jumps
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
        tl, tr = pts[0], pts[1]
        angle = math.degrees(math.atan2(tr[1] - tl[1], tr[0] - tl[0])) % 360
        out.append({"id": int(i), "cx": float(cx), "cy": float(cy),
                    "angle": angle, "pts": pts})
    return out


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


# ---- board rotation, self-calibrated from the shape tags ----------------
class BoardRotation:
    """
    Estimates the board's in-plane rotation from grid-aligned pieces. Each
    tag angle is only meaningful modulo 90°, so we work with the phasor
    e^{i*4*theta} (period 90° -> full turn) and take a smoothed circular mean.
    Because the camera is fixed, this locks onto a steady value quickly.
    """
    def __init__(self, smooth):
        self.smooth = smooth
        self._acc = None                          # complex EMA of e^{i4θ}

    def update(self, angles_deg):
        if angles_deg:
            v = np.mean([np.exp(1j * 4 * math.radians(a)) for a in angles_deg])
            if abs(v) > 1e-6:
                v /= abs(v)
                self._acc = v if self._acc is None else \
                    (1 - self.smooth) * self._acc + self.smooth * v
        if self._acc is None:
            return 0.0
        return (math.degrees(math.atan2(self._acc.imag, self._acc.real)) / 4.0) % 90.0


def to_board(px, py, rot_deg, cx0, cy0):
    """Rotate a pixel point by -rot about (cx0,cy0) -> board-aligned pixels."""
    t = math.radians(-rot_deg)
    dx, dy = px - cx0, py - cy0
    bx = dx * math.cos(t) - dy * math.sin(t)
    by = dx * math.sin(t) + dy * math.cos(t)
    return bx, by


# ---- corners mode: homography (camera px -> board cells) ----------------
BOARD_DST = {"TL": (0, 0), "TR": (BOARD_COLS, 0),
             "BR": (BOARD_COLS, BOARD_ROWS), "BL": (0, BOARD_ROWS)}


def compute_homography(markers):
    seen = {CORNER_IDS[m["id"]]: (m["cx"], m["cy"])
            for m in markers if m["id"] in CORNER_IDS}
    if len(seen) < 4:
        return None
    src = [seen[r] for r in ("TL", "TR", "BR", "BL")]
    dst = [BOARD_DST[r] for r in ("TL", "TR", "BR", "BL")]
    return cv2.getPerspectiveTransform(np.array(src, np.float32),
                                       np.array(dst, np.float32))


def warp(H, x, y):
    p = cv2.perspectiveTransform(np.array([[[x, y]]], np.float32), H)
    return float(p[0, 0, 0]), float(p[0, 0, 1])


def homography_angle(H, m):
    tl, tr = m["pts"][0], m["pts"][1]
    c = warp(H, m["cx"], m["cy"])
    tip = warp(H, m["cx"] + (tr[0] - tl[0]), m["cy"] + (tr[1] - tl[1]))
    return math.degrees(math.atan2(tip[1] - c[1], tip[0] - c[0])) % 360


# ---- snapping (board-cell units), with hysteresis -----------------------
def _snap_cell(v, cur):
    if cur is None:
        return round(v)
    if abs(v - cur) > (0.5 + SNAP_MARGIN):
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


def draw_rotated_grid(frame, rot_deg, cx0, cy0):
    """Draw the board-aligned grid back onto the frame (visual confirmation)."""
    h, w = frame.shape[:2]
    t = math.radians(rot_deg)
    ux, uy = math.cos(t), math.sin(t)             # board x axis in pixels
    vx, vy = -math.sin(t), math.cos(t)            # board y axis in pixels
    reach = int(math.hypot(w, h) / GRID) + 2
    col = (70, 70, 70)
    for i in range(-reach, reach + 1):
        ox, oy = cx0 + i * GRID * ux, cy0 + i * GRID * uy
        a = (int(ox - reach * GRID * vx), int(oy - reach * GRID * vy))
        b = (int(ox + reach * GRID * vx), int(oy + reach * GRID * vy))
        cv2.line(frame, a, b, col, 1)
        ox, oy = cx0 + i * GRID * vx, cy0 + i * GRID * vy
        a = (int(ox - reach * GRID * ux), int(oy - reach * GRID * uy))
        b = (int(ox + reach * GRID * ux), int(oy + reach * GRID * uy))
        cv2.line(frame, a, b, col, 1)


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
    board_rot = BoardRotation(ROT_SMOOTH)
    H_cached = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if depth:
                depth.submit(frame)
            work = cv2.GaussianBlur(frame, (BLUR, BLUR), 0) if BLUR >= 3 else frame
            hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
            h, w = frame.shape[:2]
            cx0, cy0 = w / 2, h / 2

            markers = detect_markers(gray, aruco_state)
            blobs = color_blobs(hsv)

            # decide the board frame for this frame
            H = None
            rot = 0.0
            if BOARD_MODE == "corners":
                H = compute_homography(markers)
                if H is not None:
                    H_cached = H
                H = H_cached
            else:  # selfcal
                shape_angles = [m["angle"] for m in markers
                                if BOARD_MODE != "corners" or m["id"] >= FIRST_SHAPE_ID]
                rot = board_rot.update(shape_angles)

            seen_ids = set()
            ready = (H is not None) if BOARD_MODE == "corners" else True

            if BOARD_MODE == "corners" and not ready:
                cv2.putText(frame, "Show all 4 corner tags (ids 0-3) to lock the board",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                if BOARD_MODE == "corners":
                    # (grid drawn per-cell would need Hinv; skip for brevity)
                    pass
                else:
                    draw_rotated_grid(frame, rot, cx0, cy0)
                    cv2.putText(frame, f"board rot {rot:.1f}deg", (10, 24),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                for mk in markers:
                    if BOARD_MODE == "corners" and mk["id"] < FIRST_SHAPE_ID:
                        continue
                    color, ct = blob_for_point(blobs, mk["cx"], mk["cy"])
                    if color is None:
                        continue
                    M = cv2.moments(ct)
                    if M["m00"] == 0:
                        continue
                    pcx, pcy = M["m10"] / M["m00"], M["m01"] / M["m00"]

                    if BOARD_MODE == "corners":
                        bx, by = warp(H, pcx, pcy)
                        ang = homography_angle(H, mk)
                    else:
                        bxp, byp = to_board(pcx, pcy, rot, cx0, cy0)
                        bx, by = bxp / GRID, byp / GRID
                        ang = (mk["angle"] - rot) % 360

                    st = states.setdefault(mk["id"], ShapeState())
                    sx, sy, sa = st.snap(bx, by, ang)
                    seen_ids.add(mk["id"])

                    z = depth.depth_at(cv2.boundingRect(ct)) if depth else None
                    z_val = Z_MISSING if z is None else round(z, 3)

                    osc.send_message("/shape", [mk["id"], sx, sy, z_val, sa,
                                                color["shape"], color["name"]])

                    cv2.drawContours(frame, [ct], -1, color["draw"], 2)
                    cv2.circle(frame, (int(pcx), int(pcy)), 4, color["draw"], -1)
                    cv2.putText(frame,
                                f"#{mk['id']} {color['shape']}/{color['name']} "
                                f"({sx},{sy}) {sa}d z={z_val}",
                                (int(pcx) + 8, int(pcy) - 8),
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
                cv2.putText(frame, s, (10, h - 12),
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
