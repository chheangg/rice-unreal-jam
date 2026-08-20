"""
lego_core.py — reusable Lego shape detection (colour + ArUco + board-frame
snapping + optional depth), shared by:
  - detect_lego.py   (desktop: laptop webcam + OpenCV window)
  - server_phone.py  (phone camera over ngrok -> browser overlay)

Feed it BGR frames via LegoTracker.process(frame); it returns a per-shape
result list and (optionally) sends OSC to Unreal. All the "why" is documented
in detect_lego.py / docs/ROADMAP.md; this module is the engine.
"""

import math
import cv2
import numpy as np
from pythonosc.udp_client import SimpleUDPClient


# ---------------- default config (override via LegoTracker kwargs) --------
# Colours are classified per-blob by MEDIAN H/S/V, not by overlapping inRange
# masks. This is what stops white and yellow bleeding into each other: white
# is defined by LOW saturation, chromatic colours by their hue band AND a
# minimum saturation, so a bright pale pixel can only ever be white, and a
# saturated hue-20..38 pixel can only ever be yellow.
#
# Chromatic colours give `hue` (list of (lo,hi), wrapping allowed for red).
# White gives `white: True`. `shape` is the pre-built shape for that colour.
DEFAULT_COLORS = [
    {"name": "red",    "shape": "L", "draw": (0, 0, 255),   "hue": [(0, 10), (170, 179)]},
    {"name": "orange", "shape": "T", "draw": (0, 140, 255), "hue": [(11, 22)]},
    {"name": "yellow", "shape": "U", "draw": (0, 255, 255), "hue": [(23, 38)]},
    {"name": "green",  "shape": "I", "draw": (0, 200, 0),   "hue": [(40, 85)]},
    {"name": "blue",   "shape": "Z", "draw": (255, 120, 0), "hue": [(90, 130)]},
    {"name": "white",  "shape": "O", "draw": (240, 240, 240), "white": True},
]

# Saturation/value gates that separate white from the chromatic colours.
# Tune these two first if white<->yellow still cross over:
S_MIN_CHROMA = 80      # a hue only counts if at least this saturated
V_MIN_CHROMA = 60      # ...and at least this bright
WHITE_S_MAX = 55       # white must be LESS saturated than this
WHITE_V_MIN = 150      # ...and at least this bright


# ---------------- ArUco (old + new OpenCV) -------------------------------
def make_aruco(dict_name="DICT_4X4_50"):
    a = cv2.aruco
    dic = a.getPredefinedDictionary(getattr(a, dict_name))
    try:
        params = a.DetectorParameters()
    except AttributeError:
        params = a.DetectorParameters_create()
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
        pts = c.reshape(4, 2)
        cx, cy = pts.mean(axis=0)
        tl, tr = pts[0], pts[1]
        angle = math.degrees(math.atan2(tr[1] - tl[1], tr[0] - tl[0])) % 360
        out.append({"id": int(i), "cx": float(cx), "cy": float(cy),
                    "angle": angle, "pts": pts})
    return out


# ---------------- board rotation, self-calibrated ------------------------
class BoardRotation:
    def __init__(self, smooth=0.15):
        self.smooth = smooth
        self._acc = None

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
    t = math.radians(-rot_deg)
    dx, dy = px - cx0, py - cy0
    return (dx * math.cos(t) - dy * math.sin(t),
            dx * math.sin(t) + dy * math.cos(t))


# ---------------- corners mode homography --------------------------------
def compute_homography(markers, corner_ids, cols, rows):
    dst_map = {"TL": (0, 0), "TR": (cols, 0), "BR": (cols, rows), "BL": (0, rows)}
    seen = {corner_ids[m["id"]]: (m["cx"], m["cy"])
            for m in markers if m["id"] in corner_ids}
    if len(seen) < 4:
        return None
    src = [seen[r] for r in ("TL", "TR", "BR", "BL")]
    dst = [dst_map[r] for r in ("TL", "TR", "BR", "BL")]
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


# ---------------- snapping (per shape) -----------------------------------
class ShapeState:
    def __init__(self):
        self.x = self.y = self.a = None
        self.unseen = 0


class LegoTracker:
    def __init__(self, colors=None, board_mode="selfcal", grid=48,
                 rot_smooth=0.15, corner_ids=None, board_cols=16, board_rows=12,
                 first_shape_id=4, min_area=800, blur=5,
                 snap_margin=0.35, angle_step=90, angle_margin=15,
                 gone_frames=12, dict_name="DICT_4X4_50",
                 osc_ip="127.0.0.1", osc_port=7000, send_osc=True,
                 depth=None):
        self.colors = colors or DEFAULT_COLORS
        self.board_mode = board_mode
        self.grid = grid
        self.corner_ids = corner_ids or {0: "TL", 1: "TR", 2: "BR", 3: "BL"}
        self.board_cols, self.board_rows = board_cols, board_rows
        self.first_shape_id = first_shape_id
        self.min_area, self.blur = min_area, blur
        self.snap_margin, self.angle_step, self.angle_margin = \
            snap_margin, angle_step, angle_margin
        self.gone_frames = gone_frames
        self.aruco = make_aruco(dict_name)
        self.board_rot = BoardRotation(rot_smooth)
        self.states = {}
        self.H_cached = None
        self.depth = depth
        self.send_osc = send_osc
        self.osc = SimpleUDPClient(osc_ip, osc_port) if send_osc else None

    # -- snapping helpers --
    def _snap_cell(self, v, cur):
        if cur is None:
            return round(v)
        return round(v) if abs(v - cur) > (0.5 + self.snap_margin) else cur

    def _snap_angle(self, a, cur):
        nearest = (round(a / self.angle_step) * self.angle_step) % 360
        if cur is None:
            return nearest
        d = ((a - cur + 180) % 360) - 180
        return nearest if abs(d) > (self.angle_step / 2 + self.angle_margin) else cur

    def _color_blobs(self, hsv):
        kernel = np.ones((5, 5), np.uint8)
        found = {}
        for c in self.colors:
            mask = None
            for lo, hi in c["hsv"]:
                m = cv2.inRange(hsv, np.array(lo), np.array(hi))
                mask = m if mask is None else (mask | m)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            found[c["name"]] = [ct for ct in cnts
                                if cv2.contourArea(ct) >= self.min_area]
        return found

    def _blob_for_point(self, blobs, px, py):
        for c in self.colors:
            for ct in blobs.get(c["name"], []):
                if cv2.pointPolygonTest(ct, (float(px), float(py)), False) >= 0:
                    return c, ct
        return None, None

    def process(self, frame):
        """Run one frame. Returns dict {w,h,rot,ready,shapes:[...]}.
        Each shape: id, color, shape, draw[bgr], px, py, cell[x,y], angle, z,
        poly[[x,y]...] (simplified outline, in processed-frame pixels)."""
        if self.depth:
            self.depth.submit(frame)
        work = cv2.GaussianBlur(frame, (self.blur, self.blur), 0) \
            if self.blur >= 3 else frame
        hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
        h, w = frame.shape[:2]
        cx0, cy0 = w / 2, h / 2

        markers = detect_markers(gray, self.aruco)
        blobs = self._color_blobs(hsv)

        H, rot, ready = None, 0.0, True
        if self.board_mode == "corners":
            H = compute_homography(markers, self.corner_ids,
                                   self.board_cols, self.board_rows)
            if H is not None:
                self.H_cached = H
            H = self.H_cached
            ready = H is not None
        else:
            rot = self.board_rot.update([m["angle"] for m in markers])

        shapes, seen = [], set()
        if ready:
            for mk in markers:
                if self.board_mode == "corners" and mk["id"] < self.first_shape_id:
                    continue
                color, ct = self._blob_for_point(blobs, mk["cx"], mk["cy"])
                if color is None:
                    continue
                M = cv2.moments(ct)
                if M["m00"] == 0:
                    continue
                pcx, pcy = M["m10"] / M["m00"], M["m01"] / M["m00"]

                if self.board_mode == "corners":
                    bx, by = warp(H, pcx, pcy)
                    ang = homography_angle(H, mk)
                else:
                    bxp, byp = to_board(pcx, pcy, rot, cx0, cy0)
                    bx, by = bxp / self.grid, byp / self.grid
                    ang = (mk["angle"] - rot) % 360

                st = self.states.setdefault(mk["id"], ShapeState())
                st.x = self._snap_cell(bx, st.x)
                st.y = self._snap_cell(by, st.y)
                st.a = self._snap_angle(ang, st.a)
                st.unseen = 0
                sx, sy, sa = int(st.x), int(st.y), int(st.a)
                seen.add(mk["id"])

                z = self.depth.depth_at(cv2.boundingRect(ct)) if self.depth else None
                z_val = -1.0 if z is None else round(z, 3)

                if self.osc:
                    self.osc.send_message("/shape", [mk["id"], sx, sy, z_val, sa,
                                                     color["shape"], color["name"]])

                poly = cv2.approxPolyDP(ct, 0.02 * cv2.arcLength(ct, True), True)
                shapes.append({
                    "id": mk["id"], "color": color["name"], "shape": color["shape"],
                    "draw": list(color["draw"]),
                    "px": int(pcx), "py": int(pcy), "cell": [sx, sy],
                    "angle": sa, "z": z_val,
                    "poly": poly.reshape(-1, 2).astype(int).tolist(),
                })

        # despawn shapes gone too long
        for sid in list(self.states):
            if sid in seen:
                continue
            self.states[sid].unseen += 1
            if self.states[sid].unseen > self.gone_frames:
                if self.osc:
                    self.osc.send_message("/shape_gone", [sid])
                del self.states[sid]

        return {"w": w, "h": h, "rot": round(rot, 1), "ready": ready,
                "shapes": shapes}


def draw_result(frame, result):
    """Draw a process() result onto a BGR frame (desktop preview)."""
    for s in result["shapes"]:
        col = tuple(s["draw"])
        pts = np.array(s["poly"], np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [pts], True, col, 2)
        cv2.circle(frame, (s["px"], s["py"]), 4, col, -1)
        cv2.putText(frame, f"#{s['id']} {s['shape']}/{s['color']} "
                    f"({s['cell'][0]},{s['cell'][1]}) {s['angle']}d z={s['z']}",
                    (s["px"] + 8, s["py"] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
    return frame
