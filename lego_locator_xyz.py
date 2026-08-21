"""
lego_locator_xyz.py - Lego colour locator + metric X, Y, Z + real-world size.

Builds on lego_locator.py (same colour detection, same S/V-floor sliders) and
adds, per detected piece:

  * Z  - metric depth in meters from Depth Anything 3 (background thread), the
         distance from the camera along its optical axis.
  * X, Y - metric position in the CAMERA's frame, in meters, by back-projecting
         the piece's pixel centre (u, v) through the camera intrinsics:
             X = (u - cx) * Z / fx
             Y = (v - cy) * Z / fy
         so (X, Y, Z) = (0,0,0) is the camera, +X right, +Y down, +Z forward.
  * size - real width/height of the piece in cm, from its pixel size scaled by
         depth and focal length:
             w_m = (pixel_w / fx) * Z ,  h_m = (pixel_h / fy) * Z
         This is what "size estimator" means here: not pixels, but centimetres,
         which stay stable as the piece moves nearer/farther (pixels don't).

Intrinsics (fx, fy, cx, cy): taken from DA3 when it provides them
(intrinsics_for_frame). If DA3 isn't loaded or didn't return them, we fall back
to an ASSUMED horizontal field of view (--fov, default 60 deg) derived from the
frame width - X/Y/size then become approximate but still usable. The HUD says
which is in effect ("intr:DA3" vs "intr:FOV60").

Depth degrades soft: no torch / no DA3 / no GPU -> Z shows "?", X/Y/size blank,
colour detection still works. Install per README to enable depth.

Controls (focus the window):
    S/V floor, Min area sliders  - colour detection (as lego_locator.py)
    f  - toggle FLOOR-frame vs CAMERA-frame coordinates
    d  - toggle a colourised depth-map window (red=near, blue=far)
    p  - print every piece: colour/shape, XYZ, size cm, angle
    s  - save current sliders + --fov/--outline-height/--osc-host/
         --osc-port/--settle to lego_locator_config.json, so the next run
         starts from this tuning instead of the hardcoded defaults (see
         locator_config.py) - CLI flags passed explicitly still override
         whatever's in the file
    Esc / q - quit

CONFIG PERSISTENCE: on startup, argparse defaults for --fov/
--outline-height/--osc-host/--osc-port/--settle and the initial slider
positions are seeded from lego_locator_config.json if present (falls back
to the hardcoded defaults otherwise). Press 's' any time to save the
CURRENT sliders + flags back to that file.

Also reports, per piece: SHAPE (square/circle/cross, from contour
geometry), ROTATION (degrees, when an ArUco tag sits on the block), and
FLOOR-frame X/Y/Z - a ground plane fitted from the depth cloud with
RANSAC (surface-agnostic, no floor marker; Z is height above the floor).
Add --osc to stream [name,x_mm,y_mm,angle,w_cm,h_cm,z_mm] to Unreal.

OUTLINE (for a real silhouette instead of a placeholder block in Unreal):
each confirmed piece's contour is simplified (cv2.approxPolyDP) and its
vertices back-projected into the same metric frame as X/Y (floor-frame by
default), then sent as a SEPARATE message, "/outline"
-> [name, n_points, x1_mm, y1_mm, ..., xn_mm, yn_mm, height_cm]. Since the
vertices are already the real, rotated world positions, Unreal doesn't need
to apply the piece's `angle` separately when building the outline mesh - just
extrude the polygon straight up by `height_cm` (a flat, not-to-scale
thickness; see --outline-height). This is NOT a 3D scan - it's the real 2D
footprint extruded flat, which is enough to look like the actual shape
instead of a generic block.

Run:
    python lego_locator_xyz.py            # default camera
    python lego_locator_xyz.py 0 --fov 55
    python lego_locator_xyz.py "http://172.20.10.1:8081/video"
    python lego_locator_xyz.py "clip.mp4"
"""

import argparse
import cmath
import math
import time
from collections import Counter, deque

import cv2
import numpy as np

from lego_locator import COLORS, find_pieces, as_source, classify_shape
from depth_estimator import DepthEstimator
import locator_config

ARUCO_DICT = "DICT_4X4_50"          # matches generate_aruco_tags.py


def make_aruco():
    """ArUco detector that works on old and new OpenCV (5.x dropped legacy)."""
    a = cv2.aruco
    dic = a.getPredefinedDictionary(getattr(a, ARUCO_DICT))
    try:
        params = a.DetectorParameters()
    except AttributeError:
        params = a.DetectorParameters_create()
    try:
        return ("new", a.ArucoDetector(dic, params))
    except AttributeError:
        return ("old", (dic, params))


def detect_markers(gray, state):
    """Return [(center_xy, angle_deg, corners)] for every tag found."""
    mode, obj = state
    if mode == "new":
        corners, ids, _ = obj.detectMarkers(gray)
    else:
        dic, params = obj
        corners, ids, _ = cv2.aruco.detectMarkers(gray, dic, parameters=params)
    out = []
    if ids is None:
        return out
    for c in corners:
        pts = c.reshape(4, 2)                 # TL, TR, BR, BL
        center = pts.mean(axis=0)
        tl, tr = pts[0], pts[1]
        angle = math.degrees(math.atan2(tr[1] - tl[1], tr[0] - tl[0])) % 360.0
        out.append((center, angle, pts))
    return out


class FloorFrame:
    """
    Surface-agnostic ground frame fitted from the depth point cloud with RANSAC
    - no marker on the floor needed, and colour-agnostic (works on the black
    table; depth is geometry, not colour). Once a dominant plane (the floor) is
    found, a block's position is reported IN THAT FRAME: X/Y along the floor,
    Z = height above it. The camera being fixed, the plane is stable frame to
    frame; we refit whenever a new depth map arrives.
    """
    def __init__(self):
        self.ok = False
        self.n = None          # unit normal, pointing toward the camera
        self.p0 = None         # a point on the plane (inlier centroid)
        self.u = self.v = None  # in-plane basis

    def fit(self, depth, fx, fy, cx, cy, frame_wh, step=10,
            iters=120, thresh=0.012):
        H, W = depth.shape
        fw, fh = frame_wh
        ys, xs = np.mgrid[0:H:step, 0:W:step]
        z = depth[ys, xs].astype(np.float32).ravel()
        xs = xs.ravel().astype(np.float32)
        ys = ys.ravel().astype(np.float32)
        good = np.isfinite(z) & (z > 0)
        if good.sum() < 60:
            self.ok = False
            return
        z = z[good]
        # depth-map pixels -> frame pixels, then back-project with frame intrinsics
        u = xs[good] * (fw / W)
        v = ys[good] * (fh / H)
        P = np.stack([(u - cx) * z / fx, (v - cy) * z / fy, z], axis=1)

        rng = np.random.default_rng(0)
        best_inl, best = 0, None
        for _ in range(iters):
            i = rng.choice(len(P), 3, replace=False)
            a, b, c = P[i]
            nrm = np.cross(b - a, c - a)
            ln = np.linalg.norm(nrm)
            if ln < 1e-9:
                continue
            nrm = nrm / ln
            d = -nrm.dot(a)
            dist = np.abs(P.dot(nrm) + d)
            inl = int((dist < thresh).sum())
            if inl > best_inl:
                best_inl, best = inl, (nrm, d)
        if best is None or best_inl < len(P) * 0.3:
            self.ok = False
            return
        nrm, d = best
        # least-squares refit to the inliers for a cleaner plane
        inl = np.abs(P.dot(nrm) + d) < thresh
        Q = P[inl]
        p0 = Q.mean(axis=0)
        _, _, Vt = np.linalg.svd(Q - p0)
        nrm = Vt[2]
        # normal should point toward the camera (origin) so height is +ve up
        if nrm.dot(-p0) < 0:
            nrm = -nrm
        # in-plane basis
        seed = np.array([1.0, 0.0, 0.0])
        if abs(nrm.dot(seed)) > 0.9:
            seed = np.array([0.0, 1.0, 0.0])
        uax = np.cross(nrm, seed); uax /= np.linalg.norm(uax)
        vax = np.cross(nrm, uax)
        self.n, self.p0, self.u, self.v, self.ok = nrm, p0, uax, vax, True

    def to_floor(self, P_cam):
        """Camera 3D point -> (X_floor, Y_floor, height) in meters."""
        rel = P_cam - self.p0
        return (float(self.u.dot(rel)), float(self.v.dot(rel)),
                float(self.n.dot(rel)))


class Tracks:
    """
    Tiny per-colour tracker so a stationary piece reports a STEADY size/depth
    instead of frame-to-frame jitter. Pieces are matched to the nearest slot of
    the same colour within `match_dist` px; the metric quantities (z, size in
    cm) are exponentially smoothed. A real change - stacking a piece taller -
    still comes through, just settled over a few frames rather than flickering.
    Pixel centroid is NOT smoothed (position already looks good, and we want it
    responsive); only z and cm are.
    """
    def __init__(self, alpha=0.25, match_dist=90, max_miss=15,
                 settle_seconds=3.0, size_alpha=0.12):
        self.alpha = alpha
        self.size_alpha = size_alpha            # slower EMA for size (steadier)
        self.match_dist = match_dist
        self.max_miss = max_miss
        self.settle_seconds = settle_seconds    # a new piece must persist this
                                                # long before it's "confirmed"
        self.slots = {}                         # colour name -> list of slots

    def update(self, color, cx, cy, z, w_cm, h_cm, angle=None, shape=None):
        now = time.time()
        lst = self.slots.setdefault(color, [])
        best, best_d = None, 1e9
        for s in lst:
            d = math.hypot(s["cx"] - cx, s["cy"] - cy)
            if d < best_d:
                best, best_d = s, d
        if best is None or best_d > self.match_dist:
            # brand-new piece: start its settle timer, not confirmed yet
            best = {"cx": cx, "cy": cy, "z": z, "w_cm": w_cm, "h_cm": h_cm,
                    "w_samps": deque(maxlen=90), "h_samps": deque(maxlen=90),
                    "size_locked": False,
                    "angle": angle, "_phasor": None, "shapes": deque(maxlen=9),
                    "shape": shape, "miss": 0,
                    "first_seen": now, "confirmed": False}
            lst.append(best)
        best["cx"], best["cy"], best["miss"] = cx, cy, 0
        # confirm once it has been seen continuously for settle_seconds. A piece
        # that leaves for > max_miss frames is dropped by age(), so putting a
        # new piece in restarts the timer from scratch.
        best["settle_left"] = max(0.0, self.settle_seconds
                                  - (now - best["first_seen"]))
        if not best["confirmed"] and best["settle_left"] <= 0.0:
            best["confirmed"] = True
        a = self.alpha
        if z is not None:                       # position/depth keep tracking
            best["z"] = z if best["z"] is None else (1 - a) * best["z"] + a * z
        # SIZE IS MEASURED, THEN LOCKED. A piece is rigid, so we collect size
        # samples during the settle window and freeze the size (median of them)
        # the moment the piece is confirmed. After that it never changes with
        # distance - no oscillation. To re-measure (e.g. after stacking), take
        # the piece out and back in, which restarts the settle.
        if not best["size_locked"]:
            if w_cm is not None:
                best["w_samps"].append(w_cm)
            if h_cm is not None:
                best["h_samps"].append(h_cm)
            if best["confirmed"]:
                if best["w_samps"]:
                    best["w_cm"] = float(np.median(best["w_samps"]))
                if best["h_samps"]:
                    best["h_cm"] = float(np.median(best["h_samps"]))
                best["size_locked"] = True
        if angle is not None:                   # circular EMA (handles 0/360 wrap)
            zc = cmath.exp(1j * math.radians(angle))
            best["_phasor"] = zc if best["_phasor"] is None \
                else a * zc + (1 - a) * best["_phasor"]
            best["angle"] = math.degrees(cmath.phase(best["_phasor"])) % 360.0
        if shape and shape != "?":              # majority vote over recent frames
            best["shapes"].append(shape)
            best["shape"] = Counter(best["shapes"]).most_common(1)[0][0]
        return best

    def age(self):
        """Drop slots that haven't been matched for a while."""
        for color, lst in list(self.slots.items()):
            for s in lst:
                s["miss"] += 1
            self.slots[color] = [s for s in lst if s["miss"] <= self.max_miss]


def resolve_intrinsics(depth, frame_w, frame_h, assumed_fov_deg):
    """(fx, fy, cx, cy), source_label. Prefer DA3's; else assume an FOV."""
    if depth.available:
        intr = depth.intrinsics_for_frame()
        if intr is not None:
            return intr, "DA3"
    # Pinhole from an assumed horizontal FOV: fx = (W/2) / tan(FOV/2).
    f = (frame_w / 2.0) / math.tan(math.radians(assumed_fov_deg / 2.0))
    return (f, f, frame_w / 2.0, frame_h / 2.0), f"FOV{assumed_fov_deg:g}"


def backproject(u, v, z, fx, fy, cx, cy):
    """Pixel (u, v) at depth z (meters) -> camera-frame 3D point (meters).
    (0,0,0) is the camera, +X right, +Y down, +Z forward."""
    return np.array([(u - cx) * z / fx, (v - cy) * z / fy, z])


def to_world_xy(P_cam, floor, use_floor):
    """Camera-frame point -> (X, Y) in whichever frame is active (meters).
    FLOOR frame when requested and the plane fit succeeded, else CAMERA
    frame (X, Y straight from back-projection, no floor-relative height)."""
    if use_floor and floor.ok:
        X, Y, _ = floor.to_floor(P_cam)
        return X, Y
    return float(P_cam[0]), float(P_cam[1])


def build_outline_mm(contour, cx, cy, fx, fy, z, floor, use_floor,
                      max_points=16):
    """Simplify a piece's contour (cv2.approxPolyDP) and back-project each
    vertex into world space at a single shared depth z (the piece is rigid
    and roughly flat, so one depth for the whole polygon is a fair
    approximation). Returns a flat [x1_mm, y1_mm, x2_mm, y2_mm, ...] list -
    winding order is preserved from the input contour (OpenCV contours from
    findContours are clockwise in image space); no separate angle is needed
    since each vertex already carries the piece's true rotation.

    A degenerate simplification (< 3 points - e.g. a sliver contour that
    approxPolyDP collapses to a line or point) falls back to the contour's
    min-area rect corners, which is always 4 points, so callers never have
    to special-case "too few points to be a polygon" - a Geometry Script
    polygon extrude on the Unreal side needs >= 3 vertices to mean anything.
    Returns (poly, pts_mm) with len(poly) always >= 3 for a non-empty input
    contour."""
    eps = 0.02 * cv2.arcLength(contour, True)
    poly = cv2.approxPolyDP(contour, eps, True).reshape(-1, 2)
    if len(poly) < 3:
        poly = cv2.boxPoints(cv2.minAreaRect(contour))
    if len(poly) > max_points:            # keep the OSC message bounded
        idx = np.linspace(0, len(poly) - 1, max_points).astype(int)
        poly = poly[idx]
    pts_mm = []
    for px, py in poly:
        P_cam = backproject(px, py, z, fx, fy, cx, cy)
        Xv, Yv = to_world_xy(P_cam, floor, use_floor)
        pts_mm.extend([Xv * 1000.0, Yv * 1000.0])
    return poly, pts_mm


def main():
    saved_cfg = locator_config.load()          # falls back to hardcoded defaults
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", nargs="?", default="0",
                    help="camera index, stream URL, or video file (default 0)")
    ap.add_argument("--fov", type=float, default=saved_cfg["fov"],
                    help="assumed horizontal FOV (deg) if DA3 gives no intrinsics")
    ap.add_argument("--model", default=None,
                    help="override DA3 model id (e.g. depth-anything/DA3-SMALL)")
    ap.add_argument("--osc", action="store_true",
                    help="send [name,x,y,angle,sx,sy,z] to Unreal on /obj")
    ap.add_argument("--osc-host", default=saved_cfg["osc_host"])
    ap.add_argument("--osc-port", type=int, default=saved_cfg["osc_port"])
    ap.add_argument("--no-floor", action="store_true",
                    help="report camera-frame XYZ instead of floor-frame")
    ap.add_argument("--settle", type=float, default=saved_cfg["settle"],
                    help="seconds a new piece must persist before it's confirmed "
                         "and reported/sent (default 3; 0 = instant)")
    ap.add_argument("--debug-size", action="store_true",
                    help="log raw pixel size and depth per piece, to check "
                         "whether pixel*depth (the real size) stays constant")
    ap.add_argument("--outline-height", type=float, default=saved_cfg["outline_height"],
                    help="fixed extrusion height in cm sent with each piece's "
                         "outline polygon on /outline (default 2.0)")
    args = ap.parse_args()

    cap = cv2.VideoCapture(as_source(args.source))
    if not cap.isOpened():
        print(f"FAILED to open {args.source!r}")
        return 1

    osc = None
    if args.osc:
        from pythonosc.udp_client import SimpleUDPClient
        osc = SimpleUDPClient(args.osc_host, args.osc_port)
        print(f"[osc] sending /obj -> {args.osc_host}:{args.osc_port}")

    aruco = make_aruco()
    floor = FloorFrame()
    use_floor = not args.no_floor

    print("[depth] initialising Depth Anything 3 (first run downloads the "
          "model; this can take a minute)...")
    depth = DepthEstimator(model_id=args.model) if args.model \
        else DepthEstimator()
    depth.start()
    if not depth.available:
        print("[depth] running WITHOUT depth (Z=?, X/Y/size approximate or "
              "blank). See README to install DA3.")

    win = "Lego locator xyz"
    cv2.namedWindow(win)
    cv2.createTrackbar("S floor", win, saved_cfg["s_floor"], 255, lambda v: None)
    cv2.createTrackbar("V floor", win, saved_cfg["v_floor"], 255, lambda v: None)
    cv2.createTrackbar("Min area/100", win, saved_cfg["min_area_100"], 100,
                       lambda v: None)

    t_prev, fps = time.time(), 0.0
    tracks = Tracks(settle_seconds=args.settle)
    show_depth = False

    while True:
        ok, frame = cap.read()
        if not ok:
            print("no more frames"); break
        depth.submit(frame)                       # hand newest frame to worker

        H, W = frame.shape[:2]
        (fx, fy, cx, cy), intr_src = resolve_intrinsics(depth, W, H, args.fov)

        s_floor = cv2.getTrackbarPos("S floor", win)
        v_floor = cv2.getTrackbarPos("V floor", win)
        min_area = cv2.getTrackbarPos("Min area/100", win) * 100

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        markers = detect_markers(gray, aruco)     # [(center, angle, corners)]
        for center, angle, pts in markers:
            cv2.polylines(frame, [pts.astype(int)], True, (255, 0, 255), 2)

        # Refit the floor plane from the current depth map (cheap, subsampled).
        dm = depth.latest_depth_map()
        if use_floor and dm is not None:
            floor.fit(dm, fx, fy, cx, cy, (W, H))

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        pieces, _ = find_pieces(hsv, s_floor, v_floor, min_area)

        report = []
        for c in COLORS:
            for ct in pieces[c["name"]]:
                x, y, w, h = cv2.boundingRect(ct)   # still used to anchor text
                (u, v), (rw, rh), _ = cv2.minAreaRect(ct)

                shape = classify_shape(ct)
                # A tag whose centre falls inside this blob gives its rotation.
                angle = None
                for center, ang, _pts in markers:
                    if cv2.pointPolygonTest(ct, (float(center[0]),
                                                 float(center[1])), False) >= 0:
                        angle = ang
                        break

                # Depth sampled ONLY inside the piece (eroded a few px so the
                # coarse depth map doesn't pick up background at the edges).
                mask = np.zeros((H, W), np.uint8)
                cv2.drawContours(mask, [ct], -1, 255, -1)
                mask = cv2.erode(mask, np.ones((7, 7), np.uint8))
                z = depth.depth_in_mask(mask)
                if z is None:                     # mask too small after erode
                    z = depth.depth_in_mask(
                        cv2.drawContours(np.zeros((H, W), np.uint8),
                                         [ct], -1, 255, -1))

                if z is not None:
                    w_cm_raw = (rw / fx) * z * 100.0
                    h_cm_raw = (rh / fy) * z * 100.0
                    slot = tracks.update(c["name"], u, v, z, w_cm_raw, h_cm_raw,
                                         angle, shape)
                else:
                    slot = tracks.update(c["name"], u, v, None, None, None,
                                         angle, shape)

                # While a new piece is settling, draw it dim + show a countdown;
                # only a CONFIRMED piece (held 3s) is reported and sent to Unreal.
                if not slot["confirmed"]:
                    cv2.drawContours(frame, [ct], -1, (150, 150, 150), 1)
                    cv2.putText(frame, f"scanning {slot['settle_left']:.1f}s",
                                (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (150, 150, 150), 2)
                    continue

                cv2.drawContours(frame, [ct], -1, c["draw"], 2)   # real outline
                if z is not None:
                    zs, w_cm, h_cm = slot["z"], slot["w_cm"], slot["h_cm"]
                    P_cam = backproject(u, v, zs, fx, fy, cx, cy)
                    if use_floor and floor.ok:
                        X, Y, Zf = floor.to_floor(P_cam)   # floor frame, height
                        frame_tag = "flr"
                    else:
                        X, Y, Zf = P_cam[0], P_cam[1], zs  # camera frame
                        frame_tag = "cam"
                    ang_txt = f" {slot['angle']:.0f}deg" if slot["angle"] \
                        is not None else ""
                    l1 = f"{c['name']}/{slot['shape']} ({X:+.2f},{Y:+.2f},{Zf:+.2f}){frame_tag}"
                    l2 = f"{w_cm:.1f}x{h_cm:.1f}cm  d={zs:.2f}m {int(rw)}px{ang_txt}"
                    if args.debug_size:
                        # pixel*depth should be constant for a rigid piece; if it
                        # drifts as the piece moves, the depth is the culprit.
                        print(f"[size] {c['name']:5} px={rw:5.1f} z={zs:.3f}m "
                              f"px*z={rw*zs:6.1f} -> {w_cm:.2f}cm")
                    report.append((c["name"], slot["shape"], X, Y, Zf,
                                   w_cm, h_cm, slot["angle"]))
                    if osc is not None:
                        a_out = 0.0 if slot["angle"] is None else slot["angle"]
                        # meters -> mm for Unreal; keep the /obj layout
                        osc.send_message("/obj", [c["name"], X * 1000.0,
                                         Y * 1000.0, a_out, w_cm, h_cm,
                                         Zf * 1000.0])

                        # Outline: simplify the real contour and back-project
                        # each vertex through the same depth as the piece centre
                        # (it's rigid and roughly flat, so one depth for the
                        # whole polygon is a fair approximation). Vertices come
                        # out already in world/floor space, so they carry the
                        # piece's real rotation - Unreal just extrudes the
                        # polygon, no separate angle needed.
                        poly, pts_mm = build_outline_mm(
                            ct, cx, cy, fx, fy, zs, floor, use_floor)
                        if len(poly) >= 3:     # never send a degenerate polygon
                            osc.send_message("/outline",
                                             [c["name"], len(poly)] + pts_mm
                                             + [args.outline_height])
                else:
                    l1 = f"{c['name']}/{slot['shape']} z=? {int(rw)}x{int(rh)}px"
                    l2 = ""
                cv2.putText(frame, l1, (x, y - 24), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, c["draw"], 2)
                if l2:
                    cv2.putText(frame, l2, (x, y - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, c["draw"], 2)
        tracks.age()

        # HUD
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-6))
        t_prev = now
        if depth.available:
            dstat = "DEPTH on" if depth.has_depth() else "DEPTH loading..."
        else:
            dstat = "DEPTH off"
        frame_state = ("FLOOR" if (use_floor and floor.ok)
                       else "floor?" if use_floor else "CAM")
        cv2.putText(frame, f"{dstat}  frame:{frame_state}  intr:{intr_src}  "
                    f"tags:{len(markers)}  {fps:.0f}fps",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow(win, frame)

        if show_depth:
            if dm is not None:
                # Nearer = brighter: invert before normalising so it reads
                # intuitively (JET's blue=far/cold, red=near/hot).
                valid = np.isfinite(dm)
                if valid.any():
                    lo, hi = dm[valid].min(), dm[valid].max()
                    norm = np.zeros_like(dm, dtype=np.uint8)
                    if hi > lo:
                        inv = 1.0 - (dm - lo) / (hi - lo)
                        norm = np.clip(inv * 255, 0, 255).astype(np.uint8)
                    depth_vis = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
                    depth_vis[~valid] = (0, 0, 0)
                    depth_vis = cv2.resize(depth_vis, (W, H),
                                            interpolation=cv2.INTER_NEAREST)
                    cv2.putText(depth_vis, f"depth map  {lo:.2f}-{hi:.2f}m "
                                "(red=near, blue=far)", (10, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                    cv2.imshow("depth", depth_vis)
            else:
                blank = np.zeros((H, W, 3), np.uint8)
                cv2.putText(blank, "depth loading / unavailable...", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
                cv2.imshow("depth", blank)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break
        elif key == ord('f'):
            use_floor = not use_floor
        elif key == ord('d'):
            show_depth = not show_depth
            if not show_depth:
                cv2.destroyWindow("depth")
        elif key == ord('p'):
            if report:
                print(" | ".join(
                    f"{n}/{sh}: ({X:+.2f},{Y:+.2f},{Z:+.2f}) "
                    f"{w:.1f}x{h:.1f}cm ang={ang}"
                    for (n, sh, X, Y, Z, w, h, ang) in report))
            else:
                print("no pieces with depth yet")
        elif key == ord('s'):
            saved = locator_config.save({
                "s_floor": s_floor, "v_floor": v_floor,
                "min_area_100": min_area // 100,
                "fov": args.fov, "outline_height": args.outline_height,
                "osc_host": args.osc_host, "osc_port": args.osc_port,
                "settle": args.settle,
            })
            print(f"[config] saved to {locator_config.DEFAULT_PATH}: {saved}")

    depth.stop()
    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
