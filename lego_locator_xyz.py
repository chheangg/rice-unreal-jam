"""
lego_locator_xyz.py - Lego colour locator + metric X, Y, Z + FIXED real size.

Builds on lego_locator.py (same colour detection, same S/V-floor sliders) and
adds, per detected piece:

  * size - the piece's real LONG x SHORT side in cm. Measured ONCE, at the
         start, then LOCKED and remembered. It does not change when the piece
         moves nearer or farther from the camera, and it does not change when
         the piece is rotated. See "Why the size used to breathe" below.
  * Z  - metric depth in meters (distance from the camera along its optical
         axis). Once a piece's size is locked, Z is derived FROM that size
         (z = size_m * f / size_px), which is far steadier than the raw
         monocular depth map; Depth Anything 3 is used to bootstrap the very
         first measurement and to fit the floor plane.
  * X, Y - metric position in the CAMERA's frame, by back-projecting the
         piece's pixel centre (u, v):  X = (u-cx)*Z/fx,  Y = (v-cy)*Z/fy.
         Optionally re-expressed in a FLOOR frame (see below).

Why the size used to breathe, and what fixes it
-----------------------------------------------
Real size from a single camera is  size_cm = (size_px / f) * depth. Every
wobble in the depth estimate therefore shows up directly as a wobble in the
reported centimetres, and a monocular depth map wobbles a lot - most of all
when the piece moves toward or away from the camera, exactly the case the
number is supposed to be immune to. Three changes make the size constant:

  1. LONG/SHORT, not W/H. cv2.minAreaRect labels its sides by the rectangle's
     own orientation, so "width" and "height" SWAP as a piece rotates past 45
     deg. We normalise to long side / short side, which is rotation-invariant,
     and use one focal length f=(fx+fy)/2 for both so the answer doesn't depend
     on which way the piece is lying.
  2. MEASURE ONCE, THEN LOCK. During a short settle window at the start we
     collect samples and take their median; on confirmation the size is frozen.
     A rigid brick's size is a constant, so after that we simply stop
     re-estimating it - distance to the camera can no longer move the number.
  3. REMEMBER IT (the "size book"). The lock is keyed by colour+shape, not by
     the tracker slot, so if the piece is picked up, moved across the table, or
     briefly lost, the same brick comes back with the SAME numbers instead of
     being re-measured at its new distance. --size-file persists the book
     across runs, so the very first measurement can be the only one you ever
     make. Each colour+shape holds a LIST of known sizes, matched by the
     piece's own rough measurement, so two yellow rectangles of different
     sizes stay two different bricks instead of one adopting the other's
     centimetres (and, through them, the other's depth).

Several pieces at once
----------------------
Every piece the camera can see gets its own track and its own stable name
(`yellow1`, `yellow2`, `red1`, ...), including several of the SAME colour. The
matching is done a whole frame at a time - all blobs of a colour are scored
against the tracks' previous positions, then assigned nearest-first with each
track claimable once. Matching them one blob at a time instead lets two nearby
bricks claim the same track (and the first claim drags that track onto the
first brick, so the second claims it too), which collapses them into a single
piece that reports one position, one size and one depth. See Tracks for the
long version.

Unreal is the narrower end: BP_OSCreciver has four pre-placed cubes and no
spawn path, so it can show three yellows and one red. The bridge leases those
cubes to tracked pieces and says on stdout what it had to drop.

Depth then rides on the size, not the other way round: for a locked piece,
z = (long_cm/100) * f / long_px. That inverts cleanly, is smooth frame to
frame, and is self-consistent with the reported size by construction.

Absolute scale
--------------
DA3METRIC returns meters, but any monocular rig can be off by a constant
factor. Measure one piece with a ruler, pass its true long side as
--calib-cm 3.2, and press 'c' with that piece confirmed on screen: everything
(depths, sizes, the size book) is rescaled so that piece reads correctly, and
the correction sticks for the session.

Intrinsics (fx, fy, cx, cy): from DA3 when it provides them; otherwise from an
assumed horizontal field of view (--fov, default 60 deg). The HUD says which is
in effect ("intr:DA3" vs "intr:FOV60").

Depth degrades soft: no torch / no DA3 / no GPU -> Z shows "?", sizes stay
blank until depth arrives (or come straight from --size-file), colour detection
still works.

Controls (focus the window):
    S/V floor, Min area sliders  - colour detection (as lego_locator.py)
    f  - toggle FLOOR-frame vs CAMERA-frame coordinates
    d  - toggle depth source: size-locked (steady) vs raw DA3
    r  - forget all locked sizes and re-measure everything
    c  - calibrate absolute scale from --calib-cm against the biggest piece
    p  - print every piece: colour/shape, XYZ, size cm, angle
    Esc / q - quit

Also reports, per piece: SHAPE (square/rectangle/circle/cross, from contour
geometry), ROTATION (degrees, when an ArUco tag sits on the block), and
FLOOR-frame X/Y/Z - a ground plane fitted from the depth cloud with RANSAC
(surface-agnostic, no floor marker; Z is height above the floor).
Add --osc to stream [name,x_mm,y_mm,angle,long_cm,short_cm,z_mm] to Unreal.

Run:
    python lego_locator_xyz.py            # default camera
    python lego_locator_xyz.py 0 --fov 55
    python lego_locator_xyz.py 0 --size-file sizes.json --snap-mm 8
    python lego_locator_xyz.py "http://172.20.10.1:8081/video"
"""

import argparse
import cmath
import json
import math
import os
import time
from collections import Counter, deque

import cv2
import numpy as np

from lego_locator import (COLORS, find_pieces, as_source, classify_shape,
                          DEFAULT_S_FLOOR, DEFAULT_V_FLOOR, DEFAULT_MIN_AREA_100)
from depth_estimator import DepthEstimator

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


class SizeBook:
    """
    The measured-once record of how big each KIND of piece really is, in cm,
    keyed by (colour, shape) and stored as (long_side, short_side).

    This is what makes the size independent of depth. A tracker slot is a
    short-lived thing - it dies the moment a hand covers the piece, or the
    piece is picked up and put down somewhere else - and if size lived only on
    the slot, every such event would re-measure the piece at whatever distance
    it now sits at, which is precisely the "size changes when I move it closer"
    problem. The book outlives slots: a red square measured once is that size
    for the rest of the session (and, with --size-file, the next session too).

    snap_mm quantises each side to a physical grid before storing - Lego studs
    are on an 8 mm pitch, so --snap-mm 8 turns "3.1 x 1.6 cm" into a clean
    "3.2 x 1.6 cm" and kills the last of the measurement noise.
    """
    def __init__(self, snap_mm=0.0, path=None, tol_cm=0.6):
        self.snap_mm = float(snap_mm)
        self.tol_cm = float(tol_cm)
        self.path = path
        # "colour/shape" -> [[long_cm, short_cm], ...]. A LIST, because two
        # yellow rectangles on the table can be genuinely different bricks:
        # keying one size per colour+shape made the second piece adopt the
        # first one's centimetres, which is wrong and also poisons its depth
        # (z is derived from the locked size).
        self.book = {}
        if path and os.path.exists(path):
            try:
                with open(path) as fh:
                    self.book = {k: self._as_entries(v)
                                 for k, v in json.load(fh).items()}
                print(f"[size] loaded {self.count()} locked sizes "
                      f"({len(self.book)} kinds) from {path}")
            except Exception as e:
                print(f"[size] could not read {path}: {e}")

    @staticmethod
    def _as_entries(v):
        """Accept both file layouts: [long, short] (old, one size per kind)
        and [[long, short], ...] (current). Old size files keep working."""
        if v and isinstance(v[0], (int, float)):
            return [[float(v[0]), float(v[1])]]
        return [[float(a), float(b)] for a, b in v]

    @staticmethod
    def _key(color, shape):
        return f"{color}/{shape or '?'}"

    def count(self):
        return sum(len(v) for v in self.book.values())

    def _snap(self, cm):
        if self.snap_mm <= 0:
            return cm
        step = self.snap_mm / 10.0                     # mm -> cm
        return max(step, round(cm / step) * step)

    def get(self, color, shape, hint_long_cm=None):
        """Locked (long_cm, short_cm) for this piece, or None.

        hint_long_cm is the piece's own rough measurement, used to pick WHICH
        of several same-colour/same-shape bricks this one is. Without a hint we
        can only answer when there is exactly one candidate - otherwise we'd be
        guessing, and guessing here is what made two different bricks read as
        the same size.
        """
        if not shape or shape == "?":                  # don't key off an unknown
            return None
        entries = self.book.get(self._key(color, shape))
        if not entries:
            return None
        if hint_long_cm is None:
            return tuple(entries[0]) if len(entries) == 1 else None
        best = min(entries, key=lambda e: abs(e[0] - hint_long_cm))
        if abs(best[0] - hint_long_cm) > self.tol_cm:
            return None                                # a brick we've not met
        return tuple(best)

    def put(self, color, shape, long_cm, short_cm):
        if not shape or shape == "?":
            return None
        lo, sh = self._snap(long_cm), self._snap(short_cm)
        if sh > lo:
            lo, sh = sh, lo
        key = self._key(color, shape)
        entries = self.book.setdefault(key, [])
        for e in entries:
            if abs(e[0] - lo) <= self.tol_cm:
                return tuple(e)                        # already know this brick
        entries.append([lo, sh])
        entries.sort()
        self.save()
        print(f"[size] locked {key} = {lo:.2f} x {sh:.2f} cm "
              f"({len(entries)} distinct size{'s' if len(entries) > 1 else ''} "
              f"for this kind)")
        return (lo, sh)

    def rescale(self, k):
        """Apply an absolute-scale correction to everything already locked."""
        for entries in self.book.values():
            for e in entries:
                e[0] *= k
                e[1] *= k
        self.save()

    def clear(self):
        self.book.clear()
        self.save()

    def save(self):
        if not self.path:
            return
        try:
            with open(self.path, "w") as fh:
                json.dump(self.book, fh, indent=1)
        except Exception as e:
            print(f"[size] could not write {self.path}: {e}")


class Tracks:
    """
    Per-colour multi-object tracker. Every piece of a colour visible in a frame
    gets its OWN slot, with a stable name (`yellow1`, `yellow2`, ...) that
    follows that physical brick for as long as it stays in view.

    Why matching is done a frame at a time
    --------------------------------------
    The obvious version - "for each blob, grab the nearest slot" - quietly
    collapses two pieces into one. Two yellow bricks 60 px apart are both
    inside the match radius of the same slot, so both claim it; worse, the
    first claim MOVES that slot onto the first brick, so the second brick is
    then measured against the already-moved slot and claims it again. The
    result is one slot reporting the last brick's position, with both bricks
    sharing one size and one depth. That is what "it only tracks one object at
    a time" actually was.

    So: gather every blob of a colour first, score all (blob, slot) pairs
    against the slots' PREVIOUS positions, then hand them out nearest-first
    with each slot claimable ONCE. Blobs left over are new pieces. This is
    plain greedy assignment - not optimal the way Hungarian would be, but for a
    handful of separated bricks the difference never shows, and it costs
    nothing.

    The match radius also scales with the piece: a brick can only have moved so
    far between two frames, and a flat 90 px for a 40 px brick is part of what
    let a neighbour steal its slot.

    Position is NOT smoothed (it already looks good and we want it responsive);
    depth is lightly smoothed; SIZE is not smoothed at all - it is measured
    during the settle window, then frozen and handed to the SizeBook.
    """
    def __init__(self, book, alpha=0.25, match_dist=90, max_miss=15,
                 settle_seconds=3.0):
        self.book = book
        self.alpha = alpha
        self.match_dist = match_dist
        self.max_miss = max_miss
        self.settle_seconds = settle_seconds    # a new piece must persist this
                                                # long before it's "confirmed"
        self.slots = {}                         # colour name -> list of slots
        self._next_id = {}                      # colour name -> next id number

    def _gate(self, long_px):
        """How far a piece may have travelled since the last frame, in px.

        Capped by the piece's own size: a small brick that appears to have
        jumped 90 px is far more likely to be a DIFFERENT brick than the same
        one moving fast.
        """
        if not long_px or long_px <= 0:
            return float(self.match_dist)
        return max(25.0, min(float(self.match_dist), 1.2 * long_px))

    def update_frame(self, color, dets):
        """Match every detection of one colour to a slot, one slot per piece.

        dets: [{cx, cy, z, long_cm, short_cm, long_px, angle, shape}]
        Returns the matching slot for each det, in the same order.
        """
        now = time.time()
        lst = self.slots.setdefault(color, [])

        # Score against the slots' previous positions - nothing moves until
        # every pair has been scored, so an early claim cannot drag a slot onto
        # its neighbour and swallow that one too.
        pairs = []
        for i, d in enumerate(dets):
            gate = self._gate(d.get("long_px"))
            for j, s in enumerate(lst):
                dist = math.hypot(s["cx"] - d["cx"], s["cy"] - d["cy"])
                if dist <= gate:
                    pairs.append((dist, i, j))
        pairs.sort()

        taken_det, taken_slot, matched = set(), set(), {}
        for _dist, i, j in pairs:
            if i in taken_det or j in taken_slot:
                continue
            taken_det.add(i)
            taken_slot.add(j)
            matched[i] = lst[j]

        out = []
        for i, d in enumerate(dets):
            slot = matched.get(i)
            if slot is None:
                slot = self._new_slot(color, d, now)
                lst.append(slot)
            self._apply(slot, d, now)
            out.append(slot)
        return out

    def _new_slot(self, color, d, now):
        """A piece we have not seen before, with its own stable name."""
        n = self._next_id.get(color, 0) + 1
        self._next_id[color] = n
        return {"name": f"{color}{n}", "color": color, "id": n,
                "cx": d["cx"], "cy": d["cy"], "z": d["z"],
                "long_cm": None, "short_cm": None,
                "long_samps": deque(maxlen=90), "short_samps": deque(maxlen=90),
                "size_locked": False,
                "angle": d.get("angle"), "_phasor": None,
                "shapes": deque(maxlen=9), "shape": d.get("shape"),
                "miss": 0, "first_seen": now, "confirmed": False,
                "settle_left": self.settle_seconds}

    def _apply(self, slot, d, now):
        """Fold one detection into the slot it was matched to."""
        z, angle, shape = d["z"], d.get("angle"), d.get("shape")
        long_cm, short_cm = d.get("long_cm"), d.get("short_cm")
        slot["cx"], slot["cy"], slot["miss"] = d["cx"], d["cy"], 0

        if shape and shape != "?":              # majority vote over recent frames
            slot["shapes"].append(shape)
            slot["shape"] = Counter(slot["shapes"]).most_common(1)[0][0]

        # confirm once it has been seen continuously for settle_seconds. A piece
        # that leaves for > max_miss frames is dropped by age(), so putting a
        # new piece in restarts the timer from scratch.
        slot["settle_left"] = max(0.0, self.settle_seconds
                                  - (now - slot["first_seen"]))
        if not slot["confirmed"] and slot["settle_left"] <= 0.0:
            slot["confirmed"] = True

        a = self.alpha
        if z is not None:                       # depth keeps tracking
            slot["z"] = z if slot["z"] is None else (1 - a) * slot["z"] + a * z

        # --- size: adopt from the book, else measure once and lock ----------
        if not slot["size_locked"]:
            if long_cm is not None:
                slot["long_samps"].append(long_cm)
                slot["short_samps"].append(short_cm)
            # The piece's own rough measurement says WHICH brick of this
            # colour+shape it is, so two different-sized yellow rectangles no
            # longer adopt each other's centimetres.
            hint = float(np.median(slot["long_samps"])) \
                if slot["long_samps"] else None
            known = self.book.get(slot["color"], slot["shape"], hint)
            if known is not None:
                # We have measured this brick before. Take that number
                # verbatim - no re-measuring at the new distance, which is the
                # whole point: the size is a property of the brick, not of
                # where it happens to be sitting right now.
                slot["long_cm"], slot["short_cm"] = known
                slot["size_locked"] = True
                slot["confirmed"] = True        # nothing left to settle for
                slot["settle_left"] = 0.0
            elif slot["confirmed"] and slot["long_samps"]:
                # median over the settle window: robust to the frames where
                # the mask fragmented or the depth map hiccupped
                lo = float(np.median(slot["long_samps"]))
                sh = float(np.median(slot["short_samps"]))
                locked = self.book.put(slot["color"], slot["shape"], lo, sh)
                slot["long_cm"], slot["short_cm"] = locked or (lo, sh)
                slot["size_locked"] = True

        if angle is not None:                   # circular EMA (handles 0/360 wrap)
            zc = cmath.exp(1j * math.radians(angle))
            slot["_phasor"] = zc if slot["_phasor"] is None \
                else a * zc + (1 - a) * slot["_phasor"]
            slot["angle"] = math.degrees(cmath.phase(slot["_phasor"])) % 360.0
        return slot

    def age(self):
        """Drop slots that haven't been matched for a while."""
        for color, lst in list(self.slots.items()):
            for s in lst:
                s["miss"] += 1
            self.slots[color] = [s for s in lst if s["miss"] <= self.max_miss]

    def forget_sizes(self):
        """Drop every lock so the next few seconds re-measure from scratch."""
        self.book.clear()
        for lst in self.slots.values():
            for s in lst:
                s["size_locked"] = False
                s["long_cm"] = s["short_cm"] = None
                s["long_samps"].clear()
                s["short_samps"].clear()
                s["first_seen"] = time.time()
                s["confirmed"] = False


def resolve_intrinsics(depth, frame_w, frame_h, assumed_fov_deg):
    """(fx, fy, cx, cy), source_label. Prefer DA3's; else assume an FOV."""
    if depth.available:
        intr = depth.intrinsics_for_frame()
        if intr is not None:
            return intr, "DA3"
    # Pinhole from an assumed horizontal FOV: fx = (W/2) / tan(FOV/2).
    f = (frame_w / 2.0) / math.tan(math.radians(assumed_fov_deg / 2.0))
    return (f, f, frame_w / 2.0, frame_h / 2.0), f"FOV{assumed_fov_deg:g}"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", nargs="?", default="0",
                    help="camera index, stream URL, or video file (default 0)")
    ap.add_argument("--fov", type=float, default=60.0,
                    help="assumed horizontal FOV (deg) if DA3 gives no intrinsics")
    ap.add_argument("--model", default=None,
                    help="override DA3 model id (e.g. depth-anything/DA3-SMALL)")
    ap.add_argument("--osc", action="store_true",
                    help="stream to Unreal on /obj in the layout BP_OSCreciver "
                         "actually parses: [name,cx,cy,angle,sizeX,sizeY] as "
                         "INTEGER pixels, named yel1/yel2/yel3/red")
    ap.add_argument("--osc-metric", action="store_true",
                    help="instead of the blueprint-compatible message, send the "
                         "rich float one [name,x_mm,y_mm,angle,long_cm,short_cm,"
                         "z_mm]. The CURRENT blueprint cannot read this (it "
                         "pulls integers and switches on yel1/yel2/yel3/red) - "
                         "only use it once the receiver has been updated.")
    ap.add_argument("--osc-verbose", action="store_true",
                    help="print every OSC message as it goes out, to prove the "
                         "tracker is sending before blaming the Unreal side")
    ap.add_argument("--osc-host", default="127.0.0.1")
    ap.add_argument("--osc-port", type=int, default=7000)
    ap.add_argument("--no-floor", action="store_true",
                    help="report camera-frame XYZ instead of floor-frame")
    ap.add_argument("--settle", type=float, default=3.0,
                    help="seconds a NEW kind of piece must persist before its "
                         "size is measured and locked (default 3; 0 = instant)")
    ap.add_argument("--size-file", default=None,
                    help="JSON file to persist locked sizes across runs, so a "
                         "piece keeps the size it was measured at last time")
    ap.add_argument("--snap-mm", type=float, default=0.0,
                    help="quantise locked sizes to this grid in mm (8 = Lego "
                         "stud pitch); 0 = off")
    ap.add_argument("--calib-cm", type=float, default=0.0,
                    help="true long side, in cm, of the piece you'll press 'c' "
                         "on, to fix absolute scale for the whole rig")
    ap.add_argument("--raw-depth", action="store_true",
                    help="report the raw DA3 depth instead of the steady depth "
                         "derived from each piece's locked size ('d' toggles)")
    ap.add_argument("--debug-size", action="store_true",
                    help="log raw pixel size and depth per piece, to check "
                         "whether pixel*depth (the raw size) drifts while the "
                         "locked size stays put")
    args = ap.parse_args()

    cap = cv2.VideoCapture(as_source(args.source))
    if not cap.isOpened():
        print(f"FAILED to open {args.source!r}")
        return 1

    osc = bridge = None
    if args.osc or args.osc_metric:
        if args.osc_metric:
            from pythonosc.udp_client import SimpleUDPClient
            osc = SimpleUDPClient(args.osc_host, args.osc_port)
            print(f"[osc] METRIC /obj -> {args.osc_host}:{args.osc_port} "
                  f"(floats; the current BP_OSCreciver will ignore these)")
        else:
            from unreal_bridge import UnrealBridge
            bridge = UnrealBridge(args.osc_host, args.osc_port,
                                  verbose=args.osc_verbose)

    aruco = make_aruco()
    floor = FloorFrame()
    use_floor = not args.no_floor
    size_depth = not args.raw_depth

    print("[depth] initialising Depth Anything 3 (first run downloads the "
          "model; this can take a minute)...")
    depth = DepthEstimator(model_id=args.model) if args.model \
        else DepthEstimator()
    depth.start()
    if not depth.available:
        print("[depth] running WITHOUT depth (Z=?, sizes blank until a depth "
              "map arrives, unless --size-file has them). See README.")

    win = "Lego locator xyz"
    cv2.namedWindow(win)
    cv2.createTrackbar("S floor", win, DEFAULT_S_FLOOR, 255, lambda v: None)
    cv2.createTrackbar("V floor", win, DEFAULT_V_FLOOR, 255, lambda v: None)
    cv2.createTrackbar("Min area/100", win, DEFAULT_MIN_AREA_100, 100, lambda v: None)

    t_prev, fps = time.time(), 0.0
    book = SizeBook(snap_mm=args.snap_mm, path=args.size_file)
    tracks = Tracks(book, settle_seconds=args.settle)

    while True:
        ok, frame = cap.read()
        if not ok:
            print("no more frames"); break
        depth.submit(frame)                       # hand newest frame to worker

        H, W = frame.shape[:2]
        (fx, fy, cx, cy), intr_src = resolve_intrinsics(depth, W, H, args.fov)
        # ONE focal length for size, so a piece doesn't measure differently
        # lying on its side than standing up (fx and fy differ slightly).
        f_size = 0.5 * (fx + fy)

        s_floor = cv2.getTrackbarPos("S floor", win)
        v_floor = cv2.getTrackbarPos("V floor", win)
        min_area = cv2.getTrackbarPos("Min area/100", win) * 100

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        markers = detect_markers(gray, aruco)     # [(center, angle, corners)]
        for center, angle, pts in markers:
            cv2.polylines(frame, [pts.astype(int)], True, (255, 0, 255), 2)

        # Refit the floor plane from the current depth map (cheap, subsampled).
        if use_floor and depth.has_depth():
            with depth._depth_lock:
                dm = depth._depth
            if dm is not None:
                floor.fit(dm, fx, fy, cx, cy, (W, H))

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        pieces, _ = find_pieces(hsv, s_floor, v_floor, min_area)

        report = []
        osc_dets = []                      # pixel-space, for the Unreal bridge
        biggest = None                     # for 'c' calibration
        for c in COLORS:
            # PASS 1 - measure every blob of this colour and touch no tracker
            # state at all. Every blob has to be scored against the slots'
            # PREVIOUS positions, so nothing may move until they are all in
            # hand; doing this inline (measure one, match it, measure the next)
            # is what let two neighbouring pieces end up sharing one slot.
            dets = []
            for ct in pieces[c["name"]]:
                x, y = cv2.boundingRect(ct)[:2]     # still used to anchor text
                (u, v), (rw, rh), _ = cv2.minAreaRect(ct)
                # Rotation-invariant sides: minAreaRect swaps w/h as the piece
                # turns past 45 deg, so never report those two directly.
                long_px, short_px = max(rw, rh), min(rw, rh)

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
                z_da3 = depth.depth_in_mask(mask)
                if z_da3 is None:                 # mask too small after erode
                    z_da3 = depth.depth_in_mask(
                        cv2.drawContours(np.zeros((H, W), np.uint8),
                                         [ct], -1, 255, -1))

                if z_da3 is not None:
                    # Only ever used to make the FIRST measurement of this
                    # brick; once locked, these samples are ignored entirely.
                    long_cm_raw = (long_px / f_size) * z_da3 * 100.0
                    short_cm_raw = (short_px / f_size) * z_da3 * 100.0
                else:
                    long_cm_raw = short_cm_raw = None

                dets.append({"ct": ct, "x": x, "y": y, "cx": u, "cy": v,
                             "long_px": long_px, "short_px": short_px,
                             "z": z_da3, "long_cm": long_cm_raw,
                             "short_cm": short_cm_raw,
                             "angle": angle, "shape": shape})

            # PASS 2 - one slot per piece, assigned nearest-first with each
            # slot claimable once, then draw and report each piece separately.
            for det, slot in zip(dets, tracks.update_frame(c["name"], dets)):
                ct, x, y = det["ct"], det["x"], det["y"]
                u, v, long_px = det["cx"], det["cy"], det["long_px"]
                angle, z_da3 = det["angle"], det["z"]

                # Feed the blueprint from PIXELS, before any of the depth/size
                # gates below. The blueprint does its own scaling and knows
                # nothing about centimetres, so making it wait for DA3 to load
                # and a size to lock would just mean an empty Unreal scene for
                # the first few seconds - or forever, on a machine with no GPU.
                # The slot's name (yellow1, yellow2, ...) is what keeps a given
                # brick on the same Unreal cube frame after frame.
                osc_dets.append((c["name"], slot["name"],
                                 (u, v, slot["angle"] or 0.0,
                                  long_px, det["short_px"])))

                # While a NEW kind of piece is settling, draw it dim + show a
                # countdown; only a CONFIRMED piece is reported and sent on.
                if not slot["confirmed"]:
                    cv2.drawContours(frame, [ct], -1, (150, 150, 150), 1)
                    cv2.putText(frame, f"sizing {slot['settle_left']:.1f}s",
                                (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (150, 150, 150), 2)
                    continue

                cv2.drawContours(frame, [ct], -1, c["draw"], 2)   # real outline

                # --- depth from the locked size ---------------------------
                # A rigid piece of known real width w_m subtending long_px
                # pixels is at z = w_m * f / long_px. Because the size is
                # frozen, this depth is smooth and self-consistent: moving the
                # piece changes only the pixel count, never the centimetres.
                z_used, z_src = slot["z"], "da3"
                if size_depth and slot["size_locked"] and long_px > 1 \
                        and slot["long_cm"]:
                    z_used = (slot["long_cm"] / 100.0) * f_size / long_px
                    z_src = "sz"
                if z_used is None:
                    cv2.putText(frame, f"{slot['name']}/{slot['shape']} z=?",
                                (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                c["draw"], 2)
                    continue

                long_cm, short_cm = slot["long_cm"], slot["short_cm"]
                P_cam = np.array([(u - cx) * z_used / fx,
                                  (v - cy) * z_used / fy, z_used])
                if use_floor and floor.ok:
                    X, Y, Zf = floor.to_floor(P_cam)   # floor frame, height
                    frame_tag = "flr"
                else:
                    X, Y, Zf = P_cam[0], P_cam[1], z_used  # camera frame
                    frame_tag = "cam"
                ang_txt = f" {slot['angle']:.0f}deg" if slot["angle"] \
                    is not None else ""
                l1 = (f"{slot['name']}/{slot['shape']} "
                      f"({X:+.2f},{Y:+.2f},{Zf:+.2f}){frame_tag}")
                size_txt = f"{long_cm:.1f}x{short_cm:.1f}cm" \
                    if long_cm is not None else "size?"
                l2 = (f"{size_txt}  d={z_used:.2f}m/{z_src} "
                      f"{int(long_px)}px{ang_txt}")
                if args.debug_size:
                    # px * z_da3 is what the size WOULD read if we re-estimated
                    # it every frame; watch it drift as you move the piece while
                    # the locked cm stays put. That drift is the depth map, and
                    # the lock is what keeps it out of the reported size.
                    raw = (long_px / f_size) * z_da3 * 100.0 if z_da3 else None
                    print(f"[size] {c['name']:6} px={long_px:6.1f} "
                          f"z_da3={z_da3 if z_da3 else float('nan'):6.3f} "
                          f"z_used={z_used:6.3f} "
                          f"would-be={raw if raw else float('nan'):6.2f}cm "
                          f"locked={long_cm:6.2f}cm")
                if long_cm is not None and (biggest is None
                                            or long_cm > biggest[0]):
                    biggest = (long_cm, c["name"], slot["shape"])
                report.append((slot["name"], slot["shape"], X, Y, Zf,
                               long_cm, short_cm, slot["angle"]))
                if osc is not None and long_cm is not None:
                    a_out = 0.0 if slot["angle"] is None else slot["angle"]
                    # meters -> mm for Unreal; keep the /obj layout
                    osc.send_message("/obj", [slot["name"], X * 1000.0,
                                     Y * 1000.0, a_out, long_cm, short_cm,
                                     Zf * 1000.0])
                cv2.putText(frame, l1, (x, y - 24), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, c["draw"], 2)
                cv2.putText(frame, l2, (x, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, c["draw"], 2)
        if bridge is not None:
            bridge.send_frame(osc_dets)
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
        live = sum(len(l) for l in tracks.slots.values())
        cv2.putText(frame, f"{dstat}  frame:{frame_state}  intr:{intr_src}  "
                    f"z:{'size-locked' if size_depth else 'raw'}  "
                    f"pieces:{live}  sizes:{book.count()}  "
                    f"scale:{depth.metric_scale:.3f}  {fps:.0f}fps",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow(win, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break
        elif key == ord('f'):
            use_floor = not use_floor
        elif key == ord('d'):
            size_depth = not size_depth
            print(f"[depth] source: {'locked size' if size_depth else 'raw DA3'}")
        elif key == ord('r'):
            tracks.forget_sizes()
            print("[size] forgot all locked sizes - re-measuring")
        elif key == ord('c'):
            # Absolute scale: the biggest piece on screen is the one you
            # measured with a ruler. Everything metric (depth map, and hence
            # every size derived from it) scales by the same factor.
            if args.calib_cm <= 0:
                print("[calib] pass --calib-cm <true long side in cm> first")
            elif biggest is None:
                print("[calib] no confirmed piece with a size on screen")
            else:
                k = args.calib_cm / biggest[0]
                depth.metric_scale *= k
                book.rescale(k)
                for lst in tracks.slots.values():
                    for s in lst:
                        if s["long_cm"]:
                            s["long_cm"] *= k
                            s["short_cm"] *= k
                print(f"[calib] {biggest[1]}/{biggest[2]} {biggest[0]:.2f}cm "
                      f"-> {args.calib_cm:.2f}cm : scale x{k:.4f} "
                      f"(total {depth.metric_scale:.4f})")
        elif key == ord('p'):
            if report:
                print(f"[pieces] {len(report)} tracked")
                for (n, sh, X, Y, Z, lo, shrt, ang) in sorted(report):
                    print(f"  {n:10} {sh:10} ({X:+.2f},{Y:+.2f},{Z:+.2f}) "
                          f"{lo:.1f}x{shrt:.1f}cm ang={ang}")
            else:
                print("no pieces with depth yet")

    depth.stop()
    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
