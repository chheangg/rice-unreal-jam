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

Controls: same as lego_locator.py, plus nothing new. Esc/q quits.

Run:
    python lego_locator_xyz.py            # default camera
    python lego_locator_xyz.py 0 --fov 55
    python lego_locator_xyz.py "http://172.20.10.1:8081/video"
    python lego_locator_xyz.py "clip.mp4"
"""

import argparse
import math
import time

import cv2
import numpy as np

from lego_locator import (COLORS, find_pieces, as_source,
                          DEFAULT_S_FLOOR, DEFAULT_V_FLOOR, DEFAULT_MIN_AREA_100)
from depth_estimator import DepthEstimator


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
    def __init__(self, alpha=0.25, match_dist=90, max_miss=15):
        self.alpha = alpha
        self.match_dist = match_dist
        self.max_miss = max_miss
        self.slots = {}                         # colour name -> list of slots

    def update(self, color, cx, cy, z, w_cm, h_cm):
        lst = self.slots.setdefault(color, [])
        best, best_d = None, 1e9
        for s in lst:
            d = math.hypot(s["cx"] - cx, s["cy"] - cy)
            if d < best_d:
                best, best_d = s, d
        if best is None or best_d > self.match_dist:
            best = {"cx": cx, "cy": cy, "z": z, "w_cm": w_cm,
                    "h_cm": h_cm, "miss": 0}
            lst.append(best)
        best["cx"], best["cy"], best["miss"] = cx, cy, 0
        a = self.alpha
        for k, val in (("z", z), ("w_cm", w_cm), ("h_cm", h_cm)):
            if val is None:
                continue                        # no depth this frame: keep last
            best[k] = val if best[k] is None else (1 - a) * best[k] + a * val
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


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", nargs="?", default="0",
                    help="camera index, stream URL, or video file (default 0)")
    ap.add_argument("--fov", type=float, default=60.0,
                    help="assumed horizontal FOV (deg) if DA3 gives no intrinsics")
    ap.add_argument("--model", default=None,
                    help="override DA3 model id (e.g. depth-anything/DA3-SMALL)")
    args = ap.parse_args()

    cap = cv2.VideoCapture(as_source(args.source))
    if not cap.isOpened():
        print(f"FAILED to open {args.source!r}")
        return 1

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
    cv2.createTrackbar("S floor", win, DEFAULT_S_FLOOR, 255, lambda v: None)
    cv2.createTrackbar("V floor", win, DEFAULT_V_FLOOR, 255, lambda v: None)
    cv2.createTrackbar("Min area/100", win, DEFAULT_MIN_AREA_100, 100, lambda v: None)

    t_prev, fps = time.time(), 0.0
    tracks = Tracks()

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

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        pieces, _ = find_pieces(hsv, s_floor, v_floor, min_area)

        report = []
        for c in COLORS:
            for ct in pieces[c["name"]]:
                x, y, w, h = cv2.boundingRect(ct)
                (u, v), (rw, rh), _ = cv2.minAreaRect(ct)
                cv2.rectangle(frame, (x, y), (x + w, y + h), c["draw"], 2)

                # Depth sampled ONLY inside the piece (eroded a few px so the
                # coarse depth map doesn't pick up background at the edges).
                # This is what makes size constant as the piece moves - the
                # bounding box would mix in table/background depth.
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
                    slot = tracks.update(c["name"], u, v, z, w_cm_raw, h_cm_raw)
                    zs, w_cm, h_cm = slot["z"], slot["w_cm"], slot["h_cm"]
                    X = (u - cx) * zs / fx
                    Y = (v - cy) * zs / fy
                    l1 = f"{c['name']} ({X:+.2f},{Y:+.2f},{zs:.2f})m"
                    l2 = f"{w_cm:.1f}x{h_cm:.1f}cm"
                    report.append((c["name"], X, Y, zs, w_cm, h_cm))
                else:
                    tracks.update(c["name"], u, v, None, None, None)
                    l1 = f"{c['name']} z=? {int(rw)}x{int(rh)}px"
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
        cv2.putText(frame, f"{dstat}  intr:{intr_src}  {fps:.0f}fps",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow(win, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break
        elif key == ord('p'):
            if report:
                print(f"[intr:{intr_src}] " + " | ".join(
                    f"{n}: X{X:+.2f} Y{Y:+.2f} Z{z:.2f}m {w:.1f}x{h:.1f}cm"
                    for (n, X, Y, z, w, h) in report))
            else:
                print("no pieces with depth yet")

    depth.stop()
    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
