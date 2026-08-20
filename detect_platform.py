"""
detect_platform.py — single-camera (DJI, angled ~45 deg) tracker with real,
perspective-corrected X/Y/Z, using the red platform border as the reference.
See platform_height.py for the full geometry explanation.

DJI intrinsics: loaded from dji_intrinsics.json if present (run
`python calibrate_iphone.py 0 dji_intrinsics.json` to generate it - same
script works on any camera index). Falls back to a rough guessed pinhole
model if that file doesn't exist yet, which is noticeably less accurate but
still gives a consistent, usable pose from the platform border alone.

Pose is solved ONCE, from the first frame where all 4 platform corners are
found, and reused after that - re-run this script if the DJI moves. Once
solved, detection is masked to ONLY the platform's interior in PIXEL space
(anything past the red border can't be detected as a block at all), and the
computed real-world X/Y is additionally CLAMPED to the platform's physical
extent as a second safety net.

STABILITY (no "jerky"/"spazzy" movement):
  - Tracking is PERSISTENT per slot (yel1/yel2/yel3/red): once a slot locks
    onto a blob, it only re-matches to a new blob near its last known pixel
    position (MATCH_DIST_PX); a frame with no nearby match just holds the
    last value instead of dropping or jumping to an unrelated blob.
  - Apparent pixel width (what the height calc divides by) is smoothed with
    a rolling median then an exponential moving average before being used,
    since raw per-frame width jitter is what caused wild height swings.
  - The final X, Y, Z values are each smoothed with their own EMA on top of
    that, so on-screen/OSC output moves smoothly frame to frame instead of
    snapping.

OSC (address /obj): [name, x_mm, y_mm, angle, sizeX_px, sizeY_px, z_mm]
  x_mm, y_mm - REAL platform-relative position in mm. (0,0) = platform's
               BOTTOM-LEFT corner (portrait-oriented frame); x increases
               left-to-right, y increases bottom-to-top; both range
               0..PLATFORM_SIZE_MM. Perspective-corrected via ray-plane
               intersection - NOT raw pixels. This is a change from earlier
               scripts in this repo that sent raw pixel x/y - the Unreal
               side will need updating to treat these as real-world mm with
               this origin/orientation, not pixel coordinates.
  angle      - always 0 (no direction marker).
  sizeX/Y    - still raw pixel box size (debug/visual only, not converted).
  z_mm       - real height above the platform, in mm. -1.0 sentinel until
               the platform pose has been solved.

Run:  python detect_platform.py    (Esc to quit)
"""

import json
from collections import deque

import cv2
import numpy as np
from pythonosc.udp_client import SimpleUDPClient

from platform_height import (find_platform_corners, solve_platform_pose,
                              PlatformHeightEstimator, PLATFORM_SIZE_MM)

osc = SimpleUDPClient("127.0.0.1", 7000)

DJI_INDEX = 0
DJI_INTRINSICS_PATH = "dji_intrinsics.json"

# The DJI is mounted in PORTRAIT but streams landscape-shaped frames with the
# content rotated - correct it here before any detection/geometry runs, so
# everything downstream (pixel coords, platform corners, pose) is consistent
# with reality. Flip to cv2.ROTATE_90_COUNTERCLOCKWISE if this comes out
# upside-down/sideways for your actual mount.
DJI_ROTATE = cv2.ROTATE_90_CLOCKWISE

YELLOW = [((20, 60, 150), (35, 255, 255))]
RED = [((0, 70, 50), (10, 255, 255)),
       ((170, 70, 50), (179, 255, 255))]

MIN_AREA = 600                  # tracked-block minimum (small blobs)
MAX_TRACKED_RED_AREA = 15000    # above this, it's the platform border, not a block
MATCH_DIST_PX = 150             # how close a new blob must be to keep a slot's
                                 # tracking - tune based on how far blocks move
                                 # between frames vs. how far apart they sit
STALE_LOCK_FRAMES = 600         # if a slot goes this many consecutive frames
                                 # with no nearby match, drop the lock instead
                                 # of freezing on it forever. Set high (~20s
                                 # at 30fps): every reset re-tares Z, which
                                 # looks like a sudden jump - only there as a
                                 # last-resort escape from a bad lock, not a
                                 # normal recovery path.
BORDER_ERODE_PX = 45            # shrink the platform mask inward by this many
                                 # px so the red border's OWN pixels are
                                 # excluded from block detection (it's the
                                 # same color as tracked red pieces) - tune to
                                 # comfortably exceed the border's thickness
Z_MISSING = -1.0

WIDTH_MEDIAN_WINDOW = 9   # rolling-median window on apparent pixel width,
                          # before EMA - kills single-frame outlier spikes
EMA_ALPHA = 0.05          # shared smoothing factor for width/x/y/z. Low on
                          # purpose - slow and steady over snappy, per request.

TARE_AFTER_N_MATCHES = WIDTH_MEDIAN_WINDOW + 3   # let smoothing settle first
# Z accuracy depends on two guessed constants (DJI focal length, block real
# width - see chat), so the ABSOLUTE mm value isn't trustworthy without real
# calibration. What we can still get right: blocks start resting ON the
# platform, so whatever Z they first read as should mean "0mm, on the
# surface." Each slot tares itself to its own early reading once its
# smoothing has settled, so height reported afterward is relative to that
# resting position - correctly signed (lifting reads positive) even though
# the absolute scale stays approximate.


def find_blobs(hsv, ranges, min_area, max_area=None, mask_region=None):
    mask = None
    for lo, hi in ranges:
        m = cv2.inRange(hsv, np.array(lo), np.array(hi))
        mask = m if mask is None else mask | m
    if mask_region is not None:
        mask = cv2.bitwise_and(mask, mask_region)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        if max_area is not None and area > max_area:
            continue
        (cx, cy), (rw, rh), _ = cv2.minAreaRect(c)
        x, y, w, h = cv2.boundingRect(c)
        out.append({"cx": int(cx), "cy": int(cy),
                    "sx": int(rw), "sy": int(rh),
                    # minAreaRect's rw/rh assignment can SWAP frame to frame
                    # for a rotated blob (OpenCV's angle convention flips
                    # which side it calls "width" near certain angles) - use
                    # the larger side consistently as a rotation-invariant
                    # size for anything that gets smoothed/used for depth,
                    # instead of raw (swappable) sx.
                    "size": int(max(rw, rh)),
                    "box": (x, y, w, h), "area": area})
    return out


def load_or_guess_intrinsics(frame_w, frame_h):
    try:
        with open(DJI_INTRINSICS_PATH) as f:
            intr = json.load(f)
        K = np.array(intr["camera_matrix"], dtype=np.float64)
        dist = np.array(intr["dist_coeffs"], dtype=np.float64)
        print(f"Loaded real DJI calibration from {DJI_INTRINSICS_PATH}.")
        return K, dist
    except FileNotFoundError:
        print(f"{DJI_INTRINSICS_PATH} not found - using a rough guessed "
              f"intrinsics model (less accurate, but still a consistent "
              f"pose from the platform border). Run "
              f"`python calibrate_iphone.py 0 {DJI_INTRINSICS_PATH}` later "
              f"for real accuracy.")
        fx = fy = float(frame_w)   # ~53deg horizontal FOV assumption
        K = np.array([[fx, 0, frame_w / 2.0],
                      [0, fy, frame_h / 2.0],
                      [0, 0, 1]], dtype=np.float64)
        dist = np.zeros(5, dtype=np.float64)
        return K, dist


def _ema(prev, new, alpha=EMA_ALPHA):
    return new if prev is None else alpha * new + (1 - alpha) * prev


class Slot:
    """One persistent tracked object: locks onto a blob, holds its last
    known value through frames with no nearby match, and smooths width/X/Y/Z
    so output motion is clean instead of jerky. See module docstring."""

    def __init__(self, name, color_name):
        self.name = name
        self.color_name = color_name
        self.blob = None
        self._width_history = deque(maxlen=WIDTH_MEDIAN_WINDOW)
        self.smooth_width = None
        self.smooth_x = None
        self.smooth_y = None
        self.smooth_z = None
        self._unmatched_frames = 0
        self._matched_count = 0   # how many frames this lock has matched -
                                   # used to trigger the one-time Z tare below
        self.z_tare = None        # Z reading treated as "0mm" (resting on
                                   # the platform) - see TARE_AFTER_N_MATCHES

    def _reset_lock(self):
        """Drop the current lock and all smoothing state, so the slot can
        cleanly re-acquire a (hopefully better) blob from scratch."""
        self.blob = None
        self._width_history.clear()
        self.smooth_width = self.smooth_x = self.smooth_y = self.smooth_z = None
        self._unmatched_frames = 0
        self._matched_count = 0
        self.z_tare = None

    def update(self, candidates):
        """candidates: list of blob dicts, mutated (matched ones removed)."""
        matched = None
        if not candidates:
            pass
        elif self.blob is None:
            matched = max(candidates, key=lambda b: b["area"])
        else:
            px, py = self.blob["cx"], self.blob["cy"]
            best, best_dist = None, MATCH_DIST_PX
            for b in candidates:
                d = ((b["cx"] - px) ** 2 + (b["cy"] - py) ** 2) ** 0.5
                if d < best_dist:
                    best, best_dist = b, d
            matched = best

        if matched is None:
            if self.blob is not None:
                self._unmatched_frames += 1
                if self._unmatched_frames > STALE_LOCK_FRAMES:
                    # Been "stuck" too long with nothing nearby - likely a bad
                    # lock (e.g. onto detection noise) - let go and try fresh.
                    self._reset_lock()
            return   # no close match - hold last known value (static)

        self._unmatched_frames = 0
        self._matched_count += 1
        self.blob = matched
        candidates.remove(matched)
        self._width_history.append(matched["size"])
        median_w = sorted(self._width_history)[len(self._width_history) // 2]
        self.smooth_width = _ema(self.smooth_width, median_w)

    def compute_xyz(self, height_est):
        """Real (x_mm, y_mm, z_mm), smoothed and clamped to the platform's
        physical extent. Returns None if we don't have enough to compute yet."""
        if self.blob is None or self.smooth_width is None or height_est is None:
            return None

        # X/Y: exact ray-plane intersection through the block's BASE (bottom
        # of its bounding box - where it actually touches the platform), not
        # its center - see platform_height.py docstring for why.
        x0, y0, w, h = self.blob["box"]
        base_px, base_py = x0 + w / 2.0, y0 + h
        ground = height_est.ground_xy_mm(base_px, base_py)

        # Z: approximate, from known-width-vs-apparent-width depth, using
        # the smoothed width (not raw) to avoid amplifying detection jitter.
        point3d = height_est.point3d_mm(self.blob["cx"], self.blob["cy"],
                                         self.smooth_width, self.color_name)
        if ground is None or point3d is None:
            return None

        raw_x, raw_y = ground
        raw_z = point3d[2]

        self.smooth_x = _ema(self.smooth_x, raw_x)
        self.smooth_y = _ema(self.smooth_y, raw_y)
        self.smooth_z = _ema(self.smooth_z, raw_z)

        # One-time tare: treat this slot's first-settled Z as "resting on
        # the platform" (0mm) - see TARE_AFTER_N_MATCHES above for why.
        if self.z_tare is None and self._matched_count >= TARE_AFTER_N_MATCHES:
            self.z_tare = self.smooth_z
            print(f"{self.name}: tared Z to 0mm at raw reading {self.smooth_z:.1f}mm")
        z = self.smooth_z if self.z_tare is None else self.smooth_z - self.z_tare

        # Hard safety net: never report a position outside the platform's
        # real physical extent (0..PLATFORM_SIZE_MM, origin at bottom-left),
        # even if geometry noise briefly pushes past it.
        x = max(0.0, min(PLATFORM_SIZE_MM, self.smooth_x))
        y = max(0.0, min(PLATFORM_SIZE_MM, self.smooth_y))
        return x, y, z


def send_and_draw(slot, xyz, frame, color):
    b = slot.blob
    if xyz is None:
        x_mm = y_mm = 0.0
        z_mm = Z_MISSING
    else:
        x_mm, y_mm, z_mm = xyz
        x_mm, y_mm, z_mm = round(x_mm, 1), round(y_mm, 1), round(z_mm, 1)
    osc.send_message("/obj", [slot.name, x_mm, y_mm, 0, b["sx"], b["sy"], z_mm])
    x, y, w, h = b["box"]
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.putText(frame, f"{slot.name} ({x_mm},{y_mm})mm z={z_mm}mm", (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
    print(f"{slot.name} x={x_mm}mm y={y_mm}mm z={z_mm}mm "
          f"(px={b['cx']},{b['cy']} size={b['sx']}x{b['sy']})")


def main():
    cap = cv2.VideoCapture(DJI_INDEX)
    if not cap.isOpened():
        print(f"DJI camera not found at index {DJI_INDEX}")
        return

    K = dist = None
    height_est = None
    platform_mask = None

    slots = [Slot("yel1", "yellow"), Slot("yel2", "yellow"), Slot("yel3", "yellow"),
             Slot("red", "red")]

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if DJI_ROTATE is not None:
            frame = cv2.rotate(frame, DJI_ROTATE)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, w = frame.shape[:2]

        if height_est is None:
            if K is None:
                K, dist = load_or_guess_intrinsics(w, h)
            corners = find_platform_corners(hsv)
            if corners is not None:
                rvec, tvec = solve_platform_pose(K, dist, corners)
                if rvec is not None:
                    height_est = PlatformHeightEstimator(K, dist, rvec, tvec)
                    platform_mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.fillPoly(platform_mask, [corners.astype(np.int32)], 255)
                    # Shrink inward so the border's OWN red pixels (same
                    # color as tracked red blocks) are excluded, not just
                    # what's outside the border entirely.
                    erode_kernel = np.ones((BORDER_ERODE_PX, BORDER_ERODE_PX), np.uint8)
                    platform_mask = cv2.erode(platform_mask, erode_kernel)
                    print("Platform pose solved - tracking is now live, "
                          "detection restricted to inside the red border.")
                    bl_x, bl_y = corners[0]
                    print(f"Classified bottom-left corner at pixel ({bl_x:.0f}, {bl_y:.0f}) "
                          f"- visually confirm this is really the platform's bottom-left "
                          f"in the (rotated) frame; if not, the X/Y axes are flipped.")
                    cv2.drawContours(frame, [corners.astype(int)], -1, (255, 0, 255), 3)
                    cv2.circle(frame, (int(bl_x), int(bl_y)), 15, (255, 255, 0), -1)
                    cv2.putText(frame, "ORIGIN (0,0)", (int(bl_x) + 20, int(bl_y)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, "finding platform border..." if height_est is None
                        else "platform locked", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255) if height_est is None else (0, 255, 0), 2)

        yellow_candidates = find_blobs(hsv, YELLOW, MIN_AREA, mask_region=platform_mask)
        red_candidates = find_blobs(hsv, RED, MIN_AREA, MAX_TRACKED_RED_AREA,
                                     mask_region=platform_mask)

        for slot in slots:
            candidates = yellow_candidates if slot.color_name == "yellow" else red_candidates
            slot.update(candidates)
            if slot.blob is None:
                continue
            xyz = slot.compute_xyz(height_est)
            color = (0, 255, 255) if slot.color_name == "yellow" else (0, 0, 255)
            send_and_draw(slot, xyz, frame, color)

        cv2.imshow("DJI platform tracker", frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
