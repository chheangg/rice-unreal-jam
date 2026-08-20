"""
detect_platform_aruco.py — accurate single-camera (DJI) X/Y/Z tracker using
ArUco tags on each block instead of guessing block size from color blobs.

Why this replaces detect_platform.py's Z: that script had to GUESS each
block's real-world width to estimate distance-from-camera, which is the
whole reason its height numbers were unreliable. An ArUco tag has an EXACT,
precisely-known printed size, so cv2.solvePnP gives a block's real 3D
position directly from geometry - no guessing, verified to sub-millimeter
error against a synthetic ground truth (see chat).

Setup required:
  1. Run generate_aruco_tags.py, print aruco_tags.png at 100% scale.
  2. Stick tag id=0 on yel1, id=1 on yel2, id=2 on yel3, id=3 on red (or
     change ID_TO_NAME below to match however you actually assign them).
  3. Run this script.

Platform pose (world frame: origin at platform's bottom-left, X right, Y up,
Z height) is still solved from the red border, same as detect_platform.py -
that part was already proven accurate; only the per-block position method
changed. Detection is masked to inside the border, same as before.

Marker IDs are inherently stable identities - no "which blob is which"
ambiguity like the color-blob version had, so tracking is simpler: a slot
just holds its last known position on any frame its tag isn't seen, and
updates directly (with EMA smoothing for clean/non-jerky motion) whenever
it is.

OSC (address /obj): [name, x_mm, y_mm, angle, sizeX_px, sizeY_px, z_mm]
  x_mm, y_mm, z_mm - real mm, platform-relative (bottom-left origin).
  angle            - the tag's TRUE in-plane rotation, 0..360 (from
                     its corners; no 180 ambiguity - a tag has a known up).
  sizeX/Y          - marker's pixel bounding size (debug/visual only).

Run:  python detect_platform_aruco.py    (Esc to quit)
"""

import cmath
import json
import math

import cv2
import numpy as np
from pythonosc.udp_client import SimpleUDPClient

from platform_height import (find_platform_corners, solve_platform_pose,
                              PlatformHeightEstimator, PLATFORM_SIZE_MM,
                              make_aruco_detector, detect_markers,
                              solve_marker_camera_point)

osc = SimpleUDPClient("127.0.0.1", 7000)

DJI_INDEX = 0
DJI_INTRINSICS_PATH = "dji_intrinsics.json"

# Must match generate_aruco_tags.py's MARKER_SIZE_MM exactly.
MARKER_SIZE_MM = 30.0

# Must match generate_aruco_tags.py's IDS mapping (or whatever you actually
# stuck where).
ID_TO_NAME = {0: "yel1", 1: "yel2", 2: "yel3", 3: "red"}

# The DJI is mounted in PORTRAIT but streams landscape-shaped frames with the
# content rotated - correct it here before any detection/geometry runs.
DJI_ROTATE = cv2.ROTATE_90_CLOCKWISE

EMA_ALPHA = 0.3   # ArUco positions are already exact per-frame (not a guess
                  # like the color-blob version), so smoothing only needs to
                  # damp ordinary corner-detection jitter - can be snappier
                  # (higher alpha) than the color-based script needed.

Z_MISSING = -1.0


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
              f"intrinsics model. Even without it, ArUco tags remove the "
              f"biggest source of error (block-size guessing); intrinsics "
              f"error alone is a much smaller residual inaccuracy.")
        fx = fy = float(frame_w)
        K = np.array([[fx, 0, frame_w / 2.0],
                      [0, fy, frame_h / 2.0],
                      [0, 0, 1]], dtype=np.float64)
        dist = np.zeros(5, dtype=np.float64)
        return K, dist


def _ema(prev, new, alpha=EMA_ALPHA):
    return new if prev is None else alpha * new + (1 - alpha) * prev


def marker_angle_deg(corners_px):
    """
    True in-plane rotation of an ArUco tag, 0..360, from its corners.
    cv2.aruco returns corners as TL, TR, BR, BL, so the top edge TL->TR is a
    fixed feature of the tag - its direction IS the tag's orientation. Unlike
    minAreaRect (ambiguous mod 180), a tag has a known "up", so a full turn is
    distinguishable: rotating the block 180 gives 180 here, not 0.
    Screen y grows downward, so this is a clockwise-positive image angle; the
    Unreal side maps it to yaw (see notes where it's sent).
    """
    tl, tr = corners_px[0], corners_px[1]
    return math.degrees(math.atan2(tr[1] - tl[1], tr[0] - tl[0])) % 360.0


class Slot:
    def __init__(self, name):
        self.name = name
        self.x = self.y = self.z = None
        self.px_size = 0
        self.angle = 0.0
        self._angle_phasor = None      # circular EMA state (unit complex)
        self.seen = False

    def update(self, world_xyz, px_size, angle_deg):
        self.x = _ema(self.x, world_xyz[0])
        self.y = _ema(self.y, world_xyz[1])
        self.z = _ema(self.z, world_xyz[2])
        self.px_size = px_size
        # Angles wrap, so a plain EMA breaks across 0/360 (e.g. 359 -> 1 would
        # average to ~180). Smooth the unit phasor e^{i*theta} instead and read
        # its phase back - the correct way to average a circular quantity.
        z = cmath.exp(1j * math.radians(angle_deg))
        self._angle_phasor = z if self._angle_phasor is None \
            else EMA_ALPHA * z + (1 - EMA_ALPHA) * self._angle_phasor
        self.angle = math.degrees(cmath.phase(self._angle_phasor)) % 360.0
        self.seen = True


def main():
    cap = cv2.VideoCapture(DJI_INDEX)
    if not cap.isOpened():
        print(f"DJI camera not found at index {DJI_INDEX}")
        return

    K = dist = None
    height_est = None
    platform_mask = None
    aruco_detector = make_aruco_detector()

    slots = {name: Slot(name) for name in ID_TO_NAME.values()}

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if DJI_ROTATE is not None:
            frame = cv2.rotate(frame, DJI_ROTATE)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = frame.shape[:2]

        if height_est is None:
            if K is None:
                K, dist = load_or_guess_intrinsics(w, h)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            corners = find_platform_corners(hsv)
            if corners is not None:
                rvec, tvec = solve_platform_pose(K, dist, corners)
                if rvec is not None:
                    height_est = PlatformHeightEstimator(K, dist, rvec, tvec)
                    platform_mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.fillPoly(platform_mask, [corners.astype(np.int32)], 255)
                    print("Platform pose solved - tag tracking is now live.")
                    cv2.drawContours(frame, [corners.astype(int)], -1, (255, 0, 255), 3)
            cv2.putText(frame, "finding platform border..." if height_est is None
                        else "platform locked", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255) if height_est is None else (0, 255, 0), 2)

        markers = detect_markers(gray, aruco_detector)

        for slot in slots.values():
            slot.seen = False

        for marker_id, corners_px in markers.items():
            name = ID_TO_NAME.get(marker_id)
            if name is None or height_est is None:
                continue
            # Ignore tags outside the platform (mirrors the color version's
            # boundary masking) - check the marker's center pixel.
            cx, cy = corners_px.mean(axis=0)
            if platform_mask is not None and platform_mask[int(cy), int(cx)] == 0:
                continue

            p_cam = solve_marker_camera_point(corners_px, K, dist, MARKER_SIZE_MM)
            if p_cam is None:
                continue
            world_xyz = height_est.camera_point_to_world(p_cam)
            px_size = int(np.linalg.norm(corners_px[0] - corners_px[1]))
            angle_deg = marker_angle_deg(corners_px)
            slots[name].update(world_xyz, px_size, angle_deg)

            cv2.polylines(frame, [corners_px.astype(int)], True, (0, 255, 0), 2)
            cv2.putText(frame, f"{name} id={marker_id}", (int(cx), int(cy) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        for slot in slots.values():
            if slot.x is None:
                continue   # never seen this tag yet
            x, y, z = round(slot.x, 1), round(slot.y, 1), round(slot.z, 1)
            angle = round(slot.angle, 1)
            osc.send_message("/obj",
                             [slot.name, x, y, angle, slot.px_size, slot.px_size, z])
            status = "LIVE" if slot.seen else "held"
            print(f"{slot.name} x={x}mm y={y}mm z={z}mm angle={angle} [{status}]")

        cv2.imshow("DJI ArUco platform tracker", frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
