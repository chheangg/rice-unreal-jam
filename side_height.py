"""
side_height.py — real (geometric) height-above-table for a block seen by the
SIDE camera (iPhone), replacing the earlier DA3 monocular depth guess.

Uses:
  1. iphone_intrinsics.json (from calibrate_iphone.py) - focal length lets us
     estimate how far a block is FROM THE IPHONE, from how big it looks in
     pixels vs. its known real-world size. This is what keeps height accurate
     even as a block moves closer/farther from the iPhone.
  2. table_pose.json (from calibrate_table_pose.py) - where the iPhone sits
     relative to the table plane, so a 3D point in camera space can be
     converted into "how far above the table is this, really."

REAL_WIDTH_MM below is a rough visual estimate of the tracked LEGO pieces,
not a ruler measurement (see chat) - ONE tunable number per color, same
pattern as metric_scale was for the old DA3 path. If heights come out
uniformly off by a constant factor, measure a real piece and fix this.

FALLBACK - SIMPLE MODE: if iphone_intrinsics.json / table_pose.json aren't
present (checkerboard calibration not done), this falls back to a naive
linear pixel-height model instead of refusing to run: height_mm =
(BASELINE_Y_PX - pixel_y) / PX_PER_MM_GUESS. This does NOT correct for a
block moving closer/farther from the iPhone (see chat - that's what the full
calibration path fixes), so it's a "good enough for now" stand-in, not the
accurate version. BASELINE_Y_PX / PX_PER_MM_GUESS are rough guesses - tune
them by placing a block at table level and at a known height, reading the
printed pixel_y each time, and solving for the two constants from that.
"""

import json
import numpy as np
import cv2

# Rough estimate - replace with a ruler measurement of the actual pieces for
# real accuracy. Only the WIDTH the camera sees face-on matters here.
REAL_WIDTH_MM = {
    "yellow": 48.0,
    "red": 24.0,
}

# Simple-mode fallback constants (see FALLBACK note above) - untested guesses,
# tune against a block at a couple of known real heights.
BASELINE_Y_PX = 900.0      # pixel_y that corresponds to table level (0mm)
PX_PER_MM_GUESS = 3.0      # pixels of vertical movement per mm of real height


class SideHeightEstimator:
    def __init__(self, intrinsics_path="iphone_intrinsics.json",
                 pose_path="table_pose.json"):
        self.available = False
        self.simple_mode = False
        self.last_error = None
        try:
            with open(intrinsics_path) as f:
                intr = json.load(f)
            with open(pose_path) as f:
                pose = json.load(f)
        except FileNotFoundError as e:
            self.last_error = repr(e)
            self.available = True
            self.simple_mode = True
            print(f"[side_height] calibration file missing: {e}. "
                  f"Falling back to SIMPLE MODE (naive pixel-height, not "
                  f"corrected for camera-distance changes) - see chat/docstring. "
                  f"Run calibrate_iphone.py then calibrate_table_pose.py for "
                  f"the accurate version.")
            return

        self.K = np.array(intr["camera_matrix"], dtype=np.float64)
        self.dist = np.array(intr["dist_coeffs"], dtype=np.float64)
        rvec = np.array(pose["rvec"], dtype=np.float64)
        self.tvec = np.array(pose["tvec"], dtype=np.float64)
        self.R = cv2.Rodrigues(rvec)[0]
        self.R_inv = self.R.T
        self.fx = self.K[0, 0]
        self.available = True

    def height_mm(self, cx, cy, pixel_width, color_name):
        """
        Real height (mm) above the table for a block detected at pixel
        center (cx, cy) with apparent on-screen width `pixel_width` (px),
        given its tracked color (looks up REAL_WIDTH_MM). Returns None if
        calibration isn't loaded or the estimate is degenerate.
        """
        if not self.available:
            return None
        if self.simple_mode:
            return (BASELINE_Y_PX - cy) / PX_PER_MM_GUESS
        real_width = REAL_WIDTH_MM.get(color_name)
        if real_width is None or pixel_width <= 1:
            return None

        # Distance from the iPhone: a known-size object's apparent pixel size
        # shrinks proportionally with distance - invert that relationship.
        depth_mm = self.fx * real_width / pixel_width

        # Back-project the pixel center into an undistorted, normalized ray,
        # then scale by depth to get the block's 3D position in CAMERA space.
        pts = np.array([[[float(cx), float(cy)]]], dtype=np.float64)
        undist = cv2.undistortPoints(pts, self.K, self.dist)
        x_n, y_n = undist[0, 0]
        p_cam = np.array([x_n * depth_mm, y_n * depth_mm, depth_mm])

        # Camera-space -> table-space (solvePnP gave p_cam = R @ p_world + t,
        # so invert: p_world = R^T @ (p_cam - t)). World Z = height above the
        # table plane the board was calibrated flat against.
        p_world = self.R_inv @ (p_cam - self.tvec)
        return float(p_world[2])
