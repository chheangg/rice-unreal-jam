"""
calibrate_table_pose.py — step 2 of 2 for real height tracking.

Finds the iPhone's position/orientation RELATIVE TO THE TABLE, by looking at
the same checkerboard laid FLAT ON THE TABLE (this defines the table plane:
Z=0 in the board's own coordinate frame, board corners lying in that plane).

Run AFTER calibrate_iphone.py has produced iphone_intrinsics.json, and only
needs to be re-run if the iPhone physically moves.

Fully automatic, headless, no GUI window / no key presses needed - just lay
the board flat on the table, in view of the iPhone, and hold it still for a
moment. It solves and saves as soon as it sees a few stable frames in a row.

Run:  python calibrate_table_pose.py [camera_index]

Output: table_pose.json (rvec, tvec - the iPhone's pose relative to the
        table plane defined by the board).
"""

import sys
import time
import json
import cv2
import numpy as np

INTRINSICS_PATH = "iphone_intrinsics.json"
OUT_PATH = "table_pose.json"
STABLE_FRAMES_NEEDED = 10   # consecutive found-frames before auto-solving
TIMEOUT_S = 60

CAM_INDEX = int(sys.argv[1]) if len(sys.argv) > 1 else 1


def main():
    with open(INTRINSICS_PATH) as f:
        intr = json.load(f)
    K = np.array(intr["camera_matrix"], dtype=np.float64)
    dist = np.array(intr["dist_coeffs"], dtype=np.float64)
    board_cols, board_rows = intr["board_cols"], intr["board_rows"]
    square_mm = intr["square_mm"]

    objp = np.zeros((board_rows * board_cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_cols, 0:board_rows].T.reshape(-1, 2) * square_mm

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print(f"Camera index {CAM_INDEX} not found.")
        return

    print("Lay the checkerboard FLAT on the table, fully visible to the iPhone, "
          "and hold it still.")

    stable_count = 0
    last_status_t = 0.0
    start_t = time.time()

    while time.time() - start_t < TIMEOUT_S:
        ok, frame = cap.read()
        if not ok:
            print("Camera read failed - stopping.")
            cap.release()
            return
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(
            gray, (board_cols, board_rows),
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)

        if found:
            stable_count += 1
        else:
            stable_count = 0

        now = time.time()
        if now - last_status_t > 2.0:
            print(f"{'board seen, holding steady...' if found else 'board not visible'} "
                  f"({stable_count}/{STABLE_FRAMES_NEEDED})")
            last_status_t = now

        if stable_count >= STABLE_FRAMES_NEEDED:
            corners = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            ok_pnp, rvec, tvec = cv2.solvePnP(objp, corners, K, dist)
            cap.release()
            if not ok_pnp:
                print("solvePnP failed - try again with a clearer view.")
                return

            out = {
                "camera_index": CAM_INDEX,
                "rvec": rvec.flatten().tolist(),
                "tvec": tvec.flatten().tolist(),
                "note": "world frame = table plane, Z=0 at table surface, "
                        "units = mm (same as square_mm in iphone_intrinsics.json)",
            }
            with open(OUT_PATH, "w") as f:
                json.dump(out, f, indent=2)
            print(f"Saved {OUT_PATH}")
            print(f"Camera height above table: {tvec.flatten()[2]:.1f}mm "
                  f"(sanity check this against a rough tape measure)")
            return

    cap.release()
    print("Timed out without a stable detection - run again.")


if __name__ == "__main__":
    main()
