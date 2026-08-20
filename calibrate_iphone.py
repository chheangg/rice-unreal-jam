"""
calibrate_iphone.py — one-time intrinsics calibration for the SIDE camera
(the iPhone, via Continuity Camera), using the printed checkerboard.

This is step 1 of 2 for real (non-monocular-guess) height tracking:
  1. THIS script: find the iPhone's focal length / distortion (intrinsics).
  2. calibrate_table_pose.py: find the table's position relative to the
     iPhone (one shot of the same board lying flat on the table).

Board: 10x7 squares (9x6 internal corners), 20mm/square - matches
checkerboard_calibration.png. If you regenerate the board at a different
size, update BOARD_COLS/BOARD_ROWS/SQUARE_MM below to match.

Fully automatic, headless, no GUI window / no key presses needed - it
auto-captures whenever the board is clearly detected, with a cooldown
between captures so you have time to move the board to a new angle/distance
before the next one. Just hold the board in view of the iPhone and move it
around; console output tracks progress. Stops itself once it has enough
good captures.

Run:  python calibrate_iphone.py [camera_index] [output_path]
      (defaults to camera index 1 / iphone_intrinsics.json - the iPhone, per
      our setup. Pass a different index + output path to calibrate another
      camera, e.g. `python calibrate_iphone.py 0 dji_intrinsics.json` for
      the DJI at index 0.)

Output: iphone_intrinsics.json by default (camera_matrix, dist_coeffs,
        image_size), or the path passed as the 2nd argument.
"""

import sys
import time
import json
import cv2
import numpy as np

BOARD_COLS, BOARD_ROWS = 9, 6     # internal corners (squares - 1)
SQUARE_MM = 20.0
TARGET_CAPTURES = 20
MIN_CAPTURES = 15
CAPTURE_COOLDOWN_S = 1.5   # time between auto-captures, to force pose variety

CAM_INDEX = int(sys.argv[1]) if len(sys.argv) > 1 else 1
OUT_PATH = sys.argv[2] if len(sys.argv) > 2 else "iphone_intrinsics.json"
# Pass "cw" or "ccw" as a 3rd arg to rotate frames before calibrating - use
# this if the camera is physically mounted sideways (e.g. the DJI here),
# so the saved intrinsics match the orientation the tracker actually uses.
_ROTATE_ARG = sys.argv[3] if len(sys.argv) > 3 else None
ROTATE = {"cw": cv2.ROTATE_90_CLOCKWISE,
          "ccw": cv2.ROTATE_90_COUNTERCLOCKWISE}.get(_ROTATE_ARG)


def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print(f"Camera index {CAM_INDEX} not found.")
        return

    objp = np.zeros((BOARD_ROWS * BOARD_COLS, 3), np.float32)
    objp[:, :2] = np.mgrid[0:BOARD_COLS, 0:BOARD_ROWS].T.reshape(-1, 2) * SQUARE_MM

    objpoints = []
    imgpoints = []
    image_size = None
    last_capture_t = 0.0
    last_status_t = 0.0

    print(f"Auto-calibrating camera index {CAM_INDEX}. No key presses needed - "
          f"hold the checkerboard up and move it to different angles/distances.")
    print(f"Target: {TARGET_CAPTURES} captures (min {MIN_CAPTURES}), "
          f"~{CAPTURE_COOLDOWN_S}s apart.")

    cv2.namedWindow("iPhone calibration - LIVE", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("iPhone calibration - LIVE", 960, 540)

    while len(objpoints) < TARGET_CAPTURES:
        ok, frame = cap.read()
        if not ok:
            print("Camera read failed - stopping.")
            break
        if ROTATE is not None:
            frame = cv2.rotate(frame, ROTATE)
        image_size = frame.shape[1], frame.shape[0]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(
            gray, (BOARD_COLS, BOARD_ROWS),
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)

        display = frame.copy()
        now = time.time()
        just_captured = False
        if found:
            corners = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            cv2.drawChessboardCorners(display, (BOARD_COLS, BOARD_ROWS), corners, found)

        if found and (now - last_capture_t) >= CAPTURE_COOLDOWN_S:
            objpoints.append(objp.copy())
            imgpoints.append(corners)
            last_capture_t = now
            just_captured = True
            print(f"captured {len(objpoints)}/{TARGET_CAPTURES} - "
                  f"now move the board to a new angle/distance")
        elif now - last_status_t > 2.0:
            print("board not visible - hold it up, flat, fully in frame"
                  if not found else "board seen, cooling down before next capture...")
            last_status_t = now

        status = f"CAPTURED! {len(objpoints)}/{TARGET_CAPTURES}" if just_captured else \
                 (f"board found - {'capturing...' if found else ''}" if found
                  else "move board into view")
        cv2.putText(display, f"{len(objpoints)}/{TARGET_CAPTURES}  {status}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0) if found else (0, 0, 255), 2)
        cv2.imshow("iPhone calibration - LIVE", display)
        cv2.waitKey(1)

    cv2.destroyAllWindows()
    cap.release()

    if len(objpoints) < MIN_CAPTURES:
        print(f"Only {len(objpoints)} good captures - need >= {MIN_CAPTURES}. "
              f"Run again.")
        return

    print("Computing calibration...")
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None)

    print(f"RMS reprojection error: {rms:.4f} px "
          f"({'good' if rms < 1.0 else 'high - consider recapturing with more/better angles'})")

    out = {
        "camera_index": CAM_INDEX,
        "image_size": list(image_size),
        "camera_matrix": K.tolist(),
        "dist_coeffs": dist.tolist(),
        "rms_reprojection_error": rms,
        "board_cols": BOARD_COLS,
        "board_rows": BOARD_ROWS,
        "square_mm": SQUARE_MM,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
