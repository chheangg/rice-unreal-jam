"""
detect_stereo.py — two-camera tracker: DJI (top-down) gives X/Y, iPhone
(side) gives real geometric height (Z), combined and sent over OSC.

Requires calibrate_iphone.py and calibrate_table_pose.py to have been run
already (produces iphone_intrinsics.json + table_pose.json in this folder).

Camera indices per our setup: DJI = 0 (top-down), iPhone = 1 (side, via
Continuity Camera). Change DJI_INDEX / IPHONE_INDEX below if your setup
differs - re-check indices with a snapshot probe if devices reconnect in a
different order (macOS doesn't guarantee stable indices across reconnects).

Matching yellow blocks between the two views: there's no per-block ID (see
docs/ROADMAP.md's ArUco note for the robust fix used on the lego-shape-snap
branch), so yellow blocks are matched by LEFT-TO-RIGHT RANK: the Nth-from-
left yellow block in the DJI view is paired with the Nth-from-left yellow
block in the iPhone view. This only holds if the iPhone is positioned so its
left-right roughly matches the table's left-right as seen from above.

OSC (address /obj): [name, x, y, angle, sizeX, sizeY, z]
  x, y, sizeX, sizeY - from the DJI (top-down), exactly as detect.py.
  angle              - always 0 (no direction marker, see detect.py).
  z                  - real height above table in mm, from the iPhone.
                       -1.0 sentinel if the iPhone doesn't currently see a
                       matching block, or calibration hasn't been run yet.

Run:  python detect_stereo.py    (Esc to quit)
"""

import cv2
import numpy as np
from pythonosc.udp_client import SimpleUDPClient

from side_height import SideHeightEstimator

osc = SimpleUDPClient("127.0.0.1", 7000)

DJI_INDEX = 0
IPHONE_INDEX = 1

YELLOW = [((20, 60, 150), (35, 255, 255))]
RED = [((0, 70, 50), (10, 255, 255)),
       ((170, 70, 50), (179, 255, 255))]

MIN_AREA = 600
Z_MISSING = -1.0


def find_blobs(hsv, ranges, min_area):
    mask = None
    for lo, hi in ranges:
        m = cv2.inRange(hsv, np.array(lo), np.array(hi))
        mask = m if mask is None else mask | m
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        (cx, cy), (rw, rh), _ = cv2.minAreaRect(c)
        x, y, w, h = cv2.boundingRect(c)
        out.append({"cx": int(cx), "cy": int(cy),
                    "sx": int(rw), "sy": int(rh),
                    "box": (x, y, w, h), "area": cv2.contourArea(c)})
    return out


def match_by_rank(top_blocks, side_blobs):
    """Pair top-view blocks with side-view blobs by left-to-right rank.
    Returns {top_index: side_blob}."""
    top_order = sorted(range(len(top_blocks)), key=lambda i: top_blocks[i]["cx"])
    side_order = sorted(side_blobs, key=lambda b: b["cx"])
    pairs = {}
    for rank, top_i in enumerate(top_order):
        if rank < len(side_order):
            pairs[top_i] = side_order[rank]
    return pairs


def send_and_draw(name, b, z, top, color):
    osc.send_message("/obj", [name, b["cx"], b["cy"], 0, b["sx"], b["sy"], z])
    x, y, w, h = b["box"]
    cv2.rectangle(top, (x, y), (x + w, y + h), color, 2)
    cv2.putText(top, f"{name} z={z}mm", (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    print(f"{name} pos={b['cx']},{b['cy']} size={b['sx']}x{b['sy']} z={z}mm")


def main():
    dji = cv2.VideoCapture(DJI_INDEX)
    iphone = cv2.VideoCapture(IPHONE_INDEX)
    if not dji.isOpened():
        print(f"DJI camera not found at index {DJI_INDEX}")
        return
    if not iphone.isOpened():
        print(f"iPhone camera not found at index {IPHONE_INDEX} - "
              f"is Continuity Camera active?")
        return

    height_est = SideHeightEstimator()
    if not height_est.available:
        print("[detect_stereo] side height calibration missing - z will be "
              "sentinel. Run calibrate_iphone.py then calibrate_table_pose.py.")

    while True:
        ok_t, top = dji.read()
        ok_s, side = iphone.read()
        if not ok_t or not ok_s:
            break

        hsv_t = cv2.cvtColor(top, cv2.COLOR_BGR2HSV)
        hsv_s = cv2.cvtColor(side, cv2.COLOR_BGR2HSV)

        # ---- top-down (DJI): x, y, labels - same scheme as detect.py ----
        yellows_t = find_blobs(hsv_t, YELLOW, MIN_AREA)
        yellows_t.sort(key=lambda b: b["area"])
        yellow_blocks = yellows_t[:3]

        reds_t = find_blobs(hsv_t, RED, MIN_AREA)
        red_block = max(reds_t, key=lambda b: b["area"]) if reds_t else None

        # ---- side (iPhone): same colors, used only for height ----
        yellows_s = find_blobs(hsv_s, YELLOW, MIN_AREA)
        reds_s = find_blobs(hsv_s, RED, MIN_AREA)

        yellow_pairs = match_by_rank(yellow_blocks, yellows_s)
        labels = ["yel1", "yel2", "yel3"]
        for i, b in enumerate(yellow_blocks):
            side_blob = yellow_pairs.get(i)
            z = Z_MISSING
            if side_blob is not None:
                h = height_est.height_mm(side_blob["cx"], side_blob["cy"],
                                          side_blob["sx"], "yellow")
                if h is not None:
                    z = round(h, 1)
            send_and_draw(labels[i], b, z, top, (0, 255, 255))

        if red_block is not None:
            side_blob = max(reds_s, key=lambda b: b["area"]) if reds_s else None
            z = Z_MISSING
            if side_blob is not None:
                h = height_est.height_mm(side_blob["cx"], side_blob["cy"],
                                          side_blob["sx"], "red")
                if h is not None:
                    z = round(h, 1)
            send_and_draw("red", red_block, z, top, (0, 0, 255))

        cv2.imshow("DJI (top - x/y)", top)
        cv2.imshow("iPhone (side - z source)", side)
        if cv2.waitKey(1) == 27:
            break

    dji.release()
    iphone.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
