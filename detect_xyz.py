"""
detect_xyz.py — block tracker with DEPTH (z), on top of the 2D tracker.

Same colour tracking as detect.py (yellow + red only, no direction marker),
but each block now also carries a z coordinate estimated by Depth Anything 3
(see depth_estimator.py).

OSC message layout (address /obj):
    [name, x, y, angle, sizeX, sizeY, z]
                                       ^--- NEW: appended at the END so every
    existing "Get at Index 0..5" node in the Unreal Blueprint keeps working.
    Add a "Get at Index 6" for z on the UE side. z is metric depth in meters
    (smaller = closer to camera); scale/offset it in Unreal to taste. angle
    is always 0 (no direction marker) but stays in the message for index
    compatibility.

If depth can't load (no torch / no DA3 package / no GPU) the script still
runs and sends z = -1.0 as a sentinel, so the tracker degrades to plain 2D
instead of crashing.

Run:  python detect_xyz.py    (Esc to quit)
"""

import cv2
import numpy as np
from pythonosc.udp_client import SimpleUDPClient

from depth_estimator import DepthEstimator

osc = SimpleUDPClient("127.0.0.1", 7000)

# Only two colors are detected - yellow and red - so nothing else in frame
# gets mistaken for a tracked block. Red wraps around the HSV hue circle
# (0 and 180 are both "red"), so it needs two ranges merged into one mask.
YELLOW = [((20, 60, 150), (35, 255, 255))]      # the 3 yellow blocks
RED = [((0, 70, 50), (10, 255, 255)),           # red, low-hue half
       ((170, 70, 50), (179, 255, 255))]        # red, high-hue half

MIN_AREA = 600        # min block size (lower if smallest yellow is missed)

Z_MISSING = -1.0      # sentinel z when depth isn't available


# find blobs of a color → list of dicts with center, size, box, contour
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
        (cx, cy), (rw, rh), _ = cv2.minAreaRect(c)     # center + side lengths
        x, y, w, h = cv2.boundingRect(c)               # axis-aligned box
        out.append({"cx": int(cx), "cy": int(cy),
                    "sx": int(rw), "sy": int(rh),
                    "box": (x, y, w, h), "area": cv2.contourArea(c)})
    return out


def send(name, b, frame, color, depth):
    # angle stays 0 (no direction marker anymore) - kept in the OSC message so
    # existing "Get at Index 0..5" nodes on the Unreal Blueprint still line up.
    angle = 0
    z = depth.depth_at(b["box"])
    z_val = Z_MISSING if z is None else round(z, 3)
    osc.send_message("/obj", [name, b["cx"], b["cy"], angle, b["sx"], b["sy"], z_val])
    x, y, w, h = b["box"]
    z_txt = "z?" if z is None else f"z={z_val}m"
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
    cv2.putText(frame, f"{name} {b['sx']}x{b['sy']} {z_txt}", (x, y-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    print(f"{name} pos={b['cx']},{b['cy']} size={b['sx']}x{b['sy']} z={z_val}")


def main():
    # cv2.VideoCapture(0) and DepthEstimator() (which loads/downloads a
    # model) used to run at module import time - meaning `import detect_xyz`
    # alone opened a real camera AND started loading DA3. Guarding both
    # behind main()/__name__ makes this file safe to import without side
    # effects (e.g. for a future test importing find_blobs()).
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not found — try index 1 or 2")
        return

    # Start depth estimation on a background thread (see depth_estimator.py).
    depth = DepthEstimator()
    depth.start()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Hand the newest frame to the depth worker (non-blocking).
            depth.submit(frame)

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # YELLOW: sort by size, assign small/med/large -> yel1/yel2/yel3
            yellows = find_blobs(hsv, YELLOW, MIN_AREA)
            yellows.sort(key=lambda b: b["area"])          # smallest first
            labels = ["yel1", "yel2", "yel3"]
            for i, b in enumerate(yellows[:3]):            # up to 3
                send(labels[i], b, frame, (0, 255, 255), depth)

            # RED: biggest one -> red
            reds = find_blobs(hsv, RED, MIN_AREA)
            if reds:
                b = max(reds, key=lambda b: b["area"])
                send("red", b, frame, (0, 0, 255), depth)

            status = "DEPTH ON" if (depth.available and depth.has_depth()) else \
                     ("DEPTH loading..." if depth.available else "DEPTH OFF (2D only)")
            cv2.putText(frame, status, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("Detection (xyz)", frame)
            if cv2.waitKey(1) == 27:
                break
    finally:
        depth.stop()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
