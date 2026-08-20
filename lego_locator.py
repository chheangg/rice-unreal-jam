"""
lego_locator.py - universal Lego colour locator (step 1: reliable detection).

Goal of this stage: find EVERY Lego piece by colour, reliably, and NOT be
fooled by skin/faces/background. No OSC, no ArUco yet - just prove detection
is trustworthy on the live camera, then we build on it.

Why the face gets detected as "red":
    Skin sits in the SAME hue band as red/orange plastic. What separates them
    is SATURATION - Lego plastic is vividly saturated (S ~ 150-255), skin is
    muted (S ~ 40-130). So the single most useful knob is a saturation floor:
    raise it until your face disappears from the mask but the brick stays.

That floor is a live slider here, because the exact value depends on your
lighting. Tune it once against your actual face + bricks, read the number off
the window, and bake it in.

Controls (focus the video window):
    S floor / V floor sliders  - global saturation / brightness minimums
    Min area slider            - ignore blobs smaller than this (x100 px^2)
    m                          - cycle mask view: off -> combined -> per-colour
    [ / ]                      - previous / next colour when viewing per-colour
    p                          - print the current tuned numbers to the console
    Esc / q                    - quit

Run:
    python lego_locator.py            # default camera (index 0)
    python lego_locator.py 2          # camera index 2
    python lego_locator.py "http://172.20.10.1:8081/video"   # phone stream
    python lego_locator.py "clip.mp4"                        # a recorded file
"""

import sys
import time

import cv2
import numpy as np

# Per-colour HUE bands only (low_H, high_H). Saturation/Value floors come from
# the sliders, shared across every colour, because skin rejection is a
# saturation problem and it's easier to reason about one global floor than six.
# Red wraps the hue circle (0 and 179 are both red), so it gets two bands.
COLORS = [
    {"name": "red",    "draw": (0, 0, 255),   "hue": [(0, 8), (170, 179)]},
    {"name": "orange", "draw": (0, 140, 255), "hue": [(9, 19)]},
    {"name": "yellow", "draw": (0, 255, 255), "hue": [(20, 33)]},
    {"name": "green",  "draw": (0, 200, 0),   "hue": [(40, 85)]},
    {"name": "blue",   "draw": (255, 120, 0), "hue": [(95, 130)]},
]

# Defaults chosen to reject skin out of the box; tune with the sliders.
DEFAULT_S_FLOOR = 140      # saturation minimum - the main skin-rejection knob
DEFAULT_V_FLOOR = 70       # brightness minimum - drops dark shadow
DEFAULT_MIN_AREA_100 = 8   # min blob area, in units of 100 px^2 -> 800 px^2

KERNEL = np.ones((5, 5), np.uint8)


def as_source(text):
    try:
        return int(text)
    except ValueError:
        return text


def color_mask(hsv, color, s_floor, v_floor):
    """Binary mask for one colour: its hue band(s), floored on S and V."""
    mask = None
    for lo_h, hi_h in color["hue"]:
        lo = np.array((lo_h, s_floor, v_floor))
        hi = np.array((hi_h, 255, 255))
        m = cv2.inRange(hsv, lo, hi)
        mask = m if mask is None else (mask | m)
    # open removes speckle, close fills pinholes inside a brick
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL)
    return mask


def find_pieces(hsv, s_floor, v_floor, min_area):
    """Return {color_name: [contours]} and {color_name: mask}."""
    pieces, masks = {}, {}
    for c in COLORS:
        mask = color_mask(hsv, c, s_floor, v_floor)
        masks[c["name"]] = mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        pieces[c["name"]] = [ct for ct in contours
                             if cv2.contourArea(ct) >= min_area]
    return pieces, masks


def main():
    src = as_source(sys.argv[1]) if len(sys.argv) > 1 else 0
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"FAILED to open {src!r}")
        return 1

    win = "Lego locator"
    cv2.namedWindow(win)
    cv2.createTrackbar("S floor", win, DEFAULT_S_FLOOR, 255, lambda v: None)
    cv2.createTrackbar("V floor", win, DEFAULT_V_FLOOR, 255, lambda v: None)
    cv2.createTrackbar("Min area/100", win, DEFAULT_MIN_AREA_100, 100, lambda v: None)

    mask_mode = 0                 # 0 off, 1 combined, 2 per-colour
    color_idx = 0                 # which colour when mask_mode == 2
    t_prev, fps = time.time(), 0.0
    print("running - focus the window, raise 'S floor' until your face stops "
          "being detected but the bricks stay. 'p' prints the numbers.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("no more frames"); break

        s_floor = cv2.getTrackbarPos("S floor", win)
        v_floor = cv2.getTrackbarPos("V floor", win)
        min_area = cv2.getTrackbarPos("Min area/100", win) * 100

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        pieces, masks = find_pieces(hsv, s_floor, v_floor, min_area)

        counts = {}
        for c in COLORS:
            cnt = pieces[c["name"]]
            counts[c["name"]] = len(cnt)
            for ct in cnt:
                x, y, w, h = cv2.boundingRect(ct)
                (cx, cy), (rw, rh), _ = cv2.minAreaRect(ct)
                cv2.rectangle(frame, (x, y), (x + w, y + h), c["draw"], 2)
                cv2.putText(frame, f"{c['name']} {int(rw)}x{int(rh)}",
                            (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            c["draw"], 2)

        # HUD
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-6))
        t_prev = now
        summary = "  ".join(f"{n}:{counts[n]}" for n in
                            [c["name"] for c in COLORS])
        cv2.putText(frame, summary, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"S>={s_floor} V>={v_floor} area>={min_area} "
                    f"{fps:.0f}fps", (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (200, 200, 200), 2)

        if mask_mode == 1:
            combined = None
            for c in COLORS:
                combined = masks[c["name"]] if combined is None \
                    else (combined | masks[c["name"]])
            cv2.imshow("mask", combined)
        elif mask_mode == 2:
            name = COLORS[color_idx]["name"]
            view = masks[name].copy()
            cv2.putText(view, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, 255, 2)
            cv2.imshow("mask", view)

        cv2.imshow(win, frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break
        elif key == ord('m'):
            mask_mode = (mask_mode + 1) % 3
            if mask_mode == 0:
                cv2.destroyWindow("mask")
        elif key == ord(']'):
            color_idx = (color_idx + 1) % len(COLORS)
        elif key == ord('['):
            color_idx = (color_idx - 1) % len(COLORS)
        elif key == ord('p'):
            print(f"S_FLOOR={s_floor}  V_FLOOR={v_floor}  "
                  f"MIN_AREA={min_area}  counts={counts}")

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
