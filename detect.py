import cv2
import numpy as np
import math
from pythonosc.udp_client import SimpleUDPClient

osc = SimpleUDPClient("127.0.0.1", 7000)

YELLOW = [((20, 60, 150), (35, 255, 255))]      # the 3 yellow blocks
SILVER = [((0, 0, 140), (179, 40, 255))]        # the 1 silver block
MARKER = [((140, 40, 120), (172, 180, 255))]    # pink = direction marker only

MIN_AREA = 600        # min block size (lower if smallest yellow is missed)
MARK_MIN = 150        # min marker size

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera not found — try index 1 or 2")
    exit()

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

# angle 0..360 from a block center to the marker sitting inside it
def angle_from_marker(block, markers):
    x, y, w, h = block["box"]
    for mx, my in markers:
        if x <= mx <= x+w and y <= my <= y+h:          # marker on THIS block
            return int(math.degrees(math.atan2(my-block["cy"], mx-block["cx"])) % 360)
    return 0                                           # no marker seen

def send(name, b, angle, frame, color):
    osc.send_message("/obj", [name, b["cx"], b["cy"], angle, b["sx"], b["sy"]])
    x, y, w, h = b["box"]
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
    cv2.putText(frame, f"{name} {angle}deg {b['sx']}x{b['sy']}", (x, y-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    print(f"{name} pos={b['cx']},{b['cy']} ang={angle} size={b['sx']}x{b['sy']}")

while True:
    ok, frame = cap.read()
    if not ok:
        break
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # markers first (just their centers)
    markers = [(m["cx"], m["cy"]) for m in find_blobs(hsv, MARKER, MARK_MIN)]
    for mx, my in markers:
        cv2.circle(frame, (mx, my), 5, (255, 0, 255), -1)

    # YELLOW: sort by size, assign small/med/large -> yel1/yel2/yel3
    yellows = find_blobs(hsv, YELLOW, MIN_AREA)
    yellows.sort(key=lambda b: b["area"])              # smallest first
    labels = ["yel1", "yel2", "yel3"]
    for i, b in enumerate(yellows[:3]):                # up to 3
        send(labels[i], b, angle_from_marker(b, markers), frame, (0, 255, 255))

    # SILVER: biggest one -> slv
    silvers = find_blobs(hsv, SILVER, MIN_AREA)
    if silvers:
        b = max(silvers, key=lambda b: b["area"])
        send("slv", b, angle_from_marker(b, markers), frame, (200, 200, 200))

    cv2.imshow("Detection", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()