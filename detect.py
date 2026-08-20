import cv2
import numpy as np
from pythonosc.udp_client import SimpleUDPClient

osc = SimpleUDPClient("127.0.0.1", 7000)

# Only two colors are detected - yellow and red - so nothing else in frame
# gets mistaken for a tracked block. Red wraps around the HSV hue circle
# (0 and 180 are both "red"), so it needs two ranges merged into one mask.
YELLOW = [((20, 60, 150), (35, 255, 255))]      # the 3 yellow blocks
RED = [((0, 70, 50), (10, 255, 255)),           # red, low-hue half
       ((170, 70, 50), (179, 255, 255))]        # red, high-hue half

MIN_AREA = 600        # min block size (lower if smallest yellow is missed)

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

def send(name, b, frame, color):
    # angle stays 0 (no direction marker anymore) - kept in the OSC message so
    # existing "Get at Index 0..5" nodes on the Unreal Blueprint still line up.
    angle = 0
    osc.send_message("/obj", [name, b["cx"], b["cy"], angle, b["sx"], b["sy"]])
    x, y, w, h = b["box"]
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
    cv2.putText(frame, f"{name} {b['sx']}x{b['sy']}", (x, y-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    print(f"{name} pos={b['cx']},{b['cy']} size={b['sx']}x{b['sy']}")

while True:
    ok, frame = cap.read()
    if not ok:
        break
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # YELLOW: sort by size, assign small/med/large -> yel1/yel2/yel3
    yellows = find_blobs(hsv, YELLOW, MIN_AREA)
    yellows.sort(key=lambda b: b["area"])              # smallest first
    labels = ["yel1", "yel2", "yel3"]
    for i, b in enumerate(yellows[:3]):                # up to 3
        send(labels[i], b, frame, (0, 255, 255))

    # RED: biggest one -> red
    reds = find_blobs(hsv, RED, MIN_AREA)
    if reds:
        b = max(reds, key=lambda b: b["area"])
        send("red", b, frame, (0, 0, 255))

    cv2.imshow("Detection", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()