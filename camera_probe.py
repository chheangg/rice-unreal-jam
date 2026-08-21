"""
camera_probe.py - check what OpenCV can actually see, before touching detect.py.

Answers three questions you need settled before wiring a phone (or any new
camera) into the tracker:
  1. Can OpenCV open this source at all?
  2. What resolution does it deliver?
  3. What frame rate does it ACTUALLY deliver (not what it advertises)?

Question 3 is the one that matters for a phone over wifi: an IP camera app can
report 30 fps and deliver 6, which makes tracking feel broken in a way that
looks like a tracking bug rather than a network problem.

Run:
    python camera_probe.py                    # scan indices 0..5, report what opens
    python camera_probe.py 0                  # live preview of index 0 (built-in cam)
    python camera_probe.py 1                  # live preview of index 1 (virtual webcam)
    python camera_probe.py http://192.168.1.42:8080/video     # iPhone IP camera app
Esc or q to quit the preview.

iPhone note: Continuity Camera is macOS-only, so on Windows you want either an
IP-camera app (gives you the http:// URL form above) or a virtual-webcam driver
like Iriun/EpocCam (gives you a numeric index). For a demo, put the phone and
laptop on the phone's Personal Hotspot - a direct private link that doesn't
depend on venue wifi letting two devices see each other.
"""

import sys
import time

import cv2

SCAN_LIMIT = 6          # indices 0..5 when scanning
WARMUP_FRAMES = 5       # first frames are often slow/garbage; don't time them
MEASURE_SECONDS = 3.0   # how long to measure real fps in the preview


def as_source(text):
    """Camera index if it looks like a number, otherwise a URL/path."""
    try:
        return int(text)
    except ValueError:
        return text


def open_source(src):
    # Plain default backend. CAP_DSHOW is tempting on Windows but OpenCV 5
    # refuses to capture by index through DirectShow ("backend is generally
    # available but can't be used to capture by index"), so it only adds
    # warning noise and a failed probe per index.
    return cv2.VideoCapture(src)


def describe(cap):
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    claimed = cap.get(cv2.CAP_PROP_FPS)
    return w, h, claimed


def scan():
    print(f"scanning camera indices 0..{SCAN_LIMIT - 1}\n")
    found = []
    for i in range(SCAN_LIMIT):
        cap = open_source(i)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                w, h, claimed = describe(cap)
                print(f"  [{i}] OPEN   {w}x{h}  claims {claimed:g} fps")
                found.append(i)
            else:
                print(f"  [{i}] opens but delivers no frames (in use by another app?)")
        cap.release()

    print()
    if not found:
        print("nothing found. if you're expecting a phone here:")
        print("  - virtual-webcam driver (Iriun/EpocCam): is the app running on BOTH")
        print("    phone and laptop, and are they on the same network?")
        print("  - IP-camera app: it won't appear as an index at all - pass the URL:")
        print("      python camera_probe.py http://<phone-ip>:8080/video")
    else:
        print(f"working indices: {found}")
        print(f"preview one with:  python camera_probe.py {found[-1]}")
    return found


def preview(src):
    print(f"opening {src!r} ...")
    t0 = time.time()
    cap = open_source(src)
    if not cap.isOpened():
        print(f"FAILED to open {src!r}")
        if not isinstance(src, int):
            print("  - is the phone app still in the foreground and streaming?")
            print("  - open the same URL in a browser on this laptop to confirm it's reachable")
            print("  - if the browser can't reach it either, the network is blocking you:")
            print("    put both devices on the phone's Personal Hotspot instead")
        return 1
    print(f"opened in {time.time() - t0:.2f}s")

    for _ in range(WARMUP_FRAMES):
        cap.read()

    w, h, claimed = describe(cap)
    print(f"resolution {w}x{h}, source claims {claimed:g} fps")
    print(f"measuring real fps over {MEASURE_SECONDS:g}s ... (Esc or q to quit)")

    frames = 0
    dropped = 0
    start = time.time()
    measured = None

    while True:
        ok, frame = cap.read()
        if not ok:
            dropped += 1
            if dropped > 30:
                print("source stopped delivering frames - giving up")
                break
            continue

        frames += 1
        elapsed = time.time() - start
        if measured is None and elapsed >= MEASURE_SECONDS:
            measured = frames / elapsed
            print(f"\n  REAL fps: {measured:.1f}  ({frames} frames in {elapsed:.1f}s)")
            if measured < 10:
                print("  that is too slow for responsive tracking. try a lower")
                print("  resolution in the phone app, or a wired/hotspot link.")
            elif measured < 20:
                print("  usable but not smooth; a direct hotspot link would help.")
            else:
                print("  good enough for the tracker.")
            print("\n  use this source in the tracker:")
            shown = src if isinstance(src, int) else f'"{src}"'
            print(f"      cap = cv2.VideoCapture({shown})")

        live = frames / elapsed if elapsed > 0 else 0
        cv2.putText(frame, f"{w}x{h}  {live:.1f} fps", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        if dropped:
            cv2.putText(frame, f"dropped reads: {dropped}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        cv2.imshow("camera probe (Esc/q to quit)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0


def main():
    if len(sys.argv) < 2:
        scan()
        return 0
    return preview(as_source(sys.argv[1]))


if __name__ == "__main__":
    sys.exit(main())
