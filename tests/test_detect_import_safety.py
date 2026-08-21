"""
Regression tests: detect.py and detect_xyz.py used to open a real camera
(cv2.VideoCapture(0)) - and detect_xyz.py additionally loaded/downloaded a
Depth Anything 3 model - as a MODULE-LEVEL side effect, meaning a bare
`import detect` or `import detect_xyz` (from anywhere - an IDE, a linter,
a future test suite) would touch real hardware / start a network fetch.
Both are now guarded behind main()/`if __name__ == "__main__"`.

This test's entire point is importing these modules and confirming nothing
happens - no camera, no model load, no window. If either regresses back to
module-level `cv2.VideoCapture(...)`, this test will hang or fail loudly
instead of silently opening a camera during a routine test run.
"""
import sys
import time

import numpy as np
import pytest


def test_import_detect_has_no_side_effects():
    sys.modules.pop("detect", None)
    t0 = time.time()
    import detect
    elapsed = time.time() - t0
    assert elapsed < 5.0            # a real camera open would visibly stall
    assert hasattr(detect, "find_blobs")
    assert hasattr(detect, "main")
    assert callable(detect.main)


def test_import_detect_xyz_has_no_side_effects():
    sys.modules.pop("detect_xyz", None)
    t0 = time.time()
    import detect_xyz
    elapsed = time.time() - t0
    # a real camera open OR a DA3 model load/download would take far
    # longer than this (model load prints "[depth] loading..." and can
    # take seconds to minutes on first run) - fast import proves neither
    # ran as an import side effect.
    assert elapsed < 5.0
    assert hasattr(detect_xyz, "find_blobs")
    assert hasattr(detect_xyz, "main")
    assert callable(detect_xyz.main)


def test_detect_find_blobs_detects_synthetic_yellow_square():
    import detect
    hsv = np.zeros((200, 200, 3), dtype=np.uint8)
    hsv[50:100, 50:100] = (27, 200, 200)   # inside detect.YELLOW's hue band
    blobs = detect.find_blobs(hsv, detect.YELLOW, min_area=100)
    assert len(blobs) == 1
    assert blobs[0]["cx"] == pytest.approx(74, abs=2)
    assert blobs[0]["cy"] == pytest.approx(74, abs=2)


def test_detect_find_blobs_ignores_blobs_below_min_area():
    import detect
    hsv = np.zeros((200, 200, 3), dtype=np.uint8)
    hsv[50:55, 50:55] = (27, 200, 200)     # tiny 5x5 blob
    blobs = detect.find_blobs(hsv, detect.YELLOW, min_area=1000)
    assert blobs == []


def test_detect_find_blobs_red_wraps_hue_circle():
    """RED spans two hue ranges (near 0 and near 179) merged into one mask -
    both ends of the wrap must be detected."""
    import detect
    hsv = np.zeros((100, 200, 3), dtype=np.uint8)
    hsv[20:60, 10:50] = (2, 200, 200)      # low-hue red
    hsv[20:60, 100:140] = (175, 200, 200)  # high-hue red
    blobs = detect.find_blobs(hsv, detect.RED, min_area=100)
    assert len(blobs) == 2


def test_detect_xyz_find_blobs_matches_detect_behavior():
    """detect_xyz.py's find_blobs is a near-duplicate of detect.py's - just
    confirm it independently gives sane output on the same synthetic input,
    since it's a separate copy of the function, not a shared import."""
    import detect_xyz
    hsv = np.zeros((200, 200, 3), dtype=np.uint8)
    hsv[50:100, 50:100] = (27, 200, 200)
    blobs = detect_xyz.find_blobs(hsv, detect_xyz.YELLOW, min_area=100)
    assert len(blobs) == 1
    assert blobs[0]["sx"] > 0 and blobs[0]["sy"] > 0
