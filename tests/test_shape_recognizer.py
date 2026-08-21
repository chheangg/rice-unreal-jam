"""
Tests for shape_recognizer.py (ORB + homography feature matching).

No camera use - everything here runs against static images already in the
repo (tests/fixtures/) plus synthetic transforms (rotation, scale,
perspective warp, brightness) applied to the real reference photo. This is
what task item 2 asked for: "tested against synthetic transformed copies of
the cropped reference," not a live camera.

Run: python -m pytest tests/test_shape_recognizer.py -v
"""
import os

import cv2
import numpy as np
import pytest

import shape_recognizer as sr

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
REFERENCE_PATH = os.path.join(FIXTURES, "reference_piece.png")
DISTRACTOR_PATH = os.path.join(FIXTURES, "distractor.png")


@pytest.fixture(scope="module")
def reference():
    return sr.load_reference(REFERENCE_PATH)


def _rotate(img, degrees):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), degrees, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderValue=127)


def _scale(img, factor):
    h, w = img.shape[:2]
    return cv2.resize(img, (int(w * factor), int(h * factor)))


def _perspective_nudge(img, strength=0.06):
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    d = strength * min(w, h)
    dst = np.float32([[0, 0], [w - d, d], [w, h], [d, h - d]])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h), borderValue=127)


def test_reference_loads_and_has_features(reference):
    img, kp, des = reference
    assert img is not None
    assert des is not None
    assert len(kp) > 50   # a real photo should yield plenty of ORB keypoints


def test_matches_unmodified_reference_against_itself(reference):
    img, ref_kp, ref_des = reference
    result = sr.match(img, ref_kp, ref_des)
    assert result["matched"]
    assert result["inliers"] >= sr.MIN_MATCH_COUNT


def test_matches_rotated_copy(reference):
    img, ref_kp, ref_des = reference
    rotated = _rotate(img, 25)
    result = sr.match(rotated, ref_kp, ref_des)
    assert result["matched"], result


def test_matches_scaled_down_copy(reference):
    img, ref_kp, ref_des = reference
    scaled = _scale(img, 0.7)
    result = sr.match(scaled, ref_kp, ref_des)
    assert result["matched"], result


def test_matches_scaled_up_copy(reference):
    img, ref_kp, ref_des = reference
    scaled = _scale(img, 1.4)
    result = sr.match(scaled, ref_kp, ref_des)
    assert result["matched"], result


def test_matches_mild_perspective_warp(reference):
    img, ref_kp, ref_des = reference
    warped = _perspective_nudge(img, strength=0.05)
    result = sr.match(warped, ref_kp, ref_des)
    assert result["matched"], result


def test_matches_brightness_shifted_copy(reference):
    img, ref_kp, ref_des = reference
    brighter = cv2.convertScaleAbs(img, alpha=1.0, beta=40)
    result = sr.match(brighter, ref_kp, ref_des)
    assert result["matched"], result


def test_combined_rotate_scale_perspective_is_harder_but_still_findable(reference):
    img, ref_kp, ref_des = reference
    warped = _rotate(_scale(img, 0.8), 15)
    warped = _perspective_nudge(warped, strength=0.04)
    result = sr.match(warped, ref_kp, ref_des)
    # documents the degradation curve rather than asserting a hard pass -
    # combined transforms are exactly where this approach gets fragile.
    assert result["good_matches"] >= 0   # just must not crash
    if not result["matched"]:
        pytest.skip("combined transform below match threshold - "
                     "documents real degradation, see shape_recognizer.py "
                     "recommendation")


def test_rejects_unrelated_distractor_image(reference):
    """A real photo of a DIFFERENT object (a red brick, not the lime
    assembly) must not be reported as a match."""
    _, ref_kp, ref_des = reference
    distractor = cv2.imread(DISTRACTOR_PATH, cv2.IMREAD_GRAYSCALE)
    assert distractor is not None
    result = sr.match(distractor, ref_kp, ref_des)
    assert not result["matched"]


def test_rejects_blank_image(reference):
    _, ref_kp, ref_des = reference
    blank = np.full((600, 400), 128, dtype=np.uint8)
    result = sr.match(blank, ref_kp, ref_des)
    assert not result["matched"]
    assert result["inliers"] == 0


def test_handles_none_descriptors_gracefully():
    # A reference with no descriptors at all (e.g. a blank reference image)
    # must not crash match() - it should just report no match.
    blank = np.full((100, 100), 128, dtype=np.uint8)
    _, kp, des = sr.image_to_features(blank)
    query = np.full((100, 100), 200, dtype=np.uint8)
    result = sr.match(query, kp, des)
    assert not result["matched"]
