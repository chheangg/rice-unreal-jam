"""
Tests for side_height.py's SideHeightEstimator - all file I/O against
fabricated JSON in a temp directory, no camera, no real calibration data.

Scope note: this module is NOT read by the active live pipeline
(lego_locator_xyz.py / detect_xyz.py) - see docs/PRODUCTION_READINESS.md
#6. These tests verify the depth-from-known-width + back-projection math
in isolation (using R=identity, t=0, so world space == camera space,
sidestepping the real calibration's Z-axis sign convention, which depends
on chessboard corner ordering at calibration time and can't be
independently re-derived from a synthetic test - that part is validated by
calibrate_table_pose.py's own live sanity-check print instead). What IS
independently verifiable and tested here: the similar-triangles depth
formula, the undistort+back-project chain, and the simple-mode fallback.
"""
import json

import numpy as np
import pytest

from side_height import SideHeightEstimator, BASELINE_Y_PX, PX_PER_MM_GUESS


def _write_identity_calibration(tmp_path, fx=1000.0, fy=1000.0,
                                cx=320.0, cy=240.0):
    intr_path = tmp_path / "iphone_intrinsics.json"
    pose_path = tmp_path / "table_pose.json"
    with open(intr_path, "w") as f:
        json.dump({
            "camera_matrix": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
            "dist_coeffs": [[0.0, 0.0, 0.0, 0.0, 0.0]],
            "board_cols": 9, "board_rows": 6, "square_mm": 25.0,
        }, f)
    with open(pose_path, "w") as f:
        json.dump({
            "rvec": [0.0, 0.0, 0.0],   # R = identity
            "tvec": [0.0, 0.0, 0.0],   # t = 0 -> p_world == p_cam
        }, f)
    return str(intr_path), str(pose_path)


def test_falls_back_to_simple_mode_when_calibration_files_missing(tmp_path):
    est = SideHeightEstimator(
        intrinsics_path=str(tmp_path / "nope1.json"),
        pose_path=str(tmp_path / "nope2.json"))
    assert est.available
    assert est.simple_mode
    assert est.last_error is not None


def test_simple_mode_formula():
    est = SideHeightEstimator.__new__(SideHeightEstimator)
    est.available = True
    est.simple_mode = True
    est.last_error = None
    h = est.height_mm(cx=0, cy=BASELINE_Y_PX - 30, pixel_width=10,
                      color_name="yellow")
    assert h == pytest.approx(30 / PX_PER_MM_GUESS)


def test_depth_from_known_width_matches_similar_triangles(tmp_path):
    intr_path, pose_path = _write_identity_calibration(tmp_path, fx=1000.0,
                                                        cx=320.0, cy=240.0)
    est = SideHeightEstimator(intr_path, pose_path)
    assert est.available and not est.simple_mode

    # centered pixel (cx=320,cy=240 -> x_n=y_n=0) so p_cam = (0, 0, depth_mm)
    # and R=I, t=0 means p_world == p_cam, sidestepping any rotation
    # ambiguity - this isolates just the depth-from-known-size formula.
    pixel_width = 100.0
    expected_depth = est.fx * 48.0 / pixel_width   # yellow: REAL_WIDTH_MM=48
    h = est.height_mm(cx=320.0, cy=240.0, pixel_width=pixel_width,
                      color_name="yellow")
    assert h == pytest.approx(expected_depth)


def test_depth_scales_inversely_with_apparent_pixel_width(tmp_path):
    intr_path, pose_path = _write_identity_calibration(tmp_path)
    est = SideHeightEstimator(intr_path, pose_path)
    h_near = est.height_mm(320.0, 240.0, pixel_width=200.0, color_name="yellow")
    h_far = est.height_mm(320.0, 240.0, pixel_width=100.0, color_name="yellow")
    # a SMALLER apparent width means the object is FARTHER away (bigger
    # depth) - halving pixel_width should double the computed depth.
    assert h_far == pytest.approx(2 * h_near)


def test_off_center_pixel_does_not_change_z_depth_when_untilted(tmp_path):
    """With R=identity, t=0, only the depth (z) component should be
    affected by REAL_WIDTH/pixel_width - x/y offset changes p_cam's x,y,
    not the height_mm() result, since height_mm reads only p_world[2]."""
    intr_path, pose_path = _write_identity_calibration(tmp_path, fx=1000.0,
                                                        cx=320.0, cy=240.0)
    est = SideHeightEstimator(intr_path, pose_path)
    h_centered = est.height_mm(320.0, 240.0, pixel_width=150.0, color_name="red")
    h_offcenter = est.height_mm(370.0, 200.0, pixel_width=150.0, color_name="red")
    assert h_centered == pytest.approx(h_offcenter)


def test_unknown_color_returns_none(tmp_path):
    intr_path, pose_path = _write_identity_calibration(tmp_path)
    est = SideHeightEstimator(intr_path, pose_path)
    assert est.height_mm(320.0, 240.0, 100.0, "purple") is None


def test_degenerate_pixel_width_returns_none(tmp_path):
    intr_path, pose_path = _write_identity_calibration(tmp_path)
    est = SideHeightEstimator(intr_path, pose_path)
    assert est.height_mm(320.0, 240.0, pixel_width=1.0, color_name="yellow") is None
    assert est.height_mm(320.0, 240.0, pixel_width=0.0, color_name="yellow") is None


def test_unavailable_estimator_returns_none():
    est = SideHeightEstimator.__new__(SideHeightEstimator)
    est.available = False
    assert est.height_mm(0, 0, 100, "yellow") is None
