"""
Tests for platform_height.py's geometry: corner classification, the
ray/plane intersection in ground_xy_mm(), and the camera<->world point
transform. No camera use - pure synthetic geometry, with PlatformHeightEstimator
instances built directly (bypassing __init__'s cv2.solvePnP-derived R/tvec)
so the ray-plane math can be checked against hand-derived expected values
without depending on solvePnP's real calibration-time sign conventions.

Scope note: platform_height.py is NOT read by the active live pipeline -
see docs/PRODUCTION_READINESS.md #6.
"""
import numpy as np
import pytest
import cv2

from platform_height import (_classify_corners, PlatformHeightEstimator,
                             solve_marker_camera_point)


# ---------------------------------------------------------------------------
# _classify_corners: image-space quadrant sorting relative to centroid
# ---------------------------------------------------------------------------

def test_classify_corners_axis_aligned_rectangle():
    # image coords: y grows downward, so "bottom" = larger y
    box = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
    result = _classify_corners(box)
    bl, br, tl, tr = result
    assert list(bl) == [0, 100]
    assert list(br) == [100, 100]
    assert list(tl) == [0, 0]
    assert list(tr) == [100, 0]


def test_classify_corners_is_order_independent():
    shuffled = np.array([[100, 100], [0, 0], [0, 100], [100, 0]], dtype=np.float32)
    result = _classify_corners(shuffled)
    bl, br, tl, tr = result
    assert list(bl) == [0, 100]
    assert list(tr) == [100, 0]


def test_classify_corners_rotated_rectangle():
    # a rectangle rotated ~15 degrees - each corner should still land in its
    # own quadrant relative to the centroid.
    angle = np.radians(15)
    w, h = 100, 60
    local = np.array([[-w/2, -h/2], [w/2, -h/2], [w/2, h/2], [-w/2, h/2]])
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle), np.cos(angle)]])
    rotated = (local @ R.T) + np.array([200, 200])   # centroid at (200,200)
    result = _classify_corners(rotated.astype(np.float32))
    assert result is not None
    assert len(result) == 4


def test_classify_corners_degenerate_returns_none():
    # 3 points on the same side of the centroid (bottom) - no valid tl/tr
    box = np.array([[0, 100], [50, 100], [100, 100], [50, 0]], dtype=np.float32)
    assert _classify_corners(box) is None


# ---------------------------------------------------------------------------
# PlatformHeightEstimator geometry - built directly (no solvePnP) so the
# formulas can be checked against hand-derived expected values.
# ---------------------------------------------------------------------------

def _estimator(R, tvec, fx=1000.0, fy=1000.0, cx=320.0, cy=240.0):
    est = PlatformHeightEstimator.__new__(PlatformHeightEstimator)
    est.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    est.dist = np.zeros(5)
    est.tvec = np.asarray(tvec, dtype=np.float64).flatten()
    est.R = np.asarray(R, dtype=np.float64)
    est.R_inv = est.R.T
    est.fx = fx
    est.cam_center_world = -est.R_inv @ est.tvec
    return est


def test_ground_xy_mm_principal_point_hits_directly_below_camera():
    # R = identity, t = (0,0,500): a world point at (0,0,0) maps to camera
    # space (0,0,500) - i.e. this camera sits "500mm back" along Z from the
    # platform origin. The principal-point ray (straight ahead) must hit
    # exactly the platform origin.
    est = _estimator(np.eye(3), [0, 0, 500])
    xy = est.ground_xy_mm(est.K[0, 2], est.K[1, 2])   # exactly (cx, cy)
    assert xy == pytest.approx((0.0, 0.0), abs=1e-6)


def test_ground_xy_mm_off_center_pixel_scales_with_depth():
    fx = 1000.0
    est = _estimator(np.eye(3), [0, 0, 500], fx=fx)
    cx, cy = est.K[0, 2], est.K[1, 2]
    # a pixel 100px right of center -> normalized x_n = 100/1000 = 0.1;
    # at depth 500mm (t=500/direction_z=1), lateral offset = 0.1*500=50mm
    xy = est.ground_xy_mm(cx + 100, cy)
    assert xy[0] == pytest.approx(50.0, abs=1e-6)
    assert xy[1] == pytest.approx(0.0, abs=1e-6)


def test_ground_xy_mm_returns_none_when_ray_parallel_to_platform():
    est = _estimator(np.eye(3), [0, 0, 500])
    # force a direction with ~0 z-component by monkeypatching _ray_world
    est._ray_world = lambda px, py: np.array([1.0, 0.0, 0.0])
    assert est.ground_xy_mm(0, 0) is None


def test_ground_xy_mm_returns_none_when_platform_is_behind_the_ray():
    est = _estimator(np.eye(3), [0, 0, 500])
    est._ray_world = lambda px, py: np.array([0.0, 0.0, -1.0])   # facing away
    assert est.ground_xy_mm(0, 0) is None


def test_camera_point_to_world_round_trips_the_forward_transform():
    """p_cam = R @ p_world + t (solvePnP's own convention) - so feeding
    camera_point_to_world() a point built via the FORWARD transform must
    recover the original p_world exactly, for any rotation."""
    rvec = np.array([0.3, -0.2, 0.1])
    R = cv2.Rodrigues(rvec)[0]
    tvec = np.array([15.0, -8.0, 400.0])
    est = _estimator(R, tvec)

    p_world_original = np.array([37.0, -12.5, 60.0])
    p_cam = R @ p_world_original + tvec
    recovered = est.camera_point_to_world(p_cam)
    assert recovered == pytest.approx(tuple(p_world_original), abs=1e-6)


def test_point3d_mm_returns_none_for_unknown_color_or_degenerate_width():
    est = _estimator(np.eye(3), [0, 0, 500])
    assert est.point3d_mm(320, 240, 100, "purple") is None
    assert est.point3d_mm(320, 240, 1, "yellow") is None


# ---------------------------------------------------------------------------
# solve_marker_camera_point: ArUco pose -> exact camera-space center
# ---------------------------------------------------------------------------

def test_solve_marker_camera_point_recovers_known_translation():
    """Build synthetic ArUco corner pixels by projecting a KNOWN marker pose
    with cv2.projectPoints, then confirm solve_marker_camera_point recovers
    that same camera-space center (its whole point: tvec directly IS the
    marker's camera-space position, no separate readout step)."""
    K = np.array([[800.0, 0, 320.0], [0, 800.0, 240.0], [0, 0, 1]])
    dist = np.zeros(5)
    marker_size = 30.0
    s = marker_size / 2.0
    objp = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]],
                    dtype=np.float32)

    true_rvec = np.array([0.05, 0.1, -0.05])
    true_tvec = np.array([20.0, -10.0, 350.0])
    imgpts, _ = cv2.projectPoints(objp, true_rvec, true_tvec, K, dist)
    corners_px = imgpts.reshape(4, 2)

    recovered = solve_marker_camera_point(corners_px, K, dist, marker_size)
    assert recovered is not None
    assert recovered == pytest.approx(true_tvec, abs=1e-3)
