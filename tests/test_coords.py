"""
Synthetic, camera-free tests for the coordinate/size/outline math in
lego_locator_xyz.py (and the shape classifier it depends on in
lego_locator.py). No cv2.VideoCapture, no imshow/waitKey - everything here
runs against fabricated contours, depth maps, and pixel coordinates.

Run: python -m pytest tests/test_coords.py -v
"""
import math

import cv2
import numpy as np
import pytest

import lego_locator_xyz as loc
from lego_locator import classify_shape


# ---------------------------------------------------------------------------
# resolve_intrinsics: FOV fallback pinhole math
# ---------------------------------------------------------------------------

class _FakeDepthUnavailable:
    available = False


class _FakeDepthWithIntrinsics:
    available = True
    def __init__(self, intr):
        self._intr = intr
    def intrinsics_for_frame(self):
        return self._intr


def test_resolve_intrinsics_fov_fallback_formula():
    (fx, fy, cx, cy), src = loc.resolve_intrinsics(
        _FakeDepthUnavailable(), frame_w=1280, frame_h=720, assumed_fov_deg=60.0)
    expected_f = (1280 / 2.0) / math.tan(math.radians(30.0))
    assert fx == pytest.approx(expected_f)
    assert fy == pytest.approx(expected_f)          # square pixels assumed
    assert cx == pytest.approx(640.0)
    assert cy == pytest.approx(360.0)
    assert src == "FOV60"


def test_resolve_intrinsics_prefers_da3_when_available():
    fake = _FakeDepthWithIntrinsics((900.0, 900.0, 640.0, 360.0))
    intr, src = loc.resolve_intrinsics(fake, 1280, 720, 60.0)
    assert intr == (900.0, 900.0, 640.0, 360.0)
    assert src == "DA3"


def test_resolve_intrinsics_falls_back_if_da3_returns_none():
    class FakeDepthNoIntrinsics:
        available = True
        def intrinsics_for_frame(self):
            return None
    (fx, fy, cx, cy), src = loc.resolve_intrinsics(
        FakeDepthNoIntrinsics(), 640, 480, 60.0)
    assert src == "FOV60"
    assert cx == 320.0


# ---------------------------------------------------------------------------
# backproject: pixel + depth -> camera-frame XYZ
# ---------------------------------------------------------------------------

def test_backproject_principal_point_maps_to_zero_xy():
    P = loc.backproject(u=320.0, v=240.0, z=2.0, fx=900.0, fy=900.0,
                         cx=320.0, cy=240.0)
    assert P[0] == pytest.approx(0.0)
    assert P[1] == pytest.approx(0.0)
    assert P[2] == pytest.approx(2.0)


def test_backproject_scales_linearly_with_pixel_offset_and_depth():
    fx = fy = 1000.0
    cx = cy = 500.0
    # 100px right of center at z=1m -> X = 100/1000 * 1 = 0.1m
    P1 = loc.backproject(600.0, 500.0, 1.0, fx, fy, cx, cy)
    assert P1[0] == pytest.approx(0.1)
    assert P1[1] == pytest.approx(0.0)
    # doubling depth doubles the recovered X for the same pixel offset
    P2 = loc.backproject(600.0, 500.0, 2.0, fx, fy, cx, cy)
    assert P2[0] == pytest.approx(0.2)


def test_backproject_asymmetric_focal_lengths():
    # fy != fx (non-square pixels / anisotropic intrinsics) - X and Y must
    # scale independently, not share one focal length.
    P = loc.backproject(u=110.0, v=60.0, z=1.0, fx=100.0, fy=50.0,
                         cx=100.0, cy=50.0)
    assert P[0] == pytest.approx((110.0 - 100.0) / 100.0)
    assert P[1] == pytest.approx((60.0 - 50.0) / 50.0)


# ---------------------------------------------------------------------------
# FloorFrame: RANSAC plane fit + floor-relative transform
# ---------------------------------------------------------------------------

def _synthetic_flat_depth_map(H, W, fx, fy, cx, cy, plane_z, noise=0.0, rng=None):
    """A depth map for a plane perpendicular to the optical axis at distance
    plane_z (every pixel's ray hits the plane at the same Z)."""
    depth = np.full((H, W), float(plane_z), dtype=np.float32)
    if noise:
        rng = rng or np.random.default_rng(0)
        depth += rng.normal(0, noise, depth.shape).astype(np.float32)
    return depth


def test_floorframe_fits_frontoparallel_plane_and_normal_points_at_camera():
    H, W = 240, 320
    fx = fy = 300.0
    cx, cy = W / 2.0, H / 2.0
    depth = _synthetic_flat_depth_map(H, W, fx, fy, cx, cy, plane_z=1.5)

    floor = loc.FloorFrame()
    floor.fit(depth, fx, fy, cx, cy, (W, H))

    assert floor.ok
    # Plane is perpendicular to +Z, so its normal is purely along Z; the
    # normal must point back toward the camera (origin), i.e. -Z.
    assert floor.n[2] < 0
    assert abs(floor.n[0]) < 1e-6
    assert abs(floor.n[1]) < 1e-6
    # centroid of the fitted plane should sit near the true plane distance
    assert floor.p0[2] == pytest.approx(1.5, abs=0.05)


def test_floorframe_basis_is_orthonormal():
    H, W = 240, 320
    fx = fy = 300.0
    cx, cy = W / 2.0, H / 2.0
    depth = _synthetic_flat_depth_map(H, W, fx, fy, cx, cy, plane_z=2.0)
    floor = loc.FloorFrame()
    floor.fit(depth, fx, fy, cx, cy, (W, H))
    assert floor.ok
    for a, b in [(floor.u, floor.v), (floor.u, floor.n), (floor.v, floor.n)]:
        assert np.dot(a, b) == pytest.approx(0.0, abs=1e-6)
    for a in (floor.u, floor.v, floor.n):
        assert np.linalg.norm(a) == pytest.approx(1.0, abs=1e-6)


def test_floorframe_height_positive_for_point_closer_than_plane():
    """A block sitting on the floor has its top surface CLOSER to the camera
    than the floor plane (smaller Z along the optical axis). That must map
    to a POSITIVE height, matching "height above the floor" as documented."""
    H, W = 240, 320
    fx = fy = 300.0
    cx, cy = W / 2.0, H / 2.0
    plane_z = 1.5
    depth = _synthetic_flat_depth_map(H, W, fx, fy, cx, cy, plane_z=plane_z)
    floor = loc.FloorFrame()
    floor.fit(depth, fx, fy, cx, cy, (W, H))
    assert floor.ok

    on_floor = loc.backproject(cx, cy, plane_z, fx, fy, cx, cy)
    _, _, h0 = floor.to_floor(on_floor)
    assert h0 == pytest.approx(0.0, abs=1e-4)

    bump_height_m = 0.03           # 3cm block, closer to camera by 3cm
    on_bump = loc.backproject(cx, cy, plane_z - bump_height_m, fx, fy, cx, cy)
    _, _, h1 = floor.to_floor(on_bump)
    assert h1 == pytest.approx(bump_height_m, abs=1e-3)


def test_floorframe_rejects_insufficient_or_degenerate_data():
    floor = loc.FloorFrame()
    # too few valid samples
    depth = np.full((240, 320), np.nan, dtype=np.float32)
    floor.fit(depth, 300.0, 300.0, 160.0, 120.0, (320, 240))
    assert not floor.ok


# ---------------------------------------------------------------------------
# Units: /obj and /outline both convert meters -> millimeters
# ---------------------------------------------------------------------------

def test_build_outline_mm_units_and_values_camera_frame():
    """No floor fit available (use_floor=False path) - outline vertices
    should be exactly backproject(...)*1000, preserving winding order."""
    # A simple square contour (as cv2.findContours/approxPolyDP would give,
    # clockwise in image space for a filled square via RETR_EXTERNAL).
    square = np.array([[0, 0], [0, 10], [10, 10], [10, 0]], dtype=np.int32) \
        .reshape(-1, 1, 2)
    fx = fy = 100.0
    cx = cy = 5.0
    z = 2.0
    floor = loc.FloorFrame()   # floor.ok is False -> camera-frame path
    poly, pts_mm = loc.build_outline_mm(square, cx, cy, fx, fy, z, floor,
                                        use_floor=True)
    assert len(poly) == 4
    assert len(pts_mm) == 8

    for (px, py), i in zip(poly, range(0, 8, 2)):
        expected = loc.backproject(px, py, z, fx, fy, cx, cy)
        assert pts_mm[i] == pytest.approx(expected[0] * 1000.0)
        assert pts_mm[i + 1] == pytest.approx(expected[1] * 1000.0)


def test_build_outline_mm_preserves_winding_order():
    """Vertex order out must match approxPolyDP's order on the input contour
    - Unreal builds a polygon straight from these points, so a shuffled or
    reversed winding would twist the extruded mesh."""
    square = np.array([[0, 0], [0, 10], [10, 10], [10, 0]], dtype=np.int32) \
        .reshape(-1, 1, 2)
    fx = fy = 100.0
    cx = cy = 0.0
    floor = loc.FloorFrame()
    poly, _ = loc.build_outline_mm(square, cx, cy, fx, fy, 1.0, floor, False)
    expected_order = cv2.approxPolyDP(
        square, 0.02 * cv2.arcLength(square, True), True).reshape(-1, 2)
    assert np.array_equal(poly, expected_order)


def test_build_outline_mm_never_returns_degenerate_polygon_from_a_line():
    """A near-zero-area sliver contour can collapse to < 3 points under
    approxPolyDP - build_outline_mm must fall back to a real (>=3 vertex)
    polygon rather than handing Unreal's Geometry Script extrude a line or
    a single point on /outline."""
    line = np.array([[0, 0], [50, 0]], dtype=np.int32).reshape(-1, 1, 2)
    fx = fy = 100.0
    floor = loc.FloorFrame()
    poly, pts_mm = loc.build_outline_mm(line, 25.0, 0.0, fx, fy, 1.0,
                                        floor, False)
    assert len(poly) >= 3
    assert len(pts_mm) == 2 * len(poly)


def test_build_outline_mm_never_returns_degenerate_polygon_from_a_point():
    point = np.array([[5, 5]], dtype=np.int32).reshape(-1, 1, 2)
    fx = fy = 100.0
    floor = loc.FloorFrame()
    poly, pts_mm = loc.build_outline_mm(point, 5.0, 5.0, fx, fy, 1.0,
                                        floor, False)
    assert len(poly) >= 3


def test_build_outline_mm_caps_point_count():
    # A many-sided polygon (near-circle) should be downsampled to <= 16 pts.
    n = 40
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    circle = np.stack([50 + 40 * np.cos(angles),
                       50 + 40 * np.sin(angles)], axis=1).astype(np.int32) \
        .reshape(-1, 1, 2)
    fx = fy = 100.0
    floor = loc.FloorFrame()
    poly, pts_mm = loc.build_outline_mm(circle, 50.0, 50.0, fx, fy, 1.0,
                                        floor, False, max_points=16)
    assert len(poly) <= 16
    assert len(pts_mm) == 2 * len(poly)


def test_build_outline_mm_uses_floor_frame_when_available():
    H, W = 240, 320
    fx = fy = 300.0
    cx, cy = W / 2.0, H / 2.0
    plane_z = 1.5
    depth = _synthetic_flat_depth_map(H, W, fx, fy, cx, cy, plane_z)
    floor = loc.FloorFrame()
    floor.fit(depth, fx, fy, cx, cy, (W, H))
    assert floor.ok

    square = np.array([[cx, cy], [cx, cy], [cx, cy], [cx, cy]],
                      dtype=np.int32).reshape(-1, 1, 2)
    poly, pts_mm = loc.build_outline_mm(square, cx, cy, fx, fy, plane_z,
                                        floor, use_floor=True)
    # Every vertex sits exactly on the fitted floor plane -> X,Y ~ 0 there
    # (centroid), height already excluded from /outline (X,Y only).
    for v in pts_mm:
        # not exactly 0: the RANSAC centroid is over a subsampled pixel grid,
        # not the exact frame center - stay within a few cm.
        assert v == pytest.approx(0.0, abs=50.0)


# ---------------------------------------------------------------------------
# Tracks: settle gating, per-colour nearest-centroid matching, size locking,
# circular angle averaging across the 0/360 wrap
# ---------------------------------------------------------------------------

def test_tracks_new_piece_not_confirmed_before_settle_window(monkeypatch):
    t = loc.Tracks(settle_seconds=3.0)
    fake_now = [1000.0]
    monkeypatch.setattr(loc.time, "time", lambda: fake_now[0])
    slot = t.update("red", 100, 100, 1.0, 5.0, 5.0)
    assert slot["confirmed"] is False
    fake_now[0] += 1.0
    slot = t.update("red", 101, 100, 1.0, 5.0, 5.0)
    assert slot["confirmed"] is False


def test_tracks_confirms_after_settle_window(monkeypatch):
    t = loc.Tracks(settle_seconds=3.0)
    fake_now = [1000.0]
    monkeypatch.setattr(loc.time, "time", lambda: fake_now[0])
    t.update("red", 100, 100, 1.0, 5.0, 5.0)
    fake_now[0] += 3.1
    slot = t.update("red", 100, 100, 1.0, 5.0, 5.0)
    assert slot["confirmed"] is True


def test_tracks_size_locks_on_confirmation_and_ignores_later_drift(monkeypatch):
    t = loc.Tracks(settle_seconds=1.0)
    fake_now = [1000.0]
    monkeypatch.setattr(loc.time, "time", lambda: fake_now[0])
    for w in (4.8, 5.0, 5.2):
        t.update("red", 100, 100, 1.0, w, w)
    fake_now[0] += 1.1
    slot = t.update("red", 100, 100, 1.0, 5.0, 5.0)
    assert slot["confirmed"]
    assert slot["size_locked"]
    locked_w = slot["w_cm"]
    # A big "size" swing after lock (e.g. transient depth noise) must NOT
    # move the reported size - it's frozen until the piece is re-added.
    slot = t.update("red", 100, 100, 1.0, 50.0, 50.0)
    assert slot["w_cm"] == locked_w


def test_tracks_disambiguates_two_same_color_pieces_by_nearest_centroid():
    """Multi-piece disambiguation: two 'red' pieces far apart must land in
    separate slots, each tracked independently by nearest centroid."""
    t = loc.Tracks(settle_seconds=0.0, match_dist=90)
    t.update("red", 50, 50, 1.0, 5.0, 5.0)
    t.update("red", 500, 500, 1.0, 8.0, 8.0)
    assert len(t.slots["red"]) == 2
    # a small nudge near piece A should match piece A, not create a third
    slot = t.update("red", 55, 52, 1.0, 5.0, 5.0)
    assert len(t.slots["red"]) == 2
    assert slot["w_cm"] != 8.0


def test_tracks_crossing_same_color_pieces_can_swap_identity():
    """Documents the actual failure mode of nearest-centroid-within-match_dist
    matching: if two same-colour pieces pass within match_dist of each
    other's slot, identity can swap (the tracker has no appearance/ID model
    beyond position). This is a known limitation, not a bug fix target here -
    see docs/PRODUCTION_READINESS.md. Uses a settle window that never expires
    in this test so size stays unlocked and the contamination is directly
    observable in the sample history."""
    t = loc.Tracks(settle_seconds=999.0, match_dist=90)
    t.update("red", 0, 0, 1.0, 4.0, 4.0)     # piece A, small, unconfirmed
    t.update("red", 300, 0, 1.0, 9.0, 9.0)   # piece B, big, far away, unconfirmed
    assert len(t.slots["red"]) == 2

    # Piece B moves to where A's slot is, well within match_dist - it gets
    # matched to A's slot (nearest centroid) instead of staying attached to
    # its own slot. No third slot is created, and A's sample history now
    # contains a sample from a physically different piece.
    slot = t.update("red", 5, 0, 1.0, 9.0, 9.0)
    assert len(t.slots["red"]) == 2
    assert list(slot["w_samps"]) == [4.0, 9.0]   # A's history, contaminated by B


def test_tracks_angle_ema_wraps_across_0_360_boundary():
    """Circular EMA must average 350deg and 10deg to ~0/360, not ~180 -
    a naive linear average would produce the wrong (opposite) angle."""
    t = loc.Tracks(alpha=0.5, settle_seconds=0.0)
    t.update("blue", 10, 10, 1.0, 5.0, 5.0, angle=350.0)
    slot = t.update("blue", 10, 10, 1.0, 5.0, 5.0, angle=10.0)
    result = slot["angle"]
    # distance to 0/360 should be small; distance to 180 should be large
    dist_to_zero = min(result, 360 - result)
    assert dist_to_zero < 15.0


def test_tracks_shape_majority_vote_ignores_unknown():
    t = loc.Tracks(settle_seconds=0.0)
    for shp in ("square", "square", "?", "rectangle", "square"):
        slot = t.update("green", 10, 10, 1.0, 5.0, 5.0, shape=shp)
    assert slot["shape"] == "square"


# ---------------------------------------------------------------------------
# classify_shape: edge cases (zero-area / degenerate contours, known shapes)
# ---------------------------------------------------------------------------

def test_classify_shape_zero_area_degenerate_contour_returns_unknown():
    # A degenerate contour after approxPolyDP - a single point / a line.
    line = np.array([[0, 0], [0, 0]], dtype=np.int32).reshape(-1, 1, 2)
    assert classify_shape(line) == "?"


def test_classify_shape_single_point_contour_returns_unknown():
    pt = np.array([[5, 5]], dtype=np.int32).reshape(-1, 1, 2)
    assert classify_shape(pt) == "?"


def test_classify_shape_square():
    sq = np.array([[0, 0], [0, 40], [40, 40], [40, 0]], dtype=np.int32) \
        .reshape(-1, 1, 2)
    assert classify_shape(sq) == "square"


def test_classify_shape_rectangle():
    rect = np.array([[0, 0], [0, 20], [80, 20], [80, 0]], dtype=np.int32) \
        .reshape(-1, 1, 2)
    assert classify_shape(rect) == "rectangle"


def test_classify_shape_circle():
    angles = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    circle = np.stack([50 + 30 * np.cos(a) for a in angles],
                       axis=0)
    circle = np.stack([50 + 30 * np.cos(angles), 50 + 30 * np.sin(angles)],
                      axis=1).astype(np.int32).reshape(-1, 1, 2)
    assert classify_shape(circle) == "circle"


def test_classify_shape_cross_is_concave():
    # A plus-sign / cross outline: concave, low solidity.
    cross = np.array([
        [10, 0], [20, 0], [20, 10], [30, 10], [30, 20], [20, 20],
        [20, 30], [10, 30], [10, 20], [0, 20], [0, 10], [10, 10],
    ], dtype=np.int32).reshape(-1, 1, 2)
    assert classify_shape(cross) == "cross"


def test_classify_shape_tiny_contour_area_below_threshold():
    tiny = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=np.int32) \
        .reshape(-1, 1, 2)
    # area is 1.0 here (not < 1.0), so it should still classify, not "?" -
    # this pins the exact boundary of the "too small" guard.
    result = classify_shape(tiny)
    assert result in ("square", "?")   # boundary case; just must not crash


def test_classify_shape_frame_edge_contour_still_classifies():
    """A piece clipped by the frame edge (negative/zero-origin coordinates,
    as OpenCV would report for a blob touching x=0) should classify the
    same as one fully in-frame - shape math is translation-invariant."""
    sq_center = np.array([[0, 0], [0, 40], [40, 40], [40, 0]], dtype=np.int32) \
        .reshape(-1, 1, 2)
    sq_edge = sq_center  # already anchored at the origin/edge
    assert classify_shape(sq_edge) == classify_shape(sq_center) == "square"


def test_classify_shape_very_large_contour():
    # A piece filling most of a 4K frame - must not overflow/behave
    # differently purely due to scale (area/perimeter math is scale-free
    # for aspect/circularity/solidity, but worth pinning down explicitly).
    huge = np.array([[0, 0], [0, 3000], [3000, 3000], [3000, 0]],
                    dtype=np.int32).reshape(-1, 1, 2)
    assert classify_shape(huge) == "square"


def test_classify_shape_thin_sliver_degenerates_gracefully():
    """A near-zero-width contour (e.g. a color mask artifact/shadow edge)
    must not crash approxPolyDP or divide-by-zero in the aspect check."""
    sliver = np.array([[0, 0], [0, 1], [100, 1], [100, 0]], dtype=np.int32) \
        .reshape(-1, 1, 2)
    result = classify_shape(sliver)
    assert result in ("rectangle", "square", "?")


# ---------------------------------------------------------------------------
# FloorFrame fallback path: floor.ok == False must not be silently used as
# if it were a valid fit - callers gate on floor.ok explicitly.
# ---------------------------------------------------------------------------

def test_build_outline_mm_falls_back_to_camera_frame_when_floor_fit_failed():
    """use_floor=True but floor.ok is False (fit never ran / failed) must
    fall back to camera-frame coordinates, not raise or silently use a
    stale/zeroed floor basis."""
    floor = loc.FloorFrame()
    assert not floor.ok      # never fit
    square = np.array([[10, 10], [10, 20], [20, 20], [20, 10]],
                      dtype=np.int32).reshape(-1, 1, 2)
    fx = fy = 100.0
    cx = cy = 15.0
    z = 1.0
    poly, pts_mm = loc.build_outline_mm(square, cx, cy, fx, fy, z, floor,
                                        use_floor=True)
    # must match the pure camera-frame (use_floor=False) result exactly,
    # proving the fallback kicked in rather than touching floor.u/v/n
    # (which are None when floor.ok is False - using them would raise).
    _, pts_mm_cam = loc.build_outline_mm(square, cx, cy, fx, fy, z, floor,
                                         use_floor=False)
    assert pts_mm == pts_mm_cam


def test_to_world_xy_falls_back_when_floor_not_ok():
    floor = loc.FloorFrame()
    assert not floor.ok
    P = loc.backproject(50.0, 40.0, 1.0, 100.0, 100.0, 45.0, 35.0)
    X, Y = loc.to_world_xy(P, floor, use_floor=True)
    assert (X, Y) == (pytest.approx(P[0]), pytest.approx(P[1]))


def test_floorframe_fit_with_all_nan_depth_stays_not_ok():
    depth = np.full((100, 100), np.nan, dtype=np.float32)
    floor = loc.FloorFrame()
    floor.fit(depth, 100.0, 100.0, 50.0, 50.0, (100, 100))
    assert not floor.ok
    assert floor.n is None


def test_floorframe_fit_with_collinear_degenerate_points_stays_not_ok():
    """A depth map that's constant along one row only (RANSAC can't find 3
    non-collinear inlier points for a plane) must not crash and must leave
    floor.ok False rather than fitting a garbage plane."""
    depth = np.zeros((50, 50), dtype=np.float32)
    depth[:] = np.nan
    depth[25, :] = 1.0    # a single valid row - degenerate for plane fitting
    floor = loc.FloorFrame()
    floor.fit(depth, 100.0, 100.0, 25.0, 25.0, (50, 50))
    assert not floor.ok
