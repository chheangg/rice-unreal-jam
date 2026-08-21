"""
Tests for detect_platform.py's non-camera logic: the EMA smoothing helper
and Slot's matching/hold/tare/clamp state machine. Uses a fake
PlatformHeightEstimator (no real geometry/solvePnP needed - that's already
covered in tests/test_platform_height.py) so these tests isolate Slot's own
bookkeeping.

detect_platform.py is NOT read by the active live pipeline - see
docs/PRODUCTION_READINESS.md #6. No camera used - cv2.VideoCapture only
happens inside main(), which is never called here.
"""
import pytest

import detect_platform as dp


def test_import_has_no_camera_side_effect():
    # importing must not have opened a camera (main() is never called here)
    assert hasattr(dp, "main")


def test_ema_first_call_returns_new_value_unchanged():
    assert dp._ema(None, 10.0) == 10.0


def test_ema_blends_toward_new_value():
    result = dp._ema(prev=100.0, new=0.0, alpha=0.05)
    assert result == pytest.approx(95.0)   # 0.05*0 + 0.95*100


def make_blob(cx, cy, size=40, box=None):
    box = box or (cx - 20, cy - 20, 40, 40)
    return {"cx": cx, "cy": cy, "sx": size, "sy": size, "size": size,
            "box": box, "area": size * size}


def test_slot_acquires_largest_candidate_when_unlocked():
    slot = dp.Slot("yel1", "yellow")
    candidates = [make_blob(10, 10, size=5), make_blob(200, 200, size=50)]
    slot.update(candidates)
    assert slot.blob["cx"] == 200
    # matched candidate removed from the list (so another slot can't also
    # claim it)
    assert len(candidates) == 1


def test_slot_matches_nearest_within_match_dist_once_locked():
    slot = dp.Slot("yel1", "yellow")
    slot.update([make_blob(100, 100)])
    assert slot.blob["cx"] == 100

    near = make_blob(110, 105)     # small move, within MATCH_DIST_PX
    far = make_blob(900, 900)      # far away, should be ignored
    slot.update([far, near])
    assert slot.blob["cx"] == 110


def test_slot_holds_last_value_when_no_candidate_is_close_enough():
    slot = dp.Slot("yel1", "yellow")
    slot.update([make_blob(100, 100)])
    far_only = [make_blob(900, 900)]
    slot.update(far_only)
    assert slot.blob["cx"] == 100          # unchanged - held, not jumped
    assert far_only == [make_blob(900, 900)]  # untouched, not consumed


def test_slot_resets_lock_after_too_many_unmatched_frames():
    slot = dp.Slot("yel1", "yellow")
    slot.update([make_blob(100, 100)])
    assert slot.blob is not None
    for _ in range(dp.STALE_LOCK_FRAMES + 1):
        slot.update([])                    # nothing nearby, every frame
    assert slot.blob is None               # gave up and reset


def test_slot_width_history_uses_rotation_invariant_max_side():
    slot = dp.Slot("yel1", "yellow")
    # size in make_blob mirrors find_blobs' "size": int(max(rw, rh))
    slot.update([make_blob(100, 100, size=40)])
    assert slot.smooth_width == 40


class FakeHeightEstimator:
    """Deterministic stand-in for PlatformHeightEstimator - the real ray
    geometry is tested separately in tests/test_platform_height.py."""
    def __init__(self, xy=(50.0, 60.0), z=15.0):
        self._xy = xy
        self._z = z

    def ground_xy_mm(self, px, py):
        return self._xy

    def point3d_mm(self, cx, cy, width, color_name):
        return (self._xy[0], self._xy[1], self._z)


def test_compute_xyz_returns_none_before_first_match():
    slot = dp.Slot("yel1", "yellow")
    assert slot.compute_xyz(FakeHeightEstimator()) is None


def test_compute_xyz_tares_z_to_zero_after_settling():
    slot = dp.Slot("yel1", "yellow")
    est = FakeHeightEstimator(z=15.0)
    for _ in range(dp.TARE_AFTER_N_MATCHES):
        slot.update([make_blob(100, 100)])
        slot.compute_xyz(est)
    assert slot.z_tare is not None
    # once tared, z should read close to 0 (the tare reading subtracted
    # from itself) rather than the raw 15.0mm
    x, y, z = slot.compute_xyz(est)
    assert z == pytest.approx(0.0, abs=0.5)


def test_compute_xyz_clamps_position_to_platform_extent():
    slot = dp.Slot("yel1", "yellow")
    # geometry noise pushing WAY outside the platform's physical bounds
    est = FakeHeightEstimator(xy=(-500.0, 99999.0), z=0.0)
    slot.update([make_blob(100, 100)])
    xyz = slot.compute_xyz(est)
    assert xyz is not None
    x, y, z = xyz
    assert 0.0 <= x <= dp.PLATFORM_SIZE_MM
    assert 0.0 <= y <= dp.PLATFORM_SIZE_MM


def test_compute_xyz_returns_none_when_geometry_unavailable():
    slot = dp.Slot("yel1", "yellow")
    slot.update([make_blob(100, 100)])

    class NoGeometry:
        def ground_xy_mm(self, px, py):
            return None
        def point3d_mm(self, cx, cy, width, color_name):
            return None

    assert slot.compute_xyz(NoGeometry()) is None
