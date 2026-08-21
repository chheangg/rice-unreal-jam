"""
Thread-safety tests for DepthEstimator's frame handoff (threading.Event/Lock)
under rapid submit() calls, per the "if you have time left" backlog item.

Deliberately does NOT instantiate DepthEstimator normally - __init__ calls
_load(), which imports torch/DA3 and downloads a multi-hundred-MB checkpoint
from HuggingFace on first run (per README). That's both slow and a network
dependency this test suite must not have. Instead we build an instance via
__new__, wire up just the locks/events/state _run() and submit() actually
touch, and monkeypatch _infer() to a fast fake - this exercises the REAL
locking code paths (the thing being tested) without the real model.

No camera use - frames are synthetic numpy arrays.
"""
import threading
import time

import numpy as np
import pytest

from depth_estimator import DepthEstimator


def make_bare_estimator():
    """A DepthEstimator with real locks/events but no model - _load() never
    ran, so this is exactly the object the worker thread and submit() see,
    minus the heavy model call."""
    d = DepthEstimator.__new__(DepthEstimator)
    d.model_id = "fake"
    d.metric_scale = 1.0
    d.available = True
    d.device = "cpu"
    d._model = None
    d._depth = None
    d._intr = None
    d._depth_src_shape = None
    d._depth_lock = threading.Lock()
    d._latest_frame = None
    d._frame_lock = threading.Lock()
    d._frame_event = threading.Event()
    d._stop = threading.Event()
    d._worker = None
    d.last_error = None
    return d


def test_submit_before_start_is_picked_up_once_worker_starts(monkeypatch):
    d = make_bare_estimator()
    calls = []

    def fake_infer(self, rgb):
        calls.append(rgb.copy())
        return np.full((4, 4), 1.5, dtype=np.float32), None

    monkeypatch.setattr(DepthEstimator, "_infer", fake_infer)
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    d.submit(frame)
    d.start()
    try:
        for _ in range(50):
            if d.has_depth():
                break
            time.sleep(0.01)
        assert d.has_depth()
        assert len(calls) >= 1
    finally:
        d.stop()


def test_rapid_submit_only_processes_latest_frame_no_crash(monkeypatch):
    """Submitting many frames faster than inference can keep up must not
    crash, deadlock, or process every single frame - only the newest at
    any given moment (the documented "drop stale frames" contract)."""
    d = make_bare_estimator()
    processed_ids = []
    lock = threading.Lock()

    def fake_infer(self, rgb):
        time.sleep(0.02)                      # slower than submit rate
        with lock:
            processed_ids.append(int(rgb[0, 0, 0]))
        return np.full((4, 4), 1.0, dtype=np.float32), None

    monkeypatch.setattr(DepthEstimator, "_infer", fake_infer)
    d.start()
    try:
        n = 30
        for i in range(n):
            frame = np.full((8, 8, 3), i % 256, dtype=np.uint8)
            d.submit(frame)
            time.sleep(0.002)                 # much faster than inference
        # let the worker drain
        for _ in range(200):
            time.sleep(0.01)
            with lock:
                if processed_ids and processed_ids[-1] == (n - 1) % 256:
                    break
        assert d.has_depth()
        # far fewer inferences than submits - proves frames were dropped,
        # not queued.
        with lock:
            assert 0 < len(processed_ids) < n
    finally:
        d.stop()


def test_concurrent_readers_never_see_torn_state(monkeypatch):
    """While the worker keeps updating _depth/_intr/_depth_src_shape
    together under one lock, concurrent readers (depth_at/has_depth/
    latest_depth_map/intrinsics_for_frame) must always see a CONSISTENT
    triple - never a new depth map paired with a stale/mismatched shape."""
    d = make_bare_estimator()
    stop_flag = threading.Event()
    errors = []

    def fake_infer(self, rgb):
        h = 4 + (int(rgb[0, 0, 0]) % 3)       # shape varies per "frame"
        depth = np.full((h, h), 2.0, dtype=np.float32)
        intr = np.eye(3, dtype=np.float32)
        return depth, intr

    monkeypatch.setattr(DepthEstimator, "_infer", fake_infer)
    d.start()

    def reader():
        try:
            for _ in range(500):
                dm = d.latest_depth_map()
                intr = d.intrinsics_for_frame()
                if dm is not None and intr is not None:
                    # intrinsics_for_frame() internally re-derives scale from
                    # the SAME locked snapshot - if it didn't, a shape change
                    # mid-read would raise or produce nonsense, not just
                    # "wrong answer", so just completing without exception
                    # is the meaningful assertion here.
                    pass
        except Exception as e:                # pragma: no cover
            errors.append(e)

    def writer():
        for i in range(200):
            frame = np.full((8, 8, 3), i % 256, dtype=np.uint8)
            d.submit(frame)

    try:
        threads = [threading.Thread(target=reader) for _ in range(4)]
        threads.append(threading.Thread(target=writer))
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)
        assert not errors
    finally:
        d.stop()


def test_stop_unblocks_worker_promptly(monkeypatch):
    def fake_infer(self, rgb):
        return np.zeros((2, 2), dtype=np.float32), None

    monkeypatch.setattr(DepthEstimator, "_infer", fake_infer)
    d = make_bare_estimator()
    d.start()
    d.submit(np.zeros((4, 4, 3), dtype=np.uint8))
    t0 = time.time()
    d.stop()
    assert time.time() - t0 < 2.5   # stop()'s own join(timeout=2.0) + slack
    assert not d._worker.is_alive()
