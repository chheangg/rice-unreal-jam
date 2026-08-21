"""
depth_estimator.py — monocular depth for the block tracker, using
Depth Anything 3 (DA3) from ByteDance.

Repo:  https://github.com/ByteDance-Seed/Depth-Anything-3
(Note: DA3 is ByteDance's model, not Facebook/Meta's. Meta's only tie is
 the DINOv2 backbone lineage the DA family builds on.)

Why a class + background thread?
--------------------------------
DA3METRIC-LARGE is a ~0.35B-param transformer. Running it on every webcam
frame would drop the tracker to a few FPS. Instead the heavy inference runs
on its own worker thread on the *most recent* frame it can grab, and the
detection loop just reads whatever depth map is currently available. The
preview stays smooth; depth refreshes as fast as the GPU allows.

Everything degrades gracefully: if torch / the DA3 package / a usable device
isn't present, `available` is False and `depth_at()` returns None, so the
caller can fall back to plain 2D tracking. The live demo never hard-breaks
just because depth couldn't load.
"""

import threading
import numpy as np

# Default model. DA3METRIC-LARGE returns *metric* depth (meters), which is
# what we want for a real z coordinate. Swap to "depth-anything/DA3-SMALL"
# for much faster (but relative, not metric) depth on weaker hardware.
DEFAULT_MODEL = "depth-anything/DA3METRIC-LARGE"


class DepthEstimator:
    def __init__(self, model_id=DEFAULT_MODEL, device=None, metric_scale=1.0):
        """
        model_id     : HuggingFace repo id of a DA3 checkpoint.
        device       : "cuda" / "cpu" / None (auto-pick cuda if available).
        metric_scale : multiplier applied to the model's metric output. Leave
                       at 1.0 for meters; tune if your Unreal scale needs it.
        """
        self.model_id = model_id
        self.metric_scale = metric_scale
        self.available = False
        self.device = device

        self._model = None
        self._depth = None          # latest metric depth map, HxW float32 (meters)
        self._intr = None           # 3x3 camera intrinsics DA3 returned for that
                                    # depth map, in DEPTH-MAP pixel space (None if
                                    # this DA3 build didn't provide intrinsics).
        self._depth_src_shape = None  # (H, W) of the frame that produced _depth -
                                       # DA3 runs inference at a resized resolution,
                                       # so depth_at() needs this to map a box given
                                       # in original-frame pixels into depth-map pixels.
        self._depth_lock = threading.Lock()
        self._latest_frame = None   # newest RGB frame handed in by the caller
        self._frame_lock = threading.Lock()
        self._frame_event = threading.Event()
        self._stop = threading.Event()
        self._worker = None
        self.last_error = None

        self._load()

    @staticmethod
    def _stub_unused_export_deps():
        """
        DA3's api.py eagerly imports its export utilities (GLB / Gaussian-video
        / COLMAP writers), which drag in heavy, Windows-fragile packages -
        moviepy (and the removed `moviepy.editor`), pycolmap, open3d, trimesh,
        plyfile. We only ever call `inference()` and read `.depth`, never any
        exporter, so we register harmless stub modules for the ones that aren't
        installed. A stub returns a dummy for any attribute, so the top-level
        `import x as y` in the export code succeeds; if an exporter were ever
        actually called it'd fail loudly - which is fine, we don't call them.
        """
        import sys
        import types

        class _Dummy:
            """Callable/subscriptable no-op, so module-level uses in the unused
            export code (decorators, base look-ups, constants) don't explode."""
            def __init__(self, name="da3_stub"):
                self.__name__ = name
            def __call__(self, *a, **k):
                return _Dummy(self.__name__)
            def __getattr__(self, name):
                return _Dummy(f"{self.__name__}.{name}")
            def __getitem__(self, key):
                return _Dummy(f"{self.__name__}[]")
            def __iter__(self):
                return iter(())

        class _Stub(types.ModuleType):
            __file__ = "<da3-stub>"             # keep inspect.* from choking
            __path__ = []                        # look like a package too
            def __getattr__(self, name):
                return _Dummy(f"{self.__name__}.{name}")

        for mod in ("moviepy", "moviepy.editor", "pycolmap", "open3d",
                    "trimesh", "plyfile"):
            try:
                __import__(mod)                 # keep the real one if present
            except Exception:
                sys.modules.setdefault(mod, _Stub(mod))

    # ---- model loading (fails soft) ---------------------------------------
    def _load(self):
        try:
            # On Apple Silicon (MPS), some DA3 ops aren't implemented in Metal;
            # this lets torch silently run those on CPU instead of crashing.
            # Harmless on non-Mac. Must be set before torch is imported.
            import os
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
            # torch's bundled libomp and another linked copy (e.g. from opencv)
            # both try to init on Mac, which OpenMP treats as fatal by default.
            os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

            import torch
            self._stub_unused_export_deps()
            from depth_anything_3.api import DepthAnything3

            if self.device is None:
                if torch.cuda.is_available():
                    self.device = "cuda"
                elif getattr(torch.backends, "mps", None) and \
                        torch.backends.mps.is_available():
                    self.device = "mps"          # Apple Silicon GPU (M1/M2/M3)
                else:
                    self.device = "cpu"
            if self.device == "cpu":
                print("[depth] WARNING: no GPU (CUDA/MPS) — DA3 on CPU will be "
                      "very slow. Consider DA3-SMALL or a GPU machine.")
            elif self.device == "mps":
                print("[depth] using Apple Silicon GPU (MPS). If you see "
                      "xformers errors, uninstall xformers — it isn't supported "
                      "on Apple Silicon and DA3 runs without it.")

            print(f"[depth] loading {self.model_id} on {self.device} ...")
            self._model = DepthAnything3.from_pretrained(self.model_id)
            self._model = self._model.to(device=self.device)
            self.available = True
            print("[depth] model ready.")
        except Exception as e:                       # missing pkg, OOM, no net, etc.
            self.last_error = repr(e)
            self.available = False
            print(f"[depth] disabled — falling back to 2D only. Reason: {e}")

    def start(self):
        """Launch the background inference worker (no-op if unavailable)."""
        if not self.available or self._worker is not None:
            return
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def stop(self):
        self._stop.set()
        self._frame_event.set()
        if self._worker is not None:
            self._worker.join(timeout=2.0)

    # ---- caller-facing API -------------------------------------------------
    def submit(self, frame_bgr):
        """Hand the worker the latest frame (BGR, as OpenCV gives it)."""
        if not self.available:
            return
        # Convert once here; keep only the newest frame (drop stale ones).
        rgb = frame_bgr[:, :, ::-1].copy()           # BGR -> RGB
        with self._frame_lock:
            self._latest_frame = rgb
        self._frame_event.set()

    def depth_at(self, box):
        """
        Median metric depth (meters) over an axis-aligned bounding box
        `box = (x, y, w, h)`, given in the ORIGINAL camera frame's pixel
        coordinates (i.e. what detect_xyz.py's blob boxes are in). Returns
        None if no depth is available yet. Median over the box is robust to
        a few bad pixels / edges.
        """
        with self._depth_lock:
            depth = self._depth
            src_shape = self._depth_src_shape
        if depth is None or src_shape is None:
            return None
        H, W = depth.shape
        src_H, src_W = src_shape
        # DA3 runs inference on a resized copy of the frame, so the depth map's
        # resolution differs from the source frame's - scale the box into
        # depth-map pixel space before indexing (see #DA3-depth-scale-mismatch).
        sx, sy = W / src_W, H / src_H
        x, y, w, h = box
        x0, y0 = max(0, int(x * sx)), max(0, int(y * sy))
        x1, y1 = min(W, int((x + w) * sx)), min(H, int((y + h) * sy))
        if x1 <= x0 or y1 <= y0:
            return None
        patch = depth[y0:y1, x0:x1]
        patch = patch[np.isfinite(patch)]
        if patch.size == 0:
            return None
        return float(np.median(patch)) * self.metric_scale

    def depth_in_mask(self, mask):
        """
        Median metric depth (meters) over the TRUE pixels of `mask` - a uint8
        image in the ORIGINAL frame's resolution (e.g. a filled piece contour).
        Preferred over depth_at(box) for size/z because it samples only the
        object, not the background inside its bounding box, which is what makes
        a stationary piece's size read constant instead of jittering. Returns
        None if no depth yet or the mask covers nothing with valid depth.
        """
        import cv2
        with self._depth_lock:
            depth = self._depth
        if depth is None or mask is None:
            return None
        H, W = depth.shape
        # Shrink the mask to the depth map's (smaller) resolution with nearest
        # so we index depth-map pixels directly - no coordinate scaling needed.
        small = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        vals = depth[small > 0]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return None
        return float(np.median(vals)) * self.metric_scale

    def has_depth(self):
        with self._depth_lock:
            return self._depth is not None

    def latest_depth_map(self):
        """Copy of the latest metric depth map (HxW float32, meters), or None.
        Small (DA3's inference resolution, not the camera's) - resize to the
        preview frame's size before overlaying it."""
        with self._depth_lock:
            return None if self._depth is None else self._depth.copy()

    def intrinsics_for_frame(self):
        """
        Camera intrinsics (fx, fy, cx, cy) in the ORIGINAL camera frame's
        pixel coordinates, or None if DA3 didn't return intrinsics yet. DA3
        gives intrinsics in the resized depth-map's pixel space, so they're
        scaled here back up to the source frame the caller's pixel coords are
        in. Use these to back-project a pixel (u, v) + depth z into metric
        camera-space:  X = (u - cx) * z / fx,  Y = (v - cy) * z / fy.
        """
        with self._depth_lock:
            intr = self._intr
            depth = self._depth
            src_shape = self._depth_src_shape
        if intr is None or depth is None or src_shape is None:
            return None
        H, W = depth.shape
        src_H, src_W = src_shape
        sx, sy = src_W / W, src_H / H          # depth-map px -> source-frame px
        fx = float(intr[0, 0]) * sx
        fy = float(intr[1, 1]) * sy
        cx = float(intr[0, 2]) * sx
        cy = float(intr[1, 2]) * sy
        return (fx, fy, cx, cy)

    # ---- worker loop -------------------------------------------------------
    def _run(self):
        while not self._stop.is_set():
            self._frame_event.wait()
            self._frame_event.clear()
            if self._stop.is_set():
                break
            with self._frame_lock:
                frame = self._latest_frame
                self._latest_frame = None
            if frame is None:
                continue
            try:
                depth, intr = self._infer(frame)
                with self._depth_lock:
                    self._depth = depth
                    self._intr = intr
                    self._depth_src_shape = frame.shape[:2]
            except Exception as e:
                self.last_error = repr(e)
                print(f"[depth] inference error: {e}")

    def _infer(self, rgb):
        """Run DA3 on one RGB frame -> (metric depth map HxW float32 meters,
        3x3 intrinsics in depth-map pixel space or None)."""
        # DA3's api accepts a list of inputs. It takes image paths, and also
        # PIL/numpy arrays. Try the array path first (no disk I/O); fall back
        # to a temp file if this build only accepts paths.
        pred = None
        try:
            pred = self._model.inference([rgb])
        except Exception:
            import tempfile, os, cv2
            tmp = os.path.join(tempfile.gettempdir(), "da3_frame.png")
            cv2.imwrite(tmp, rgb[:, :, ::-1])        # write back as BGR for cv2
            pred = self._model.inference([tmp])

        depth = np.asarray(pred.depth[0], dtype=np.float32)   # [H, W], meters

        # DA3METRIC returns metric depth in METERS directly - DA3's own code
        # reads prediction.depth[y,x] and prints it as "{d:.2f}m" with no
        # rescaling (see app/modules/visualization.py). So we take it as-is.
        # (An earlier version applied metric = focal*raw/300; that formula is
        # from Depth Anything *V2*-metric, a different model, and corrupts DA3's
        # already-metric output - removed.) Absolute scale can still need
        # per-rig calibration: tune `metric_scale` if z is off vs a tape measure.
        #
        # We still pull intrinsics (if present) so callers can back-project a
        # pixel + depth into metric camera-space X/Y and estimate real size.
        try:
            intr = np.asarray(pred.intrinsics[0], dtype=np.float32)
        except Exception:
            intr = None
        return depth, intr
