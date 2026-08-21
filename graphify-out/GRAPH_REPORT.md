# Graph Report - rice-unreal-jam  (2026-08-21)

## Corpus Check
- 69 files · ~127,980 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 365 nodes · 590 edges · 21 communities (15 shown, 6 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 10 edges (avg confidence: 0.92)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- AGENTS.md Collaboration Guide
- Task Filing Workflow
- Marker Detection Pipeline
- README Project Setup
- Tracked-Block Scene Capture
- README Unreal Engine Integration
- Roadmap Living Document
- Blueprint Inspection Tool
- README OSC Networking
- OSC Send Test Script
- Community 10
- Community 11
- Community 12
- Community 13
- Community 14
- Community 15
- Community 16
- Community 17
- Community 18
- Community 19
- Community 20

## God Nodes (most connected - your core abstractions)
1. `DepthEstimator` - 24 edges
2. `FloorFrame` - 23 edges
3. `Tracks` - 17 edges
4. `Slot` - 16 edges
5. `classify_shape()` - 16 edges
6. `main()` - 14 edges
7. `SideHeightEstimator` - 14 edges
8. `PlatformHeightEstimator` - 13 edges
9. `match()` - 13 edges
10. `build_outline_mm()` - 12 edges

## Surprising Connections (you probably didn't know these)
- `_estimator()` --uses--> `PlatformHeightEstimator`  [INFERRED]
  tests/test_platform_height.py → platform_height.py
- `test_simple_mode_formula()` --uses--> `SideHeightEstimator`  [INFERRED]
  tests/test_side_height.py → side_height.py
- `test_unavailable_estimator_returns_none()` --uses--> `SideHeightEstimator`  [INFERRED]
  tests/test_side_height.py → side_height.py
- `main()` --calls--> `DepthEstimator`  [EXTRACTED]
  lego_locator_xyz.py → depth_estimator.py
- `make_bare_estimator()` --uses--> `DepthEstimator`  [INFERRED]
  tests/test_depth_estimator_threading.py → depth_estimator.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Task Filing Workflow** — agents_task_workflow, tasks_readme_doc, tasks_readme_task_lifecycle, tasks_template_doc [EXTRACTED 1.00]

## Communities (21 total, 6 thin omitted)

### Community 0 - "AGENTS.md Collaboration Guide"
Cohesion: 0.06
Nodes (56): classify_shape(), Label a piece contour as 'square', 'rectangle', 'circle', 'cross', or '?'.…, backproject(), build_outline_mm(), FloorFrame, Surface-agnostic ground frame fitted from the depth point cloud with RANSAC -…, In-plane basis (uax, vax) for unit normal `nrm`. `fit()` runs every frame (the…, Camera 3D point -> (X_floor, Y_floor, height) in meters. (+48 more)

### Community 1 - "Task Filing Workflow"
Cohesion: 0.07
Nodes (24): DepthEstimator, depth_estimator.py — monocular depth for the block tracker, using Depth…, Launch the background inference worker (no-op if unavailable)., Hand the worker the latest frame (BGR, as OpenCV gives it)., Median metric depth (meters) over an axis-aligned bounding box `box = (x, y, w,…, Median metric depth (meters) over the TRUE pixels of `mask` - a uint8 image in…, Copy of the latest metric depth map (HxW float32, meters), or None. Small…, Camera intrinsics (fx, fy, cx, cy) in the ORIGINAL camera frame's pixel… (+16 more)

### Community 2 - "Marker Detection Pipeline"
Cohesion: 0.09
Nodes (29): _ema(), load_or_guess_intrinsics(), main(), marker_angle_deg(), detect_platform_aruco.py — accurate single-camera (DJI) X/Y/Z tracker using…, True in-plane rotation of an ArUco tag, 0..360, from its corners. cv2.aruco…, Slot, find_blobs() (+21 more)

### Community 3 - "README Project Setup"
Cohesion: 0.12
Nodes (21): _ema(), One persistent tracked object: locks onto a blob, holds its last known value…, Drop the current lock and all smoothing state, so the slot can cleanly re-…, candidates: list of blob dicts, mutated (matched ones removed)., Real (x_mm, y_mm, z_mm), smoothed and clamped to the platform's physical…, Slot, FakeHeightEstimator, make_blob() (+13 more)

### Community 4 - "Tracked-Block Scene Capture"
Cohesion: 0.14
Nodes (23): fixture, image_to_features(), load_reference(), match(), shape_recognizer.py - feature-matching piece identification (ORB + homography),…, Load a reference image and compute ORB keypoints/descriptors. Returns…, Try to match a query grayscale image against a reference's…, _perspective_nudge() (+15 more)

### Community 5 - "README Unreal Engine Integration"
Cohesion: 0.14
Nodes (20): find_blobs(), main(), match_by_rank(), detect_stereo.py — two-camera tracker: DJI (top-down) gives X/Y, iPhone (side)…, Pair top-view blocks with side-view blobs by left-to-right rank. Returns…, send_and_draw(), side_height.py — real (geometric) height-above-table for a block seen by the…, Real height (mm) above the table for a block detected at pixel center (cx, cy)… (+12 more)

### Community 6 - "Roadmap Living Document"
Cohesion: 0.13
Nodes (20): as_source(), color_mask(), find_pieces(), main(), lego_locator.py - universal Lego colour locator (step 1: reliable detection).…, Binary mask for one colour: its hue band(s), floored on S and V., Return {color_name: [contours]} and {color_name: mask}., detect_markers() (+12 more)

### Community 7 - "Blueprint Inspection Tool"
Cohesion: 0.08
Nodes (21): Tiny per-colour tracker so a stationary piece reports a STEADY size/depth…, Drop slots that haven't been matched for a while., Tracks, Multi-piece disambiguation: two 'red' pieces far apart must land in separate…, Documents the actual failure mode of nearest-centroid-within-match_dist…, The fix for the identity-swap failure mode: when both pieces carry distinct…, A tagged piece that jumps far in one frame (fast motion / a dropped frame) must…, A slot that adopted a tag_id can still be updated by an UNTAGGED detection… (+13 more)

### Community 8 - "README OSC Networking"
Cohesion: 0.18
Nodes (17): _classify_corners(), box: 4x2 array of corner pixels, any order. Returns them reordered as…, _estimator(), Tests for platform_height.py's geometry: corner classification, the ray/plane…, p_cam = R @ p_world + t (solvePnP's own convention) - so feeding…, Build synthetic ArUco corner pixels by projecting a KNOWN marker pose with…, test_camera_point_to_world_round_trips_the_forward_transform(), test_classify_corners_axis_aligned_rectangle() (+9 more)

### Community 9 - "OSC Send Test Script"
Cohesion: 0.20
Nodes (15): load(), locator_config.py - persisted tuning for lego_locator_xyz.py, so a rig's…, DEFAULTS merged with whatever `path` has, ignoring unknown keys and falling…, Write only the known keys (ignores extras the caller's dict might carry),…, save(), Tests for locator_config.py (config persistence, docs/PRODUCTION_READINESS.md…, test_load_corrupt_json_falls_back_to_defaults(), test_load_ignores_unknown_keys_in_file() (+7 more)

### Community 10 - "Community 10"
Cohesion: 0.15
Nodes (12): detect.py, Git LFS Usage, inspect_bp.py, MechTwin03, MT03_RealTimeLayout (Unreal Project), OSC Protocol Link, Roadmap-Update Practice, send_test.py (+4 more)

### Community 11 - "Community 11"
Cohesion: 0.20
Nodes (5): Regression tests: detect.py and detect_xyz.py used to open a real camera…, RED spans two hue ranges (near 0 and near 179) merged into one mask - both ends…, detect_xyz.py's find_blobs is a near-duplicate of detect.py's - just confirm it…, test_detect_find_blobs_red_wraps_hue_circle(), test_detect_xyz_find_blobs_matches_detect_behavior()

### Community 12 - "Community 12"
Cohesion: 0.42
Nodes (8): as_source(), describe(), main(), open_source(), preview(), camera_probe.py - check what OpenCV can actually see, before touching…, Camera index if it looks like a number, otherwise a URL/path., scan()

### Community 13 - "Community 13"
Cohesion: 0.32
Nodes (5): FakeBlock, main(), make_names(), send_test.py - fake tracker, for testing the Unreal side without a camera.…, One block circling its own centre, with a wobbling size and depth.

### Community 14 - "Community 14"
Cohesion: 0.83
Nodes (3): find_blobs(), main(), send()

## Knowledge Gaps
- **7 isolated node(s):** `MechTwin03`, `send_test.py`, `inspect_bp.py`, `OSC Protocol Link`, `Git LFS Usage` (+2 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **6 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `DepthEstimator` connect `Task Filing Workflow` to `Roadmap Living Document`?**
  _High betweenness centrality (0.077) - this node is a cross-community bridge._
- **Why does `main()` connect `Roadmap Living Document` to `AGENTS.md Collaboration Guide`, `Task Filing Workflow`, `OSC Send Test Script`, `Blueprint Inspection Tool`?**
  _High betweenness centrality (0.054) - this node is a cross-community bridge._
- **Why does `FloorFrame` connect `AGENTS.md Collaboration Guide` to `Roadmap Living Document`?**
  _High betweenness centrality (0.032) - this node is a cross-community bridge._
- **Are the 5 inferred relationships involving `DepthEstimator` (e.g. with `make_bare_estimator()` and `test_concurrent_readers_never_see_torn_state()`) actually correct?**
  _`DepthEstimator` has 5 INFERRED edges - model-reasoned connections that need verification._
- **What connects `MechTwin03`, `send_test.py`, `inspect_bp.py` to the rest of the system?**
  _7 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `AGENTS.md Collaboration Guide` be split into smaller, more focused modules?**
  _Cohesion score 0.05683060109289618 - nodes in this community are weakly interconnected._
- **Should `Task Filing Workflow` be split into smaller, more focused modules?**
  _Cohesion score 0.07179487179487179 - nodes in this community are weakly interconnected._