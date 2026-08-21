# Graph Report - rice-unreal-jam  (2026-08-21)

## Corpus Check
- 76 files · ~129,675 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 397 nodes · 624 edges · 27 communities (20 shown, 7 thin omitted)
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
- Community 21
- Community 22
- Community 23
- Community 24
- Community 25

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
- `test_simple_mode_formula()` --uses--> `SideHeightEstimator`  [INFERRED]
  tests/test_side_height.py → side_height.py
- `test_unavailable_estimator_returns_none()` --uses--> `SideHeightEstimator`  [INFERRED]
  tests/test_side_height.py → side_height.py
- `main()` --calls--> `DepthEstimator`  [EXTRACTED]
  lego_locator_xyz.py → depth_estimator.py
- `make_bare_estimator()` --uses--> `DepthEstimator`  [INFERRED]
  tests/test_depth_estimator_threading.py → depth_estimator.py
- `test_concurrent_readers_never_see_torn_state()` --uses--> `DepthEstimator`  [INFERRED]
  tests/test_depth_estimator_threading.py → depth_estimator.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Task Filing Workflow** — agents_task_workflow, tasks_readme_doc, tasks_readme_task_lifecycle, tasks_template_doc [EXTRACTED 1.00]

## Communities (27 total, 7 thin omitted)

### Community 0 - "AGENTS.md Collaboration Guide"
Cohesion: 0.06
Nodes (46): _ema(), load_or_guess_intrinsics(), main(), marker_angle_deg(), detect_platform_aruco.py — accurate single-camera (DJI) X/Y/Z tracker using…, True in-plane rotation of an ArUco tag, 0..360, from its corners. cv2.aruco…, Slot, find_blobs() (+38 more)

### Community 1 - "Task Filing Workflow"
Cohesion: 0.07
Nodes (24): DepthEstimator, depth_estimator.py — monocular depth for the block tracker, using Depth…, Launch the background inference worker (no-op if unavailable)., Hand the worker the latest frame (BGR, as OpenCV gives it)., Median metric depth (meters) over an axis-aligned bounding box `box = (x, y, w,…, Median metric depth (meters) over the TRUE pixels of `mask` - a uint8 image in…, Copy of the latest metric depth map (HxW float32, meters), or None. Small…, Camera intrinsics (fx, fy, cx, cy) in the ORIGINAL camera frame's pixel… (+16 more)

### Community 2 - "Marker Detection Pipeline"
Cohesion: 0.12
Nodes (21): _ema(), One persistent tracked object: locks onto a blob, holds its last known value…, Drop the current lock and all smoothing state, so the slot can cleanly re-…, candidates: list of blob dicts, mutated (matched ones removed)., Real (x_mm, y_mm, z_mm), smoothed and clamped to the platform's physical…, Slot, FakeHeightEstimator, make_blob() (+13 more)

### Community 3 - "README Project Setup"
Cohesion: 0.14
Nodes (23): fixture, image_to_features(), load_reference(), match(), shape_recognizer.py - feature-matching piece identification (ORB + homography),…, Load a reference image and compute ORB keypoints/descriptors. Returns…, Try to match a query grayscale image against a reference's…, _perspective_nudge() (+15 more)

### Community 4 - "Tracked-Block Scene Capture"
Cohesion: 0.14
Nodes (20): find_blobs(), main(), match_by_rank(), detect_stereo.py — two-camera tracker: DJI (top-down) gives X/Y, iPhone (side)…, Pair top-view blocks with side-view blobs by left-to-right rank. Returns…, send_and_draw(), side_height.py — real (geometric) height-above-table for a block seen by the…, Real height (mm) above the table for a block detected at pixel center (cx, cy)… (+12 more)

### Community 5 - "README Unreal Engine Integration"
Cohesion: 0.08
Nodes (21): Tiny per-colour tracker so a stationary piece reports a STEADY size/depth…, Drop slots that haven't been matched for a while., Tracks, Multi-piece disambiguation: two 'red' pieces far apart must land in separate…, Documents the actual failure mode of nearest-centroid-within-match_dist…, The fix for the identity-swap failure mode: when both pieces carry distinct…, A tagged piece that jumps far in one frame (fast motion / a dropped frame) must…, A slot that adopted a tag_id can still be updated by an UNTAGGED detection… (+13 more)

### Community 6 - "Roadmap Living Document"
Cohesion: 0.10
Nodes (19): FloorFrame, Surface-agnostic ground frame fitted from the depth point cloud with RANSAC -…, In-plane basis (uax, vax) for unit normal `nrm`. `fit()` runs every frame (the…, Camera 3D point -> (X_floor, Y_floor, height) in meters., A depth map for a plane perpendicular to the optical axis at distance plane_z…, A block sitting on the floor has its top surface CLOSER to the camera than the…, Regression test: _basis_for_normal used to pick its seed purely from…, With no prior self.u (first fit ever), _basis_for_normal falls back to the… (+11 more)

### Community 7 - "Blueprint Inspection Tool"
Cohesion: 0.12
Nodes (17): FString, FSubsystemCollectionBase, int32, ADynamicMeshActor, FLegoPieceState, Actor, LastSeenSeconds, FOSCMessage (+9 more)

### Community 8 - "README OSC Networking"
Cohesion: 0.11
Nodes (20): backproject(), build_outline_mm(), Pixel (u, v) at depth z (meters) -> camera-frame 3D point (meters). (0,0,0) is…, Camera-frame point -> (X, Y) in whichever frame is active (meters). FLOOR frame…, Simplify a piece's contour (cv2.approxPolyDP) and back-project each vertex into…, to_world_xy(), No floor fit available (use_floor=False path) - outline vertices should be…, Vertex order out must match approxPolyDP's order on the input contour - Unreal… (+12 more)

### Community 9 - "OSC Send Test Script"
Cohesion: 0.18
Nodes (17): classify_shape(), Label a piece contour as 'square', 'rectangle', 'circle', 'cross', or '?'.…, _FakeDepthUnavailable, Synthetic, camera-free tests for the coordinate/size/outline math in…, A piece clipped by the frame edge (negative/zero-origin coordinates, as OpenCV…, A near-zero-width contour (e.g. a color mask artifact/shadow edge) must not…, test_classify_shape_circle(), test_classify_shape_cross_is_concave() (+9 more)

### Community 10 - "Community 10"
Cohesion: 0.20
Nodes (15): as_source(), color_mask(), find_pieces(), main(), lego_locator.py - universal Lego colour locator (step 1: reliable detection).…, Binary mask for one colour: its hue band(s), floored on S and V., Return {color_name: [contours]} and {color_name: mask}., detect_markers() (+7 more)

### Community 11 - "Community 11"
Cohesion: 0.20
Nodes (15): load(), locator_config.py - persisted tuning for lego_locator_xyz.py, so a rig's…, DEFAULTS merged with whatever `path` has, ignoring unknown keys and falling…, Write only the known keys (ignores extras the caller's dict might carry),…, save(), Tests for locator_config.py (config persistence, docs/PRODUCTION_READINESS.md…, test_load_corrupt_json_falls_back_to_defaults(), test_load_ignores_unknown_keys_in_file() (+7 more)

### Community 12 - "Community 12"
Cohesion: 0.15
Nodes (12): detect.py, Git LFS Usage, inspect_bp.py, MechTwin03, MT03_RealTimeLayout (Unreal Project), OSC Protocol Link, Roadmap-Update Practice, send_test.py (+4 more)

### Community 13 - "Community 13"
Cohesion: 0.20
Nodes (5): Regression tests: detect.py and detect_xyz.py used to open a real camera…, RED spans two hue ranges (near 0 and near 179) merged into one mask - both ends…, detect_xyz.py's find_blobs is a near-duplicate of detect.py's - just confirm it…, test_detect_find_blobs_red_wraps_hue_circle(), test_detect_xyz_find_blobs_matches_detect_behavior()

### Community 14 - "Community 14"
Cohesion: 0.42
Nodes (8): as_source(), describe(), main(), open_source(), preview(), camera_probe.py - check what OpenCV can actually see, before touching…, Camera index if it looks like a number, otherwise a URL/path., scan()

### Community 15 - "Community 15"
Cohesion: 0.32
Nodes (5): FakeBlock, main(), make_names(), send_test.py - fake tracker, for testing the Unreal side without a camera.…, One block circling its own centre, with a wobbling size and depth.

### Community 16 - "Community 16"
Cohesion: 0.29
Nodes (5): (fx, fy, cx, cy), source_label. Prefer DA3's; else assume an FOV., resolve_intrinsics(), _FakeDepthWithIntrinsics, test_resolve_intrinsics_falls_back_if_da3_returns_none(), test_resolve_intrinsics_prefers_da3_when_available()

### Community 17 - "Community 17"
Cohesion: 0.40
Nodes (3): MT03_RealTimeLayoutTarget, MT03_RealTimeLayoutEditorTarget, TargetRules

### Community 18 - "Community 18"
Cohesion: 0.83
Nodes (3): find_blobs(), main(), send()

## Knowledge Gaps
- **10 isolated node(s):** `UOSCServer`, `Actor`, `LastSeenSeconds`, `MechTwin03`, `send_test.py` (+5 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **7 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `DepthEstimator` connect `Task Filing Workflow` to `Community 10`?**
  _High betweenness centrality (0.065) - this node is a cross-community bridge._
- **Why does `main()` connect `Community 10` to `Task Filing Workflow`, `README Unreal Engine Integration`, `Roadmap Living Document`, `README OSC Networking`, `OSC Send Test Script`, `Community 11`, `Community 16`?**
  _High betweenness centrality (0.046) - this node is a cross-community bridge._
- **Why does `FloorFrame` connect `Roadmap Living Document` to `README OSC Networking`, `Community 10`?**
  _High betweenness centrality (0.027) - this node is a cross-community bridge._
- **Are the 5 inferred relationships involving `DepthEstimator` (e.g. with `make_bare_estimator()` and `test_concurrent_readers_never_see_torn_state()`) actually correct?**
  _`DepthEstimator` has 5 INFERRED edges - model-reasoned connections that need verification._
- **What connects `UOSCServer`, `Actor`, `LastSeenSeconds` to the rest of the system?**
  _10 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `AGENTS.md Collaboration Guide` be split into smaller, more focused modules?**
  _Cohesion score 0.06203007518796992 - nodes in this community are weakly interconnected._
- **Should `Task Filing Workflow` be split into smaller, more focused modules?**
  _Cohesion score 0.07179487179487179 - nodes in this community are weakly interconnected._