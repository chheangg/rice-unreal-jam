# Graph Report - rice-unreal-jam  (2026-08-20)

## Corpus Check
- Corpus is ~27,036 words - fits in a single context window. You may not need a graph.

## Summary
- 172 nodes · 228 edges · 18 communities (10 shown, 8 thin omitted)
- Extraction: 99% EXTRACTED · 1% INFERRED · 0% AMBIGUOUS · INFERRED: 2 edges (avg confidence: 0.8)
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
- Community 14
- Community 15
- Community 16
- Community 17

## God Nodes (most connected - your core abstractions)
1. `DepthEstimator` - 16 edges
2. `PlatformHeightEstimator` - 11 edges
3. `main()` - 10 edges
4. `main()` - 10 edges
5. `main()` - 8 edges
6. `Slot` - 7 edges
7. `find_platform_corners()` - 7 edges
8. `find_pieces()` - 6 edges
9. `FloorFrame` - 6 edges
10. `Tracks` - 6 edges

## Surprising Connections (you probably didn't know these)
- `main()` --calls--> `DepthEstimator`  [EXTRACTED]
  lego_locator_xyz.py → depth_estimator.py
- `Digital Twin Scene Screenshot` --conceptually_related_to--> `WorldGridMaterial Shader (DDC Key)`  [INFERRED]
  MT03_RealTimeLayout/Saved/AutoScreenshot.png → MT03_RealTimeLayout/Saved/ShaderDebugInfo/PCD3D_SM6/WorldGridMaterial_67bacdd81a65e5e9/Default/DDCKey-Editor.txt
- `main()` --calls--> `find_platform_corners()`  [EXTRACTED]
  detect_platform.py → platform_height.py
- `main()` --calls--> `PlatformHeightEstimator`  [EXTRACTED]
  detect_platform.py → platform_height.py
- `main()` --calls--> `solve_platform_pose()`  [EXTRACTED]
  detect_platform.py → platform_height.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Task Filing Workflow** — agents_task_workflow, tasks_readme_doc, tasks_readme_task_lifecycle, tasks_template_doc [EXTRACTED 1.00]

## Communities (18 total, 8 thin omitted)

### Community 0 - "AGENTS.md Collaboration Guide"
Cohesion: 0.11
Nodes (27): _ema(), load_or_guess_intrinsics(), main(), marker_angle_deg(), detect_platform_aruco.py — accurate single-camera (DJI) X/Y/Z tracker using…, True in-plane rotation of an ArUco tag, 0..360, from its corners. cv2.aruco…, Slot, find_blobs() (+19 more)

### Community 1 - "Task Filing Workflow"
Cohesion: 0.09
Nodes (11): DepthEstimator, depth_estimator.py — monocular depth for the block tracker, using Depth…, Launch the background inference worker (no-op if unavailable)., Hand the worker the latest frame (BGR, as OpenCV gives it)., Median metric depth (meters) over an axis-aligned bounding box `box = (x, y, w,…, Median metric depth (meters) over the TRUE pixels of `mask` - a uint8 image in…, Camera intrinsics (fx, fy, cx, cy) in the ORIGINAL camera frame's pixel…, Run DA3 on one RGB frame -> (metric depth map HxW float32 meters, 3x3… (+3 more)

### Community 2 - "Marker Detection Pipeline"
Cohesion: 0.13
Nodes (20): as_source(), classify_shape(), color_mask(), find_pieces(), main(), lego_locator.py - universal Lego colour locator (step 1: reliable detection).…, Binary mask for one colour: its hue band(s), floored on S and V., Return {color_name: [contours]} and {color_name: mask}. (+12 more)

### Community 3 - "README Project Setup"
Cohesion: 0.15
Nodes (12): detect.py, Git LFS Usage, inspect_bp.py, MechTwin03, MT03_RealTimeLayout (Unreal Project), OSC Protocol Link, Roadmap-Update Practice, send_test.py (+4 more)

### Community 4 - "Tracked-Block Scene Capture"
Cohesion: 0.22
Nodes (9): find_blobs(), main(), match_by_rank(), detect_stereo.py — two-camera tracker: DJI (top-down) gives X/Y, iPhone (side)…, Pair top-view blocks with side-view blobs by left-to-right rank. Returns…, send_and_draw(), side_height.py — real (geometric) height-above-table for a block seen by the…, Real height (mm) above the table for a block detected at pixel center (cx, cy)… (+1 more)

### Community 5 - "README Unreal Engine Integration"
Cohesion: 0.24
Nodes (6): _ema(), One persistent tracked object: locks onto a blob, holds its last known value…, Drop the current lock and all smoothing state, so the slot can cleanly re-…, candidates: list of blob dicts, mutated (matched ones removed)., Real (x_mm, y_mm, z_mm), smoothed and clamped to the platform's physical…, Slot

### Community 6 - "Roadmap Living Document"
Cohesion: 0.42
Nodes (8): as_source(), describe(), main(), open_source(), preview(), camera_probe.py - check what OpenCV can actually see, before touching…, Camera index if it looks like a number, otherwise a URL/path., scan()

### Community 7 - "Blueprint Inspection Tool"
Cohesion: 0.32
Nodes (5): FakeBlock, main(), make_names(), send_test.py - fake tracker, for testing the Unreal side without a camera.…, One block circling its own centre, with a wobbling size and depth.

### Community 8 - "README OSC Networking"
Cohesion: 0.33
Nodes (3): Tiny per-colour tracker so a stationary piece reports a STEADY size/depth…, Drop slots that haven't been matched for a while., Tracks

## Knowledge Gaps
- **7 isolated node(s):** `MechTwin03`, `send_test.py`, `inspect_bp.py`, `OSC Protocol Link`, `Git LFS Usage` (+2 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **8 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `DepthEstimator` connect `Task Filing Workflow` to `Marker Detection Pipeline`?**
  _High betweenness centrality (0.069) - this node is a cross-community bridge._
- **Why does `PlatformHeightEstimator` connect `AGENTS.md Collaboration Guide` to `OSC Send Test Script`, `Community 10`?**
  _High betweenness centrality (0.033) - this node is a cross-community bridge._
- **Why does `main()` connect `Marker Detection Pipeline` to `README OSC Networking`, `Task Filing Workflow`?**
  _High betweenness centrality (0.029) - this node is a cross-community bridge._
- **What connects `MechTwin03`, `send_test.py`, `inspect_bp.py` to the rest of the system?**
  _7 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `AGENTS.md Collaboration Guide` be split into smaller, more focused modules?**
  _Cohesion score 0.10984848484848485 - nodes in this community are weakly interconnected._
- **Should `Task Filing Workflow` be split into smaller, more focused modules?**
  _Cohesion score 0.08547008547008547 - nodes in this community are weakly interconnected._
- **Should `Marker Detection Pipeline` be split into smaller, more focused modules?**
  _Cohesion score 0.12666666666666668 - nodes in this community are weakly interconnected._