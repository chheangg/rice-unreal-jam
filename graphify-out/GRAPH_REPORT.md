# Graph Report - rice-unreal-jam  (2026-08-20)

## Corpus Check
- Corpus is ~21,386 words - fits in a single context window. You may not need a graph.

## Summary
- 118 nodes · 147 edges · 14 communities (8 shown, 6 thin omitted)
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
- OSC Send Test Script
- Community 10
- Community 11
- Community 12

## God Nodes (most connected - your core abstractions)
1. `DepthEstimator` - 11 edges
2. `PlatformHeightEstimator` - 11 edges
3. `main()` - 9 edges
4. `main()` - 8 edges
5. `Slot` - 7 edges
6. `find_platform_corners()` - 7 edges
7. `solve_platform_pose()` - 6 edges
8. `main()` - 5 edges
9. `SideHeightEstimator` - 5 edges
10. `Slot` - 4 edges

## Surprising Connections (you probably didn't know these)
- `main()` --calls--> `PlatformHeightEstimator`  [EXTRACTED]
  detect_platform.py → platform_height.py
- `main()` --calls--> `find_platform_corners()`  [EXTRACTED]
  detect_platform_aruco.py → platform_height.py
- `main()` --calls--> `PlatformHeightEstimator`  [EXTRACTED]
  detect_platform_aruco.py → platform_height.py
- `main()` --calls--> `solve_platform_pose()`  [EXTRACTED]
  detect_platform_aruco.py → platform_height.py
- `Digital Twin Scene Screenshot` --conceptually_related_to--> `WorldGridMaterial Shader (DDC Key)`  [INFERRED]
  MT03_RealTimeLayout/Saved/AutoScreenshot.png → MT03_RealTimeLayout/Saved/ShaderDebugInfo/PCD3D_SM6/WorldGridMaterial_67bacdd81a65e5e9/Default/DDCKey-Editor.txt

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Task Filing Workflow** — agents_task_workflow, tasks_readme_doc, tasks_readme_task_lifecycle, tasks_template_doc [EXTRACTED 1.00]

## Communities (14 total, 6 thin omitted)

### Community 0 - "AGENTS.md Collaboration Guide"
Cohesion: 0.10
Nodes (9): DepthEstimator, depth_estimator.py — monocular depth for the block tracker, using Depth…, Launch the background inference worker (no-op if unavailable)., Hand the worker the latest frame (BGR, as OpenCV gives it)., Median metric depth (meters) over an axis-aligned bounding box `box = (x, y, w,…, Run DA3 on one RGB frame -> metric depth map (HxW float32, meters)., # NOTE: absolute scale can still need per-rig calibration — tune, model_id : HuggingFace repo id of a DA3 checkpoint. device : "cuda" / "cpu" /… (+1 more)

### Community 1 - "Task Filing Workflow"
Cohesion: 0.15
Nodes (15): _ema(), find_blobs(), load_or_guess_intrinsics(), main(), detect_platform.py — single-camera (DJI, angled ~45 deg) tracker with real,…, One persistent tracked object: locks onto a blob, holds its last known value…, Drop the current lock and all smoothing state, so the slot can cleanly re-…, candidates: list of blob dicts, mutated (matched ones removed). (+7 more)

### Community 2 - "Marker Detection Pipeline"
Cohesion: 0.17
Nodes (14): _ema(), load_or_guess_intrinsics(), main(), detect_platform_aruco.py — accurate single-camera (DJI) X/Y/Z tracker using…, Slot, _classify_corners(), detect_markers(), make_aruco_detector() (+6 more)

### Community 3 - "README Project Setup"
Cohesion: 0.15
Nodes (12): detect.py, Git LFS Usage, inspect_bp.py, MechTwin03, MT03_RealTimeLayout (Unreal Project), OSC Protocol Link, Roadmap-Update Practice, send_test.py (+4 more)

### Community 4 - "Tracked-Block Scene Capture"
Cohesion: 0.22
Nodes (9): find_blobs(), main(), match_by_rank(), detect_stereo.py — two-camera tracker: DJI (top-down) gives X/Y, iPhone (side)…, Pair top-view blocks with side-view blobs by left-to-right rank. Returns…, send_and_draw(), side_height.py — real (geometric) height-above-table for a block seen by the…, Real height (mm) above the table for a block detected at pixel center (cx, cy)… (+1 more)

### Community 5 - "README Unreal Engine Integration"
Cohesion: 0.22
Nodes (6): PlatformHeightEstimator, Real X/Y/Z for pixels in the DJI frame, given a solved platform pose., Undistorted camera ray through pixel (px,py), as a WORLD-space direction vector…, EXACT real-world (X, Y) in mm, platform-relative, for a pixel that lies ON the…, Approximate full 3D point (X, Y, Z) in mm for a block's CENTER pixel (cx, cy),…, Transform a 3D point already expressed in CAMERA space into platform-world…

## Knowledge Gaps
- **7 isolated node(s):** `MechTwin03`, `send_test.py`, `inspect_bp.py`, `OSC Protocol Link`, `Git LFS Usage` (+2 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **6 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `PlatformHeightEstimator` connect `README Unreal Engine Integration` to `Task Filing Workflow`, `Marker Detection Pipeline`?**
  _High betweenness centrality (0.067) - this node is a cross-community bridge._
- **What connects `MechTwin03`, `send_test.py`, `inspect_bp.py` to the rest of the system?**
  _7 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `AGENTS.md Collaboration Guide` be split into smaller, more focused modules?**
  _Cohesion score 0.1038961038961039 - nodes in this community are weakly interconnected._
- **Should `Task Filing Workflow` be split into smaller, more focused modules?**
  _Cohesion score 0.14736842105263157 - nodes in this community are weakly interconnected._