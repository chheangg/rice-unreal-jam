# Graph Report - rice-unreal-jam  (2026-08-20)

## Corpus Check
- Corpus is ~12,183 words - fits in a single context window. You may not need a graph.

## Summary
- 49 nodes · 47 edges · 14 communities (5 shown, 9 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 2 edges (avg confidence: 0.8)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- AGENTS.md Collaboration Guide
- Task Filing Workflow
- Marker Detection Pipeline
- Tracked-Block Scene Capture
- README Unreal Engine Integration
- Roadmap Living Document
- Blueprint Inspection Tool
- README OSC Networking
- OSC Send Test Script
- Community 10
- Community 11
- Community 12

## God Nodes (most connected - your core abstractions)
1. `DepthEstimator` - 11 edges
2. `detect.py` - 3 edges
3. `MT03_RealTimeLayout (Unreal Project)` - 2 edges
4. `Task Lifecycle (Open/In progress/Done)` - 2 edges
5. `Task File Fields (status/requested-by/assigned-to/date/What/Why/Notes)` - 2 edges
6. `Code-only graphify rebuild for CI. No LLM key available here, so semantic…` - 1 edges
7. `depth_estimator.py — monocular depth for the block tracker, using Depth…` - 1 edges
8. `model_id : HuggingFace repo id of a DA3 checkpoint. device : "cuda" / "cpu" /…` - 1 edges
9. `Launch the background inference worker (no-op if unavailable).` - 1 edges
10. `Hand the worker the latest frame (BGR, as OpenCV gives it).` - 1 edges

## Surprising Connections (you probably didn't know these)
- `Digital Twin Scene Screenshot` --conceptually_related_to--> `WorldGridMaterial Shader (DDC Key)`  [INFERRED]
  MT03_RealTimeLayout/Saved/AutoScreenshot.png → MT03_RealTimeLayout/Saved/ShaderDebugInfo/PCD3D_SM6/WorldGridMaterial_67bacdd81a65e5e9/Default/DDCKey-Editor.txt
- `Task File Fields (status/requested-by/assigned-to/date/What/Why/Notes)` --conceptually_related_to--> `Task Lifecycle (Open/In progress/Done)`  [INFERRED]
  tasks/TEMPLATE.md → tasks/README.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Task Filing Workflow** — agents_task_workflow, tasks_readme_doc, tasks_readme_task_lifecycle, tasks_template_doc [EXTRACTED 1.00]

## Communities (14 total, 9 thin omitted)

### Community 0 - "AGENTS.md Collaboration Guide"
Cohesion: 0.22
Nodes (9): detect.py, Git LFS Usage, inspect_bp.py, MechTwin03, MT03_RealTimeLayout (Unreal Project), OSC Protocol Link, Roadmap-Update Practice, send_test.py (+1 more)

### Community 1 - "Task Filing Workflow"
Cohesion: 0.25
Nodes (3): depth_estimator.py — monocular depth for the block tracker, using Depth…, # NOTE: absolute scale can still need per-rig calibration — tune, detect_xyz.py — block tracker with DEPTH (z), on top of the 2D tracker. Same…

### Community 2 - "Marker Detection Pipeline"
Cohesion: 0.50
Nodes (3): File-Instead-of-Ask Practice, Task Lifecycle (Open/In progress/Done), Task File Fields (status/requested-by/assigned-to/date/What/Why/Notes)

## Knowledge Gaps
- **7 isolated node(s):** `MechTwin03`, `send_test.py`, `inspect_bp.py`, `OSC Protocol Link`, `Git LFS Usage` (+2 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **9 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `DepthEstimator` connect `Tracked-Block Scene Capture` to `Task Filing Workflow`, `README Unreal Engine Integration`, `Roadmap Living Document`, `Blueprint Inspection Tool`, `README OSC Networking`, `OSC Send Test Script`?**
  _High betweenness centrality (0.172) - this node is a cross-community bridge._
- **What connects `MechTwin03`, `send_test.py`, `inspect_bp.py` to the rest of the system?**
  _7 weakly-connected nodes found - possible documentation gaps or missing edges._