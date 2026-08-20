# Graph Report - rice-unreal-jam  (2026-08-20)

## Corpus Check
- Corpus is ~10,030 words - fits in a single context window. You may not need a graph.

## Summary
- 36 nodes · 39 edges · 10 communities (7 shown, 3 thin omitted)
- Extraction: 90% EXTRACTED · 10% INFERRED · 0% AMBIGUOUS · INFERRED: 4 edges (avg confidence: 0.82)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- AGENTS.md Collaboration Guide
- Task Filing Workflow
- README Project Setup
- Tracked-Block Scene Capture
- README Unreal Engine Integration
- Roadmap Living Document
- Blueprint Inspection Tool
- README OSC Networking

## God Nodes (most connected - your core abstractions)
1. `MT03_RealTimeLayout (Unreal Project)` - 5 edges
2. `detect.py` - 4 edges
3. `detect.py` - 3 edges
4. `Digital Twin Scene Screenshot` - 3 edges
5. `MT03_RealTimeLayout (Unreal Project)` - 2 edges
6. `send_test.py` - 2 edges
7. `inspect_bp.py` - 2 edges
8. `OSC Protocol Link` - 2 edges
9. `Unreal Engine 5.8` - 2 edges
10. `Task Lifecycle (Open/In progress/Done)` - 2 edges

## Surprising Connections (you probably didn't know these)
- `Digital Twin Scene Screenshot` --shares_data_with--> `MT03_RealTimeLayout (Unreal Project)`  [INFERRED]
  MT03_RealTimeLayout/Saved/AutoScreenshot.png → README.md
- `Digital Twin Scene Screenshot` --conceptually_related_to--> `detect.py`  [INFERRED]
  MT03_RealTimeLayout/Saved/AutoScreenshot.png → README.md
- `Digital Twin Scene Screenshot` --conceptually_related_to--> `WorldGridMaterial Shader (DDC Key)`  [INFERRED]
  MT03_RealTimeLayout/Saved/AutoScreenshot.png → MT03_RealTimeLayout/Saved/ShaderDebugInfo/PCD3D_SM6/WorldGridMaterial_67bacdd81a65e5e9/Default/DDCKey-Editor.txt
- `Task File Fields (status/requested-by/assigned-to/date/What/Why/Notes)` --conceptually_related_to--> `Task Lifecycle (Open/In progress/Done)`  [INFERRED]
  tasks/TEMPLATE.md → tasks/README.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Task Filing Workflow** — agents_task_workflow, tasks_readme_doc, tasks_readme_task_lifecycle, tasks_template_doc [EXTRACTED 1.00]
- **Physical-to-Digital Twin OSC Pipeline** — readme_detect_py, readme_osc, readme_mt03_realtimelayout, mt03_realtimelayout_saved_autoscreenshot_scene [INFERRED 0.85]

## Communities (10 total, 3 thin omitted)

### Community 0 - "AGENTS.md Collaboration Guide"
Cohesion: 0.22
Nodes (9): detect.py, Git LFS Usage, inspect_bp.py, MechTwin03, MT03_RealTimeLayout (Unreal Project), OSC Protocol Link, Roadmap-Update Practice, send_test.py (+1 more)

### Community 1 - "Task Filing Workflow"
Cohesion: 0.50
Nodes (3): File-Instead-of-Ask Practice, Task Lifecycle (Open/In progress/Done), Task File Fields (status/requested-by/assigned-to/date/What/Why/Notes)

### Community 3 - "README Project Setup"
Cohesion: 0.50
Nodes (3): Git LFS Usage, MechTwin03, Python Tracking Dependencies (opencv-python, numpy, python-osc)

### Community 4 - "Tracked-Block Scene Capture"
Cohesion: 0.67
Nodes (3): Digital Twin Scene Screenshot, WorldGridMaterial Shader (DDC Key), detect.py

### Community 5 - "README Unreal Engine Integration"
Cohesion: 0.67
Nodes (3): inspect_bp.py, MT03_RealTimeLayout (Unreal Project), Unreal Engine 5.8

## Knowledge Gaps
- **9 isolated node(s):** `MechTwin03`, `send_test.py`, `inspect_bp.py`, `OSC Protocol Link`, `Git LFS Usage` (+4 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **3 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `detect.py` connect `Tracked-Block Scene Capture` to `README OSC Networking`, `README Project Setup`, `README Unreal Engine Integration`?**
  _High betweenness centrality (0.057) - this node is a cross-community bridge._
- **Why does `MT03_RealTimeLayout (Unreal Project)` connect `README Unreal Engine Integration` to `README Project Setup`, `Tracked-Block Scene Capture`?**
  _High betweenness centrality (0.043) - this node is a cross-community bridge._
- **Are the 3 inferred relationships involving `Digital Twin Scene Screenshot` (e.g. with `WorldGridMaterial Shader (DDC Key)` and `detect.py`) actually correct?**
  _`Digital Twin Scene Screenshot` has 3 INFERRED edges - model-reasoned connections that need verification._
- **What connects `MechTwin03`, `send_test.py`, `inspect_bp.py` to the rest of the system?**
  _9 weakly-connected nodes found - possible documentation gaps or missing edges._