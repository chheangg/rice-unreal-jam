"""
inspect_bp.py  --  Deep probe of graph accessibility.
Run: py "H:/Unreal + Ai try/Unreal_MechTwin03/inspect_bp.py"
"""
import unreal

BP  = "/Game/BP_OSCreciver"
bel = unreal.BlueprintEditorLibrary

bp    = unreal.load_asset(BP)
graph = bel.find_event_graph(bp)

# ── 1. All Python-visible attributes on the graph object ─────────
attrs = [a for a in dir(graph) if not a.startswith("_")]
unreal.log("\ndir(graph):\n  " + "\n  ".join(attrs))

# ── 2. list_events — do we get names or node objects? ────────────
unreal.log("\n--- list_events ---")
try:
    events = list(bel.list_events(bp))
    unreal.log(f"  count={len(events)}")
    for e in events:
        unreal.log(f"  item: {e!r}  type={type(e).__name__}  class={getattr(e,'get_class',lambda:None)()}")
        # Try using each event as a K2Node
        for fn in ["list_all_pins", "list_input_pins", "list_output_pins",
                   "find_then_pin", "find_execute_pin", "get_node_title"]:
            try:
                result = getattr(bel, fn)(e)
                unreal.log(f"    bel.{fn}(event) -> {result}")
                break
            except Exception as ex:
                unreal.log(f"    bel.{fn}: {ex}")
except Exception as e:
    unreal.log(f"  FAILED: {e}")

# ── 3. graph.get_schema() ────────────────────────────────────────
unreal.log("\n--- graph.get_schema() ---")
try:
    schema = graph.get_schema()
    unreal.log(f"  schema class: {schema.get_class().get_name()}")
    schema_attrs = [a for a in dir(schema) if not a.startswith("_")]
    unreal.log("  dir(schema):\n    " + "\n    ".join(schema_attrs))
except Exception as e:
    unreal.log(f"  FAILED: {e}")

# ── 4. Try iteration / sequence protocols on graph ───────────────
unreal.log("\n--- sequence probes on graph ---")
try:
    unreal.log(f"  len(graph) = {len(graph)}")
except Exception as e:
    unreal.log(f"  len: {e}")
try:
    first = next(iter(graph))
    unreal.log(f"  iter -> {first}")
except Exception as e:
    unreal.log(f"  iter: {e}")

# ── 5. call_method probes ────────────────────────────────────────
unreal.log("\n--- call_method probes on graph ---")
for m in ["GetAllNodes", "GetNodes", "Nodes", "GetNodeCount"]:
    try:
        r = graph.call_method(m)
        unreal.log(f"  {m} -> {r}")
    except Exception as e:
        unreal.log(f"  {m}: {e}")

# ── 6. editor property probes ────────────────────────────────────
unreal.log("\n--- editor property probes on graph ---")
for p in ["Schema", "SubGraphs", "bEditable", "InterfaceGuid",
          "bAllowDeletion", "GraphGuid", "bAllowRenaming"]:
    try:
        v = graph.get_editor_property(p)
        unreal.log(f"  {p} = {v!r}")
    except Exception as e:
        unreal.log(f"  {p}: {e}")

unreal.log("\n=== paste full output back to Claude ===\n")
