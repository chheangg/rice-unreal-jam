"""Code-only graphify rebuild for CI. No LLM key available here, so semantic
extraction is skipped for anything not already in graphify-out/cache/semantic
(from a prior local /graphify run) - run /graphify locally to refresh docs/images.
"""
import json
from pathlib import Path

from graphify.detect import detect, save_manifest
from graphify.extract import extract, collect_files
from graphify.cache import check_semantic_cache
from graphify.build import build_from_json
from graphify.cluster import cluster, score_all
from graphify.analyze import god_nodes, surprising_connections, suggest_questions
from graphify.report import generate
from graphify.export import to_json

ROOT = "."

result = detect(Path(ROOT))

code_files = []
for f in result["files"].get("code", []):
    p = Path(f)
    code_files.extend(collect_files(p) if p.is_dir() else [p])

if code_files:
    ast = extract(code_files, cache_root=Path(ROOT))
else:
    ast = {"nodes": [], "edges": [], "input_tokens": 0, "output_tokens": 0}

doc_files = [f for cat in ("document", "paper", "image") for f in result["files"].get(cat, [])]
cached_nodes, cached_edges, cached_hyperedges, uncached = check_semantic_cache(doc_files, root=ROOT)
if uncached:
    print(f"Skipping {len(uncached)} uncached doc/image file(s) - no LLM in CI, run /graphify locally to include them")

seen = {n["id"] for n in ast["nodes"]}
merged_nodes = list(ast["nodes"])
for n in cached_nodes:
    if n["id"] not in seen:
        merged_nodes.append(n)
        seen.add(n["id"])

merged = {
    "nodes": merged_nodes,
    "edges": ast["edges"] + cached_edges,
    "hyperedges": cached_hyperedges,
    "input_tokens": 0,
    "output_tokens": 0,
}

G = build_from_json(merged, root=ROOT, directed=False)
if G.number_of_nodes() == 0:
    print("CI graphify: extraction produced no nodes, leaving existing graph.json untouched")
    raise SystemExit(0)

communities = cluster(G)
cohesion = score_all(G, communities)
gods = god_nodes(G)
surprises = surprising_connections(G, communities)

labels_path = Path("graphify-out/.graphify_labels.json")
if labels_path.exists():
    saved_labels = json.loads(labels_path.read_text(encoding="utf-8"))
    labels = {cid: saved_labels.get(str(cid), f"Community {cid}") for cid in communities}
else:
    labels = {cid: f"Community {cid}" for cid in communities}

questions = suggest_questions(G, communities, labels)

wrote = to_json(G, communities, "graphify-out/graph.json")
if not wrote:
    print("CI graphify: shrink-guard held graph.json unchanged (new graph not larger)")
else:
    report = generate(
        G, communities, cohesion, labels, gods, surprises, result,
        {"input": 0, "output": 0}, ROOT, suggested_questions=questions,
    )
    Path("graphify-out/GRAPH_REPORT.md").write_text(report, encoding="utf-8")
    print(f"CI graphify: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, {len(communities)} communities")

save_manifest(result.get("all_files") or result["files"], root=ROOT)
