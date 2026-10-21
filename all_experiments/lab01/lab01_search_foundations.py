"""
Lab 1 of 8 - Search Foundations on an Airline Network
Objective: Implement and compare BFS, DFS and Uniform-Cost Search on one cleaned airline graph.

Plain Python script version of the Lab 1 notebook (no Jupyter needed).
Save this file in:  AI/all_experiments/
Run it with the Run button (top right in VS Code) or:  python lab01_search_foundations.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

# utils/ must sit next to this script, inside all_experiments/
from utils import bfs, dfs, ucs, haversine_km, load_openflights, plot_route_paths, route_is_valid

# ---------------------------------------------------------------------------
# Setup: the three folders every lab uses
# Paths are built from this file's location, so it works from any terminal folder.
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent      # .../AI/all_experiments
ROOT = HERE.parent                          # .../AI  (the course folder)
DATA = ROOT / "all_datasets"
RESULTS = ROOT / "results" / "lab01"
RESULTS.mkdir(parents=True, exist_ok=True)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)


def heading(text):
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


# ---------------------------------------------------------------------------
# Question 1 - Load and validate the graph
# ---------------------------------------------------------------------------
heading("Question 1 - Load and validate the graph")

airports, routes, coords, graph, stats = load_openflights(
    DATA / "lab01-03_airports.csv",
    DATA / "lab01-03_routes.csv",
)
queries = pd.read_csv(HERE / "utils" / "search_queries.csv").head(3)

print(pd.Series(stats, name="count").to_frame())
print()
print(queries)

# ---------------------------------------------------------------------------
# Question 2 - Define the edge cost (great-circle distance in km)
# ---------------------------------------------------------------------------
heading("Question 2 - Define the edge cost")

sample = pd.DataFrame(graph["KUL"][:5], columns=["destination", "distance_km"])
print(sample.round(1))

direct_km = haversine_km(
    coords["KUL"]["latitude"], coords["KUL"]["longitude"],
    coords["KEF"]["latitude"], coords["KEF"]["longitude"],
)
print(f"\nKUL has {len(graph['KUL'])} outgoing routes")
print(f"Great-circle distance KUL -> KEF: {direct_km:,.0f} km (no direct route exists)")

# ---------------------------------------------------------------------------
# Question 3 - Run BFS, DFS and UCS on the three standard queries
# ---------------------------------------------------------------------------
heading("Question 3 - Run BFS, DFS and UCS")

rows = []
for query in queries.itertuples():
    for name, algorithm in [("BFS", bfs), ("DFS", dfs), ("UCS", ucs)]:
        result = algorithm(graph, query.start, query.goal)
        rows.append({
            "query_id": query.query_id,
            "start": query.start,
            "goal": query.goal,
            "algorithm": name,
            **result.as_dict(),
            "path": " -> ".join(result.path),
        })

metrics = pd.DataFrame(rows).drop(columns="found")
metrics.to_csv(RESULTS / "metrics.csv", index=False)
print(metrics.drop(columns=["start", "goal", "path"]).round(2))

# ---------------------------------------------------------------------------
# Question 4 - Verify every path independently
# ---------------------------------------------------------------------------
heading("Question 4 - Verify every path")

edge_km = {airport: dict(neighbours) for airport, neighbours in graph.items()}

for row in metrics.itertuples():
    path = row.path.split(" -> ")
    assert route_is_valid(path, graph), f"{row.algorithm} {row.query_id}: a leg is not a real route"
    recomputed = sum(edge_km[a][b] for a, b in zip(path, path[1:]))
    assert np.isclose(recomputed, row.path_cost_km), f"{row.algorithm} {row.query_id}: cost mismatch"

print(f"Verified {len(metrics)} paths: every leg exists and every cost matches its edge sum")

# ---------------------------------------------------------------------------
# Question 5 - Compare the objectives (hops vs kilometres)
# ---------------------------------------------------------------------------
heading("Question 5 - Compare the objectives")

print(metrics.pivot(index="query_id", columns="algorithm", values=["hops", "path_cost_km"]).round(0))
print()

reversed_graph = {airport: neighbours[::-1] for airport, neighbours in graph.items()}
for query in queries.itertuples():
    original = dfs(graph, query.start, query.goal).hops
    reordered = dfs(reversed_graph, query.start, query.goal).hops
    print(f"{query.query_id} DFS: {original} hops with the original order, "
          f"{reordered} with the reversed order")

# ---------------------------------------------------------------------------
# Question 6 - Save the evidence (route comparison plot)
# ---------------------------------------------------------------------------
heading("Question 6 - Save the evidence")

q1 = metrics[metrics.query_id == "Q1"]
paths = {row.algorithm: row.path.split(" -> ") for row in q1.itertuples()}
figure, _ = plot_route_paths(paths, coords, title="Route comparison: KUL to KEF")
figure.savefig(RESULTS / "route_comparison.png", dpi=160, bbox_inches="tight")

print("Saved:", sorted(p.name for p in RESULTS.iterdir()))
