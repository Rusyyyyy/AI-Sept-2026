"""Lab 2 of 8 — Heuristic Search on the Airline Network
Run this directly in VS Code: click "Run Python File" (top-right play button)
or type `python lab02_informed_search.py` in the integrated terminal.

Expected location: all_experiments/lab02_informed_search.py
(so that ../all_datasets and utils/ are found correctly)
"""
from heapq import heappop, heappush
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import ucs, gbfs, astar, haversine_km, load_openflights

ROOT = Path(__file__).resolve().parent.parent      # the AI course folder
DATA = ROOT / "all_datasets"
RESULTS = ROOT / "results" / "lab02"
RESULTS.mkdir(parents=True, exist_ok=True)

_, _, coords, graph, _ = load_openflights(DATA / "lab01-03_airports.csv", DATA / "lab01-03_routes.csv")
queries = pd.read_csv("utils/search_queries.csv").head(3)
print(queries)


def report(name, query, result):
    print(f"{query.query_id} {name:<5} {result.path_cost_km:>9,.1f} km "
          f"{result.expanded:>5} expanded   {' -> '.join(result.path)}")


# ---------------------------------------------------------------------------
# Question 1 — Establish the baseline (UCS)
# ---------------------------------------------------------------------------
print("\n=== Question 1: UCS baseline ===")
baseline = {}
for query in queries.itertuples():
    baseline[query.query_id] = ucs(graph, query.start, query.goal)
    report("UCS", query, baseline[query.query_id])


# ---------------------------------------------------------------------------
# Question 2 — Build the heuristic, run GBFS and A*
# ---------------------------------------------------------------------------
print("\n=== Question 2: GBFS and A* with a Haversine heuristic ===")

def heuristic_to(goal, weight=1.0):
    """h(airport) = weight x great-circle distance from the airport to the goal."""
    def h(airport):
        return weight * haversine_km(
            coords[airport]["latitude"], coords[airport]["longitude"],
            coords[goal]["latitude"], coords[goal]["longitude"],
        )
    return h


informed = {}
for query in queries.itertuples():
    h = heuristic_to(query.goal)
    informed[query.query_id, "GBFS"] = gbfs(graph, query.start, query.goal, h)
    informed[query.query_id, "A*"] = astar(graph, query.start, query.goal, h)
    report("GBFS", query, informed[query.query_id, "GBFS"])
    report("A*", query, informed[query.query_id, "A*"])


# ---------------------------------------------------------------------------
# Question 3 — Check admissibility empirically
# ---------------------------------------------------------------------------
print("\n=== Question 3: Checking the heuristic never overestimates ===")

def reverse(graph):
    reversed_graph = {airport: [] for airport in graph}
    for airport, neighbours in graph.items():
        for neighbour, cost in neighbours:
            reversed_graph[neighbour].append((airport, cost))
    return reversed_graph


def cost_to_all(graph, start):
    """Dijkstra: cheapest cost from start to every reachable state."""
    best = {start: 0.0}
    frontier = [(0.0, start)]
    while frontier:
        cost, state = heappop(frontier)
        if cost > best[state]:
            continue
        for neighbour, edge in graph[state]:
            if cost + edge < best.get(neighbour, np.inf):
                best[neighbour] = cost + edge
                heappush(frontier, (cost + edge, neighbour))
    return best


reversed_graph = reverse(graph)
checks = []
for query in queries.itertuples():
    exact = cost_to_all(reversed_graph, query.goal)          # exact cost-to-go for every airport
    h = heuristic_to(query.goal)
    airports = sorted((a for a in exact if a != query.goal), key=exact.get)   # nearest first
    for airport in [airports[i] for i in np.linspace(0, len(airports) - 1, 10, dtype=int)]:
        checks.append({
            "query_id": query.query_id, "airport": airport, "goal": query.goal,
            "haversine_km": h(airport), "shortest_route_km": exact[airport],
        })

check = pd.DataFrame(checks)
check["slack_km"] = check.shortest_route_km - check.haversine_km
check["admissible"] = check.slack_km >= -1e-6
check.to_csv(RESULTS / "heuristic_check.csv", index=False)
print("Admissibility violations:", int((~check.admissible).sum()), "of", len(check))
print(check.head(10).round(1))


# ---------------------------------------------------------------------------
# Question 4 — Compare the methods
# ---------------------------------------------------------------------------
print("\n=== Question 4: UCS vs GBFS vs A*, side by side ===")
rows = []
for query in queries.itertuples():
    results = {
        "UCS": baseline[query.query_id],
        "GBFS": informed[query.query_id, "GBFS"],
        "A*": informed[query.query_id, "A*"],
    }
    for name, result in results.items():
        rows.append({
            "query_id": query.query_id, "algorithm": name, **result.as_dict(),
            "path": " -> ".join(result.path),
        })

metrics = pd.DataFrame(rows).drop(columns="found").rename(columns={"path_cost_km": "cost_km"})
ucs_cost = metrics[metrics.algorithm == "UCS"].set_index("query_id").cost_km
metrics["gap_pct"] = 100 * (metrics.cost_km / metrics.query_id.map(ucs_cost) - 1)
print(metrics.drop(columns="path").round(2))

ucs_rows, astar_rows = metrics[metrics.algorithm == "UCS"], metrics[metrics.algorithm == "A*"]
print("A* matches UCS on every query:", np.allclose(ucs_rows.cost_km.values, astar_rows.cost_km.values))
print("Mean A* expansion reduction vs UCS: {:.1%}".format(
    1 - astar_rows.expanded.mean() / ucs_rows.expanded.mean()
))


# ---------------------------------------------------------------------------
# Question 5 — Stress the guarantee (Weighted A*)
# ---------------------------------------------------------------------------
print("\n=== Question 5: Weighted A* (1.5h) breaks the guarantee ===")
weighted = []
for query in queries.itertuples():
    result = astar(graph, query.start, query.goal, heuristic_to(query.goal, weight=1.5))
    weighted.append({
        "query_id": query.query_id, "algorithm": "Weighted A* (1.5h)", **result.as_dict(),
        "path": " -> ".join(result.path),
    })

weighted = pd.DataFrame(weighted).drop(columns="found").rename(columns={"path_cost_km": "cost_km"})
weighted["gap_pct"] = 100 * (weighted.cost_km / weighted.query_id.map(ucs_cost) - 1)
metrics = pd.concat([metrics, weighted], ignore_index=True)
metrics.to_csv(RESULTS / "metrics.csv", index=False)
print(metrics[metrics.algorithm != "GBFS"][["query_id", "algorithm", "cost_km", "gap_pct", "expanded"]].round(2))


# ---------------------------------------------------------------------------
# Question 6 — Save the evidence
# ---------------------------------------------------------------------------
print("\n=== Question 6: Save the expanded-states chart ===")
effort = metrics[metrics.algorithm.isin(["UCS", "A*", "GBFS"])].pivot(
    index="query_id", columns="algorithm", values="expanded"
)
ax = effort[["UCS", "A*", "GBFS"]].plot.bar(figsize=(8, 4), logy=True)
ax.set(ylabel="Expanded states (log scale)", title="Search effort on the three standard queries")
ax.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig(RESULTS / "expanded_nodes.png", dpi=160)

print("Saved:", sorted(p.name for p in RESULTS.iterdir()))
print("\nLab 2 complete.")
