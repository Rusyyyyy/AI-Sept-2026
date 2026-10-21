"""Graph-search algorithms shared by Labs 1-3: BFS, DFS, UCS, GBFS and A*.

A graph is a dictionary ``{state: [(neighbour, cost), ...]}``. Every algorithm
returns the same ``SearchResult`` so that results can be compared in one table.
``expanded`` counts the states whose neighbours were generated; the goal state
itself is not expanded, so a start that equals the goal expands nothing.
 
graph = {
    "KUL": [("AMS", 10236.7), ("HKT", 837.0), ...],
    "AMS": [("KEF", 2295.0), ...],
    ...
}"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from heapq import heappop, heappush
from math import inf
from time import perf_counter


@dataclass(frozen=True)
class SearchResult:
    """What every search returns: the path plus the effort spent finding it."""

    found: bool
    path: list
    hops: int
    path_cost_km: float
    expanded: int
    max_frontier: int
    runtime_ms: float

    def as_dict(self) -> dict:
        return asdict(self)


def _build_path(parent: dict, goal) -> list:
    """Follow parent links back from the goal to the start."""
    path = []
    state = goal
    while state is not None:
        path.append(state)
        state = parent[state]
    return path[::-1]


def _path_cost(path: list, graph: dict) -> float:
    """Sum the edge costs along a path (0 for a single-state path)."""
    edges = {state: dict(neighbours) for state, neighbours in graph.items()}
    return sum(edges[a][b] for a, b in zip(path, path[1:]))

"""
edges = {
    "KUL": {"AMS": 10236.7, "HKT": 837.0},
    "AMS": {"KEF": 2295.0}
}
"""


def _result(found, parent, goal, graph, expanded, max_frontier, started) -> SearchResult:
    path = _build_path(parent, goal) if found else []
    cost = _path_cost(path, graph) if found else inf
    runtime_ms = (perf_counter() - started) * 1000
    return SearchResult(found, path, len(path) - 1, cost, expanded, max_frontier, runtime_ms)


def bfs(graph: dict, start, goal) -> SearchResult:
    """Breadth-first search: fewest edges, ignores edge costs."""
    started = perf_counter()
    frontier = deque([start])
    parent = {start: None}
    expanded, max_frontier = 0, 1
    while frontier:
        state = frontier.popleft()
        if state == goal:
            return _result(True, parent, goal, graph, expanded, max_frontier, started)
        expanded += 1
        for neighbour, _ in graph.get(state, ()):
            if neighbour not in parent:
                parent[neighbour] = state
                frontier.append(neighbour)
        max_frontier = max(max_frontier, len(frontier))
    return _result(False, parent, goal, graph, expanded, max_frontier, started)


def dfs(graph: dict, start, goal) -> SearchResult:
    """Depth-first search: follows the first listed neighbour as deep as it can."""
    started = perf_counter()
    frontier = [start]
    parent = {start: None}
    expanded, max_frontier = 0, 1
    while frontier:
        state = frontier.pop()
        if state == goal:
            return _result(True, parent, goal, graph, expanded, max_frontier, started)
        expanded += 1
        # Pushed in reverse so that the first listed neighbour is popped first.
        for neighbour, _ in reversed(graph.get(state, ())):
            if neighbour not in parent:
                parent[neighbour] = state
                frontier.append(neighbour)
        max_frontier = max(max_frontier, len(frontier))
    return _result(False, parent, goal, graph, expanded, max_frontier, started)


def best_first(graph: dict, start, goal, heuristic=None, mode: str = "ucs") -> SearchResult:
    """One loop for UCS (priority g), GBFS (priority h) and A* (priority g + h)."""
    if mode not in ("ucs", "gbfs", "astar"):
        raise ValueError("mode must be 'ucs', 'gbfs' or 'astar'")
    for neighbours in graph.values():
        if any(cost < 0 for _, cost in neighbours):
            raise ValueError("edge costs must be non-negative")
    h = heuristic or (lambda state: 0.0)

    def priority(state, g):
        if mode == "ucs":
            return g
        if mode == "gbfs":
            return h(state)
        return g + h(state)

    started = perf_counter()
    counter = 0                                  # tie-breaker so states are never compared
    frontier = [(priority(start, 0.0), counter, start, 0.0)]
    best_g = {start: 0.0}
    parent = {start: None}
    expanded_states = set()
    max_frontier = 1
    while frontier:
        _, _, state, g = heappop(frontier)
        if state in expanded_states:
            continue                             # an older, costlier entry for this state
        if state == goal:
            return _result(True, parent, goal, graph, len(expanded_states), max_frontier, started)
        expanded_states.add(state)
        for neighbour, cost in graph.get(state, ()):
            new_g = g + cost
            if neighbour not in expanded_states and new_g < best_g.get(neighbour, inf):
                best_g[neighbour] = new_g
                parent[neighbour] = state
                counter += 1
                heappush(frontier, (priority(neighbour, new_g), counter, neighbour, new_g))
        max_frontier = max(max_frontier, len(frontier))
    return _result(False, parent, goal, graph, len(expanded_states), max_frontier, started)


def ucs(graph: dict, start, goal) -> SearchResult:
    """Uniform-cost search: cheapest path for non-negative edge costs."""
    return best_first(graph, start, goal, mode="ucs")


def gbfs(graph: dict, start, goal, heuristic) -> SearchResult:
    """Greedy best-first search: fastest to run, no optimality guarantee."""
    return best_first(graph, start, goal, heuristic, mode="gbfs")


def astar(graph: dict, start, goal, heuristic) -> SearchResult:
    """A* search: optimal here when the heuristic is consistent."""
    return best_first(graph, start, goal, heuristic, mode="astar")
