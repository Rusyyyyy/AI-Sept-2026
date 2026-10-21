"""Unit tests for utils.core on tiny hand-made graphs (no dataset needed).

Run from the all_experiments folder:   python -m pytest utils -q
"""
import math

import pytest

from utils import astar, bfs, dfs, gbfs, ucs


def test_start_equals_goal():
    for algorithm in (bfs, dfs, ucs):
        result = algorithm({"A": []}, "A", "A")
        assert result.found and result.path == ["A"]
        assert result.hops == 0 and result.path_cost_km == 0
        assert result.expanded == 0


def test_unreachable_goal():
    graph = {"A": [("B", 1.0)], "B": [], "C": []}
    result = bfs(graph, "A", "C")
    assert not result.found and result.path == []
    assert result.hops == -1 and math.isinf(result.path_cost_km)


def test_cycles_terminate():
    graph = {"A": [("B", 1.0)], "B": [("A", 1.0), ("C", 1.0)], "C": []}
    assert dfs(graph, "A", "C").path == ["A", "B", "C"]


def test_bfs_and_ucs_optimize_different_objectives():
    graph = {"A": [("G", 9.0), ("B", 1.0)], "B": [("C", 1.0)], "C": [("G", 1.0)], "G": []}
    assert bfs(graph, "A", "G").path == ["A", "G"]
    assert ucs(graph, "A", "G").path == ["A", "B", "C", "G"]


def test_frontier_update_finds_cheaper_path():
    graph = {"A": [("B", 10.0), ("C", 1.0)], "C": [("B", 1.0)], "B": [("G", 1.0)], "G": []}
    result = ucs(graph, "A", "G")
    assert result.path == ["A", "C", "B", "G"]
    assert result.path_cost_km == pytest.approx(3.0)
    

def test_astar_matches_ucs_with_consistent_heuristic():
    graph = {"S": [("A", 2.0), ("B", 5.0)], "A": [("G", 5.0)], "B": [("G", 1.0)], "G": []}
    heuristic = {"S": 5.0, "A": 4.0, "B": 1.0, "G": 0.0}.__getitem__
    assert astar(graph, "S", "G", heuristic).path_cost_km == ucs(graph, "S", "G").path_cost_km


def test_gbfs_is_not_assumed_optimal():
    graph = {"S": [("A", 1.0), ("B", 2.0)], "A": [("G", 10.0)], "B": [("G", 2.0)], "G": []}
    heuristic = {"S": 2.0, "A": 0.1, "B": 1.0, "G": 0.0}.__getitem__
    assert gbfs(graph, "S", "G", heuristic).path_cost_km > ucs(graph, "S", "G").path_cost_km


def test_negative_cost_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        ucs({"A": [("B", -1.0)], "B": []}, "A", "B")
