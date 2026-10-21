"""Helper code for the AI course notebooks.

Labs 1-3 import the search algorithms, the OpenFlights loader and the route
plotter from here::

    from utils import bfs, dfs, ucs, gbfs, astar, load_openflights, haversine_km
"""

from .core import SearchResult, astar, best_first, bfs, dfs, gbfs, ucs
from .openflights import haversine_km, load_openflights, route_is_valid
from .plotting import plot_route_paths

__all__ = [
    "SearchResult", "astar", "best_first", "bfs", "dfs", "gbfs", "ucs",
    "haversine_km", "load_openflights", "route_is_valid", "plot_route_paths",
]
