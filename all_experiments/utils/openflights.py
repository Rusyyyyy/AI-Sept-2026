"""Load the OpenFlights airport and route files into a weighted graph."""
from __future__ import annotations

from math import atan2, cos, radians, sin, sqrt
from pathlib import Path

import pandas as pd

AIRPORT_COLUMNS = [
    "airport_id",
    "name",
    "city",
    "country",
    "iata",
    "icao",
    "latitude",
    "longitude",
    "altitude",
    "timezone",
    "dst",
    "tz_database",
    "type",
    "source",
]

ROUTE_ALIASES = {
    "source airport": "source",
    "source airport id": "source_id",
    # The supplied CSV contains this historical misspelling.
    "destination apirport": "destination",
    "destination airport": "destination",
    "destination airport id": "destination_id",
}


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return great-circle distance in kilometres between two coordinates."""

    radius_km = 6371.0088
    phi1, phi2 = radians(lat1), radians(lat2)
    delta_phi = radians(lat2 - lat1)
    delta_lambda = radians(lon2 - lon1)
    value = sin(delta_phi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(delta_lambda / 2) ** 2
    value = min(1.0, max(0.0, value))  # protect against floating-point drift
    return 2 * radius_km * atan2(sqrt(value), sqrt(1 - value))


def load_openflights(airports_path: str | Path, routes_path: str | Path):
    """Load the supplied OpenFlights files into a deterministic directed graph."""

    airports_path = Path(airports_path)
    routes_path = Path(routes_path)
    if not airports_path.is_file() or not routes_path.is_file():
        raise FileNotFoundError("The OpenFlights airport and route CSV files are required.")

    airports = pd.read_csv(airports_path, header=None, names=AIRPORT_COLUMNS, na_values="\\N")
    airports["iata"] = airports["iata"].astype("string").str.strip().str.upper()
    airports["latitude"] = pd.to_numeric(airports["latitude"], errors="coerce")
    airports["longitude"] = pd.to_numeric(airports["longitude"], errors="coerce")
    valid_airports = airports[
        airports["iata"].notna()
        & airports["iata"].str.fullmatch(r"[A-Z0-9]{3}", na=False)
        & airports["latitude"].between(-90, 90, inclusive="both")
        & airports["longitude"].between(-180, 180, inclusive="both")
    ].copy()
    valid_airports = valid_airports.drop_duplicates("iata", keep="first")
    coords = valid_airports.set_index("iata")[["latitude", "longitude"]].to_dict("index")

    routes = pd.read_csv(routes_path, na_values="\\N")
    routes.columns = [str(column).strip().lower() for column in routes.columns]
    routes = routes.rename(columns=ROUTE_ALIASES)
    if not {"source", "destination"}.issubset(routes.columns):
        raise ValueError(f"Unexpected route columns: {routes.columns.tolist()}")

    routes["source"] = routes["source"].astype("string").str.strip().str.upper()
    routes["destination"] = routes["destination"].astype("string").str.strip().str.upper()
    raw_route_count = len(routes)
    cleaned_routes = (
        routes[
            routes["source"].isin(coords)
            & routes["destination"].isin(coords)
            & routes["source"].ne(routes["destination"])
        ][["source", "destination"]]
        .dropna()
        .drop_duplicates()
        .copy()
    )

    adjacency: dict[str, list[tuple[str, float]]] = {code: [] for code in coords}
    for source, destination in cleaned_routes.itertuples(index=False):
        start, end = coords[source], coords[destination]
        distance = haversine_km(
            start["latitude"], start["longitude"], end["latitude"], end["longitude"]
        )
        adjacency[source].append((destination, distance))
    for source in adjacency:
        adjacency[source].sort(key=lambda item: item[0])

    stats = {
        "airport_rows": len(airports),
        "valid_airports": len(valid_airports),
        "route_rows": raw_route_count,
        "unique_valid_routes": len(cleaned_routes),
        "removed_route_rows": raw_route_count - len(cleaned_routes),
    }
    return valid_airports, cleaned_routes, coords, adjacency, stats


def route_is_valid(path: list[str], adjacency) -> bool:
    """Return True only when a non-empty path uses graph edges throughout."""

    if not path:
        return False
    edge_sets = {state: {neighbour for neighbour, _ in edges} for state, edges in adjacency.items()}
    return all(v in edge_sets.get(u, set()) for u, v in zip(path, path[1:]))
