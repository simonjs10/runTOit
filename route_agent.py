from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import folium
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pyproj import Transformer
from shapely.geometry import LineString, mapping

# ----------------------------
# Configuration / Data model
# ----------------------------

@dataclass(frozen=True)
class RouteRequest:
    start_lat: float
    start_lon: float

    # End point (optional; None => loop)
    end_lat: Optional[float] = None
    end_lon: Optional[float] = None

    # Distance target
    target_km: float = 10.0
    tolerance_km: float = 1.0

    # Scoring weights
    w_distance_match: float = 3.0
    target_trail_prop: float = 0.3   # ideal fraction of route that is trail (0.0–1.0)
    w_trail_match: float = 2         # weight for matching target_trail_prop
    w_avoid_arterials: float = 2

    # Search parameters
    candidates: int = 360
    waypoint_radius_km: float = 6.0
    seed: int = 42

    def is_point_to_point(self) -> bool:
        return self.end_lat is not None and self.end_lon is not None


@dataclass
class ScoredRoute:
    route_nodes: List[int]
    length_m: float
    trail_ratio: float
    arterial_ratio: float
    score: float


# ----------------------------
# Graph build & features
# ----------------------------

def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres between two WGS-84 points."""
    R = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def load_walk_graphs_for_request(req: 'RouteRequest'):

    ox.settings.use_cache   = True
    ox.settings.log_console = False   # suppress verbose download chatter

    # Radius must contain all waypoints + the full route with a safety margin
    base_m  = (req.waypoint_radius_km * 1000.0
               + req.target_km        * 500.0)   # half the target distance
    BUFFER  = 1.25

    if req.is_point_to_point():
        clat     = (req.start_lat + req.end_lat) / 2
        clon     = (req.start_lon + req.end_lon) / 2
        se_m     = _haversine_m(req.start_lat, req.start_lon,
                                req.end_lat,   req.end_lon)
        needed_m = max(base_m, se_m / 2 + req.waypoint_radius_km * 1000.0) * BUFFER
    else:
        clat, clon = req.start_lat, req.start_lon
        needed_m   = base_m * BUFFER

    print(f"  Graph centre ({clat:.4f}, {clon:.4f}), "
          f"radius {needed_m / 1000:.1f} km …")
    G_ll   = ox.graph_from_point((clat, clon), dist=needed_m,
                                  network_type='walk', simplify=True)
    G_proj = ox.project_graph(G_ll)
    return G_ll, G_proj


def _highway_set(edge_attrs: Dict) -> set:
    """Extract the highway tag(s) as a set of strings."""
    v = edge_attrs.get('highway')
    if v is None:
        return set()
    return {str(x) for x in (v if isinstance(v, list) else [v])}


_TRAIL_HIGHWAYS    = {'footway', 'path', 'track', 'bridleway', 'steps'}
_TRAIL_SURFACES    = {'dirt', 'gravel', 'ground', 'unpaved', 'fine_gravel', 'compacted'}
_ARTERIAL_HIGHWAYS = {'primary', 'secondary', 'trunk', 'primary_link', 'secondary_link', 'trunk_link'}


def edge_is_trailish(edge_attrs: Dict) -> bool:
    if _highway_set(edge_attrs) & _TRAIL_HIGHWAYS:
        return True
    surface = str(edge_attrs.get('surface', '')).lower()
    return any(s in surface for s in _TRAIL_SURFACES)


def edge_is_arterial(edge_attrs: Dict) -> bool:
    return bool(_highway_set(edge_attrs) & _ARTERIAL_HIGHWAYS)


def add_custom_edge_costs(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Tag each edge with trail_flag and arterial_flag; fill missing length."""
    for u, v, k, data in G.edges(keys=True, data=True):
        data.setdefault('length', 1.0)
        data['trail_flag']    = int(edge_is_trailish(data))
        data['arterial_flag'] = int(edge_is_arterial(data))
    return G


# ----------------------------
# Routing primitives
# ----------------------------

def route_stats(G: nx.MultiDiGraph, route_nodes: List[int]) -> Tuple[float, float, float]:
    """
    Single-pass computation of (length_m, trail_ratio, arterial_ratio).

    Replaces the separate route_length_m + route_edge_stats calls to avoid
    iterating the same edges twice per candidate.
    """
    total_m = trail_m = arterial_m = 0.0
    for u, v in zip(route_nodes[:-1], route_nodes[1:]):
        edata = min(G.get_edge_data(u, v).values(), key=lambda d: d.get('length', 0))
        L = float(edata.get('length', 0.0))
        total_m += L
        if edata.get('trail_flag'):    trail_m    += L
        if edata.get('arterial_flag'): arterial_m += L
    if total_m <= 0:
        return 0.0, 0.0, 0.0
    return total_m, trail_m / total_m, arterial_m / total_m


def _astar_path(G: nx.MultiDiGraph, a: int, b: int) -> Optional[List[int]]:
    """
    A* shortest path with a Euclidean admissible heuristic.

    On a projected (metre-unit) graph the straight-line distance is always
    <= the true road distance, so the heuristic is admissible and A* returns
    the optimal path while typically expanding 2-5x fewer nodes than Dijkstra.
    """
    try:
        return nx.astar_path(
            G, a, b,
            heuristic=lambda u, v: euclid_m(G, u, v),
            weight='length',
        )
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


def euclid_m(G: nx.MultiDiGraph, a: int, b: int) -> float:
    return math.hypot(G.nodes[a]['x'] - G.nodes[b]['x'],
                      G.nodes[a]['y'] - G.nodes[b]['y'])


def stitch_paths(paths: List[List[int]]) -> List[int]:
    """Concatenate node-paths, removing duplicate joints."""
    if not paths:
        return []
    stitched = list(paths[0])
    for p in paths[1:]:
        if p:
            stitched.extend(p[1:] if stitched[-1] == p[0] else p)
    return stitched


# ----------------------------
# Candidate generation + scoring
# ----------------------------

def sample_waypoint_nodes(
    G: nx.MultiDiGraph,
    anchor_node: int,
    radius_m: float,
    n: int,
    rng: np.random.Generator,
) -> List[int]:
    """
    Sample n random nodes within radius_m of anchor_node:
      1. draw uniform-disk (x, y) offsets
      2. snap to nearest graph nodes (vectorised)
    """
    x0, y0 = G.nodes[anchor_node]['x'], G.nodes[anchor_node]['y']
    angles = rng.uniform(0, 2 * np.pi, n)
    radii  = radius_m * np.sqrt(rng.uniform(0, 1, n))
    xs = x0 + radii * np.cos(angles)
    ys = y0 + radii * np.sin(angles)
    nodes = ox.distance.nearest_nodes(G, X=xs, Y=ys)
    return list(dict.fromkeys(int(u) for u in nodes))  # dedupe, preserve order


def score_route(req: RouteRequest, length_m: float, trail_ratio: float, arterial_ratio: float) -> float:
    """Higher is better."""
    dist_err_m = abs(length_m - req.target_km * 1000.0)
    dist_denom = max(req.tolerance_km * 1000.0, 200.0)
    distance_component = math.exp(-dist_err_m / dist_denom)

    # Trail match: exponential decay on absolute error vs target proportion.
    # Denom of 0.15 means a 15-percentage-point miss roughly halves the component.
    trail_component = math.exp(-abs(trail_ratio - req.target_trail_prop) / 0.15)

    return (
        req.w_distance_match * distance_component
        + req.w_trail_match  * trail_component
        - req.w_avoid_arterials * arterial_ratio
    )


def _try_route(G, cache: dict, *node_sequence) -> Optional[List[int]]:
    """
    Build a stitched route through node_sequence via A*, with per-session caching.

    `cache` is a {(a, b): path_or_None} dict shared across all candidates in one
    search session.  Repeated waypoint pairs are resolved instantly instead of
    re-running A*.
    """
    paths = []
    for a, b in zip(node_sequence[:-1], node_sequence[1:]):
        key = (a, b)
        if key not in cache:
            cache[key] = _astar_path(G, a, b)
        p = cache[key]
        if p is None:
            return None
        paths.append(p)
    return stitch_paths(paths)


def _evaluate_candidate(G, req, route) -> Optional[ScoredRoute]:
    """Return a ScoredRoute if route is within tolerance, else None."""
    length_m, trail_ratio, arterial_ratio = route_stats(G, route)
    if abs(length_m - req.target_km * 1000.0) > req.tolerance_km * 1000.0:
        return None
    score = score_route(req, length_m, trail_ratio, arterial_ratio)
    return ScoredRoute(route, length_m, trail_ratio, arterial_ratio, score)


def generate_best_loop(
    G: nx.MultiDiGraph, req: RouteRequest, start_node: int
) -> Optional[ScoredRoute]:
    rng      = np.random.default_rng(req.seed)
    radius_m = req.waypoint_radius_km * 1000.0
    pool     = sample_waypoint_nodes(G, start_node, radius_m, max(req.candidates, 200), rng)
    if len(pool) < 10:
        return None

    target_m = req.target_km   * 1000.0
    tol_m    = req.tolerance_km * 1000.0
    cache: dict = {}          # reuse A* paths across candidates
    best: Optional[ScoredRoute] = None
    for _ in range(req.candidates):
        w1, w2 = int(rng.choice(pool)), int(rng.choice(pool))
        # Euclidean lower-bound: the loop triangle cannot be shorter than its
        # straight-line perimeter, so reject geometrically impossible pairs early
        lb = (euclid_m(G, start_node, w1)
              + euclid_m(G, w1, w2)
              + euclid_m(G, w2, start_node))
        if lb > target_m + tol_m:
            continue
        route  = _try_route(G, cache, start_node, w1, w2, start_node)
        if route is None:
            continue
        cand = _evaluate_candidate(G, req, route)
        if cand and (best is None or cand.score > best.score):
            best = cand
    return best


def generate_best_point_to_point(
    G: nx.MultiDiGraph,
    req: RouteRequest,
    start_node: int,
    end_node: int,
) -> Optional[ScoredRoute]:
    """
    Point-to-point search on projected graph G (meters).
    NOTE: snap start_node / end_node using G_ll, not G_proj.
    """
    if not req.is_point_to_point():
        raise ValueError('end_lat/end_lon must be set for point-to-point mode.')

    rng      = np.random.default_rng(req.seed)
    radius_m = req.waypoint_radius_km * 1000.0
    n        = max(req.candidates, 300)

    # Sample near both endpoints to cover the full corridor
    pool = list(dict.fromkeys(
        sample_waypoint_nodes(G, start_node, radius_m, n, rng)
        + sample_waypoint_nodes(G, end_node, radius_m, n, rng)
    ))
    if len(pool) < 20:
        return None

    target_m = req.target_km   * 1000.0
    tol_m    = req.tolerance_km * 1000.0
    cache: dict = {}          # reuse A* paths across candidates
    best: Optional[ScoredRoute] = None

    for _ in range(req.candidates):
        w1, w2 = int(rng.choice(pool)), int(rng.choice(pool))
        if len({start_node, end_node, w1, w2}) < 4:
            continue
        # Early reject via Euclidean lower-bound (valid on projected graph)
        lb = euclid_m(G, start_node, w1) + euclid_m(G, w1, w2) + euclid_m(G, w2, end_node)
        if lb > target_m + tol_m:
            continue
        route = _try_route(G, cache, start_node, w1, w2, end_node)
        if route is None:
            continue
        cand = _evaluate_candidate(G, req, route)
        if cand and (best is None or cand.score > best.score):
            best = cand

    return best


# ----------------------------
# GeoJSON export + explanation
# ----------------------------

def route_edges_gdf_ll(G_ll, route_nodes):
    """GeoDataFrame of route edges in lat/lon (EPSG:4326)."""
    uvk = []
    for u, v in zip(route_nodes[:-1], route_nodes[1:]):
        data = G_ll.get_edge_data(u, v)
        if data is None:
            continue
        k_min = min(data, key=lambda k: data[k].get('length', float('inf')))
        uvk.append((u, v, k_min))
    gdf = ox.utils_graph.graph_to_gdfs(G_ll, nodes=False, fill_edge_geometry=True)
    gdf_route = gdf.loc[uvk].copy()
    try:
        gdf_route = gdf_route.to_crs(epsg=4326)
    except Exception:
        pass
    return gdf_route


def route_to_geojson_featurecollection(G_ll, route_nodes, properties=None):
    """Export route as GeoJSON FeatureCollection (one Feature per edge)."""
    gdf_route = route_edges_gdf_ll(G_ll, route_nodes)
    scalar_cols = [
        c for c in gdf_route.columns
        if c != 'geometry' and not gdf_route[c].map(lambda x: isinstance(x, (dict, list))).any()
    ]
    features = []
    for _, row in gdf_route.iterrows():
        geom = row.geometry
        if geom is None:
            continue
        geom_dict = json.loads(geom.to_json()) if hasattr(geom, 'to_json') else mapping(geom)
        props = {c: row[c] for c in scalar_cols if c in row and pd.notnull(row[c])}
        if properties:
            props.update(properties)
        features.append({'type': 'Feature', 'geometry': geom_dict, 'properties': props})
    return {'type': 'FeatureCollection', 'features': features}


def explain_route_choice(
    req: RouteRequest, length_m: float, trail_ratio: float, arterial_ratio: float
) -> Dict:
    """Human-readable score decomposition mirroring score_route()."""
    dist_err_m         = abs(length_m - req.target_km * 1000.0)
    dist_denom         = max(req.tolerance_km * 1000.0, 200.0)
    distance_component = math.exp(-dist_err_m / dist_denom)
    trail_err          = abs(trail_ratio - req.target_trail_prop)
    trail_component    = math.exp(-trail_err / 0.15)
    distance_term      = req.w_distance_match  * distance_component
    trail_term         = req.w_trail_match      * trail_component
    arterial_term      = -req.w_avoid_arterials * arterial_ratio
    return {
        'target_km':          req.target_km,
        'actual_km':          round(length_m / 1000.0, 3),
        'distance_error_km':  round(dist_err_m / 1000.0, 3),
        'target_trail_prop':  req.target_trail_prop,
        'actual_trail_ratio': round(trail_ratio, 4),
        'trail_error':        round(trail_err, 4),
        'arterial_ratio':     round(arterial_ratio, 4),
        'weights': {
            'w_distance_match': req.w_distance_match,
            'w_trail_match':    req.w_trail_match,
            'w_avoid_arterials': req.w_avoid_arterials,
        },
        'score_breakdown': {
            'distance_component_(0to1)': round(distance_component, 6),
            'distance_term':             round(distance_term, 6),
            'trail_component_(0to1)':    round(trail_component, 6),
            'trail_term':                round(trail_term, 6),
            'arterial_term':             round(arterial_term, 6),
            'total_score':               round(distance_term + trail_term + arterial_term, 6),
        },
        'notes': [
            'Distance component = exp(-|error_m|/denom); closer to target -> higher score.',
            'Trail component = exp(-|trail_ratio - target_trail_prop| / 0.15); 1.0 = perfect match.',
            'Arterial term penalises fraction tagged primary/secondary/trunk.',
        ],
    }


# ----------------------------
# Agent orchestration
# ----------------------------

def plan_route_agent(G_proj: nx.MultiDiGraph, G_ll: nx.MultiDiGraph, req: RouteRequest) -> Dict:
    # Snap using unprojected lat/lon graph
    start_node = ox.distance.nearest_nodes(G_ll, X=req.start_lon, Y=req.start_lat)

    if req.is_point_to_point():
        end_node = ox.distance.nearest_nodes(G_ll, X=req.end_lon, Y=req.end_lat)
        best = generate_best_point_to_point(G_proj, req, int(start_node), int(end_node))
    else:
        best = generate_best_loop(G_proj, req, start_node)

    if best is None:
        return {
            'ok': False,
            'error': (
                'No route found. Try increasing candidates, waypoint_radius_km, '
                'tolerance_km, or the graph extent.'
            ),
        }

    summary = {
        'length_m':       best.length_m,
        'length_km':      best.length_m / 1000.0,
        'trail_ratio':    best.trail_ratio,
        'arterial_ratio': best.arterial_ratio,
        'score':          best.score,
    }
    return {
        'ok':          True,
        'best':        best,
        'summary':     summary,
        'explanation': explain_route_choice(req, best.length_m, best.trail_ratio, best.arterial_ratio),
        'geojson':     route_to_geojson_featurecollection(G_ll, best.route_nodes, properties=summary),
    }


# ----------------------------
# Folium map rendering
# ----------------------------

def _best_edge_data(G: nx.MultiDiGraph, u: int, v: int) -> Optional[Dict]:
    """Return the shortest parallel edge data dict, or None if edge missing."""
    data_dict = G.get_edge_data(u, v)
    if not data_dict:
        return None
    return data_dict[min(data_dict, key=lambda k: data_dict[k].get('length', float('inf')))]


def _edge_geometry_xy(G: nx.MultiDiGraph, u: int, v: int) -> List[Tuple[float, float]]:
    """Return (x, y) coords for the shortest parallel edge; straight line as fallback."""
    data = _best_edge_data(G, u, v)
    if data and data.get('geometry'):
        return list(data['geometry'].coords)
    return [(G.nodes[u]['x'], G.nodes[u]['y']), (G.nodes[v]['x'], G.nodes[v]['y'])]


def _route_polyline_xy(G: nx.MultiDiGraph, route: List[int]) -> List[Tuple[float, float]]:
    """Densified XY polyline along route using edge geometries."""
    pts: List[Tuple[float, float]] = []
    for i, (u, v) in enumerate(zip(route[:-1], route[1:])):
        seg = _edge_geometry_xy(G, u, v)
        if i > 0 and pts and seg and pts[-1] == seg[0]:
            pts.extend(seg[1:])
        else:
            pts.extend(seg)
    return pts


def _km_markers_xy(
    G: nx.MultiDiGraph, route: List[int], km_step: float = 1.0
) -> List[Tuple[float, float, int]]:
    """Return (x, y, km_number) positions interpolated at each km_step along route."""
    step_m, next_m, cum_m, km_num = km_step * 1000.0, km_step * 1000.0, 0.0, 1
    markers: List[Tuple[float, float, int]] = []

    for u, v in zip(route[:-1], route[1:]):
        data     = _best_edge_data(G, u, v)
        seg_xy   = _edge_geometry_xy(G, u, v)
        edge_len = float(data.get('length', 0.0)) if data else math.hypot(
            G.nodes[u]['x'] - G.nodes[v]['x'], G.nodes[u]['y'] - G.nodes[v]['y']
        )
        if edge_len <= 0:
            continue
        line = LineString(seg_xy)
        while cum_m + edge_len >= next_m:
            t = (next_m - cum_m) / edge_len
            p = line.interpolate(t, normalized=True) if line.length > 0 else None
            markers.append((p.x, p.y, km_num) if p else (seg_xy[-1][0], seg_xy[-1][1], km_num))
            km_num += 1
            next_m += step_m
        cum_m += edge_len

    return markers


def route_to_folium_map(
    G_ll: nx.MultiDiGraph,
    G_proj: nx.MultiDiGraph,
    route: List[int],
    *,
    start_label: str = 'Start',
    end_label: str = 'End',
    km_step: float = 1.0,
) -> folium.Map:
    """Route polyline + Start/End markers + per-km distance markers on a Folium map."""
    transformer = Transformer.from_crs(G_proj.graph.get('crs'), 'EPSG:4326', always_xy=True)
    start_node, end_node = route[0], route[-1]
    m = folium.Map(
        location=[G_ll.nodes[start_node]['y'], G_ll.nodes[start_node]['x']],
        zoom_start=13,
    )

    # Route polyline (projected -> WGS84)
    poly_latlon = [transformer.transform(x, y)[::-1] for x, y in _route_polyline_xy(G_proj, route)]
    folium.PolyLine(poly_latlon, weight=5, opacity=0.85).add_to(m)

    # Start / End markers
    for node, label, color, icon in [
        (start_node, start_label, 'green', 'play'),
        (end_node,   end_label,   'red',   'stop'),
    ]:
        folium.Marker(
            location=[G_ll.nodes[node]['y'], G_ll.nodes[node]['x']],
            popup=label, tooltip=label,
            icon=folium.Icon(color=color, icon=icon),
        ).add_to(m)

    # KM markers
    for x, y, km_num in _km_markers_xy(G_proj, route, km_step):
        lat, lon = transformer.transform(x, y)[::-1]
        folium.CircleMarker(
            location=[lat, lon], radius=6,
            tooltip=f'{km_num} km', popup=f'{km_num} km',
            fill=True, fill_opacity=0.9,
        ).add_to(m)
        folium.map.Marker(
            [lat, lon],
            icon=folium.DivIcon(
                icon_size=(24, 24), icon_anchor=(12, 12),
                html=(
                    '<div style="width:24px;height:24px;display:flex;'
                    'align-items:center;justify-content:center;color:white;'
                    'font-size:11px;font-weight:700;pointer-events:none;'
                    'text-shadow:0 0 2px rgba(0,0,0,0.4);">'
                    f'{km_num}</div>'
                ),
            ),
        ).add_to(m)

    m.fit_bounds(poly_latlon)
    return m


# ----------------------------
# Pydantic API models
# ----------------------------

class RouteRequestBody(BaseModel):
    """Request body for POST /route — mirrors RouteRequest fields."""

    start_lat: float = Field(..., ge=-90,  le=90)
    start_lon: float = Field(..., ge=-180, le=180)

    # Optional end point (omit or set to None for a loop)
    end_lat: Optional[float] = Field(None, ge=-90,  le=90)
    end_lon: Optional[float] = Field(None, ge=-180, le=180)

    target_km:    float = Field(10.0, gt=0)
    tolerance_km: float = Field(1.0,  ge=0)

    w_distance_match:  float = Field(3.0, ge=0)
    target_trail_prop: float = Field(0.3, ge=0, le=1)
    w_trail_match:     float = Field(2.0, ge=0)
    w_avoid_arterials: float = Field(2.0, ge=0)

    candidates:         int   = Field(360, ge=10)
    waypoint_radius_km: float = Field(6.0, gt=0)
    seed:               int   = 42


class RouteResponseOK(BaseModel):
    ok: bool = True
    summary:     dict
    explanation: dict
    geojson:     dict


class RouteResponseError(BaseModel):
    ok: bool = False
    error: str


# ----------------------------
# FastAPI application
# ----------------------------

app = FastAPI(
    title="runTOit – Route Generator API",
    description="Generates optimised running routes using OSMnx + A*.",
    version="1.0.0",
)

# Allow all origins so a GitHub Pages frontend can call this API.
# Restrict origins list in production if required.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    """Quick liveness probe — useful for Render / Railway health checks."""
    return {"status": "ok"}


@app.post("/route", response_model=RouteResponseOK)
def generate_route(body: RouteRequestBody) -> dict:
    """
    Generate the best running route for the supplied parameters.

    Returns a JSON object with three keys:
      - **summary**     – length, trail/arterial ratios, score
      - **explanation** – human-readable score breakdown
      - **geojson**     – FeatureCollection ready to drop into Leaflet / Mapbox
    """
    req = RouteRequest(
        start_lat=body.start_lat,
        start_lon=body.start_lon,
        end_lat=body.end_lat,
        end_lon=body.end_lon,
        target_km=body.target_km,
        tolerance_km=body.tolerance_km,
        w_distance_match=body.w_distance_match,
        target_trail_prop=body.target_trail_prop,
        w_trail_match=body.w_trail_match,
        w_avoid_arterials=body.w_avoid_arterials,
        candidates=body.candidates,
        waypoint_radius_km=body.waypoint_radius_km,
        seed=body.seed,
    )

    try:
        G_ll, G_proj = load_walk_graphs_for_request(req)
        G_proj = add_custom_edge_costs(G_proj)
        result = plan_route_agent(G_proj, G_ll, req)
    except Exception as exc:  # network / graph errors
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if not result["ok"]:
        raise HTTPException(status_code=422, detail=result["error"])

    return {
        "ok":          True,
        "summary":     result["summary"],
        "explanation": result["explanation"],
        "geojson":     result["geojson"],
    }


# ----------------------------
# Launch the server
# ----------------------------
# Run this file to start the API locally on http://localhost:8000
# Interactive docs will be available at http://localhost:8000/docs
#
# For production deployment (e.g. Render / Railway) point the platform's
# start command at this file: uvicorn route_agent:app --host 0.0.0.0 --port 8000

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
