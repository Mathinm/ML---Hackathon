"""Raw-file readers for the PathSnap dataset.

Each trip folder <root>/<partition>/<id>/ holds <id>.track, .nodes, .arcs, .route
(all header-less, whitespace separated). See Dataset Exploration.pdf.
"""
import os
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Trip:
    trip_id: str
    track: np.ndarray          # (N,3) lon, lat, t
    nodes: np.ndarray          # (M,2) lon, lat ; row index = node id
    arcs: np.ndarray           # (A,2) from_node, to_node ; row index = arc id
    route: np.ndarray | None   # (R,) arc ids, chronological (None if unavailable)
    issues: list = field(default_factory=list)


def list_trips(root, partition):
    d = os.path.join(root, partition)
    return sorted(x for x in os.listdir(d) if os.path.isdir(os.path.join(d, x)))


def _load(path, dtype=float, ndmin=2):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    return np.loadtxt(path, dtype=dtype, ndmin=ndmin)


def load_trip(root, partition, trip_id, with_route=True):
    base = os.path.join(root, partition, trip_id, trip_id)
    track = _load(base + ".track")
    nodes = _load(base + ".nodes")
    arcs = _load(base + ".arcs", dtype=np.int64)
    route = _load(base + ".route", dtype=np.int64, ndmin=1) if with_route else None
    issues = []

    # --- cleaning: drop non-finite / implausible GPS rows, enforce time order, drop duplicate timestamps
    ok = np.all(np.isfinite(track), axis=1)
    ok &= (np.abs(track[:, 0]) <= 180) & (np.abs(track[:, 1]) <= 90)
    if (~ok).any():
        issues.append(f"dropped {int((~ok).sum())} invalid GPS rows")
    track = track[ok]
    order = np.argsort(track[:, 2], kind="stable")
    if not np.all(order == np.arange(len(order))):
        issues.append("track re-sorted by timestamp")
        track = track[order]
    dup = np.r_[False, np.diff(track[:, 2]) <= 0]
    if dup.any():
        issues.append(f"dropped {int(dup.sum())} duplicate timestamps")
        track = track[~dup]
    return Trip(trip_id, track, nodes, arcs, route, issues)
