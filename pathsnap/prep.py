"""Network building, label derivation, candidate generation and feature engineering."""
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .geo import make_projector, project, point_segment, bearing, angle_diff

DEFAULTS = dict(radius=60.0, k_max=10, crop_buffer=1500.0, sample_step=10.0,
                ctx_offset=3, head_win=2, label_max_dist=100.0)


@dataclass
class Network:
    xy: np.ndarray          # node coords (m)
    arcs: np.ndarray        # (A,2) from,to  (original ids, full network)
    keep: np.ndarray        # arc ids kept after cropping
    length: np.ndarray      # (A,) metres (full)
    bear: np.ndarray        # (A,) bearing deg
    has_rev: np.ndarray     # (A,) bool, reverse arc exists
    out_deg: np.ndarray     # (M,) out-degree per node
    adj: dict               # node -> list[(to_node, arc_id, len)]


def build_network(trip, fwd, track_xy, cfg=DEFAULTS):
    xy = project(fwd, trip.nodes)
    a = trip.arcs
    p, q = xy[a[:, 0]], xy[a[:, 1]]
    d = q - p
    length = np.hypot(d[:, 0], d[:, 1])
    bear = bearing(d[:, 0], d[:, 1])
    lo = track_xy.min(0) - cfg["crop_buffer"]
    hi = track_xy.max(0) + cfg["crop_buffer"]
    inside = lambda z: np.all((z >= lo) & (z <= hi), axis=1)
    keep = np.where(inside(p) | inside(q))[0]
    rev_keys = set(zip(a[keep, 1].tolist(), a[keep, 0].tolist()))
    has_rev = np.zeros(len(a), bool)
    has_rev[keep] = [(f, t) in rev_keys for f, t in a[keep].tolist()]
    out_deg = np.bincount(a[keep, 0], minlength=len(xy))
    adj = {}
    for aid in keep.tolist():
        f, t = a[aid]
        adj.setdefault(int(f), []).append((int(t), aid, float(length[aid])))
    return Network(xy, a, keep, length, bear, has_rev, out_deg, adj)


def seg_dist(net, pts, arc_ids):
    """distance of pts (n,2) to arcs arc_ids (n,) elementwise."""
    a = net.arcs[arc_ids]
    P, Q = net.xy[a[:, 0]], net.xy[a[:, 1]]
    return point_segment(pts[:, 0], pts[:, 1], P[:, 0], P[:, 1], Q[:, 0], Q[:, 1])


# ------------------------------------------------------------------ labels
def derive_labels(net, track_xy, route, cfg=DEFAULTS):
    """Assign each GPS point to one arc of the reference route with a monotone
    (order-preserving) alignment minimising total point-to-arc distance (DP).
    Returns (label_arc, label_dist, route_index)."""
    N, R = len(track_xy), len(route)
    a = net.arcs[route]
    P, Q = net.xy[a[:, 0]], net.xy[a[:, 1]]
    back = np.zeros((N, R), dtype=np.int32)
    dist_rows = np.empty((N, R), dtype=np.float32)
    for s in range(0, N, 2000):
        px = track_xy[s:s + 2000, 0][:, None]
        py = track_xy[s:s + 2000, 1][:, None]
        dist_rows[s:s + 2000] = point_segment(px, py, P[None, :, 0], P[None, :, 1], Q[None, :, 0], Q[None, :, 1])[0]
    ar = np.arange(R)
    C = dist_rows[0].astype(np.float64)
    for i in range(1, N):
        cm = np.minimum.accumulate(C)
        arg = np.maximum.accumulate(np.where(C == cm, ar, 0))
        back[i] = arg
        C = dist_rows[i] + cm
    k = int(np.argmin(C))
    idx = np.empty(N, dtype=np.int64)
    for i in range(N - 1, -1, -1):
        idx[i] = k
        k = back[i, k]
    lab = route[idx]
    return lab, dist_rows[np.arange(N), idx].astype(float), idx


# ------------------------------------------------------------------ candidates
def build_index(net, cfg=DEFAULTS):
    pts, ids = [], []
    a = net.arcs
    for aid in net.keep:
        P, Q = net.xy[a[aid, 0]], net.xy[a[aid, 1]]
        n = max(2, int(math.ceil(net.length[aid] / cfg["sample_step"])) + 1)
        t = np.linspace(0, 1, n)[:, None]
        pts.append(P + t * (Q - P))
        ids.append(np.full(n, aid))
    pts = np.vstack(pts)
    return cKDTree(pts), np.concatenate(ids)


def generate_candidates(net, track_xy, tree, sample_ids, cfg=DEFAULTS):
    """For each GPS point: arcs within `radius` m (exact point-segment distance), keep k_max
    nearest. If none, fall back to the k_max nearest arcs (flagged)."""
    R, K = cfg["radius"], cfg["k_max"]
    half = cfg["sample_step"] / 2 + 0.5
    rows = []
    for i, nb in enumerate(tree.query_ball_point(track_xy, R + half)):
        fallback = 0
        if len(nb) == 0:
            _, nb = tree.query(track_xy[i], k=min(60, len(sample_ids)))
            nb = np.atleast_1d(nb)
            fallback = 1
        arcs = np.unique(sample_ids[nb])
        d, t, qx, qy = seg_dist(net, np.repeat(track_xy[i:i + 1], len(arcs), 0), arcs)
        if not fallback:
            m = d <= R
            if not m.any():
                m[:] = True
                fallback = 1
            arcs, d, t = arcs[m], d[m], t[m]
        o = np.argsort(d, kind="stable")[:K]
        for r, j in enumerate(o):
            rows.append((i, int(arcs[j]), float(d[j]), float(t[j]), r, fallback))
    return pd.DataFrame(rows, columns=["obs", "arc", "dist", "frac", "dist_rank", "fallback"])


# ------------------------------------------------------------------ features
FEATURES = ["dist", "dist_rank", "dist_minus_min", "dist_ratio", "n_cand", "head_diff",
            "head_valid", "head_rank", "speed", "dt_prev", "arc_len", "frac", "frac_edge",
            "has_rev", "deg_to", "ctx_prev_dist", "ctx_next_dist", "ctx_prev_head", "ctx_next_head",
            "ctx_mean_dist", "fallback"]


def obs_kinematics(track_xy, t, cfg=DEFAULTS):
    N = len(track_xy)
    w = cfg["head_win"]
    i0 = np.clip(np.arange(N) - w, 0, N - 1)
    i1 = np.clip(np.arange(N) + w, 0, N - 1)
    d = track_xy[i1] - track_xy[i0]
    disp = np.hypot(d[:, 0], d[:, 1])
    head = bearing(d[:, 0], d[:, 1])
    dtw = np.maximum(t[i1] - t[i0], 1e-6)
    speed = disp / dtw
    dt_prev = np.r_[1.0, np.diff(t)]
    return head, disp, speed, dt_prev


def make_features(net, track_xy, t, cand, cfg=DEFAULTS):
    head, disp, speed, dt_prev = obs_kinematics(track_xy, t, cfg)
    N = len(track_xy)
    c = cand
    obs = c["obs"].to_numpy()
    arc = c["arc"].to_numpy()
    f = pd.DataFrame({"obs": obs, "arc": arc})
    f["dist"] = c["dist"].to_numpy()
    f["dist_rank"] = c["dist_rank"].to_numpy()
    g = f.groupby("obs")["dist"]
    dmin = g.transform("min").to_numpy()
    f["dist_minus_min"] = f["dist"] - dmin
    f["dist_ratio"] = f["dist"] / (dmin + 1.0)
    f["n_cand"] = g.transform("size").to_numpy()
    hd = angle_diff(head[obs], net.bear[arc])
    valid = disp[obs] > 3.0  # heading unreliable when (nearly) stationary
    f["head_diff"] = np.where(valid, hd, 90.0)
    f["head_valid"] = valid.astype(int)
    f["head_rank"] = f.groupby("obs")["head_diff"].rank(method="min").to_numpy()
    f["speed"] = speed[obs]
    f["dt_prev"] = dt_prev[obs]
    f["arc_len"] = net.length[arc]
    f["frac"] = c["frac"].to_numpy()
    f["frac_edge"] = np.minimum(f["frac"], 1 - f["frac"]) * f["arc_len"]
    f["has_rev"] = net.has_rev[arc].astype(int)
    f["deg_to"] = net.out_deg[net.arcs[arc, 1]]
    o = cfg["ctx_offset"]
    ip = np.clip(obs - o, 0, N - 1)
    inx = np.clip(obs + o, 0, N - 1)
    f["ctx_prev_dist"] = seg_dist(net, track_xy[ip], arc)[0]
    f["ctx_next_dist"] = seg_dist(net, track_xy[inx], arc)[0]
    f["ctx_prev_head"] = np.where(disp[ip] > 3, angle_diff(head[ip], net.bear[arc]), 90.0)
    f["ctx_next_head"] = np.where(disp[inx] > 3, angle_diff(head[inx], net.bear[arc]), 90.0)
    f["ctx_mean_dist"] = (f["dist"] + f["ctx_prev_dist"] + f["ctx_next_dist"]) / 3
    f["fallback"] = c["fallback"].to_numpy()
    return f


# ------------------------------------------------------------------ one-shot trip processing
def process_trip(trip, cfg=DEFAULTS, with_labels=True):
    fwd = make_projector(trip.track[:, 0], trip.track[:, 1])
    txy = project(fwd, trip.track[:, :2])
    net = build_network(trip, fwd, txy, cfg)
    tree, sids = build_index(net, cfg)
    cand = generate_candidates(net, txy, tree, sids, cfg)
    feats = make_features(net, txy, trip.track[:, 2], cand, cfg)
    out = dict(fwd=fwd, txy=txy, net=net, feats=feats)
    if with_labels and trip.route is not None:
        lab, ldist, ridx = derive_labels(net, txy, trip.route, cfg)
        feats["y"] = (feats["arc"].to_numpy() == lab[feats["obs"].to_numpy()]).astype(int)
        out.update(label=lab, label_dist=ldist, route_idx=ridx)
    return out
