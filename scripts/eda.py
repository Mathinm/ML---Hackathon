"""Step 1: dataset exploration (train + student_test only; evaluator data is never touched).
Outputs: results/eda_inventory.csv, figures/eda_*.png, maps/eda_trip_<id>.html
"""
import argparse, os, sys

import folium
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pathsnap.io import load_trip, list_trips
from pathsnap.geo import make_projector, project


def inventory(root):
    rows = []
    for part in ["train", "student_test"]:
        for tid in list_trips(root, part):
            base = os.path.join(root, part, tid, tid)
            raw = np.loadtxt(base + ".track", ndmin=2)
            tr = load_trip(root, part, tid)
            n_nodes, n_arcs = len(tr.nodes), len(tr.arcs)
            dt = np.diff(tr.track[:, 2])
            fwd = make_projector(tr.track[:, 0], tr.track[:, 1])
            xy = project(fwd, tr.track[:, :2])
            step = np.hypot(*np.diff(xy, axis=0).T)
            g = coo_matrix((np.ones(n_arcs), (tr.arcs[:, 0], tr.arcs[:, 1])), shape=(n_nodes, n_nodes))
            n_weak, lab_w = connected_components(g, directed=True, connection="weak")
            n_strong, _ = connected_components(g, directed=True, connection="strong")
            deg = np.bincount(tr.arcs.ravel(), minlength=n_nodes)
            r = tr.route
            rows.append(dict(
                partition=part, trip=tid,
                size_track_kb=os.path.getsize(base + ".track") / 1024, size_nodes_kb=os.path.getsize(base + ".nodes") / 1024,
                size_arcs_kb=os.path.getsize(base + ".arcs") / 1024, size_route_kb=os.path.getsize(base + ".route") / 1024,
                n_obs=len(tr.track), n_nan=int(np.isnan(raw).sum()), cleaning="; ".join(tr.issues),
                duration_min=(tr.track[-1, 2] - tr.track[0, 2]) / 60, dt_median=np.median(dt),
                frac_dt_1s=np.mean(dt == 1), n_gaps_gt10s=int((dt > 10).sum()), max_gap_s=dt.max(),
                path_km=step.sum() / 1000, max_jump_m=step.max(), mean_speed_kmh=3.6 * step.sum() / max(dt.sum(), 1),
                extent_km_x=np.ptp(xy[:, 0]) / 1000, extent_km_y=np.ptp(xy[:, 1]) / 1000,
                lon_c=tr.track[:, 0].mean(), lat_c=tr.track[:, 1].mean(),
                n_nodes=n_nodes, n_arcs=n_arcs, isolated_nodes=int((deg == 0).sum()),
                weak_components=n_weak, largest_weak_frac=np.bincount(lab_w).max() / n_nodes, strong_components=n_strong,
                route_len=len(r), route_unique=len(np.unique(r)),
                route_connected_frac=float(np.mean(tr.arcs[r[:-1], 1] == tr.arcs[r[1:], 0]))))
            print(tid, part, flush=True)
    return pd.DataFrame(rows)


def plots(inv, res, fig):
    f, ax = plt.subplots(2, 3, figsize=(15, 8))
    for p, c in [("train", "#2457d6"), ("student_test", "#e5484d")]:
        d = inv[inv.partition == p]
        ax[0, 0].hist(d.n_obs, bins=30, alpha=.6, color=c, label=p)
        ax[0, 1].hist(d.duration_min, bins=30, alpha=.6, color=c, label=p)
        ax[0, 2].scatter(d.n_nodes, d.n_arcs, color=c, label=p, s=14)
        ax[1, 0].hist(np.log10(d.max_gap_s), bins=30, alpha=.6, color=c, label=p)
        ax[1, 1].scatter(d.lon_c, d.lat_c, color=c, label=p, s=14)
        ax[1, 2].hist(d.mean_speed_kmh, bins=30, alpha=.6, color=c, label=p)
    for a_, t in zip(ax.ravel(), ["GPS points per trip", "Trip duration (min)", "Road network: nodes vs arcs",
                                   "log10 max time gap (s)", "Trip centroids (lon, lat)", "Mean speed (km/h)"]):
        a_.set_title(t); a_.legend(fontsize=8)
    ax[0, 2].set_xscale("log"); ax[0, 2].set_yscale("log")
    plt.tight_layout(); plt.savefig(f"{fig}/eda_overview.png", dpi=130); plt.close()

    obs = pd.read_parquet(f"{res}/obs_train.parquet")
    feats = pd.read_parquet(f"{res}/features_train.parquet", columns=["trip", "obs", "dist", "y", "head_diff", "head_valid"])
    f, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].hist(np.clip(obs.label_dist, 0, 200), bins=80, color="#2457d6")
    ax[0].axvline(100, color="k", ls="--"); ax[0].set_title("GPS -> aligned reference arc distance (m) [train]")
    nc = feats.groupby(["trip", "obs"]).size()
    ax[1].hist(nc, bins=np.arange(0.5, 11.5), color="#2457d6"); ax[1].set_title("Candidates per GPS point (r=60 m, k<=10) [train]")
    v = feats[feats.head_valid == 1]
    ax[2].hist([v[v.y == 1].head_diff, v[v.y == 0].head_diff], bins=36, label=["true arc", "other candidates"],
               density=True, color=["#2ea043", "#999"])
    ax[2].legend(); ax[2].set_title("Heading difference GPS vs arc (deg) [train]")
    plt.tight_layout(); plt.savefig(f"{fig}/eda_labels_candidates.png", dpi=130); plt.close()

    f, ax = plt.subplots(figsize=(6, 4))
    ax.hist([feats[feats.y == 1].dist.clip(0, 60), feats[feats.y == 0].dist.clip(0, 60)], bins=60, density=True,
            label=["true arc", "other candidates"], color=["#2ea043", "#999"])
    ax.set_title("Perpendicular distance to candidate arc (m) [train]"); ax.legend()
    plt.tight_layout(); plt.savefig(f"{fig}/eda_distance.png", dpi=130); plt.close()


def folium_trip(root, part, tid, out):
    tr = load_trip(root, part, tid)
    lat, lon = tr.track[:, 1], tr.track[:, 0]
    m = folium.Map(location=[lat.mean(), lon.mean()], zoom_start=14, tiles="OpenStreetMap")
    pad = 0.005
    nd = tr.nodes
    a = tr.arcs
    inb = lambda z: (z[:, 0] > lon.min() - pad) & (z[:, 0] < lon.max() + pad) & (z[:, 1] > lat.min() - pad) & (z[:, 1] < lat.max() + pad)
    keep = np.where(inb(nd[a[:, 0]]) & inb(nd[a[:, 1]]))[0]
    net = folium.FeatureGroup("Road network (arcs)")
    for i in keep:
        folium.PolyLine([[nd[a[i, 0], 1], nd[a[i, 0], 0]], [nd[a[i, 1], 1], nd[a[i, 1], 0]]], color="#aaaaaa", weight=1).add_to(net)
    net.add_to(m)
    ref = folium.FeatureGroup("Reference route")
    pts = [[nd[a[r, 0], 1], nd[a[r, 0], 0]] for r in tr.route] + [[nd[a[tr.route[-1], 1], 1], nd[a[tr.route[-1], 1], 0]]]
    folium.PolyLine(pts, color="#2ea043", weight=5, opacity=.8, tooltip="reference route").add_to(ref)
    ref.add_to(m)
    gps = folium.FeatureGroup("Raw GPS")
    for y, x in zip(lat[::2], lon[::2]):
        folium.CircleMarker([y, x], radius=2, color="#1f6feb", fill=True, weight=1).add_to(gps)
    gps.add_to(m)
    folium.Marker([lat[0], lon[0]], tooltip="start", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker([lat[-1], lon[-1]], tooltip="end", icon=folium.Icon(color="red")).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[lat.min(), lon.min()], [lat.max(), lon.max()]])
    m.save(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "data"))
    ap.add_argument("--res", default=os.path.join(ROOT, "results"))
    ap.add_argument("--fig", default=os.path.join(ROOT, "figures"))
    ap.add_argument("--maps", default=os.path.join(ROOT, "maps"))
    ap.add_argument("--trips", default="train:00000048,train:00000054,student_test:00000052")
    a = ap.parse_args()
    for d in [a.res, a.fig, a.maps]:
        os.makedirs(d, exist_ok=True)
    inv = inventory(a.data)
    inv.to_csv(f"{a.res}/eda_inventory.csv", index=False)
    print(inv.groupby("partition").describe().T.to_string())
    plots(inv, a.res, a.fig)
    for spec in a.trips.split(","):
        part, tid = spec.split(":")
        folium_trip(a.data, part, tid, f"{a.maps}/eda_{part}_{tid}.html")
