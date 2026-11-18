"""Step 6: comparison plots + Folium maps of predicted routes (from saved experiment outputs only).
Outputs: figures/*.png, maps/pred_<trip>.html
"""
import argparse, json, os, sys

import folium
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pathsnap.io import load_trip

C = ["#2457d6", "#e5484d", "#2ea043", "#a371f7", "#d29922", "#1f9e9e", "#888888", "#c2410c", "#64748b"]


def bar(ax, labels, vals, err=None, title="", fmt="{:.3f}", ylim=None):
    b = ax.bar(labels, vals, yerr=err, color=C[:len(labels)], capsize=3)
    for r, v in zip(b, vals):
        ax.text(r.get_x() + r.get_width() / 2, r.get_height(), fmt.format(v), ha="center", va="bottom", fontsize=8)
    ax.set_title(title); ax.tick_params(axis="x", rotation=35)
    if ylim: ax.set_ylim(*ylim)


def comparison(res, fig):
    d = pd.read_csv(f"{res}/model_comparison.csv")
    f, ax = plt.subplots(2, 3, figsize=(16, 8))
    bar(ax[0, 0], d.model, d.cv_macro_f1, d.cv_macro_f1_std, "CV Macro F1 (5-fold GroupKFold, train)", ylim=(0.8, 0.96))
    bar(ax[0, 1], d.model, d.test_top1, None, "Student-test Top-1 accuracy (per GPS point)", ylim=(0.8, 0.93))
    bar(ax[0, 2], d.model, d.test_macro_f1, None, "Student-test Macro F1", ylim=(0.8, 0.96))
    bar(ax[1, 0], d.model, d.test_log_loss, None, "Student-test Log loss (lower=better)")
    bar(ax[1, 1], d.model, d.test_route_continuity, None, "Route continuity (classifier only)", ylim=(0.6, 0.8))
    bar(ax[1, 2], d.model, d.test_infer_ms_per_1k_obs, None, "Inference ms / 1k GPS points", fmt="{:.1f}")
    plt.tight_layout(); plt.savefig(f"{fig}/model_comparison.png", dpi=130); plt.close()

    cv = pd.read_csv(f"{res}/cv_folds.csv")
    f, ax = plt.subplots(figsize=(9, 4))
    ax.boxplot([cv[cv.model == m].macro_f1 for m in d.model], tick_labels=d.model)
    ax.set_title("CV Macro F1 per fold (stability across trip groups)"); ax.tick_params(axis="x", rotation=35)
    plt.tight_layout(); plt.savefig(f"{fig}/cv_folds_box.png", dpi=130); plt.close()

    if os.path.exists(f"{res}/mlp_history.json"):
        h = json.load(open(f"{res}/mlp_history.json"))
        f, ax = plt.subplots(1, 2, figsize=(10, 3.5))
        ax[0].plot(h["train_loss"]); ax[0].set_title("MLP training loss (log loss) per epoch"); ax[0].set_xlabel("epoch")
        ax[1].plot(h["val_accuracy"], color="#e5484d"); ax[1].set_title("MLP validation accuracy per epoch (10% hold-out)")
        ax[1].set_xlabel("epoch")
        plt.tight_layout(); plt.savefig(f"{fig}/mlp_history.png", dpi=130); plt.close()

    te = pd.read_parquet(f"{res}/features_student_test.parquet", columns=["y", "reliable"])
    rel = te.reliable.to_numpy() == 1
    names = [m for m in d.model if os.path.exists(f"{res}/proba_student_test_{m}.npy")]
    f, ax = plt.subplots(1, len(names), figsize=(3.2 * len(names), 3.2))
    for a_, m in zip(np.atleast_1d(ax), names):
        p = np.load(f"{res}/proba_student_test_{m}.npy")
        ConfusionMatrixDisplay(confusion_matrix(te.y[rel], p[rel] >= .5, normalize="true"),
                               display_labels=["other", "true arc"]).plot(ax=a_, colorbar=False, values_format=".2f")
        a_.set_title(m, fontsize=9)
    plt.tight_layout(); plt.savefig(f"{fig}/confusion_matrices.png", dpi=110); plt.close()

    if os.path.exists(f"{res}/tuned_vs_untuned.csv"):
        t = pd.read_csv(f"{res}/tuned_vs_untuned.csv")
        f, ax = plt.subplots(1, 5, figsize=(17, 3.4))
        for a_, (col, title) in zip(ax, [("cv_macro_f1", "CV Macro F1"), ("test_top1", "Test Top-1"),
                                         ("test_macro_f1", "Test Macro F1"), ("test_log_loss", "Test Log loss"),
                                         ("test_infer_ms_per_1k_obs", "Inference ms/1k obs")]):
            bar(a_, t.variant, t[col], None, title, fmt="{:.4f}")
        plt.tight_layout(); plt.savefig(f"{fig}/tuned_vs_untuned.png", dpi=130); plt.close()
        tr = pd.read_csv(f"{res}/optuna_trials.csv")
        f, ax = plt.subplots(figsize=(7, 3.5))
        ax.plot(tr.number, tr.value, "o", ms=4, label="trial"); ax.plot(tr.number, tr.value.cummax(), "-", label="best so far")
        ax.set_xlabel("trial"); ax.set_ylabel("CV Macro F1"); ax.set_title("Optuna TPE search history"); ax.legend()
        plt.tight_layout(); plt.savefig(f"{fig}/optuna_history.png", dpi=130); plt.close()

    if os.path.exists(f"{res}/hmm_vs_classifier.csv"):
        h = pd.read_csv(f"{res}/hmm_vs_classifier.csv")
        f, ax = plt.subplots(1, 3, figsize=(16, 4))
        x = np.arange(len(h)); w = .38
        for a_, (c1, c2, title) in zip(ax, [("clf_top1_accuracy", "hmm_top1_accuracy", "Top-1 accuracy"),
                                             ("clf_route_continuity", "hmm_filled_route_route_continuity", "Route continuity"),
                                             ("clf_route_f1", "hmm_filled_route_route_f1", "Route F1 (arc set)")]):
            a_.bar(x - w / 2, h[c1], w, label="classifier only", color="#999")
            a_.bar(x + w / 2, h[c2], w, label="classifier + HMM", color="#2457d6")
            a_.set_xticks(x, h.trip.astype(str).str.zfill(8).str[-2:]); a_.set_title(title + " — student-test trips"); a_.legend(fontsize=8)
            a_.set_ylim(min(h[c1].min(), h[c2].min()) - .05, 1.01)
        plt.tight_layout(); plt.savefig(f"{fig}/hmm_vs_classifier.png", dpi=130); plt.close()
    if os.path.exists(f"{res}/hmm_grid.csv"):
        g = pd.read_csv(f"{res}/hmm_grid.csv").pivot(index="alpha", columns="beta", values="hmm_top1")
        f, ax = plt.subplots(figsize=(5, 3.5))
        im = ax.imshow(g.values, cmap="Blues"); plt.colorbar(im)
        ax.set_xticks(range(g.shape[1]), g.columns); ax.set_yticks(range(g.shape[0]), g.index)
        for i in range(g.shape[0]):
            for j in range(g.shape[1]):
                ax.text(j, i, f"{g.values[i, j]:.4f}", ha="center", va="center", fontsize=8)
        ax.set_xlabel("beta (m)"); ax.set_ylabel("alpha"); ax.set_title("HMM decoder grid: top-1 (held-out train trips)")
        plt.tight_layout(); plt.savefig(f"{fig}/hmm_grid.png", dpi=130); plt.close()


def pred_map(data, part, tid, model, hmm, out):
    from pathsnap.pipeline import predict_trip  # HMM stage (deliverable 2)
    tr = load_trip(data, part, tid)
    pr = predict_trip(tr, model, hmm)
    nd, a = tr.nodes, tr.arcs
    lat, lon = tr.track[:, 1], tr.track[:, 0]
    m = folium.Map(location=[lat.mean(), lon.mean()], zoom_start=14, tiles="OpenStreetMap")
    pad = .004
    inb = lambda z: (z[:, 0] > lon.min() - pad) & (z[:, 0] < lon.max() + pad) & (z[:, 1] > lat.min() - pad) & (z[:, 1] < lat.max() + pad)
    keep = np.where(inb(nd[a[:, 0]]) & inb(nd[a[:, 1]]))[0]
    g = folium.FeatureGroup("Road network", show=False)
    for i in keep:
        folium.PolyLine([[nd[a[i, 0], 1], nd[a[i, 0], 0]], [nd[a[i, 1], 1], nd[a[i, 1], 0]]], color="#bbbbbb", weight=1).add_to(g)
    g.add_to(m)

    def line(route, color, name, weight, dash=None):
        fg = folium.FeatureGroup(name)
        seg = []
        for k, r in enumerate(route):
            if seg and a[route[k - 1], 1] != a[r, 0]:
                folium.PolyLine(seg, color=color, weight=weight, opacity=.85, dash_array=dash).add_to(fg); seg = []
            if not seg:
                seg = [[nd[a[r, 0], 1], nd[a[r, 0], 0]]]
            seg.append([nd[a[r, 1], 1], nd[a[r, 1], 0]])
        if seg:
            folium.PolyLine(seg, color=color, weight=weight, opacity=.85, dash_array=dash).add_to(fg)
        fg.add_to(m)

    line(tr.route, "#2ea043", "Reference route", 7)
    line(pr["route"], "#e5484d", "Predicted route (classifier + HMM)", 3)
    clf = folium.FeatureGroup("Classifier-only arc per point", show=False)
    for r in np.unique(pr["clf_arcs"]):
        folium.PolyLine([[nd[a[r, 0], 1], nd[a[r, 0], 0]], [nd[a[r, 1], 1], nd[a[r, 1], 0]]], color="#d29922", weight=3).add_to(clf)
    clf.add_to(m)
    gps = folium.FeatureGroup("Raw GPS")
    for y, x in zip(lat[::2], lon[::2]):
        folium.CircleMarker([y, x], radius=2, color="#1f6feb", fill=True, weight=1).add_to(gps)
    gps.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    legend = ('<div style="position:fixed;bottom:30px;left:30px;z-index:9999;background:white;padding:8px;border:1px solid #888;'
              'font-size:13px"><b>Trip ' + tid + '</b><br><span style="color:#1f6feb">&#9679;</span> Raw GPS<br>'
              '<span style="color:#2ea043">&#9644;</span> Reference route<br><span style="color:#e5484d">&#9644;</span> Predicted route'
              '<br><span style="color:#d29922">&#9644;</span> Classifier-only arcs</div>')
    m.get_root().html.add_child(folium.Element(legend))
    m.fit_bounds([[lat.min(), lon.min()], [lat.max(), lon.max()]])
    m.save(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "data"))
    ap.add_argument("--res", default=os.path.join(ROOT, "results"))
    ap.add_argument("--fig", default=os.path.join(ROOT, "figures"))
    ap.add_argument("--maps", default=os.path.join(ROOT, "maps"))
    ap.add_argument("--map_trips", default="00000052,00000065,00000076")
    a = ap.parse_args()
    comparison(a.res, a.fig)
    mf = os.path.join(ROOT, "models", "final_model.joblib")
    hp = os.path.join(ROOT, "configs", "hmm_params.json")
    if os.path.exists(mf) and os.path.exists(os.path.join(ROOT, "pathsnap", "pipeline.py")):
        model = joblib.load(mf)
        hmm = json.load(open(hp))["params"] if os.path.exists(hp) else None
        for t in a.map_trips.split(","):
            pred_map(a.data, "student_test", t, model, hmm, f"{a.maps}/pred_student_test_{t}.html")
