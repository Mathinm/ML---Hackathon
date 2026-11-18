"""Step 3: train + compare models.
- 5-fold GroupKFold over the 80 training trips (groups = trip, so no trip leaks across folds)
- refit on all training trips, evaluate on the 10 student-test trips
Outputs: results/model_comparison.csv, results/cv_folds.csv, models/<name>.joblib,
         results/proba_student_test_<name>.npy, results/mlp_history.json
"""
import argparse, json, os, sys, time

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pathsnap.prep import FEATURES
from pathsnap.models import MODEL_SPECS, build, SEED
from pathsnap.metrics import classification_metrics, top1, route_continuity
from pathsnap.io import load_trip


def load_split(res, part, stride=1):
    f = pd.read_parquet(f"{res}/features_{part}.parquet")
    f["obs_key"] = f["trip"] + "_" + f["obs"].astype(str)
    if stride > 1:
        f = f[f["obs"] % stride == 0]
    return f.reset_index(drop=True)


def eval_on(model, df, arcs_by_trip=None):
    t0 = time.time()
    p = model.predict_proba(df[FEATURES].to_numpy())[:, 1]
    inf = time.time() - t0
    rel = df["reliable"].to_numpy() == 1
    m = classification_metrics(df["y"].to_numpy()[rel], p[rel])
    acc, _ = top1(df[rel], p[rel])
    m["top1"] = acc
    m["infer_ms_per_1k_obs"] = 1e6 * inf / df["obs_key"].nunique()
    if arcs_by_trip is not None:
        _, best = top1(df, p)
        conts = []
        for tid, arcs in arcs_by_trip.items():
            b = best[best.index.str.startswith(tid + "_")]
            order = np.argsort([int(k.split("_")[1]) for k in b.index])
            conts.append(route_continuity(b.to_numpy()[order], arcs))
        m["route_continuity"] = float(np.mean(conts))
    return m, p


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default=os.path.join(ROOT, "results"))
    ap.add_argument("--data", default=os.path.join(ROOT, "data"))
    ap.add_argument("--models", default=",".join(MODEL_SPECS))
    ap.add_argument("--stride", type=int, default=3, help="use every k-th GPS obs for training (1 Hz is highly redundant)")
    ap.add_argument("--knn_stride", type=int, default=12)
    a = ap.parse_args()
    np.random.seed(SEED)
    os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)

    tr_all = load_split(a.res, "train", 1)
    te = load_split(a.res, "student_test", 1)
    arcs_te = {t: load_trip(a.data, "student_test", t, with_route=False).arcs for t in te["trip"].unique()}
    rows, folds = [], []
    for name in a.models.split(","):
        stride = a.knn_stride if name == "knn" else a.stride
        tr = tr_all[(tr_all["obs"] % stride == 0) & (tr_all["reliable"] == 1)].reset_index(drop=True)
        X, y, g = tr[FEATURES].to_numpy(), tr["y"].to_numpy(), tr["trip"].to_numpy()
        cv = []
        for k, (i_tr, i_va) in enumerate(GroupKFold(n_splits=5).split(X, y, g)):
            m = build(name).fit(X[i_tr], y[i_tr])
            mm, _ = eval_on(m, tr.iloc[i_va])
            mm.update(model=name, fold=k)
            cv.append(mm)
            print(name, "fold", k, {kk: round(v, 4) for kk, v in mm.items() if isinstance(v, float)}, flush=True)
        folds += cv
        cvd = pd.DataFrame(cv)
        t0 = time.time()
        model = build(name).fit(X, y)
        ttrain = time.time() - t0
        te_m, p = eval_on(model, te, arcs_te)
        np.save(f"{a.res}/proba_student_test_{name}.npy", p)
        joblib.dump(model, os.path.join(ROOT, "models", f"{name}.joblib"))
        if name == "mlp":
            mlp = model[-1]
            json.dump({"train_loss": mlp.loss_curve_, "val_accuracy": mlp.validation_scores_},
                      open(f"{a.res}/mlp_history.json", "w"))
        row = dict(model=name, family=MODEL_SPECS[name][0], config=json.dumps(MODEL_SPECS[name][1]),
                   train_rows=len(tr), cv_macro_f1=cvd.macro_f1.mean(), cv_macro_f1_std=cvd.macro_f1.std(),
                   cv_top1=cvd.top1.mean(), cv_log_loss=cvd.log_loss.mean(), train_time_s=ttrain,
                   **{f"test_{k}": v for k, v in te_m.items()})
        rows.append(row)
        print("==>", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items() if k != "config"}, flush=True)
        pd.DataFrame(rows).to_csv(f"{a.res}/model_comparison.csv", index=False)
        pd.DataFrame(folds).to_csv(f"{a.res}/cv_folds.csv", index=False)
