"""Step 5: HMM/Viterbi decoding — tune decoder params, then compare classifier-only vs classifier+HMM.

Decoder tuning (alpha, beta) uses 12 held-out TRAINING trips with a classifier re-fit on the other
68 training trips (so the probabilities are out-of-sample). Final comparison on the 10 student-test trips.
Outputs: results/hmm_grid.csv, configs/hmm_params.json, results/hmm_vs_classifier.csv
"""
import argparse, itertools, json, os, sys, time

import joblib
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pathsnap.io import load_trip, list_trips
from pathsnap.prep import FEATURES
from pathsnap.models import build, SEED
from pathsnap.pipeline import predict_trip, score_trip
from pathsnap.hmm import DEFAULT_HMM


def flat(tid, sc):
    r = dict(trip=tid, n_obs=sc["n_obs"], n_scored=sc["n_scored_obs"])
    for k in ["clf", "hmm", "hmm_filled_route"]:
        for m, v in sc[k].items():
            r[f"{k}_{m}"] = v
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="xgboost")
    ap.add_argument("--model_file", default=os.path.join(ROOT, "models", "xgboost_tuned.joblib"))
    ap.add_argument("--params_json", default=os.path.join(ROOT, "configs", "best_params.json"))
    ap.add_argument("--data", default=os.path.join(ROOT, "data"))
    ap.add_argument("--res", default=os.path.join(ROOT, "results"))
    ap.add_argument("--skip_grid", action="store_true")
    a = ap.parse_args()

    best_prm = json.load(open(a.params_json))["best_params"] if os.path.exists(a.params_json) else {}
    hmm_prm = dict(DEFAULT_HMM)
    if not a.skip_grid:
        rng = np.random.RandomState(SEED)
        train_ids = list_trips(a.data, "train")
        f = pd.read_parquet(f"{a.res}/features_train.parquet")
        small = f.groupby("trip")["obs"].max()
        pool = [t for t in train_ids if small[t] < 4000]           # keep the grid affordable
        hold = sorted(rng.choice(pool, 12, replace=False).tolist())
        fit = f[(~f["trip"].isin(hold)) & (f["reliable"] == 1) & (f["obs"] % 3 == 0)]
        m = build(a.model_name, best_prm).fit(fit[FEATURES].to_numpy(), fit["y"].to_numpy())
        trips = {t: load_trip(a.data, "train", t) for t in hold}
        grid = []
        for alpha, beta in itertools.product([0.5, 1.0, 2.0], [2.0, 5.0, 10.0, 20.0]):
            prm = {**DEFAULT_HMM, "alpha": alpha, "beta": beta}
            accs, f1s = [], []
            for t, tr in trips.items():
                sc = score_trip(tr, predict_trip(tr, m, prm))
                accs.append(sc["hmm"]["top1_accuracy"]); f1s.append(sc["hmm_filled_route"]["route_f1"])
            grid.append(dict(alpha=alpha, beta=beta, hmm_top1=np.mean(accs), route_f1=np.mean(f1s)))
            print(grid[-1], flush=True)
        g = pd.DataFrame(grid)
        g.to_csv(f"{a.res}/hmm_grid.csv", index=False)
        b = g.sort_values(["hmm_top1", "route_f1"], ascending=False).iloc[0]
        hmm_prm.update(alpha=float(b.alpha), beta=float(b.beta))
        json.dump(dict(params=hmm_prm, tuned_on_train_trips=hold), open(os.path.join(ROOT, "configs", "hmm_params.json"), "w"), indent=2)
    elif os.path.exists(os.path.join(ROOT, "configs", "hmm_params.json")):
        hmm_prm = json.load(open(os.path.join(ROOT, "configs", "hmm_params.json")))["params"]

    model = joblib.load(a.model_file)
    rows = []
    for t in list_trips(a.data, "student_test"):
        tr = load_trip(a.data, "student_test", t)
        pr = predict_trip(tr, model, hmm_prm)
        r = flat(t, score_trip(tr, pr))
        r.update(breaks=pr["breaks"], unfilled=pr["unfilled"], **pr["timing"])
        rows.append(r)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(f"{a.res}/hmm_vs_classifier.csv", index=False)
    w = d["n_scored"]
    print("\nweighted top1 clf", np.average(d.clf_top1_accuracy, weights=w), "hmm", np.average(d.hmm_top1_accuracy, weights=w))
    print(d.mean(numeric_only=True).round(4).to_string())
