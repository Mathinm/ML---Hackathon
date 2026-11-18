"""Step 2: derive labels, generate candidates and engineer features for train + student_test.
Output: results/features_<partition>.parquet, results/obs_<partition>.parquet, results/prep_summary.csv
Usage: python scripts/build_dataset.py --data data
"""
import argparse, os, sys, time
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathsnap.io import load_trip, list_trips
from pathsnap.prep import process_trip, DEFAULTS


def work(args):
    root, part, tid = args
    t0 = time.time()
    tr = load_trip(root, part, tid)
    o = process_trip(tr)
    f = o["feats"]
    reliable = o["label_dist"] <= DEFAULTS["label_max_dist"]
    f["reliable"] = reliable[f["obs"].to_numpy()].astype(int)
    f.insert(0, "trip", tid)
    hit = f.groupby("obs")["y"].max().reindex(range(len(tr.track)), fill_value=0).to_numpy()
    obs = pd.DataFrame({"trip": tid, "obs": np.arange(len(tr.track)), "t": tr.track[:, 2],
                        "label_arc": o["label"], "label_dist": o["label_dist"], "reliable": reliable.astype(int),
                        "cand_hit": hit})
    summ = dict(trip=tid, partition=part, n_obs=len(tr.track), n_rows=len(f), arcs_kept=len(o["net"].keep),
                reliable_frac=float(reliable.mean()),
                cand_miss_rate=float(1 - hit[reliable].mean()) if reliable.any() else np.nan,
                mean_cands=float(f.groupby("obs").size().mean()), issues="; ".join(tr.issues),
                secs=round(time.time() - t0, 1))
    return f, obs, summ


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    summaries = []
    for part in ["train", "student_test"]:
        jobs = [(a.data, part, t) for t in list_trips(a.data, part)]
        with Pool(a.workers) as p:
            res = p.map(work, jobs, chunksize=1)
        pd.concat([r[0] for r in res]).to_parquet(f"{a.out}/features_{part}.parquet", index=False)
        pd.concat([r[1] for r in res]).to_parquet(f"{a.out}/obs_{part}.parquet", index=False)
        summaries += [r[2] for r in res]
        print(part, "done", len(res))
    s = pd.DataFrame(summaries)
    s.to_csv(f"{a.out}/prep_summary.csv", index=False)
    print(s.groupby("partition")[["n_obs", "n_rows", "reliable_frac", "cand_miss_rate", "mean_cands"]].describe().T)
