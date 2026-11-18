"""End-to-end inference for one trip: raw files -> candidates/features -> classifier -> HMM/Viterbi.
Used by the evaluation scripts and the FastAPI server (labels are only used *after* prediction,
for scoring, and never enter the feature/prediction path)."""
import time

import numpy as np

from .prep import process_trip, FEATURES, derive_labels, DEFAULTS
from .hmm import viterbi, cands_from_frame, fill_route, DEFAULT_HMM
from .metrics import route_continuity, sequence_metrics


def dedup(seq):
    seq = np.asarray(seq)
    if len(seq) == 0:
        return seq
    return seq[np.r_[True, seq[1:] != seq[:-1]]]


def predict_trip(trip, model, hmm_params=None):
    t0 = time.time()
    o = process_trip(trip, with_labels=False)       # <- no labels on the prediction path
    f = o["feats"]
    p = model.predict_proba(f[FEATURES].to_numpy())[:, 1]
    s = np.bincount(f["obs"].to_numpy(), weights=p, minlength=len(trip.track))
    pn = p / np.maximum(s[f["obs"].to_numpy()], 1e-12)
    f = f.assign(p=p, pn=pn)
    best = f.loc[f.groupby("obs")["p"].idxmax()].sort_values("obs")
    clf_arcs = best["arc"].to_numpy()
    t1 = time.time()
    cands = cands_from_frame(f, pn, len(trip.track))
    hmm_arcs, breaks = viterbi(cands, o["net"], o["txy"], trip.track[:, 2], hmm_params)
    route, unfilled = fill_route(o["net"], dedup(hmm_arcs))
    t2 = time.time()
    return dict(clf_arcs=clf_arcs, hmm_arcs=hmm_arcs, route=route, unfilled=unfilled, breaks=breaks, net=o["net"], txy=o["txy"],
                timing=dict(features_and_classifier_s=t1 - t0, viterbi_s=t2 - t1, total_s=t2 - t0))


def score_trip(trip, pred):
    """Server-side scoring against the reference route (labels derived by monotone alignment)."""
    lab, ldist, _ = derive_labels(pred["net"], pred["txy"], trip.route)
    rel = ldist <= DEFAULTS["label_max_dist"]
    out = {}
    for k in ["clf", "hmm"]:
        arcs = pred[f"{k}_arcs"]
        out[k] = dict(top1_accuracy=float(np.mean(arcs[rel] == lab[rel])),
                      route_continuity=route_continuity(arcs, trip.arcs),
                      **sequence_metrics(dedup(arcs), trip.route))
    out["hmm_filled_route"] = dict(route_continuity=route_continuity(pred["route"], trip.arcs),
                                   **sequence_metrics(pred["route"], trip.route))
    out["n_obs"] = int(len(lab))
    out["n_scored_obs"] = int(rel.sum())
    return out
