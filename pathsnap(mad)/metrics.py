import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score


def normalise(df, p):
    """Per-observation normalisation of candidate scores -> distribution over candidates."""
    s = pd.Series(p, index=df.index).groupby(df["obs_key"]).transform("sum").to_numpy()
    return p / np.maximum(s, 1e-12)


def top1(df, p):
    """Return (accuracy, predicted arc per obs Series indexed by obs_key)."""
    d = df[["obs_key", "arc", "y"]].copy()
    d["p"] = p
    idx = d.groupby("obs_key")["p"].idxmax()
    best = d.loc[idx]
    return float(best["y"].mean()), best.set_index("obs_key")["arc"]


def classification_metrics(y, p, thr=0.5):
    yhat = (p >= thr).astype(int)
    pc = np.clip(p, 1e-7, 1 - 1e-7)
    return dict(macro_f1=f1_score(y, yhat, average="macro"),
                precision=precision_score(y, yhat, zero_division=0),
                recall=recall_score(y, yhat, zero_division=0),
                log_loss=log_loss(y, pc, labels=[0, 1]))


def route_continuity(seq, arcs):
    """Fraction of predicted arc *changes* between consecutive observations that are
    topologically valid (to_node(prev) == from_node(next)). 1.0 = fully connected route."""
    seq = np.asarray(seq)
    a, b = seq[:-1], seq[1:]
    ch = a != b
    if ch.sum() == 0:
        return 1.0
    return float(np.mean(arcs[a[ch], 1] == arcs[b[ch], 0]))


def sequence_metrics(pred_seq, ref_route):
    """Arc-set precision / recall / F1 between the de-duplicated predicted route and the reference route."""
    pred = set(np.asarray(pred_seq).tolist())
    ref = set(np.asarray(ref_route).tolist())
    tp = len(pred & ref)
    pr = tp / max(len(pred), 1)
    rc = tp / max(len(ref), 1)
    return dict(route_precision=pr, route_recall=rc, route_f1=2 * pr * rc / max(pr + rc, 1e-12))
