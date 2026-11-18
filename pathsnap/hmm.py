"""HMM + Viterbi route reconstruction.

Hidden state at time i  = one candidate arc of GPS observation i (candidate -> arc id is preserved).
Emission  log P(z_i | s)  = alpha * log(p_norm(s))   where p_norm = classifier probability
                                                  normalised over the candidates of obs i.
Transition log P(s_j | s_i) = -| d_route(s_i -> s_j) - d_gps(i, i+1) | / beta_t
    d_route : network distance from the projection on arc s_i to the projection on arc s_j
              (same arc: distance along the arc; otherwise remaining part of s_i + shortest path
              between to_node(s_i) and from_node(s_j) + entered part of s_j)
    d_gps   : straight-line distance between the two GPS fixes
    beta_t  = beta * sqrt(max(dt, 1))   (longer gaps -> looser transitions)
Unreachable pairs (no path within the search limit) get log-prob `break_logp` instead of -inf, so the
decoder can still pass through disconnected network pieces / bad candidates without dying.
Large gaps (dt > gap_s) reset the chain: transitions become uniform (new route segment).
"""
import heapq

import numpy as np

DEFAULT_HMM = dict(alpha=1.0, beta=5.0, break_logp=-30.0, gap_s=60.0, backward_tol=5.0,
                   limit_factor=3.0, limit_min=300.0, eps=1e-6)


class PathCache:
    def __init__(self, net):
        self.net = net
        self.cache = {}  # node -> (limit, dist dict)

    def dist(self, src, dst, limit):
        hit = self.cache.get(src)
        if hit is None or hit[0] < limit:
            d = {src: 0.0}
            pq = [(0.0, src)]
            adj = self.net.adj
            while pq:
                du, u = heapq.heappop(pq)
                if du > d.get(u, np.inf) or du > limit:
                    continue
                for v, _, w in adj.get(u, ()):
                    nd = du + w
                    if nd < d.get(v, np.inf) and nd <= limit:
                        d[v] = nd
                        heapq.heappush(pq, (nd, v))
            hit = (limit, d)
            self.cache[src] = hit
        return hit[1].get(dst, np.inf)


def route_distance(net, pc, a, fa, b, fb, limit, tol):
    La, Lb = net.length[a], net.length[b]
    if a == b:
        delta = (fb - fa) * La
        return abs(delta) if delta >= -tol else np.inf
    sp = pc.dist(int(net.arcs[a, 1]), int(net.arcs[b, 0]), limit)
    return (1 - fa) * La + sp + fb * Lb


def viterbi(cands, net, txy, t, prm=DEFAULT_HMM):
    """cands: list over observations of (arcs ndarray, frac ndarray, p_norm ndarray).
    Returns (best arc per observation, number of chain breaks used)."""
    prm = {**DEFAULT_HMM, **(prm or {})}
    pc = PathCache(net)
    N = len(cands)
    a0, f0, p0 = cands[0]
    score = prm["alpha"] * np.log(p0 + prm["eps"])
    backs = [None]
    breaks = 0
    for i in range(1, N):
        ai, fi, pi = cands[i]
        a_prev, f_prev, _ = cands[i - 1]
        emis = prm["alpha"] * np.log(pi + prm["eps"])
        dt = max(t[i] - t[i - 1], 1.0)
        dg = float(np.hypot(*(txy[i] - txy[i - 1])))
        if dt > prm["gap_s"]:
            bp = np.full(len(ai), int(np.argmax(score)))
            new = score.max() + emis
            breaks += 1
        else:
            beta = prm["beta"] * np.sqrt(dt)
            limit = max(prm["limit_min"], prm["limit_factor"] * dg + 100)
            T = np.empty((len(a_prev), len(ai)))
            for u in range(len(a_prev)):
                for v in range(len(ai)):
                    dr = route_distance(net, pc, a_prev[u], f_prev[u], ai[v], fi[v], limit, prm["backward_tol"])
                    T[u, v] = prm["break_logp"] if not np.isfinite(dr) else -abs(dr - dg) / beta
            tot = score[:, None] + T
            bp = np.argmax(tot, axis=0)
            new = tot[bp, np.arange(len(ai))] + emis
        backs.append(bp)
        score = new - new.max()  # keep numerically stable
    k = int(np.argmax(score))
    path = np.empty(N, dtype=np.int64)
    for i in range(N - 1, -1, -1):
        path[i] = cands[i][0][k]
        if i > 0:
            k = int(backs[i][k])
    return path, breaks


def cands_from_frame(df, p_norm, n_obs):
    """Group candidate rows of one trip (columns obs, arc, frac) into per-observation tuples."""
    out = [None] * n_obs
    obs = df["obs"].to_numpy()
    arc = df["arc"].to_numpy()
    frac = df["frac"].to_numpy()
    order = np.argsort(obs, kind="stable")
    obs, arc, frac, p = obs[order], arc[order], frac[order], p_norm[order]
    cuts = np.flatnonzero(np.diff(obs)) + 1
    for o_, a_, f_, p_ in zip(np.split(obs, cuts), np.split(arc, cuts), np.split(frac, cuts), np.split(p, cuts)):
        out[int(o_[0])] = (a_, f_, p_)
    return out


def shortest_arc_path(net, src_node, dst_node, limit=3000.0):
    """Dijkstra returning the list of arc ids from src_node to dst_node (None if unreachable)."""
    d = {src_node: 0.0}
    prev = {}
    pq = [(0.0, src_node)]
    while pq:
        du, u = heapq.heappop(pq)
        if u == dst_node:
            break
        if du > d.get(u, np.inf) or du > limit:
            continue
        for v, aid, w in net.adj.get(u, ()):
            nd = du + w
            if nd < d.get(v, np.inf):
                d[v] = nd
                prev[v] = (u, aid)
                heapq.heappush(pq, (nd, v))
    if dst_node not in d:
        return None
    path, n = [], dst_node
    while n != src_node:
        u, aid = prev[n]
        path.append(aid)
        n = u
    return path[::-1]


def fill_route(net, arc_seq, limit=3000.0):
    """Turn the de-duplicated per-observation arc sequence into a connected arc route by inserting
    shortest-path arcs between consecutive arcs that are not adjacent. Returns (route, n_unfilled)."""
    if len(arc_seq) == 0:
        return np.array([], dtype=np.int64), 0
    out = [int(arc_seq[0])]
    unfilled = 0
    for b in arc_seq[1:]:
        a = out[-1]
        b = int(b)
        if net.arcs[a, 1] != net.arcs[b, 0]:
            p = shortest_arc_path(net, int(net.arcs[a, 1]), int(net.arcs[b, 0]), limit)
            if p is None:
                unfilled += 1
            else:
                out.extend(p)
        out.append(b)
    return np.array(out, dtype=np.int64), unfilled
