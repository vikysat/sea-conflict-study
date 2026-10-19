"""Sensitivity of SEA's output to a global rescaling of rho (standardized data, 20 graphs with E=10). Needs the SEA checkpoint."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import pickle, numpy as np
import gen, sea_run as S
from sklearn.metrics import roc_auc_score
enc, top, args = S.load_aggregator(S.CKPT)
rows = pickle.load(open("results/r0.pkl", "rb")); n = 10; iu = np.triu_indices(n, 1)
off = lambda R: np.abs(R[iu]).mean(); dg = lambda R: np.diag(R).mean()
std_off = [off(r['reps'][0]['cache']['rho']) for r in rows[:20]]; std_dg = [dg(r['reps'][0]['cache']['rho']) for r in rows[:20]]
raw_off, raw_dg = [], []
for r in rows[:20]:   # same graphs, same generator, but NOT standardized
    rng = np.random.default_rng(r['gid']); A = r['A']; W = gen.random_weights(A, rng); sd = 0.4 * np.sqrt(rng.uniform(1, 2, n)); X = np.zeros((10000, n))
    for j in gen.topo_order(A): X[:, j] = rng.standard_normal(10000) * 1.2 if A[:, j].sum() == 0 else X @ W[:, j] + sd[j] * rng.standard_normal(10000)
    R = S.precision(X[rng.choice(10000, 500, replace=False)]); raw_off.append(off(R)); raw_dg.append(dg(R))
print("rho scale | mean |offdiag| | mean diag\n  standardized: %.2f | %.2f\n  raw:          %.2f | %.2f" % (np.mean(std_off), np.mean(std_dg), np.mean(raw_off), np.mean(raw_dg)))
print("\nscale factor on rho -> compelled-edge orientation | false-edge mass on non-adjacent | TV on ambiguous edges | adjacency AUC")
for c in [0.1, 0.25, 0.5, 1, 2, 4]:
    ok, fp, tva, au = [], [], [], []
    for r in rows[:20]:
        cache = dict(r['reps'][0]['cache']); cache['rho'] = cache['rho'] * c
        P = S.predict(enc, top, cache, seed=0); T = r['tgt']; D, U = gen.cpdag_undirected(r['A'])
        amb = np.array([U[i, j] for i, j in zip(*iu)]); comp = np.array([D[i, j] or D[j, i] for i, j in zip(*iu)]); non = ~(amb | comp)
        if comp.any(): ok.append((P[comp][:, 1:].argmax(1) == T[comp][:, 1:].argmax(1)).mean())
        fp.append(1 - P[non, 0].mean())
        if amb.any(): tva.append(0.5 * np.abs(P[amb] - T[amb]).sum(1).mean())
        y = (~non).astype(int)
        if 0 < y.sum() < len(y): au.append(roc_auc_score(y, 1 - P[:, 0]))
    print("  x%-5s -> %.3f | %.3f | %.3f | %.3f" % (c, np.mean(ok), np.mean(fp), np.mean(tva), np.mean(au)))
