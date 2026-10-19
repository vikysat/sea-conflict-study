"""2x2 diagnostic (root distribution x standardization): AP / OA / AUC vs the exact MEC oracle, random-member reference,
and ambiguous-vs-compelled edge behaviour.  Input: results/diag_std.pkl (made by experiments/diag_std.py)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import pickle, numpy as np
import gen
from sklearn.metrics import average_precision_score, roc_auc_score
rows = pickle.load(open("results/diag_std.pkl", "rb")); n = 10; iu = np.triu_indices(n, 1)
conds = [("unif", False), ("unif", True), ("gauss", False), ("gauss", True)]
H = lambda P: -(P * np.log(np.clip(P, 1e-9, 1))).sum(1)
ap = lambda A, P: average_precision_score(np.concatenate([A[iu], A.T[iu]]), np.concatenate([P[:, 1], P[:, 2]]))
auc = lambda A, P: roc_auc_score(np.concatenate([A[iu], A.T[iu]]), np.concatenate([P[:, 1], P[:, 2]]))
def oa(A, P):
    ok = []
    for k, (i, j) in enumerate(zip(*iu)):
        if A[i, j]: ok.append(P[k, 1] > P[k, 2])
        if A[j, i]: ok.append(P[k, 2] > P[k, 1])
    return np.mean(ok)
print("graphs %d | oracle (exact MEC posterior) AP %.3f OA %.3f" % (len(rows), np.mean([r['oracle_ap'] for r in rows]), np.mean([oa(r['A'], r['tgt']) for r in rows])))
rm = [np.mean([ap(r['A'], gen.three_way_target(m, n)) for m in gen.mec_members(r['A'])]) for r in rows]
print("random-MEC-member (hard answer) AP %.3f" % np.mean(rm))
for c in conds:
    print("%-6s %-4s  AP %.3f  OA %.3f  AUC %.3f" % (c[0], "std" if c[1] else "raw", np.mean([ap(r['A'], r[c]['P']) for r in rows]),
          np.mean([oa(r['A'], r[c]['P']) for r in rows]), np.mean([auc(r['A'], r[c]['P']) for r in rows])))
for c in [("gauss", True), ("unif", True), ("gauss", False)]:
    hp, ht, mp, mt, ok = [], [], [], [], []
    for r in rows:
        D, U = gen.cpdag_undirected(r['A']); P = r[c]['P']; T = r['tgt']
        amb = np.array([U[i, j] for i, j in zip(*iu)]); comp = np.array([D[i, j] or D[j, i] for i, j in zip(*iu)])
        if amb.any(): hp.append(H(P[amb])); ht.append(H(T[amb])); mp.append(P[amb][:, 1:].max(1)); mt.append(T[amb][:, 1:].max(1))
        if comp.any(): ok.append(P[comp][:, 1:].argmax(1) == T[comp][:, 1:].argmax(1))
    hp, ht, mp = map(np.concatenate, (hp, ht, mp))
    print("%s %s | ambiguous edges n=%d: entropy model %.3f vs target %.3f; share >0.9 conf %.0f%% | compelled edges correct %.1f%%" %
          (c[0], "std" if c[1] else "raw", len(hp), hp.mean(), ht.mean(), 100 * (mp > 0.9).mean(), 100 * np.concatenate(ok).mean()))
