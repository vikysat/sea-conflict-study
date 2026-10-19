"""R2 detail: edge-only-in-G1 vs edge-only-in-G2 (target / SEA / vote), confident-resolution share, and conflict-detection AUROC."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import pickle, numpy as np
from sklearn.metrics import roc_auc_score
from helpers import votes
n = 10; iu = np.triu_indices(n, 1)
adj = lambda A: np.array([(A[i, j] or A[j, i]) for i, j in zip(*iu)], bool)
Hb = lambda p: -(p * np.log(np.clip(p, 1e-9, 1)) + (1 - p) * np.log(np.clip(1 - p, 1e-9, 1)))
recs = pickle.load(open("results/r2.pkl", "rb"))
print("pi   | edge ONLY in G1 (majority): target SEA vote | edge ONLY in G2 (minority): target SEA vote")
for pi in [0.0, 0.1, 0.25, 0.5]:
    s1, s2 = [], []
    for r in recs:
        if r['pi'] != pi: continue
        a1, a2 = adj(r['A1']), adj(r['A2']); pe = 1 - r['P'][:, 0]; te = 1 - r['tgt'][:, 0]; vo = np.nan_to_num(votes(r['cache'])[0])
        for m, L in [(a1 & ~a2, s1), (a2 & ~a1, s2)]:
            for k in np.flatnonzero(m): L.append((te[k], pe[k], vo[k]))
    s1, s2 = np.array(s1), np.array(s2)
    print("%.2f | n=%2d  %.2f %.2f %.2f | n=%2d  %.2f %.2f %.2f" % (pi, len(s1), *s1.mean(0), len(s2), *s2.mean(0)))
print()
for pi in [0.25, 0.5]:
    pe = np.array([x for r in recs if r['pi'] == pi for x in (1 - r['P'][:, 0])[adj(r['A1']) != adj(r['A2'])]])
    print("pi=%.2f presence-conflict pairs n=%d: SEA P(edge) <0.2: %.0f%% | 0.2-0.8: %.0f%% | >0.8: %.0f%%" % (pi, len(pe), 100 * (pe < 0.2).mean(), 100 * ((pe >= 0.2) & (pe <= 0.8)).mean(), 100 * (pe > 0.8).mean()))
print("\nDetecting presence-conflict pairs vs agreeing pairs (AUROC): SEA entropy of P(edge) | vote-disagreement score | % conflict pairs flagged by 0.2<vote<0.8 | false-flag %")
for pi in [0.0, 0.1, 0.25, 0.5]:
    y, s1, s2, fl = [], [], [], []
    for r in recs:
        if r['pi'] != pi: continue
        a1, a2 = adj(r['A1']), adj(r['A2']); pres = a1 != a2; ag = a1 == a2; m = pres | ag
        pe = 1 - r['P'][:, 0]; vo = np.nan_to_num(votes(r['cache'])[0])
        y += list(pres[m]); s1 += list(Hb(pe)[m]); s2 += list((1 - np.abs(2 * vo - 1))[m]); fl += list(((vo > 0.2) & (vo < 0.8))[m])
    y = np.array(y); fl = np.array(fl); a = lambda s: roc_auc_score(y, s) if 0 < y.sum() < len(y) else float('nan')
    print("  %.2f | %.2f | %.2f | %3.0f%% | %3.0f%%" % (pi, a(np.array(s1)), a(np.array(s2)), 100 * fl[y].mean(), 100 * fl[~y].mean()))
