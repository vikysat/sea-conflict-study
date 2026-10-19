"""R1 dose-response summary (delete / flip / add) + bimodality + relation to |rho_ij|. Input: results/r1.pkl, results/r0v2.pkl"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import pickle, numpy as np
from scipy.stats import spearmanr
out = pickle.load(open("results/r1.pkl", "rb"))
for op in ["delete", "flip", "add"]:
    rs = [r for r in out if r['op'] == op]; key = "p_true_dir" if op == "flip" else "p_edge"
    print("\n== %s (n=%d target pairs) ==\ndose | realized vote | %s" % (op, len({r['gid'] for r in rs}), "P(true direction)" if op == "flip" else "P(edge)"))
    for d in sorted({r['dose'] for r in rs}):
        s = [r for r in rs if r['dose'] == d]
        print("%.1f  | %.2f | %.2f (sd across pairs %.2f)" % (d, np.mean([r['vote'] for r in s]), np.mean([r[key] for r in s]), np.std([r[key] for r in s])))
rs = [r for r in out if r['op'] == 'delete']
print("\ndelete, by realized vote fraction:")
for lo, hi in [(0.8, 1.01), (0.6, 0.8), (0.4, 0.6), (0.2, 0.4), (0, 0.2)]:
    s = [r['p_edge'] for r in rs if lo <= r['vote'] < hi]
    if s: print("  vote in [%.1f,%.1f): mean P(edge) %.2f (n=%d)" % (lo, min(hi, 1), np.mean(s), len(s)))
def dist(op, d, key="p_edge"):
    p = np.array([r[key] for r in out if r['op'] == op and r['dose'] == d])
    return "<0.2: %2.0f%% | 0.2-0.8: %2.0f%% | >0.8: %2.0f%% (n=%d)" % (100 * (p < 0.2).mean(), 100 * ((p >= 0.2) & (p <= 0.8)).mean(), 100 * (p > 0.8).mean(), len(p))
print("\nper-pair spread of P(edge):")
for op, ds in [("delete", [0.5, 0.7, 1.0]), ("add", [0.7, 1.0])]:
    for d in ds: print("  %s dose %.1f: %s" % (op, d, dist(op, d)))
print("  flip dose 1.0, P(true dir): %s" % dist("flip", 1.0, "p_true_dir"))
recs = {r['gid']: r for r in pickle.load(open("results/r0v2.pkl", "rb")) if r['f'] == 0.9 and r['n_edges'] == 10}
x = np.array([abs(recs[r['gid']]['cache']['rho'][r['i'], r['j']]) for r in out if r['op'] == 'delete' and r['dose'] == 1.0])
y = np.array([r['p_edge'] for r in out if r['op'] == 'delete' and r['dose'] == 1.0])
print("\ndelete dose 1.0: Spearman(|rho_ij|, P(edge)) = %.2f (p=%.3f)" % tuple(spearmanr(x, y)))
