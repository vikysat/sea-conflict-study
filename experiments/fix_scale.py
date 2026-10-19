import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
"""Block 5: label-free rho-scale normalization (rho <- rho * REF/mean diag) and temperature scaling fit on held-out graphs."""
import sys, pickle; 
import numpy as np, sea_run as S, gen
enc, top, args = S.load_aggregator(S.CKPT)
n=10; iu=np.triu_indices(n,1); REF=10.33   # mean diag of rho on training-like raw (paper-style) data, from our own generator
H=lambda P:-(P*np.log(np.clip(P,1e-9,1))).sum(-1)
r0=[r for r in pickle.load(open("results/r0.pkl","rb")) if r['n_edges']==10]
r2=pickle.load(open("results/r0v2.pkl","rb"))
sets={"standardized (rho x4.5)":[dict(A=r['A'],T=r['tgt'],c=r['reps'][0]['cache'],P=r['reps'][0]['P'][0]) for r in r0[:40]],
      "matched f=0.9":[dict(A=r['A'],T=r['tgt'],c=r['cache'],P=r['P'][0]) for r in r2 if r['f']==0.9 and r['n_edges']==10],
      "low-scale f=0.8":[dict(A=r['A'],T=r['tgt'],c=r['cache'],P=r['P'][0]) for r in r2 if r['f']==0.8]}
for name,L in sets.items():
    for d in L:
        c=dict(d['c']); fac=REF/np.diag(c['rho']).mean(); c['rho']=c['rho']*fac; d['Pn']=S.predict(enc,top,c,seed=0); d['fac']=fac
pickle.dump(sets,open("results/fix_sets.pkl","wb")); print("forward passes done",flush=True)
