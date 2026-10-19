import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
"""R1: dose-response. Base = matched-scale (f=0.9, E=10) R0 caches. For a clean target pair, corrupt a fraction of the
subsets containing it and read P. delete: true edge removed; flip: orientation reversed; add: spurious edge."""
import sys, time, pickle; 
import numpy as np, sea_run as S, gen
enc, top, args = S.load_aggregator(S.CKPT)
n=10; iu=np.triu_indices(n,1); pos={(i,j):k for k,(i,j) in enumerate(zip(*iu))}
recs=[r for r in pickle.load(open("results/r0v2.pkl","rb")) if r['f']==0.9 and r['n_edges']==10]
doses=[0,0.1,0.2,0.3,0.5,0.7,1.0]; out=[]; t0=time.time(); rng=np.random.default_rng(9)
for r in recs:
    c=r['cache']; A=r['A']; D,U=gen.cpdag_undirected(A); base=r['P'][0]
    cands={"delete":[], "flip":[], "add":[]}
    for i in range(n):
        for j in range(n):
            if i==j: continue
            p=(min(i,j),max(i,j)); k=pos[p]; cov=len(S.subsets_with_pair(c,i,j))
            if cov<40: continue
            v=S.vote_fraction(c,i,j)
            if A[i,j] and v>=0.8 and 1-base[k,0]>0.8: cands["delete"].append((i,j))
            if A[i,j] and D[i,j] and v>=0.8 and (base[k,1] if i<j else base[k,2])>0.8: cands["flip"].append((i,j))
            if i<j and not A[i,j] and not A[j,i] and v<=0.05 and 1-base[k,0]<0.1: cands["add"].append((i,j))
    for op,cl in cands.items():
        if not cl: continue
        i,j=cl[rng.integers(len(cl))]; k=pos[(min(i,j),max(i,j))]
        for d in doses:
            tok=S.inject(c,i,j,d,op,rng); P=S.predict(enc,top,c,tokens=tok,seed=0)
            pc=(P[k,1] if i<j else P[k,2]) if op=="flip" else np.nan
            out.append(dict(gid=r['gid'],op=op,i=i,j=j,dose=d,vote=S.vote_fraction(c,i,j,tok),p_edge=1-P[k,0],p_true_dir=pc,p_none=P[k,0]))
    pickle.dump(out,open("results/r1.pkl","wb")); print("graph",r['gid'],"%.0fs"%(time.time()-t0),flush=True)
print("ALL DONE",flush=True)
