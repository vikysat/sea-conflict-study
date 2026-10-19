import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import time, pickle
import numpy as np, torch
from sklearn.metrics import average_precision_score
import sea_run as S, gen
enc, top, args = S.load_aggregator(S.CKPT)
n=10; iu=np.triu_indices(n,1)
def ap(A,P3): return average_precision_score(np.concatenate([A[iu],A.T[iu]]), np.concatenate([P3[:,1],P3[:,2]]))
def make(A, rng, roots, std):
    W=gen.random_weights(A,rng); sd=0.4*np.sqrt(rng.uniform(1,2,n)); X=np.zeros((10000,n))
    for j in gen.topo_order(A):
        if A[:,j].sum()==0: X[:,j]=rng.uniform(-2,2,10000) if roots=="unif" else rng.standard_normal(10000)*1.2
        else: X[:,j]=X@W[:,j]+sd[j]*rng.standard_normal(10000)
    return (X-X.mean(0))/X.std(0) if std else X
conds=[("unif",False),("unif",True),("gauss",False),("gauss",True)]
rows=[]
for s in range(24):
    rng=np.random.default_rng(2000+s); A=gen.random_er_dag(n,10,rng)
    if A.sum()<3: continue
    mem=gen.mec_members(A); tgt=gen.three_way_target(mem,n)
    r={"seed":s,"mec":len(mem),"oracle_ap":ap(A,tgt),"A":A,"tgt":tgt}
    for c in conds:
        X=make(A,rng,*c); cache=S.sample_and_estimate(S.standard_draw(X),n,rng,T=500)
        P=S.predict(enc,top,cache,seed=s); r[c]=dict(ap=ap(A,P),P=P)
    rows.append(r); pickle.dump(rows,open("results/diag_std.pkl","wb"))
    print(s,"done",flush=True)
