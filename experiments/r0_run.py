import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
"""R0: irreducible ambiguity. Standardized linear-Gaussian (MEC is the true identifiability class).
Per graph: exact MEC posterior target; 2 independent FCI resamples x 3 aggregator permutation seeds;
caches tokens/nodes/rho so later analyses (vote baseline, temp scaling, ensembling) need no re-estimation."""
import sys, time, pickle; 
import numpy as np
import sea_run as S, gen
enc, top, args = S.load_aggregator(S.CKPT)
import os
n=10; out=pickle.load(open('results/r0.pkl','rb')) if os.path.exists('results/r0.pkl') else []; t0=time.time(); print('resuming from',len(out),flush=True)
seen={r['A'].tobytes() for r in out}
def make(A, rng):
    W=gen.random_weights(A,rng); sd=0.4*np.sqrt(rng.uniform(1,2,n)); X=np.zeros((10000,n))
    for j in gen.topo_order(A):
        X[:,j]=rng.standard_normal(10000)*1.2 if A[:,j].sum()==0 else X@W[:,j]+sd[j]*rng.standard_normal(10000)
    return (X-X.mean(0))/X.std(0)
gid=len(out)
for n_edges, count in [(10,40),(20,20)]:
    done=0; seed=0
    while done<count:
        rng=np.random.default_rng(10_000*n_edges+seed); seed+=1
        A=gen.random_er_dag(n,n_edges,rng)
        if A.sum()<3: continue
        mem=gen.mec_members(A, max_undirected=14)
        if mem is None: continue
        if A.tobytes() in seen:
            done+=1; continue
        if gid < len(out):  # resume: graph already computed (seeds are deterministic)
            assert (out[gid]['A']==A).all(); gid+=1; done+=1; continue
        X=make(A,rng); reps=[]
        for r in range(2):
            c=S.sample_and_estimate(S.standard_draw(X),n,rng,T=500)
            Ps=[S.predict(enc,top,c,seed=100*r+s) for s in range(3)]
            reps.append(dict(cache=c,P=np.stack(Ps)))
        out.append(dict(gid=gid,n_edges=n_edges,A=A,mec=len(mem),tgt=gen.three_way_target(mem,n),reps=reps))
        gid+=1; done+=1
        pickle.dump(out,open("results/r0.pkl","wb"))
        print("graph",gid,"E",n_edges,"MEC",len(mem),"%.0fs"%(time.time()-t0),flush=True)
print("ALL DONE",flush=True)
