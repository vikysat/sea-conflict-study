import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import time, pickle, os
import numpy as np, sea_run as S, gen
enc, top, args = S.load_aggregator(S.CKPT)
n=10; out=[]; t0=time.time(); graphs=[]
for n_edges,count in [(10,25),(20,15)]:
    seed=0
    while sum(g[0]==n_edges for g in graphs)<count:
        rng=np.random.default_rng(77_000*n_edges+seed); seed+=1
        A=gen.random_er_dag(n,n_edges,rng)
        if A.sum()<3: continue
        mem=gen.mec_members(A,max_undirected=14)
        if mem is None: continue
        graphs.append((n_edges,A,mem,gen.random_weights(A,rng,0.5,1.5),seed))
for gi,(n_edges,A,mem,W,seed) in enumerate(graphs):
    for f in ([0.9,0.8] if n_edges==10 else [0.9]):
        rng=np.random.default_rng(555+gi*10+int(f*10))
        X=gen.sample_eqvar(A,W,rng,10000,f)
        c=S.sample_and_estimate(S.standard_draw(X),n,rng,T=500)
        P=np.stack([S.predict(enc,top,c,seed=s) for s in range(2)])
        out.append(dict(gid=gi,f=f,n_edges=n_edges,A=A,mec=len(mem),tgt=gen.three_way_target(mem,n),cache=c,P=P,W=W))
    pickle.dump(out,open("results/r0v2.pkl","wb")); print("graph",gi+1,"/",len(graphs),"%.0fs"%(time.time()-t0),flush=True)
print("ALL DONE",flush=True)
