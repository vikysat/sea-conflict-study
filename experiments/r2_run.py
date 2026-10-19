import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
"""R2: batch-heterogeneous mixture. Component 1 = graph G1 (matched scale f=0.9), component 2 = G2 = G1 with 2 rewired edges.
Each batch is drawn entirely from one component (c_t ~ Bernoulli(pi)), so subset estimates genuinely disagree. rho from pooled rows.
Soft target for each pair = (1-pi_hat)*MECposterior(G1) + pi_hat*MECposterior(G2), pi_hat = realized share of G2 batches."""
import sys, time, pickle; 
import numpy as np, sea_run as S, gen
enc, top, args = S.load_aggregator(S.CKPT)
n=10; recs=[r for r in pickle.load(open("results/r0v2.pkl","rb")) if r['f']==0.9 and r['n_edges']==10]
out=[]; t0=time.time(); rng0=np.random.default_rng(31337)
for r in recs[:22]:
    A1,W1=r['A'],r['W']; post1=gen.three_way_target(gen.mec_members(A1),n)
    for tries in range(30):
        A2=gen.rewire(A1,2,rng0); m2=gen.mec_members(A2,max_undirected=14)
        if m2 is not None and not np.array_equal(A1,A2): break
    else: continue
    post2=gen.three_way_target(m2,n); W2=gen.share_weights(A1,W1,A2,rng0)
    rng=np.random.default_rng(r['gid']+999); X1=gen.sample_eqvar(A1,W1,rng,10000,0.9); X2=gen.sample_eqvar(A2,W2,rng,10000,0.9)
    for pi in [0.0,0.1,0.25,0.5]:
        w=[1-pi,pi]; c=S.sample_and_estimate(S.batch_mixture_draw([X1,X2],w),n,rng,T=500,rho_rows=S.pooled_rho_rows([X1,X2],w,rng))
        pih=float((c['src']==1).mean()); P=S.predict(enc,top,c,seed=0)
        out.append(dict(gid=r['gid'],pi=pi,pi_hat=pih,A1=A1,A2=A2,post1=post1,post2=post2,tgt=(1-pih)*post1+pih*post2,P=P,cache=c))
    pickle.dump(out,open("results/r2.pkl","wb")); print("graph",r['gid'],"%.0fs"%(time.time()-t0),flush=True)
print("ALL DONE",flush=True)
