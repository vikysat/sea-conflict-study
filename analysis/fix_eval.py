import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import pickle
import numpy as np, gen, sea_run as S
enc, top, args = S.load_aggregator(S.CKPT)
n=10; iu=np.triu_indices(n,1); REF=10.33
sets=pickle.load(open("results/fix_sets.pkl","rb")); r2=pickle.load(open("results/r2.pkl","rb"))
H=lambda P:-(P*np.log(np.clip(P,1e-9,1))).sum(-1)
def tempT(P,T): z=np.log(np.clip(P,1e-9,1))/T; z-=z.max(-1,keepdims=True); e=np.exp(z); return e/e.sum(-1,keepdims=True)
def classes(A):
    D,U=gen.cpdag_undirected(A); amb=np.array([U[i,j] for i,j in zip(*iu)]); comp=np.array([D[i,j] or D[j,i] for i,j in zip(*iu)]); return amb,comp
def ece(Ps,Ts,bins=10):
    p=np.concatenate([P[:,1:].ravel() for P in Ps]); t=np.concatenate([T[:,1:].ravel() for T in Ts]); idx=np.clip((p*bins).astype(int),0,bins-1)
    return sum((idx==b).mean()*abs(p[idx==b].mean()-t[idx==b].mean()) for b in range(bins) if (idx==b).any())
def metrics(items,key,T=1.0):
    Ps=[tempT(d[key],T) for d in items]; Ts=[d['T'] for d in items]; ok=[];conf=[];tv=[]
    for d,P in zip(items,Ps):
        amb,comp=classes(d['A'])
        if comp.any(): ok.append((P[comp][:,1:].argmax(1)==d['T'][comp][:,1:].argmax(1)).mean())
        if amb.any(): conf.append((P[amb][:,1:].max(1)>0.9).mean())
        tv.append(0.5*np.abs(P-d['T']).sum(1).mean())
    return dict(ece=ece(Ps,Ts),comp=np.mean(ok),conf=np.mean(conf),tv=np.mean(tv))
def fitT(items,key):
    best=None
    for T in np.arange(0.5,3.01,0.05):
        ce=np.mean([-(d['T']*np.log(np.clip(tempT(d[key],T),1e-9,1))).sum(1).mean() for d in items])
        if best is None or ce<best[0]: best=(ce,T)
    return best[1]
fit=[d for k,d in enumerate(sets["matched f=0.9"]) if k%2==0]            # graphs used ONLY to fit T
held={name:[d for k,d in enumerate(L) if (k%2==1 or name.startswith("standardized"))] for name,L in sets.items()}  # standardized set uses entirely different graphs
T_raw,T_norm=fitT(fit,'P'),fitT(fit,'Pn'); print("fitted temperatures (fit on %d graphs, raw-rho / normalized-rho): %.2f / %.2f"%(len(fit),T_raw,T_norm))
print("\nHELD-OUT graphs.  cols: ECE vs exact posterior | compelled edges correct | share of ambiguous edges >0.9 conf | TV to posterior")
for name,L in held.items():
    print("\n%s (n=%d)"%(name,len(L)))
    for lab,key,T in [("as is","P",1.0),("temperature only","P",T_raw),("rho-scale normalization","Pn",1.0),("normalization + temperature","Pn",T_norm)]:
        m=metrics(L,key,T); print("   %-30s ECE %.3f | %.3f | %.3f | %.3f"%(lab,m['ece'],m['comp'],m['conf'],m['tv']))
# R2 conflicts: need forward passes with normalized rho on held-out parity (odd index among f=0.9 graphs)
f9=[r['gid'] for r in pickle.load(open("results/r0v2.pkl","rb")) if r['f']==0.9 and r['n_edges']==10]; odd={g for k,g in enumerate(f9) if k%2==1}
adj=lambda A:np.array([(A[i,j] or A[j,i]) for i,j in zip(*iu)],bool)
print("\nR2 mixture, held-out graphs, presence-conflict pairs (target P(edge)=0.5 at pi=0.5): mean P(edge) | share confident (<0.2 or >0.8)")
for pi in [0.25,0.5]:
    rows={"as is":[],"temperature only":[],"rho-scale normalization":[],"normalization + temperature":[]}
    for r in r2:
        if r['pi']!=pi or r['gid'] not in odd: continue
        m=adj(r['A1'])!=adj(r['A2']); c=dict(r['cache']); c['rho']=c['rho']*REF/np.diag(c['rho']).mean(); Pn=S.predict(enc,top,c,seed=0)
        for lab,P in [("as is",r['P']),("temperature only",tempT(r['P'],T_raw)),("rho-scale normalization",Pn),("normalization + temperature",tempT(Pn,T_norm))]:
            rows[lab]+=list((1-P[:,0])[m])
    for lab,v in rows.items():
        v=np.array(v); print("   pi=%.2f %-30s %.2f | %.0f%%  (n=%d)"%(pi,lab,v.mean(),100*((v<0.2)|(v>0.8)).mean(),len(v)))
