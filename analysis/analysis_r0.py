import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import pickle, numpy as np
import gen, sea_run as S
rows=pickle.load(open("results/r0.pkl","rb")); n=10; iu=np.triu_indices(n,1); K=len(iu[0])
H=lambda P:-(P*np.log(np.clip(P,1e-9,1))).sum(-1)
def classes(A):
    D,U=gen.cpdag_undirected(A)
    amb=np.array([U[i,j] for i,j in zip(*iu)]); comp=np.array([D[i,j] or D[j,i] for i,j in zip(*iu)])
    return amb, comp, ~(amb|comp)
def votes(c):
    """per pair i<j: edge-vote fraction, and arrowhead-direction score from cached FCI tokens"""
    k=c['k']; tok=c['tokens']; nodes=c['nodes']; adj=np.full(K,np.nan); dirn=np.full(K,np.nan); cov=np.zeros(K)
    for p,(i,j) in enumerate(zip(*iu)):
        ts=S.subsets_with_pair(c,i,j); cov[p]=len(ts)
        if not len(ts): continue
        e=a_j=a_i=0
        for t in ts:
            a=int(np.flatnonzero(nodes[t]==i)[0]); b=int(np.flatnonzero(nodes[t]==j)[0]); x=int(tok[t,S._slot(k,a,b)])
            if x!=S.NONE:
                e+=1; a_j+= x in (S.FWD,S.CIRC_FWD,S.BI); a_i+= x in (S.BWD,S.CIRC_BWD,S.BI)
        adj[p]=e/len(ts); dirn[p]=(a_j-a_i)/e if e else 0
    return adj,dirn,cov
acc={k:[] for k in ["tv_amb","tv_comp","tv_non","bias_none_adj","fp_none","H_amb_m","H_amb_t","conf_amb","conf_comp","comp_ok","comp_ok_ens","tv_single","tv_ens","perm_sd","resamp_sd","vote_ok","vote_n","sea_ok_same","auc_sea","auc_vote","cov"]}
from sklearn.metrics import roc_auc_score
rel=[]  # (pred prob of a direction, soft target of that direction)
for r in rows:
    amb,comp,non=classes(r['A']); T=r['tgt']; Ps=np.concatenate([x['P'] for x in r['reps']])  # (6,K,3)
    P1=Ps[0]; Pm=Ps.mean(0)
    tv=lambda P,m:0.5*np.abs(P[m]-T[m]).sum(1)
    for nm,m in [("amb",amb),("comp",comp),("non",non)]:
        if m.any(): acc["tv_"+nm].append(tv(Ps.reshape(-1,K,3)[0],m).mean())
    adjm=~non
    acc["bias_none_adj"].append((P1[adjm,0]-T[adjm,0]).mean()); acc["fp_none"].append(1-P1[non,0].mean())
    if amb.any():
        acc["H_amb_m"].append(H(P1[amb]).mean()); acc["H_amb_t"].append(H(T[amb]).mean()); acc["conf_amb"].append((P1[amb][:,1:].max(1)>0.9).mean())
    if comp.any():
        tdir=T[comp][:,1:].argmax(1)
        acc["comp_ok"].append((P1[comp][:,1:].argmax(1)==tdir).mean()); acc["comp_ok_ens"].append((Pm[comp][:,1:].argmax(1)==tdir).mean())
        acc["conf_comp"].append((P1[comp][:,1:].max(1)>0.9).mean())
    acc["tv_single"].append(0.5*np.abs(Ps-T).sum(-1).mean()); acc["tv_ens"].append(0.5*np.abs(Pm-T).sum(-1).mean())
    perm=np.stack([x['P'] for x in r['reps']]); acc["perm_sd"].append(perm.std(1).mean())            # sd over 3 perm seeds, same cache
    acc["resamp_sd"].append(np.stack([x['P'].mean(0) for x in r['reps']]).std(0).mean())            # sd over 2 FCI resamples
    adj,dirn,cov=votes(r['reps'][0]['cache']); acc["cov"].append(np.nanmin(cov))
    if comp.any():
        ok=comp&(adj>0.5); 
        if ok.any():
            tdir=(T[:,1]>T[:,2]); vd=dirn>0
            acc["vote_ok"].append((vd[ok]==tdir[ok]).mean()); acc["vote_n"].append(ok.sum()); acc["sea_ok_same"].append(((P1[:,1]>P1[:,2])[ok]==tdir[ok]).mean())
    y=(~non).astype(int)
    if 0<y.sum()<len(y): acc["auc_sea"].append(roc_auc_score(y,1-P1[:,0])); acc["auc_vote"].append(roc_auc_score(y,np.nan_to_num(adj)))
    for d in (1,2): rel.append(np.stack([P1[:,d],T[:,d]],1))
m=lambda k:(np.mean(acc[k]),np.std(acc[k])/np.sqrt(len(acc[k])))
f=lambda k:"%.3f ± %.3f"%m(k)
print("graphs",len(rows)," total ambiguous edges: see below")
print("\n-- distance to exact MEC posterior (total variation, 0=perfect) --")
for k,l in [("tv_amb","ambiguous edges"),("tv_comp","compelled edges"),("tv_non","non-adjacent pairs")]: print("  %-20s %s"%(l,f(k)))
print("\n-- sparsity: mean(P_none - target_none) on truly adjacent pairs: %s   | mean false-edge mass on non-adjacent: %s"%(f("bias_none_adj"),f("fp_none")))
print("-- ambiguous edges: entropy model %s vs target %s ; share max-prob>0.9: %s"%(f("H_amb_m"),f("H_amb_t"),f("conf_amb")))
print("-- compelled edges: share max-prob>0.9: %s ; orientation correct: single %s, 6-run ensemble %s"%(f("conf_comp"),f("comp_ok"),f("comp_ok_ens")))
print("\n-- noise decomposition (sd of P): aggregator permutation noise %s ; FCI resampling %s"%(f("perm_sd"),f("resamp_sd")))
print("-- TV to posterior: single run %s -> 6-run ensemble %s"%(f("tv_single"),f("tv_ens")))
print("\n-- information ceiling, compelled edges that FCI marginals detect as adjacent (vote>0.5):")
print("   simple arrowhead-vote orientation correct: %s | SEA on same edges: %s"%(f("vote_ok"),f("sea_ok_same")))
print("-- adjacency AUC: SEA %s vs vote-fraction %s ; min pair coverage (subsets): %.0f"%(f("auc_sea"),f("auc_vote"),np.mean(acc["cov"])))
rel=np.concatenate(rel); bins=np.linspace(0,1,6); idx=np.clip(np.digitize(rel[:,0],bins)-1,0,4)
print("\n-- reliability of direction probabilities vs exact posterior (pred bin -> mean target, n):")
for b in range(5):
    s=rel[idx==b]; print("   pred %.1f-%.1f -> target %.3f  (n=%d)"%(bins[b],bins[b+1],s[:,1].mean() if len(s) else np.nan,len(s)))
pickle.dump(dict(acc=acc,rel=rel),open("results/r0_summary.pkl","wb"))
