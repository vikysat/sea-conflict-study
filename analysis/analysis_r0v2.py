import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import pickle, numpy as np
import gen, sea_run as S
from sklearn.metrics import roc_auc_score, average_precision_score
n=10; iu=np.triu_indices(n,1); K=len(iu[0])
H=lambda P:-(P*np.log(np.clip(P,1e-9,1))).sum(-1)
def classes(A):
    D,U=gen.cpdag_undirected(A)
    amb=np.array([U[i,j] for i,j in zip(*iu)]); comp=np.array([D[i,j] or D[j,i] for i,j in zip(*iu)]); return amb,comp,~(amb|comp)
import sea_run as S
from helpers import votes
def oa(A,P):
    ok=[]
    for k,(i,j) in enumerate(zip(*iu)):
        if A[i,j]: ok.append(P[k,1]>P[k,2])
        if A[j,i]: ok.append(P[k,2]>P[k,1])
    return np.mean(ok)
def summarize(recs,label):
    R={k:[] for k in "tv_amb tv_comp tv_non bias fp Hm Ht conf_amb conf_comp comp_ok vote_ok sea_same auc ap oap oa ooa rho_off rho_dg perm".split()}; rel=[]
    for r in recs:
        amb,comp,non=classes(r['A']); T=r['tgt']; P=r['P'][0]; adjm=~non
        for nm,m in [("amb",amb),("comp",comp),("non",non)]:
            if m.any(): R["tv_"+nm].append(0.5*np.abs(P[m]-T[m]).sum(1).mean())
        R["bias"].append((P[adjm,0]-T[adjm,0]).mean()); R["fp"].append(1-P[non,0].mean())
        if amb.any(): R["Hm"].append(H(P[amb]).mean()); R["Ht"].append(H(T[amb]).mean()); R["conf_amb"].append((P[amb][:,1:].max(1)>0.9).mean())
        if comp.any():
            td=T[comp][:,1:].argmax(1); R["comp_ok"].append((P[comp][:,1:].argmax(1)==td).mean()); R["conf_comp"].append((P[comp][:,1:].max(1)>0.9).mean())
            adj,dirn,cov=votes(r['cache']); ok=comp&(adj>0.5)
            if ok.any():
                tdir=T[:,1]>T[:,2]; R["vote_ok"].append(((dirn>0)[ok]==tdir[ok]).mean()); R["sea_same"].append(((P[:,1]>P[:,2])[ok]==tdir[ok]).mean())
        y=(~non).astype(int)
        if 0<y.sum()<K: R["auc"].append(roc_auc_score(y,1-P[:,0]))
        A=r['A']; yy=np.concatenate([A[iu],A.T[iu]]); R["ap"].append(average_precision_score(yy,np.concatenate([P[:,1],P[:,2]]))); R["oap"].append(average_precision_score(yy,np.concatenate([T[:,1],T[:,2]])))
        R["oa"].append(oa(A,P)); R["ooa"].append(oa(A,T)); R["perm"].append(np.abs(r['P'][0]-r['P'][1]).mean())
        rho=r['cache']['rho']; R["rho_off"].append(np.abs(rho[iu]).mean()); R["rho_dg"].append(np.diag(rho).mean())
        for d in (1,2): rel.append(np.stack([P[:,d],T[:,d]],1))
    f=lambda k:"%.3f ± %.3f"%(np.mean(R[k]),np.std(R[k])/np.sqrt(len(R[k])))
    print("\n=== %s  (n=%d graphs; rho offdiag %.2f diag %.2f) ==="%(label,len(recs),np.mean(R["rho_off"]),np.mean(R["rho_dg"])))
    print("AP %s (oracle %s) | OA %s (oracle %s) | adjacency AUC %s"%(f("ap"),f("oap"),f("oa"),f("ooa"),f("auc")))
    print("TV to exact posterior: ambiguous %s | compelled %s | non-adjacent %s"%(f("tv_amb"),f("tv_comp"),f("tv_non")))
    print("sparsity: P_none-target on adjacent pairs %s | false-edge mass on non-adjacent %s"%(f("bias"),f("fp")))
    print("ambiguous edges: entropy model %s vs target %s | share >0.9 conf %s"%(f("Hm"),f("Ht"),f("conf_amb")))
    print("compelled edges: oriented correctly %s | vote baseline %s vs SEA %s on same edges | share >0.9 conf %s"%(f("comp_ok"),f("vote_ok"),f("sea_same"),f("conf_comp")))
    rel=np.concatenate(rel); bins=np.linspace(0,1,6); idx=np.clip(np.digitize(rel[:,0],bins)-1,0,4)
    print("reliability (pred bin -> exact-posterior mean): "+"; ".join("%.1f-%.1f→%.2f (n=%d)"%(bins[b],bins[b+1],rel[idx==b][:,1].mean(),(idx==b).sum()) for b in range(5) if (idx==b).any()))
    return R
if __name__=="__main__":
    recs=pickle.load(open("results/r0v2.pkl","rb"))
    out={}
    out["f0.9 all"]=summarize([r for r in recs if r['f']==0.9],"MATCHED SCALE f=0.9 (all graphs)")
    out["f0.9 E10"]=summarize([r for r in recs if r['f']==0.9 and r['n_edges']==10],"f=0.9, E=10 only")
    out["f0.8 E10"]=summarize([r for r in recs if r['f']==0.8],"SENSITIVITY f=0.8, E=10 (same graphs)")
    pickle.dump(out,open("results/r0v2_summary.pkl","wb"))
