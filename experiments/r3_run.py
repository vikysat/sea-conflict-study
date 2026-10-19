import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import pickle
import numpy as np, sea_run as S, gen
from sklearn.metrics import roc_auc_score
enc, top, args = S.load_aggregator(S.CKPT)
n=10; iu=np.triu_indices(n,1)
recs=[r for r in pickle.load(open("results/r0v2.pkl","rb")) if r['f']==0.9 and r['n_edges']==10]
adj=lambda A:np.array([(A[i,j] or A[j,i]) for i,j in zip(*iu)],bool)
res=[]
for a,r in enumerate(recs):
    d=recs[(a+1)%len(recs)]
    for name,E_src,R_src in [("E'=G1,rho=G2",r,d),("E'=G2,rho=G1",d,r)]:
        c=dict(E_src['cache']); c['rho']=R_src['cache']['rho']
        P=S.predict(enc,top,c,seed=0); pe=1-P[:,0]; aE,aR=adj(E_src['A']),adj(R_src['A'])
        res.append(dict(case=name,P=P,only_E=pe[aE&~aR].mean() if (aE&~aR).any() else np.nan,only_R=pe[aR&~aE].mean() if (aR&~aE).any() else np.nan,
                        both=pe[aE&aR].mean() if (aE&aR).any() else np.nan,neither=pe[~aE&~aR].mean(),
                        auc_E=roc_auc_score(aE,pe),auc_R=roc_auc_score(aR,pe),
                        base_E=1-E_src['P'][0][:,0]))
pickle.dump(res,open("results/r3.pkl","wb"))
f=lambda k:"%.2f"%np.nanmean([x[k] for x in res]); 
print("n swaps:",len(res))
print("mean P(edge) on pairs adjacent ONLY in the E' source graph: %s | ONLY in the rho source graph: %s | both: %s | neither: %s"%(f("only_E"),f("only_R"),f("both"),f("neither")))
print("adjacency AUC of P(edge) vs E'-source graph: %s | vs rho-source graph: %s"%(f("auc_E"),f("auc_R")))
print("(reference, no swap) P(edge) AUC vs own graph ~0.997")
