import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import pickle, numpy as np
import gen
from scipy.stats import spearmanr
n=10; iu=np.triu_indices(n,1); K=len(iu[0])
import sea_run as S
from helpers import votes
H=lambda P:-(P*np.log(np.clip(P,1e-9,1))).sum(-1)
adj=lambda A:np.array([(A[i,j] or A[j,i]) for i,j in zip(*iu)],bool)
recs=pickle.load(open("results/r2.pkl","rb")); print("records",len(recs),"graphs",len({r['gid'] for r in recs}))
rows=[]
for r in recs:
    a1,a2=adj(r['A1']),adj(r['A2']); tv12=0.5*np.abs(r['post1']-r['post2']).sum(1)
    pres=a1!=a2; ori=(a1==a2)&(tv12>0.2)&a1; agree=~(pres|ori)
    P,T=r['P'],r['tgt']; pe=1-P[:,0]; te=1-T[:,0]; vote,_,_=votes(r['cache']); vote=np.nan_to_num(vote)
    rows.append(dict(pi=r['pi'],pih=r['pi_hat'],pres=pres,ori=ori,agree=agree,pe=pe,te=te,vote=vote,P=P,T=T))
def agg(pi):
    R=[x for x in rows if x['pi']==pi]; g=lambda f:np.concatenate([f(x) for x in R]) if R else np.array([])
    pres=g(lambda x:x['pres']); ori=g(lambda x:x['ori']); ag=g(lambda x:x['agree']); pe,te,vo=g(lambda x:x['pe']),g(lambda x:x['te']),g(lambda x:x['vote'])
    tv=g(lambda x:0.5*np.abs(x['P']-x['T']).sum(1)); Hm,Ht=g(lambda x:H(x['P'])),g(lambda x:H(x['T']))
    return dict(n_pres=int(pres.sum()),n_ori=int(ori.sum()),
        pres_target=te[pres].mean() if pres.any() else np.nan, pres_model=pe[pres].mean() if pres.any() else np.nan,
        pres_vote=vo[pres].mean() if pres.any() else np.nan, mae_model=np.abs(pe-te)[pres].mean() if pres.any() else np.nan,
        mae_vote=np.abs(vo-te)[pres].mean() if pres.any() else np.nan, tv_ori=tv[ori].mean() if ori.any() else np.nan,
        tv_agree=tv[ag].mean(), Hm_conf=Hm[pres|ori].mean(), Ht_conf=Ht[pres|ori].mean(), pe_pres=pe[pres], te_pres=te[pres], pih=np.mean([x['pih'] for x in R]))
print("\npi   realized | presence-conflict pairs: target  SEA  vote | MAE SEA vs vote | orientation-conflict TV | non-conflict TV | entropy on conflict pairs: SEA vs target")
S_={}
for pi in [0.0,0.1,0.25,0.5]:
    a=agg(pi); S_[pi]=a
    print("%.2f  %.2f     | n=%3d  %.2f  %.2f  %.2f | %.3f vs %.3f | n=%2d TV %.3f | %.3f | %.2f vs %.2f"%(pi,a['pih'],a['n_pres'],a['pres_target'],a['pres_model'],a['pres_vote'],a['mae_model'],a['mae_vote'],a['n_ori'],a['tv_ori'],a['tv_agree'],a['Hm_conf'],a['Ht_conf']))
# is predicted uncertainty tied to conflict? spearman over presence-conflict pairs of SEA entropy vs target entropy, mixed pi>0
pe=np.concatenate([S_[p]['pe_pres'] for p in [0.1,0.25,0.5]]); te=np.concatenate([S_[p]['te_pres'] for p in [0.1,0.25,0.5]])
print("\npresence-conflict pairs, pi>0 (n=%d): Spearman(SEA P(edge), target frequency) = %.2f ; share of pairs where SEA is >0.25 below target: %.0f%%, >0.25 above: %.0f%%"%(len(pe),spearmanr(pe,te)[0],100*np.mean(pe<te-0.25),100*np.mean(pe>te+0.25)))
# split by which side is the minority: edge only in the minor component (G2 at small pi)
pickle.dump(S_,open("results/r2_summary.pkl","wb"))
