"""Shared analysis helper: per-pair vote statistics from cached FCI tokens."""
import numpy as np
import sea_run as S
n = 10
iu = np.triu_indices(n, 1)
K = len(iu[0])


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
