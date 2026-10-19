"""
Synthetic data + ground truth for the SEA conflict study.

Conventions (match the SEA repo / DCDI):  A[i, j] = 1  <=>  edge i -> j.
"""
import itertools
import numpy as np


# ---------------------------------------------------------------- graphs
def random_er_dag(n, n_edges, rng):
    """Erdos-Renyi DAG with ~n_edges expected edges, random topological order."""
    order = rng.permutation(n)
    p = n_edges / (n * (n - 1) / 2)
    A = np.zeros((n, n), dtype=int)
    for a in range(n):
        for b in range(a + 1, n):
            if rng.random() < p:
                A[order[a], order[b]] = 1
    return A


def is_dag(A):
    """Kahn's algorithm."""
    A = np.asarray(A).astype(bool)
    indeg = A.sum(0).astype(int)
    stack = [v for v in range(len(A)) if indeg[v] == 0]
    seen = 0
    while stack:
        v = stack.pop()
        seen += 1
        for w in np.flatnonzero(A[v]):
            indeg[w] -= 1
            if indeg[w] == 0:
                stack.append(w)
    return seen == len(A)


def topo_order(A):
    A = A.copy()
    n = len(A)
    order, remaining = [], set(range(n))
    while remaining:
        roots = [v for v in remaining if not any(A[u, v] for u in remaining)]
        assert roots, "cycle"
        order.extend(sorted(roots))
        remaining -= set(roots)
    return order


# ------------------------------------------------------------- linear SEM
def random_weights(A, rng, lo=0.5, hi=2.0):
    W = rng.uniform(lo, hi, A.shape) * rng.choice([-1, 1], A.shape)
    return W * A


def sample_linear_sem(A, W, noise_sd, m, rng, standardize=True):
    """x_j = sum_i W[i,j] x_i + noise_sd[j] * N(0,1).  Gaussian roots => linear-Gaussian."""
    n = len(A)
    X = np.zeros((m, n))
    for j in topo_order(A):
        X[:, j] = X @ W[:, j] + noise_sd[j] * rng.standard_normal(m)
    if standardize:  # removes the variance-ordering shortcut (Reisach et al. 2021)
        X = (X - X.mean(0)) / X.std(0)
    return X


# ---------------------------------------------------- Markov equivalence class
def _vstructs(A):
    """set of (i, j, k), i<k, with i->j<-k and i,k non-adjacent"""
    n = len(A)
    skel = (A | A.T).astype(bool)
    out = set()
    for j in range(n):
        pa = np.flatnonzero(A[:, j])
        for i, k in itertools.combinations(pa, 2):
            if not skel[i, k]:
                out.add((i, j, k))
    return out


def cpdag_undirected(A):
    """Return (D, U): compelled directed edges D[i,j]=1 for i->j and undirected edges U (symmetric),
    via v-structures + Meek rules 1-3."""
    n = len(A)
    skel = (A | A.T).astype(bool)
    D = np.zeros((n, n), dtype=bool)
    for (i, j, k) in _vstructs(A):
        D[i, j] = D[k, j] = True
    U = skel & ~(D | D.T)
    changed = True
    while changed:
        changed = False
        for a, b in itertools.permutations(range(n), 2):
            if not U[a, b]:
                continue
            # R1: c->a, a-b, c not adj b  =>  a->b
            r1 = any(D[c, a] and not skel[c, b] for c in range(n) if c not in (a, b))
            # R2: a->c->b, a-b => a->b
            r2 = any(D[a, c] and D[c, b] for c in range(n) if c not in (a, b))
            # R3: a-c->b, a-d->b, c,d non-adjacent, a-b => a->b
            cs = [c for c in range(n) if c not in (a, b) and U[a, c] and D[c, b]]
            r3 = any(not skel[c, d] for c, d in itertools.combinations(cs, 2))
            if r1 or r2 or r3:
                D[a, b] = True
                U[a, b] = U[b, a] = False
                changed = True
    return D, U


def mec_members(A, max_undirected=14):
    """All DAGs in the Markov equivalence class of A. None if class is too large to enumerate."""
    D, U = cpdag_undirected(A)
    edges = [(i, j) for i in range(len(A)) for j in range(i + 1, len(A)) if U[i, j]]
    if len(edges) > max_undirected:
        return None
    target = _vstructs(A)
    members = []
    for bits in itertools.product([0, 1], repeat=len(edges)):
        M = D.astype(int).copy()
        for (i, j), b in zip(edges, bits):
            if b:
                M[i, j] = 1
            else:
                M[j, i] = 1
        if is_dag(M) and _vstructs(M) == target:
            members.append(M)
    return members


def three_way_target(members_or_A, n):
    """Soft 3-way target over pairs i<j (triu order): columns = [none, i->j, j->i]."""
    if isinstance(members_or_A, np.ndarray):
        members_or_A = [members_or_A]
    mean = np.mean(members_or_A, axis=0)  # mean adjacency over members
    iu = np.triu_indices(n, 1)
    fwd, bwd = mean[iu], mean.T[iu]
    return np.stack([1 - fwd - bwd, fwd, bwd], axis=1)


# ------------------------------------------------------------------ mixtures
def rewire(A, n_changes, rng, max_tries=200):
    """G2 from G1: each change either flips an edge (if still acyclic) or moves it to a
    previously non-adjacent pair. Returns A2 (acyclic)."""
    A2 = A.copy()
    n = len(A)
    changed = 0
    for _ in range(max_tries):
        if changed == n_changes:
            break
        edges = np.argwhere(A2 == 1)
        if len(edges) == 0:
            break
        i, j = edges[rng.integers(len(edges))]
        B = A2.copy()
        if rng.random() < 0.5:  # flip
            B[i, j], B[j, i] = 0, 1
        else:  # move: delete and add a new edge between non-adjacent nodes
            B[i, j] = 0
            cand = [(u, v) for u in range(n) for v in range(n)
                    if u != v and not (B[u, v] or B[v, u]) and (u, v) != (i, j) and (u, v) != (j, i)]
            if not cand:
                continue
            u, v = cand[rng.integers(len(cand))]
            B[u, v] = 1
        if is_dag(B):
            A2, changed = B, changed + 1
    return A2


def share_weights(A1, W1, A2, rng):
    """Weights for G2: reuse W1 where the same directed edge exists, fresh otherwise."""
    W2 = random_weights(A2, rng)
    same = (A1 == 1) & (A2 == 1)
    W2[same] = W1[same]
    return W2


def pair_status(A):
    """3-way one-hot over pairs i<j: [none, i->j, j->i]"""
    return three_way_target(A, len(A))


def sample_paper_like(A, rng, m):
    """Approximation of the DCDI-style generator described in the paper (App. B.1), NOT standardized:
    roots ~ U(-2,2); noise ~ 0.4*N(0, s2), s2~U(1,2) per node; weights +-U(0.5,2) (our assumption)."""
    n = len(A)
    W = random_weights(A, rng)
    sd = 0.4 * np.sqrt(rng.uniform(1, 2, n))
    X = np.zeros((m, n))
    for j in topo_order(A):
        if A[:, j].sum() == 0:
            X[:, j] = rng.uniform(-2, 2, m)
        else:
            X[:, j] = X @ W[:, j] + sd[j] * rng.standard_normal(m)
    return X


def sample_eqvar(A, W, rng, m, sig_frac=0.8):
    """Equal-variance linear-Gaussian SEM: every node has population variance 1 by construction, and every
    non-root has signal fraction `sig_frac` (so no varsortability; R2-sortability is also flattened).
    Relative weights W are kept; per-node scale is solved from the exact population covariance."""
    n = len(A)
    Sig = np.zeros((n, n))
    X = np.zeros((m, n))
    done = []
    for j in topo_order(A):
        pa = np.flatnonzero(A[:, j])
        if len(pa) == 0:
            X[:, j] = rng.standard_normal(m)
            Sig[j, j] = 1.0
        else:
            w = W[pa, j]
            v = w @ Sig[np.ix_(pa, pa)] @ w
            c = np.sqrt(sig_frac / v)
            X[:, j] = c * (X[:, pa] @ w) + np.sqrt(1 - sig_frac) * rng.standard_normal(m)
            cov = c * (Sig[done][:, pa] @ w) if done else np.array([])
            for q, k in enumerate(done):
                Sig[j, k] = Sig[k, j] = cov[q]
            Sig[j, j] = 1.0
        done.append(j)
    return X
