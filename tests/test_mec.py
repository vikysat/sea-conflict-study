"""Sanity tests for the exact Markov-equivalence-class enumeration used as the calibration target.
Run from the repo root:  python tests/test_mec.py"""
import itertools, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np
import gen

def brute_force_members(A):
    """all DAGs with the same skeleton and v-structures, by trying every orientation of the skeleton"""
    n = len(A); skel = (A | A.T); edges = [(i, j) for i in range(n) for j in range(i + 1, n) if skel[i, j]]
    target = gen._vstructs(A); out = []
    for bits in itertools.product([0, 1], repeat=len(edges)):
        M = np.zeros_like(A)
        for (i, j), b in zip(edges, bits):
            if b: M[i, j] = 1
            else: M[j, i] = 1
        if gen.is_dag(M) and gen._vstructs(M) == target: out.append(M)
    return out

# known answers: chain a->b->c has 3 members and P(a->b)=1/3, P(b->a)=2/3; a collider has exactly 1
chain = np.zeros((3, 3), int); chain[0, 1] = chain[1, 2] = 1
m = gen.mec_members(chain); assert len(m) == 3
t = gen.three_way_target(m, 3); assert np.allclose(t[0], [0, 1 / 3, 2 / 3]) and np.allclose(t[1], [1, 0, 0])
coll = np.zeros((3, 3), int); coll[0, 1] = coll[2, 1] = 1; assert len(gen.mec_members(coll)) == 1

# fast (Meek-rule) enumeration == full brute force on random graphs
rng = np.random.default_rng(1); bad = 0
for _ in range(150):
    A = gen.random_er_dag(8, int(rng.integers(5, 11)), rng)
    bad += sorted(x.tobytes() for x in gen.mec_members(A)) != sorted(x.tobytes() for x in brute_force_members(A))
assert bad == 0, bad
print("OK: MEC enumeration matches brute force on 150 random graphs")
