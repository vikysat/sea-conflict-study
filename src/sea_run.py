"""
Lean re-implementation of SEA's inference path so that we can
  (1) cache the expensive FCI marginal estimates,
  (2) perturb them (inject conflict) before the aggregator sees them,
  (3) read out the full 3-way softmax [none, i->j, j->i] per pair.

Sampling logic mirrors sea/src/data/samplers.py + dataset.py (TestDataset), with one hook added:
`draw_rows`, so a batch can come from a chosen mixture component.
"""
import importlib.util
import os
import sys

import numpy as np
import torch

if not hasattr(np, "mat"):  # causal-learn 0.1.3.7 (the repo's pinned version) predates NumPy 2
    np.mat = np.asmatrix

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
# the original SEA code + checkpoints are NOT redistributed here: clone https://github.com/rmwu/sea into external/sea (or set SEA_REPO)
SEA = os.environ.get("SEA_REPO", os.path.join(ROOT, "external", "sea"))
CKPT = os.path.join(SEA, "checkpoints", "fci_synthetic", "model_best_epoch=373_auprc=0.842.ckpt")
sys.path.insert(0, os.path.join(SEA, "src"))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# repo helpers, imported by path to avoid the package __init__ that pulls in Lightning
_utils = _load("sea_data_utils", os.path.join(SEA, "src/data/utils.py"))
_axial = _load("sea_axial", os.path.join(SEA, "src/model/axial.py"))
run_fci, convert_to_graphs = _utils.run_fci, _utils.convert_to_graphs

# token semantics for ordered pair (i, j), from edge_map_fci (marks stored as G+2):
#   1 none | 2 i->j | 3 j->i | 4 i<->j | 5 i<-o j | 6 i o-> j | 7 i o-o j
NONE, FWD, BWD, BI, CIRC_BWD, CIRC_FWD, CIRC = 1, 2, 3, 4, 5, 6, 7
PARTNER = {1: 1, 2: 3, 3: 2, 4: 4, 5: 6, 6: 5, 7: 7}  # token of (j, i) given token of (i, j)


# ------------------------------------------------------------------ model
def load_aggregator(ckpt_path, device="cpu", config="aggregator_tf_fci.yaml"):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = ck["hyper_parameters"]["args"]
    # the YAML config (inner-most keys) is what the checkpoint was actually trained with
    import yaml
    cfg = yaml.safe_load(open(os.path.join(SEA, "config", config)))
    for sect in cfg.values():
        for key, val in sect.items():
            if hasattr(args, key):
                setattr(args, key, val)
    enc = _axial.AxialTransformer(args)
    top = _axial.TopLayer(embed_dim=args.embed_dim * 2, output_dim=3)
    sd = ck["state_dict"]
    enc.load_state_dict({k[8:]: v for k, v in sd.items() if k.startswith("encoder.")})
    top.load_state_dict({k[10:]: v for k, v in sd.items() if k.startswith("top_layer.")})
    return enc.to(device).eval(), top.to(device).eval(), args


# ----------------------------------------------------- sample + estimate
def precision(x):
    """Global statistic, same as sea compute_features for N<100.  x: (rows, N)"""
    return np.linalg.pinv(np.cov(x.T), rcond=1e-10)


def _sample_nodes(scores, visit, k, rng):
    joint = scores / np.sqrt(1 + visit)
    sampled = []
    for _ in range(k):
        if not sampled:
            p = joint.sum(0)
        else:
            p = joint[sampled].copy()
            p[:, sampled] = 0
            p = p.sum(0)
        p = p / p.sum()
        sampled.append(int(rng.choice(len(p), p=p)))
    visit[np.ix_(sampled, sampled)] = visit[np.ix_(sampled, sampled)] + 1
    return np.array(sampled)


def sample_and_estimate(draw_rows, n, rng, T=500, b=500, k=5, rho_rows=None):
    """
    draw_rows(rng, b) -> (rows[b, n], source_id)   source_id tags the mixture component of the batch.
    Half of the batches use random node scores, half |precision| scores (as in the repo).
    Returns a cache dict (everything needed to re-run the aggregator on possibly perturbed tokens).
    """
    visit = np.zeros((n, n))
    results, nodes_all, src = [], [], []
    last_rows = None
    for t in range(T):
        if t < T // 2:
            scores = np.ones((n, n)) - np.eye(n)
        else:
            sc_rows, _ = draw_rows(rng, b)
            scores = np.abs(precision(sc_rows)) * (1 - np.eye(n))
        nodes = _sample_nodes(scores, visit, k, rng)
        rows, s = draw_rows(rng, b)
        if t == T // 2 - 1:
            last_rows = rows  # sea computes rho from the last batch of the first sampler
        G = run_fci(rows[:, nodes])
        if G is None:
            continue
        results.append((G, torch.from_numpy(nodes).long()))
        nodes_all.append(nodes)
        src.append(s)
    graphs, orders = convert_to_graphs(results, _FakeDS)
    if rho_rows is None:
        rho_rows = last_rows
    return dict(tokens=graphs.numpy(), nodes=np.array(nodes_all), src=np.array(src),
                rho=precision(rho_rows), n=n, k=k)


class _FakeDS:
    algorithm = "fci"


def standard_draw(X):
    def draw(rng, b):
        return X[rng.choice(len(X), b, replace=False)], 0
    return draw


def batch_mixture_draw(Xs, weights):
    """Each batch comes entirely from one component (genuine disagreement across subsets)."""
    w = np.asarray(weights, float) / np.sum(weights)

    def draw(rng, b):
        c = int(rng.choice(len(Xs), p=w))
        return Xs[c][rng.choice(len(Xs[c]), b, replace=False)], c
    return draw


def pooled_rho_rows(Xs, weights, rng, b=500):
    w = np.asarray(weights, float) / np.sum(weights)
    comp = rng.choice(len(Xs), size=b, p=w)
    return np.concatenate([Xs[c][rng.choice(len(Xs[c]), (comp == c).sum(), replace=False)]
                           for c in range(len(Xs)) if (comp == c).any()])


# --------------------------------------------------------------- aggregator
def to_batch(cache, tokens=None):
    n, k = cache["n"], cache["k"]
    tokens = cache["tokens"] if tokens is None else tokens
    nodes = torch.from_numpy(cache["nodes"]).long()
    # same pair order as convert_to_graphs: row-major over (a, b), a != b
    orders = torch.cat([torch.cartesian_prod(o, o) for o in nodes], 0)
    orders = orders[orders[:, 0] != orders[:, 1]]
    unique, inverse = torch.unique(orders, dim=0, return_inverse=True)
    return {
        "input": torch.from_numpy(tokens).long()[None],
        "index": inverse.reshape(len(nodes), -1)[None],
        "feats": torch.from_numpy(cache["rho"]).float()[None],
        "unique": (unique + 1)[None],
    }


@torch.no_grad()
def predict(enc, top, cache, tokens=None, seed=None):
    """Return P[npairs, 3] over pairs i<j in triu order; columns [none, i->j, j->i]."""
    if seed is not None:
        torch.manual_seed(seed)
    out = enc(to_batch(cache, tokens))[0]  # (N, N, d)
    iu = torch.triu_indices(cache["n"], cache["n"], 1)
    logits = top(torch.cat([out[iu[0], iu[1]], out[iu[1], iu[0]]], -1))
    return torch.softmax(logits, -1).numpy()


# ------------------------------------------------------- conflict injection
def _slot(k, a, b):
    """column of ordered local pair (a, b), a != b, in the (k*(k-1)) token vector"""
    return a * (k - 1) + (b if b < a else b - 1)


def subsets_with_pair(cache, i, j):
    return np.flatnonzero([(i in nd) and (j in nd) for nd in cache["nodes"]])


def vote_fraction(cache, i, j, tokens=None):
    """fraction of subsets containing {i,j} in which FCI reports *any* edge between them"""
    tokens = cache["tokens"] if tokens is None else tokens
    ts = subsets_with_pair(cache, i, j)
    if len(ts) == 0:
        return np.nan
    k, hits = cache["k"], 0
    for t in ts:
        a, b = int(np.flatnonzero(cache["nodes"][t] == i)[0]), int(np.flatnonzero(cache["nodes"][t] == j)[0])
        hits += tokens[t, _slot(k, a, b)] != NONE
    return hits / len(ts)


def inject(cache, i, j, frac, op, rng):
    """
    Corrupt a fraction `frac` of the subsets containing {i, j}. Returns a new token array.
      op = 'delete'   edge removed (token -> none)            [tests the any-absence 'veto']
           'flip'     direction reversed (2<->3, 5<->6)       [orientation conflict]
           'uninform' orientation hidden (token -> o-o)
           'add'      spurious edge (token -> o-o)             [false-positive conflict]
    """
    tokens = cache["tokens"].copy()
    ts = subsets_with_pair(cache, i, j)
    chosen = rng.choice(ts, int(round(frac * len(ts))), replace=False) if len(ts) else []
    k = cache["k"]
    for t in chosen:
        a, b = int(np.flatnonzero(cache["nodes"][t] == i)[0]), int(np.flatnonzero(cache["nodes"][t] == j)[0])
        s_ab, s_ba = _slot(k, a, b), _slot(k, b, a)
        tok = int(tokens[t, s_ab])
        new = {"delete": NONE, "uninform": CIRC, "add": CIRC,
               "flip": {FWD: BWD, BWD: FWD, CIRC_FWD: CIRC_BWD, CIRC_BWD: CIRC_FWD}.get(tok, tok)}[op]
        if op == "add" and tok != NONE:
            continue
        if op == "uninform" and tok == NONE:
            continue
        tokens[t, s_ab], tokens[t, s_ba] = new, PARTNER[new]
    return tokens
