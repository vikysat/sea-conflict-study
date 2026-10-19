# Does SEA surface calibrated uncertainty when subset estimates conflict?

A small synthetic case study of **SEA** (*Sample, Estimate, Aggregate*, Wu, Bao, Barzilay & Jaakkola, TMLR 2025;
[paper](https://arxiv.org/abs/2402.01929), [code](https://github.com/rmwu/sea)).

**Question.** SEA runs a classical algorithm (FCI) on many small subsets of variables and trains a network to aggregate the
subset graphs plus a global statistic (inverse covariance, ρ) into one causal graph. The paper never evaluates the case where
the subset estimates *genuinely conflict* (rather than being noisy-but-consistent), and its Discussion lists resolving
conflicting views as unexplored and "erring on the side of sparsity" as a limitation. Does SEA report calibrated per-edge
uncertainty there, or does it default to sparsity?

> **Status: small exploratory study, not peer reviewed.** N=10 variables, linear-Gaussian data, Fisher-z FCI, FCI checkpoint only,
> one seed per condition, ~20–25 test pairs per cell. Nothing here shows SEA fails on real biological data.

## Findings

| # | Experiment | What we did | Result |
|---|---|---|---|
| R0 | Irreducible ambiguity | Standardized/equal-variance data where orientation is only identifiable up to the Markov equivalence class (MEC). Exact MEC posterior computed by enumeration (`tests/test_mec.py` checks it against brute force on 150 graphs). | At a ρ-scale close to training, direction probabilities are close to the exact posterior (predicted 0.8–1.0 → true 0.82). Calibration degrades off-scale: at ρ 2.3× smaller, 28% of ambiguous edges get >0.9 confidence (true posterior never does); at the 4.5× larger standardized scale SEA puts 0.32 average edge mass on non-adjacent pairs. |
| R1 | Dose-response ("veto" test) | Delete / flip / add an edge in a controlled fraction of the subsets that contain it. | **No veto.** P(edge) falls smoothly and late (vote fraction 0.49 → 0.83, 0.29 → 0.54, 0.00 → 0.39). Spurious edges asserted in 100% of subsets still get only 0.19. Flipping one pair's orientation in 100% of its subsets moves P(true direction) only 0.95 → 0.92. Resistance to single-pair corruption, but no uncertainty flag: with every subset denying a true edge, 44% of pairs collapse <0.2, 32% sit in between, 24% stay >0.8. |
| R3 | Cue conflict | FCI estimates from one graph, ρ from another. | Neither cue dominates. Pairs supported only by ρ: mean P(edge) 0.46; only by the estimates: 0.31; both: 0.97; neither: 0.09. |
| R2 | Genuine heterogeneity | Two graphs differing by 2 rewired edges; each batch is drawn entirely from one of them. Target for an edge present in one graph = that graph's share of batches. | **Edge inflation, not sparsity.** At a 50/50 split the target is 0.5 for every conflicting edge; SEA gives 0.72 / 0.62 (edge only in G1 / only in G2) vs 0.56 / 0.52 for the plain vote fraction. Only 29% of conflict pairs get an intermediate probability (52% >0.8, 19% <0.2). SEA's uncertainty barely grows with conflict, and detects conflict pairs worse than a vote-disagreement score (AUROC 0.75 vs 0.94 at 50/50). |
| Fix | Temperature scaling; ρ-scale normalization | Fit on held-out graphs only. | Neither fixes R2. Temperature (fitted T=0.75) sharpens outputs and worsens overconfidence. ρ-normalization repairs the low-scale regime (compelled-edge accuracy 0.74 → 0.85) but hurts on standardized data (0.72 → 0.58). The failure is structural rather than a global scale/temperature problem. |

**Other observations**
- *Shortcut on raw data.* On the paper's usual (unstandardized) data SEA scores **above** the exact MEC oracle (orientation accuracy 0.87–0.90 vs 0.77), i.e. it uses information beyond the equivalence class (variance order; cf. Reisach et al. 2021). Calibration tests therefore need standardized or equal-variance data.
- *Scale sensitivity.* Rescaling ρ by a constant leaves edge ranking intact (AUC ≈ 0.99) but swings orientation accuracy 58% → 81% and false-edge mass 0.10 → 0.34.
- *Under-use of identifiable information.* On compelled (identifiable) edges, SEA orients 80% correctly vs 93% for a trivial arrowhead vote on the same edges (E=10, 25 graphs). **Across all 40 graphs including E=20 the gap is smaller: 80% vs 86%.**
- *Errors are systematic, not noise.* Aggregator permutation noise sd 0.010, FCI resampling noise sd 0.035; a 6-run ensemble does not reduce distance to the exact posterior.
- *Replication.* Raw-data AP 0.95 (25 graphs) vs the paper's 0.96–0.98; standardized orientation accuracy 0.62–0.65 / AUC 0.93 vs the paper's Table 6 0.69 / 0.97.

**Main assumption (R2).** The "correct" answer is defined as population frequency across batches, each component at its own MEC
posterior. The vote fraction independently agrees with it, but someone who defines the truth as the *union* graph would
consider the edge inflation reasonable.

## Layout
```
src/gen.py          data generators (equal-variance SEM, mixtures, rewiring) + exact MEC enumeration (Meek rules)
src/sea_run.py      lean SEA inference path: sampling, cached FCI estimates, token injection, full 3-way softmax
src/helpers.py      per-pair vote statistics
experiments/        scripts that produce results/*.pkl (diag_std, r0_run, r0v2_run, r1_run, r2_run, r3_run, fix_scale)
analysis/           scripts that print every number quoted above
results/            cached outputs (pickles) so the analysis scripts run without re-running experiments
tests/test_mec.py   sanity test of the exact MEC posterior
run_all.sh          full reproduction
```

## Reproduce
```bash
git clone https://github.com/rmwu/sea external/sea      # original SEA code + checkpoints (not redistributed here)
pip install -r requirements.txt
python tests/test_mec.py                                # exact-posterior sanity test (no SEA checkout needed)
python analysis/r2_detail.py                            # analysis scripts read the cached results/ (most import src/sea_run.py, so they need the SEA checkout)
./run_all.sh                                            # everything from scratch, ~1 h on one CPU core
```
Run all commands from the repository root. Set `SEA_REPO=/path/to/sea` if your checkout is elsewhere.
Inference is cheap (500 FCI runs ≈ 2.4 s, one aggregator forward pass ≈ 1.4 s on CPU); no GPU is needed.

## Implementation notes
- The SEA aggregator is loaded **without PyTorch Lightning** (weights mapped straight from the checkpoint), and the YAML config is applied over the checkpoint's saved args (they disagree on the FFN width).
- The aggregator's positional-embedding permutation is active at inference, so outputs are stochastic for fixed inputs; seeds are fixed throughout.
- `causal-learn` is pinned to 0.1.3.7 (see `requirements.txt`); a NumPy-2 shim (`np.mat`) is applied in `sea_run.py`.
- `results/*.pkl` are NumPy pickles generated by these scripts; do not load pickles from untrusted sources.
- Training-like ρ scale reference (mean diagonal ≈ 10.3) comes from our own re-implementation of the paper's raw-data generator, not from their data. The equal-variance generator's signal fraction (0.9) was chosen to match that scale before looking at any accuracy results; 0.8 was run as a pre-declared sensitivity check.

## AI assistance
Code and analysis were developed with the assistance of Claude (Anthropic); all results were reproduced from the committed scripts.

## Credits
Method and pretrained models: Wu, Bao, Barzilay, Jaakkola, *Sample, estimate, aggregate: A recipe for causal discovery foundation models* (TMLR, 2025).
This repository contains none of their code or weights.
