#!/usr/bin/env bash
# Reproduce everything from scratch (single CPU core: roughly 45-60 min total). Run from the repo root.
# Prereq:  git clone https://github.com/rmwu/sea external/sea     (or: export SEA_REPO=/path/to/sea)
set -e
python experiments/diag_std.py      # 2x2 diagnostic: root distribution x standardization   (~6 min)
python experiments/r0_run.py        # R0 on standardized data, 60 graphs                     (~11 min)
python experiments/r0v2_run.py      # R0 at matched rho-scale (equal-variance generator)     (~5 min)
python experiments/r1_run.py        # R1 dose-response ("veto") test                         (~10 min)
python experiments/r2_run.py        # R2 heterogeneous batches (genuine conflict)            (~6 min)
python experiments/r3_run.py        # R3 rho vs marginal-estimate cue conflict               (~1 min)
python experiments/fix_scale.py     # forward passes for the recalibration experiments       (~2 min)
for f in diag_std_summary rho_scale_sweep analysis_r0 analysis_r0v2 r1_summary r2_detail analysis_r2 fix_eval; do
  echo "######## $f"; python analysis/$f.py
done
