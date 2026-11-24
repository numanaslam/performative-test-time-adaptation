# Performative Feedback in Test-Time Adaptation (PATTA)

Reference implementation and manuscript for:

> **Performative Feedback in Test-Time Adaptation: Stability Transitions, Hidden Collapse, and the Limits of Class-Balance Regularization**
> Numan Aslam, Ghulam Mustafa, Zia Ul Rehman, Muhammad Adnan Khan, Arselan Ashraf, Muhammad Amir Khan.
> Submitted to *Computers, Materials & Continua* (CMC), Tech Science Press.

## Overview

In online test-time adaptation (TTA) the adapting model **induces** the distribution it next
adapts to: its predictions decide which samples are confident, which pseudo-labels are emitted,
and (in active or embodied vision) which inputs arrive next. This repository makes that loop
explicit and studies it as a **performative-prediction** process.

Key findings:

- **A stability transition.** As the feedback strength `ε` grows (or the batch shrinks), balanced
  accuracy crosses an operational degradation boundary into collapse; a tractable 1-D model gives an
  exact local threshold `ε*` and the direction of a class-balance correction.
- **A hidden metric.** Accuracy measured on the model-influenced stream can stay high while
  *balanced* accuracy on a fixed class-uniform set collapses — e.g. at 1000-class ImageNet-C scale,
  stream accuracy reads ~77% while balanced accuracy is ~4%.
- **PATTA (Performativity-Aware TTA).** An adaptive marginal-entropy term, derived from the stability
  analysis, that recovers balanced accuracy at CIFAR scale, is inert when no collapse is present, but
  does **not** restore per-class accuracy at ImageNet scale (a limit we report honestly).
- **Controls.** Frozen-model, oracle-label, and batch-normalization-only controls locate the collapse
  in prediction-dependent gradient updates rather than in prior concentration or BN-statistics drift.

## Repository layout

```
perftta/          core library: adapters (TENT, EATA, SAR, SCALAR, PATTA, …),
                  performative feedback map, metrics, tractable model
experiments/      one experiment per file (bifurcation grid, baselines, scale,
                  feedback controls, trajectories) + per-experiment summarizers
scripts/          PowerShell/-shell runners that reproduce each table and figure
requirements.txt  Python dependencies
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # or conda
pip install -r requirements.txt
```

Datasets and checkpoints are **not** stored in this repository (see `.gitignore`). The experiments
expect the standard corruption benchmarks (CIFAR-10-C, CIFAR-100-C, ImageNet-C) and the source
checkpoints under a local data root; paths are set in the run scripts. The study was run on a single
RTX 5070 (12 GB).

## Reproducing the results

Each table/figure has a dedicated runner in `scripts/` and a summarizer in `experiments/`. For
example:

```bash
# Bifurcation grid (Table 1 / Figure 2)
python -m experiments.exp_bifurcation --help
# ImageNet-C scale study (Table 5)
python -m experiments.exp_scale --help
# Feedback-generator controls (Table 6)
python -m experiments.exp_fbcontrol --help
# Per-step trajectories + λ_t distribution (Figure 4)
python -m experiments.exp_trajectory --help
```

The `*.ps1` scripts under `scripts/` wrap these with the exact sweeps, seeds, and output paths used
in the paper, and are resumable.

## The tractable model

`experiments/verify_tractable.py` reproduces the closed-form constants of the 1-D tractable model
(the exact threshold `ε*` and the fixed-weight threshold shift). The full derivation and the
numerical subcriticality check are in the paper's supplementary material.

## Citation

A DOI will be added on publication. In the meantime please cite the manuscript:

```bibtex
@article{aslam2026performative,
  title   = {Performative Feedback in Test-Time Adaptation: Stability Transitions,
             Hidden Collapse, and the Limits of Class-Balance Regularization},
  author  = {Aslam, Numan and Mustafa, Ghulam and Rehman, Zia Ul and
             Khan, Muhammad Adnan and Ashraf, Arselan and Khan, Muhammad Amir},
  journal = {Computers, Materials \& Continua},
  year    = {2026},
  note    = {Under review}
}
```

## License

To be added by the authors (e.g. MIT for the code). Until then, all rights reserved.
