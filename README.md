# PatchICU

A patch-based transformer benchmark for ICU sepsis prediction on the HiRID-ICU-Benchmark.

This repository extends [YAIB](https://github.com/rvandewater/YAIB) (Yet Another ICU Benchmark) with [PatchTST](https://arxiv.org/abs/2211.14730) — a patch-based time-series Transformer not previously evaluated on the HiRID sepsis task — along with the configs, sweep infrastructure, and analysis scripts used to produce the results in [`REPORT.pdf`](REPORT.pdf).

The project has two purposes: (1) demonstrate that a state-of-the-art time-series architecture produces competitive sepsis-prediction performance on a public ICU benchmark, and (2) use that demonstrated foundation to scope two forward research directions (below) addressing gaps in the deployed sepsis-prediction literature.

## What's demonstrated

| Aspect | Result |
|---|---|
| Architecture | PatchTST encoder integrated as a YAIB-conformant DL model |
| Cohort | HiRID-ICU-Benchmark sepsis task — 29,642 stays, hourly resolution |
| Evaluation | 5 × 5 nested cross-validation, 30-trial Optuna GP tuning |
| Performance (HiRID, 30-trial) | **AUROC 0.940 [0.938, 0.942], AUPRC 0.312 [0.296, 0.329]** |
| Cross-site (preliminary, MIMIC-IV) | AUROC 0.955, AUPRC 0.297 — single random trial, untuned |
| Baselines outperformed | Logistic Regression, LSTM, GRU, TCN, Transformer, LightGBM |

The 30-trial PatchTST result more than doubles the AUPRC of the next-best baseline (LightGBM at 0.128). Methodology, tuning-budget asymmetry, and limitations are detailed in [`REPORT.pdf`](REPORT.pdf). A public [wandb report](https://api.wandb.ai/links/verity-hogans-university-of-utah/3rjnpwzn) presents per-trial sweep visualizations and training curves from the 30-trial PatchTST tuning.

## Status

| Component | Status |
|---|---|
| 1-hour HiRID benchmark (headline) | **Complete** |
| Cross-site validation (MIMIC-IV, eICU) | Preliminary; some runs pending |
| 2-minute native-resolution variant | Blocked on cohort-generation memory issues; not in this release |
| Empirical leakage audit (label permutation test) | Static review only; empirical test not yet run |
| Causal-attention variant (deployment-ready) | Not yet implemented |

## Forward research directions

The 1-hour result is the empirical foundation. Two parallel research directions extend it.

### Task 1 — Shortening time to targeted antibiotic intervention

Current sepsis early-warning systems (TREWS and most published work) produce a single binary sepsis flag. The clinically actionable next step — selecting *which* antibiotic — is gated on lab turnaround for organism identification and susceptibility testing. Empiric broad-spectrum coverage starts immediately on suspicion of sepsis; narrowing to targeted coverage waits 24–72+ hours for cultures, and is further complicated by the high false-negative rate of cultures drawn after empiric antibiotics begin.

Task 1 investigates whether a transformer-based model can predict *organism class* (or resistance profile) from clinical features prior to culture return, supporting earlier empiric→targeted narrowing. Likely cohort pivot to MIMIC-IV (detailed microbiology). Positions against direct-detection approaches (Inflammatix HostDx, Karius, T2 Biosystems) as a no-additional-assay alternative.

### Task 2 — Accuracy with a defined minimum feature set

HiRID provides ~50 dynamic variables at high temporal density — far more than most non-academic hospitals collect. A deployable sepsis EWS needs to retain signal at community-hospital feature density (vitals + basic CBC + BMP + lactate; ~15–20 features rather than 50).

Task 2 systematically evaluates *performance as a function of feature count* using feature selection, knowledge distillation, and pretrain-then-fine-tune approaches. Deliverable: an explicit performance/feature-count tradeoff curve and a defensible minimum-feature set with documented retained performance.

## Repository structure

```
patchicu/
├── README.md                                       # This file (replaces upstream YAIB README)
├── REPORT.pdf                                      # Empirical writeup
├── REPRODUCE.md                                    # Setup + reproduction instructions
├── cohort-modifications.patch                      # Bug-fix patch against rvandewater/YAIB-cohorts
├── figures/                                        # Result + EDA figures
├── icu_benchmarks/models/dl_models/patchtst.py     # YAIB-conformant PatchTST adapter
├── configs/prediction_models/
│   ├── PatchTST.gin                                # Default PatchTST config
│   └── PatchTST_1hr_tuned.gin                      # Frozen 30-trial best HPs (reproduces REPORT.pdf headline)
├── experiments/                                    # Sweep configs (institutional paths scrubbed)
├── requirements.txt                                # Pinned dependencies
└── ...                                             # Standard YAIB structure (icu_benchmarks/, scripts/, tests/, etc.)
```

This repository is a fork of [`rvandewater/YAIB`](https://github.com/rvandewater/YAIB). Please cite [van de Water et al., 2024](https://arxiv.org/abs/2306.05109) for the underlying benchmark framework. PatchICU adds the PatchTST integration and surrounding infrastructure; YAIB's core APIs are unchanged.

## Reproduction

See [`REPRODUCE.md`](REPRODUCE.md) for full setup, cohort regeneration, and pinned-dependency commits. In brief:

1. **Data access.** HiRID v1.1.1 requires PhysioNet credentialed access (DUA). Raw data is not redistributed.
2. **Cohort.** Generated against `rvandewater/YAIB-cohorts` at commit `74ac699` with a one-line bug-fix patch (see [`cohort-modifications.patch`](cohort-modifications.patch)).
3. **Training.** YAIB's standard `icu-benchmarks` CLI with the included `PatchTST_1hr_tuned.gin` config. Hardware: University of Utah CHPC NVIDIA GH200 partition.
4. **Pinned dependencies.** See [`requirements.txt`](requirements.txt) — extends YAIB upstream's pin list with `transformers==5.5.3` for the HuggingFace PatchTST port.

## Background and prior art

The most-cited deployed sepsis early-warning system in current literature is **TREWS** (Henry et al., *Nature Medicine* 2022), a mixture-of-Cox-proportional-hazards system deployed across five Johns Hopkins hospitals. The TREWS paper's primary contribution is a deep adoption analysis showing that provider engagement, not raw model performance, is the binding constraint at deployment scale.

PatchICU differs from TREWS along five axes relevant to evaluating this work:
- **Modeling paradigm:** patch-based transformer vs. mixture-of-Cox.
- **Setting:** ICU-only (HiRID) vs. whole-hospital deployment.
- **Reproducibility:** open code on a public benchmark vs. proprietary code on private EHR.
- **Forward direction:** organism-class targeting and minimum-feature deployability vs. continued binary sepsis flagging.
- **Stage:** research artifact vs. deployed CDS tool.

Direct performance comparisons between PatchICU and TREWS are *not* meaningful (different cohorts, different label phenotyping, different evaluation categories) and are deliberately not made in [`REPORT.pdf`](REPORT.pdf).

## About

This work was completed as a final project for BMI 6114 (Deep Learning in Biomedicine), University of Utah, with empirical collaboration from David Bean. Ongoing development is led by Josh McAlister (MSBA, David Eccles School of Business; pivoting into Biomedical Informatics). I am not currently pursuing a terminal degree — though that path remains open if the research warrants the commitment.

I am currently seeking a paid research home (RA, research staff, or equivalent) to continue this work in a Biomedical Informatics or affiliated lab. If you are a faculty member whose interests align with the directions above, I would welcome the conversation.

**Contact:** Josh McAlister · u1561737@utah.edu

## Citations

- **YAIB** — van de Water R, Schmidt H, Elbers P, Thoral P, Arnrich B, Rockenschaub P. *Yet Another ICU Benchmark: A Flexible Multi-Center Framework for Clinical ML.* ICLR 2024.
- **HiRID** — Faltys M, Zimmermann M, Lyu X, et al. *HiRID, a high time-resolution ICU dataset (v1.1.1).* PhysioNet, 2021. https://doi.org/10.13026/nkwc-js72
- **PatchTST** — Nie Y, Nguyen NH, Sinthong P, Kalagnanam J. *A time series is worth 64 words: long-term forecasting with Transformers.* ICLR 2023.

## License

This repository inherits its license from upstream YAIB (MIT). See [`LICENSE`](LICENSE).
