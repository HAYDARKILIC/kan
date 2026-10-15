# Artifact for *Where the Dimension Hides*

Code, run logs and analysis outputs for the manuscript

> **Where the Dimension Hides: The Approximation Constant of Kolmogorov–Arnold Networks**
> Haydar Kılıç, Ostim Technical University — [ORCID 0000-0002-2551-3772](https://orcid.org/0000-0002-2551-3772)

This repository contains **no manuscript**. It exists so that a reviewer can check any number in
the paper against the code and data that produced it.

---

## Start here: verifying a claim in five minutes

Every number in the paper derives from `results/runs_raw.csv`, a 33,300-row log of the
pre-registered study, or from one of the three follow-up logs. Nothing needs a GPU to check.

```bash
pip install -r requirements.txt
jupyter lab notebooks/KAN_paper_pipeline.ipynb     # leave ANALYSIS_ONLY = True, run all
```

That reproduces every table and figure of the pre-registered study on a CPU in about five
minutes, including the closed-form probes, the controls, the design-power analysis, both
parametrisations of the joint regression, and the machine-generated verdict.

For a single quantity without opening a notebook:

```bash
python scripts/corrected_joint_fit.py --runs results/runs_raw.csv --out ./check --n-boot 2000
```

---

## Where each claim comes from

| Manuscript | Claim | Artifact |
|---|---|---|
| §4.2, Table 1 | Bedrock rate −3.11 / −4.09 / −4.26 | `results/tables/bedrock_rate.csv`; notebook §5 |
| §4.2, Table 2 | Smoothness saturation, 18/18 cells | `results/tables/smoothness_saturation.csv`; notebook §6 |
| §4.2, Table 3 | Precision floor, window spread 0.000 | `results/tables/floor_robustness.csv`; notebook §6 |
| §4.3, Table 4 | Design power, VIF 10.7 → 6.8 | `results/tables/design_power*.csv`; notebook §7 |
| §4.4, Table 5 | Joint fit, asymptotic parametrisation | `results/tables/scaling_joint_fit*.csv`; notebook §9 |
| §4.4, Table 7 | **Exact parametrisation**, oracle passes at all orders | `results/tables/corrected_joint_fit.csv`; `scripts/corrected_joint_fit.py` |
| §4.4 | Nested test, F ≤ 0.03 for the oracle | `results/tables/corrected_nested_test.csv` |
| §4.5, Table 9 | Dimensional slope at fixed resolution, 0.00–0.02 | notebook §10; derived from `runs_raw.csv` |
| §4.5, Table 10 | Capacity-to-sample ratios | notebook §10 |
| §4.6, Table 11 | Sandwich, 135/135 | `results/tables/sandwich.csv` |
| §4.7, Table 12 | Absorption gap, ρ = +0.136 | `results/tables/p2_absorption.csv`, `p2_trend_test.csv` |
| §4.8, Table 13 | Critical dimension, 29/35 finite | `results/tables/critical_dimension.csv` |
| §4.9, Table 15 | KAN vs MLP, 186/210 significant | `results/tables/significance.csv` |
| §4.10, Table 16 | C5 on real data, 6/6 positive | `results/completion/real_full.csv` |
| §4.10, Table 22 | Five definitions of an appended edge | `results/optional/D1_duplicate_variants_full.csv` |
| §4.11, Table 17 | Capacity-decoupled sweep | `results/followup/E1_capacity_decoupled_full.csv` |
| §4.11, Table 18 | P2 re-specified | `results/followup/E3_p2_respecified_full.csv` |
| §4.11, Table 19 | WidthKAN, depth-one | `results/completion/E4_widthkan_full.csv` |
| §4.11, Table 20 | Certification, warm start 8/9 | `results/optional/D4_certificate_full.csv` |
| §4.11, Table 21 | Width decoupled from dimension | `results/optional/D2_width_decoupled_full.csv` |
| §4.11 | Integrity checks, 4/4 | `results/completion/integrity_recomputed.json` |
| §4.12, Table 23 | Pre-registered verdict | `results/verdict.txt` |
| §3.5, App. C | Outer design matrix numerically singular in 66/75 cells | column `cond_final` of `results/optional/../E2_alternating_solve_full.csv` |

The reported pre-registered run carries the manifest digest **`4e5b8b0777fa9959`**
(`results/manifest.json`), which records the environment, the configuration and the threshold
constants exactly as executed.

---

## Layout

```
notebooks/
  KAN_paper_pipeline.ipynb        the pre-registered study            §3–4
  KAN_followup_experiments.ipynb  capacity decoupling, P2 re-spec     §4.11
  KAN_completion.ipynb            width probe, real data, integrity   §4.10, §4.11
  KAN_optional.ipynb              certification, edge definitions,
                                  width decoupling, replication       §4.10–§5.2
scripts/
  corrected_joint_fit.py          joint fit, both parametrisations (CLI)
  corrected_joint_fit_colab.py    the same, as a notebook cell
  phase_control.py                phase-randomisation control
  reanalysis_abscissa.py          the abscissa re-analysis of §5.2
  regen_fig_p3.py                 regenerates Figure 4
results/
  runs_raw.csv                    the pre-registered run, 33,300 cells
  manifest.json  verdict.txt  integrity.json
  tables/                         every analysis table as CSV
  figures/                        every figure, vector PDF and 300 dpi PNG
  followup/  completion/  optional/    the three post-registration runs
```

---

## Reproducing from scratch

Only the training sweeps need an accelerator. All analysis re-derives on a CPU.

| Notebook | Setting | Hardware | Time |
|---|---|---|---|
| `KAN_paper_pipeline` | `ANALYSIS_ONLY = True` | CPU | ~5 min |
| `KAN_paper_pipeline` | `ANALYSIS_ONLY = False` | A100 | days |
| `KAN_followup_experiments` | `SCALE = "full"` | A100 | ~5 h |
| `KAN_completion` | `SCALE = "full"` | A100 | ~3 h |
| `KAN_optional` | `SCALE = "full"` | A100 | ~1 h |

Every sweep is resumable: completed cells are skipped when a notebook is re-run, and each
notebook writes to persistent storage or refuses to start.

Set `SCALE = "smoke"` in any of the three follow-up notebooks to exercise every code path on a
CPU in minutes. That will not reproduce the reported numbers, but it verifies that the code
runs and that the analysis paths are exercised.

---

## Points a reviewer may want to probe directly

These are the places where the paper's own instrumentation contradicted an earlier position of
ours. Each is checkable from the artifacts above.

1. **The abscissa correction.** The pre-registered regression estimates an exponent with
   respect to `G + k + 1`, not `G`. `scripts/reanalysis_abscissa.py` shows the difference on
   the real cells; `results/tables/corrected_joint_fit.csv` gives both parametrisations with
   bootstrap intervals.
2. **The withdrawn diagnosis.** The paper originally attributed the two-layer probe's behaviour
   to estimation error. `results/followup/E1_capacity_decoupled_full.csv` refutes that: the
   generalisation gap collapses from 32× to 2.2× while the dimensional slope persists on the
   *training* split.
3. **The withdrawn control.** `results/optional/D1_duplicate_variants_full.csv` shows that the
   duplicate-edge arm reproduces the predicted trend only when the duplicate is row-permuted,
   which destroys the signal it was supposed to preserve.
4. **The certificate.** `results/optional/D4_certificate_full.csv` shows that only a
   warm-started alternating solve certifies, and that AdamW misses the attainable error by at
   most 0.0088 — which is why optimisation is excluded as the mechanism.

---

## Requirements

Python 3.10+ and the packages in `requirements.txt`. The analysis path needs only NumPy,
pandas, SciPy, PyTorch and Matplotlib.

The five UCI datasets used by the real-data controls are downloaded on first use via
`ucimlrepo` and cached; they are not redistributed here. If the machine has no network access,
place `<name>.csv` in the cache directory (features first, target last) and the loader will use
it.

## Licence

See `LICENSE`.
