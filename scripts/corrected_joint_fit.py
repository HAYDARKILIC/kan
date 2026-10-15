"""
Corrected joint edge-law fit  (Cell 13 replacement / standalone re-analysis)
===========================================================================

WHY THIS EXISTS
---------------
The edge law is derived by substituting  G ~ N/E  into the spline rate bound. The exact
parameter accounting (Lemma 1) is

        N = E (G + k + 1)      =>      G = N/E - (k+1),

so  G ~ N/E  is a LARGE-G approximation. In the full-mode design G ranges over 4..64 while
k+1 ranges over 3..5, and 33-56% of probe cells (depending on k) have G < 4(k+1). Over those
cells the approximation is not defensible, and the two-regressor fit

        log eps = a + c_N log N + c_E log E                                   (A)

therefore estimates an exponent with respect to (G+k+1), not with respect to G. That is the
quantity the spline rate theorem is stated in, so (A) is biased away from -(k+1) at the
small-G end. This script refits the SAME cells with the exact per-edge resolution

        log eps = a + c_G log G_exact,     G_exact = N/E - (k+1)              (B)

and reports (A) and (B) side by side, with the paper's full uncertainty machinery:
saturation filter, cell-level bootstrap, seed-level cluster bootstrap, VIF diagnostics, and
a saturation-threshold sensitivity sweep.

It also runs a nested-model check: the edge law asserts that error depends on (N,E) ONLY
through G. If adding log E as a separate regressor on top of log G_exact does not improve
the fit, that functional form is confirmed; if it does, the law is incomplete.

NO MODEL IS RETRAINED. This reads the existing checkpoint log only.

HOW TO RUN
----------
    python3 corrected_joint_fit.py --runs /path/to/outputs/runs_raw.csv --out ./corrected

  or, straight from the checkpoint log:

    python3 corrected_joint_fit.py --runs /path/to/checkpoints/runs.jsonl --out ./corrected

Requires only numpy and pandas. Runtime is dominated by the bootstraps; see --n-boot.

OUTPUTS (written to --out)
    corrected_joint_fit.csv          per (family, k): coefficients (A) and (B) with CIs
    corrected_saturation_sens.csv    exponents across the saturation grid
    corrected_nested_test.csv        does log E add anything beyond log G_exact?
    corrected_joint_fit.tex          booktabs table, \\input-able
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd

# Pre-registered constants (Table 3 of the manuscript) -- do not edit.
SAT_DEFAULT = 0.6
SAT_GRID = [0.5, 0.6, 0.7]
MIN_FIT_POINTS = 4
TOL_RATE = 0.15          # bedrock_rate_rel_tol, applied to the rate exponent
TOL_CE = 0.25            # cE_rel_tol, applied to c_E in parametrisation (A)
VIF_MAX = 10.0

PROBE_KINDS = ("edge_kan", "kanet", "best_spline")
SYNTH_TARGETS = ("additive", "kastruct", "kastruct_np")


# ----------------------------------------------------------------------------- utilities
def load_runs(path):
    if path.endswith(".jsonl"):
        rows = [json.loads(l) for l in open(path) if l.strip()]
        df = pd.DataFrame(rows)
    else:
        df = pd.read_csv(path, low_memory=False)
    need = {"kind", "target", "k", "d", "grid", "params", "edges", "seed", "rel_l2"}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"runs file is missing columns: {sorted(missing)}")
    return df[df.kind.isin(PROBE_KINDS)].copy()


def ols(X, y):
    return np.linalg.lstsq(X, y, rcond=None)[0]


def vif(a, b):
    B = np.column_stack([np.ones(len(b)), b])
    r = a - B @ ols(B, a)
    ss = ((a - a.mean()) ** 2).sum()
    r2 = 1 - (r ** 2).sum() / ss if ss > 0 else 0.0
    return float(1.0 / max(1e-9, 1 - r2))


def design(sub, k):
    """Return (N, E, G_exact, y) for a filtered cell frame."""
    N = sub["params"].values.astype(float)
    E = sub["edges"].values.astype(float)
    Gx = N / E - (k + 1)
    return N, E, Gx, np.log(sub["err"].values)


def fit_pair(sub, k):
    """Fit parametrisation (A) and (B) on one filtered cell frame."""
    N, E, Gx, y = design(sub, k)
    ok = Gx > 0
    out = {}
    XA = np.column_stack([np.ones(len(N)), np.log(N), np.log(E)])
    cA = ols(XA, y)
    out["cN"], out["cE"] = float(cA[1]), float(cA[2])
    out["vif_N"] = vif(np.log(N), np.log(E))
    out["cond"] = float(np.linalg.cond(XA))
    if ok.sum() >= MIN_FIT_POINTS:
        XB = np.column_stack([np.ones(ok.sum()), np.log(Gx[ok])])
        out["cG"] = float(ols(XB, y[ok])[1])
        out["n_B"] = int(ok.sum())
    else:
        out["cG"], out["n_B"] = np.nan, int(ok.sum())
    return out


def aggregate(raw, sat):
    ag = (raw.groupby(["d", "grid", "params", "edges"], dropna=False)["rel_l2"]
             .mean().reset_index().rename(columns={"rel_l2": "err"}))
    return ag[(ag.err < sat) & (ag.err > 0) & (ag.edges > 0) & (ag.params > 0)]


# ------------------------------------------------------------------------------- main fit
def analyse(df, sat, n_boot, seed=0):
    rng = np.random.default_rng(seed)
    rows, nested = [], []

    for (kind, target), gt in df.groupby(["kind", "target"]):
        if target not in SYNTH_TARGETS:
            continue
        for k, graw in gt.groupby("k"):
            if k == 0:
                continue
            ag = aggregate(graw, sat)
            if len(ag) < MIN_FIT_POINTS or ag["d"].nunique() < 2:
                continue
            point = fit_pair(ag, k)

            # ---- cell-level bootstrap ------------------------------------------------
            bA, bB = [], []
            n = len(ag)
            for _ in range(n_boot):
                idx = rng.integers(0, n, n)
                s = ag.iloc[idx]
                try:
                    f = fit_pair(s, k)
                    bA.append(f["cE"])
                    if np.isfinite(f["cG"]):
                        bB.append(f["cG"])
                except Exception:
                    pass
            qA = np.percentile(bA, [2.5, 97.5]) if bA else [np.nan, np.nan]
            qB = np.percentile(bB, [2.5, 97.5]) if bB else [np.nan, np.nan]

            # ---- seed-level cluster bootstrap ---------------------------------------
            seeds = np.array(sorted(graw["seed"].dropna().unique()))
            by_seed = {s: graw[graw.seed == s] for s in seeds}
            cA_, cB_ = [], []
            if len(seeds) >= 2:
                for _ in range(n_boot):
                    pick = rng.choice(seeds, size=len(seeds), replace=True)
                    sub = pd.concat([by_seed[s] for s in pick], ignore_index=True)
                    agb = aggregate(sub, sat)
                    if len(agb) < MIN_FIT_POINTS or agb["d"].nunique() < 2:
                        continue
                    f = fit_pair(agb, k)
                    cA_.append(f["cE"])
                    if np.isfinite(f["cG"]):
                        cB_.append(f["cG"])
            cqA = np.percentile(cA_, [2.5, 97.5]) if cA_ else [np.nan, np.nan]
            cqB = np.percentile(cB_, [2.5, 97.5]) if cB_ else [np.nan, np.nan]

            rows.append(dict(
                family=f"{kind}/{target}", k=int(k), n_cells=len(ag),
                # (A) paper's parametrisation
                cN=round(point["cN"], 3), cE=round(point["cE"], 3),
                cE_lo=round(qA[0], 3), cE_hi=round(qA[1], 3),
                cE_lo_clu=round(cqA[0], 3), cE_hi_clu=round(cqA[1], 3),
                cE_pred=k + 1,
                cE_pass=bool(abs(point["cE"] - (k + 1)) <= TOL_CE * (k + 1)),
                vif_N=round(point["vif_N"], 2),
                identifiable=bool(point["vif_N"] < VIF_MAX),
                # (B) exact per-edge resolution
                n_cells_B=point["n_B"],
                cG=round(point["cG"], 3),
                cG_lo=round(qB[0], 3), cG_hi=round(qB[1], 3),
                cG_lo_clu=round(cqB[0], 3), cG_hi_clu=round(cqB[1], 3),
                cG_pred=-(k + 1),
                cG_pass=bool(abs(point["cG"] + (k + 1)) <= TOL_RATE * (k + 1)),
            ))

            # ---- nested-model check: does log E add anything beyond log G_exact? -----
            N, E, Gx, y = design(ag, k)
            ok = Gx > 0
            if ok.sum() >= MIN_FIT_POINTS + 1:
                X1 = np.column_stack([np.ones(ok.sum()), np.log(Gx[ok])])
                X2 = np.column_stack([X1, np.log(E[ok])])
                r1 = y[ok] - X1 @ ols(X1, y[ok])
                r2 = y[ok] - X2 @ ols(X2, y[ok])
                rss1, rss2 = float((r1 ** 2).sum()), float((r2 ** 2).sum())
                dof = ok.sum() - 3
                Fstat = ((rss1 - rss2) / 1) / (rss2 / dof) if dof > 0 and rss2 > 0 else np.nan
                nested.append(dict(family=f"{kind}/{target}", k=int(k), n=int(ok.sum()),
                                   rss_G_only=round(rss1, 4), rss_G_plus_E=round(rss2, 4),
                                   F=round(float(Fstat), 2),
                                   extra_coef_on_logE=round(float(ols(X2, y[ok])[2]), 3)))
    return pd.DataFrame(rows), pd.DataFrame(nested)


def to_booktabs(df, path):
    lines = [r"\begin{tabular}{llccccc}", r"\toprule",
             r"Family & $k$ & $\hat c_{E}$ (A) & 95\% CI & $\hat c_{G}$ (B) & 95\% CI & predicted \\",
             r"\midrule"]
    for _, r in df.iterrows():
        lines.append(
            f"{r.family.replace('_', chr(92)+'_')} & {r.k} & ${r.cE:.3f}$ & "
            f"$[{r.cE_lo:.3f}, {r.cE_hi:.3f}]$ & ${r.cG:.3f}$ & "
            f"$[{r.cG_lo:.3f}, {r.cG_hi:.3f}]$ & ${r.cG_pred}$ \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(path, "w").write("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="runs_raw.csv or runs.jsonl")
    ap.add_argument("--out", default="./corrected")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    t0 = time.time()
    df = load_runs(args.runs)
    print(f"loaded {len(df):,} probe rows in {time.time()-t0:.1f}s")

    t1 = time.time()
    main, nested = analyse(df, SAT_DEFAULT, args.n_boot)
    print(f"main fit + {args.n_boot} cell bootstraps + {args.n_boot} cluster bootstraps: "
          f"{time.time()-t1:.1f}s")

    t2 = time.time()
    sens = []
    for s in SAT_GRID:
        m, _ = analyse(df, s, n_boot=0)
        m["sat"] = s
        sens.append(m[["sat", "family", "k", "cE", "cG"]])
    sens = pd.concat(sens)
    print(f"saturation sensitivity: {time.time()-t2:.1f}s")

    pd.set_option("display.width", 250)
    cols = ["family", "k", "n_cells", "cE", "cE_lo", "cE_hi", "cE_pass",
            "cG", "cG_lo", "cG_hi", "cG_pass", "cG_pred", "vif_N", "identifiable"]
    print("\n=== CORRECTED JOINT FIT ===")
    print(main[cols].to_string(index=False))
    print("\n=== NESTED CHECK: does log E add anything beyond log G_exact? ===")
    print(nested.to_string(index=False))

    main.to_csv(f"{args.out}/corrected_joint_fit.csv", index=False)
    sens.to_csv(f"{args.out}/corrected_saturation_sens.csv", index=False)
    nested.to_csv(f"{args.out}/corrected_nested_test.csv", index=False)
    to_booktabs(main, f"{args.out}/corrected_joint_fit.tex")
    print(f"\ntotal wall time: {time.time()-t0:.1f}s")
    print(f"written to {args.out}/")
