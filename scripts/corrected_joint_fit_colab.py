#@title Corrected joint edge-law fit (Colab cell — paste and run)
"""
Corrected joint edge-law fit — NOTEBOOK VERSION
===============================================

This is the notebook-safe version of corrected_joint_fit.py. It takes no command-line
arguments; edit the CONFIG block below if the automatic path search does not find your run.

WHY THIS EXISTS
---------------
The edge law is derived by substituting  G ~ N/E  into the spline rate bound. The exact
accounting (Lemma 1) is  N = E (G + k + 1), so  G = N/E - (k+1)  and the substitution is a
LARGE-G approximation. In the full-mode design G spans 4..64 while k+1 spans 3..5, and
33-56% of probe cells (depending on k) have G < 4(k+1). Over those cells the two-regressor
fit

        log eps = a + c_N log N + c_E log E                                   (A)

estimates an exponent with respect to (G+k+1) rather than with respect to G, which is the
quantity the spline rate theorem is stated in. This script refits the SAME cells with the
exact per-edge resolution

        log eps = a + c_G log G_exact,     G_exact = N/E - (k+1)              (B)

and reports (A) and (B) side by side with the paper's full machinery: saturation filter,
cell-level bootstrap, seed-level cluster bootstrap, VIF diagnostics, saturation sensitivity,
and a nested-model check of the law's functional form.

NO MODEL IS RETRAINED. Only the existing checkpoint log is read. Runtime is about 80 s at
N_BOOT = 2000, or about 8 s at N_BOOT = 200.
"""

# ============================== CONFIG =================================================
RUNS_PATH = None      # e.g. "/content/drive/MyDrive/kan_hidden_curse_v5/outputs/runs_raw.csv"
                      # leave as None to search the usual locations automatically
OUT_DIR   = None      # leave as None to write next to the runs file, in ../outputs/corrected
N_BOOT    = 2000      # 2000 matches the paper; use 200 for a quick look
MOUNT_DRIVE = True    # set False if Drive is already mounted or you uploaded the file
# =======================================================================================

import os, json, glob, time
import numpy as np
import pandas as pd

# ---- locate the run log ----------------------------------------------------------------
if MOUNT_DRIVE:
    try:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
    except Exception as _e:
        print("[info] Drive not mounted (fine if running locally):", _e)

def _find_runs():
    if RUNS_PATH:
        return RUNS_PATH
    patterns = [
        "/content/drive/MyDrive/kan_hidden_curse_v5/outputs/runs_raw.csv",
        "/content/drive/MyDrive/kan_hidden_curse_v5/checkpoints/runs.jsonl",
        "/content/drive/MyDrive/**/runs_raw.csv",
        "/content/drive/MyDrive/**/runs.jsonl",
        "/content/**/runs_raw.csv",
        "/content/**/runs.jsonl",
        "./**/runs_raw.csv",
    ]
    for p in patterns:
        hits = sorted(glob.glob(p, recursive=True))
        if hits:
            return hits[0]
    return None

RUNS = _find_runs()
if RUNS is None:
    raise SystemExit(
        "Could not find runs_raw.csv or runs.jsonl.\n"
        "Set RUNS_PATH at the top of this cell to the full path, e.g.\n"
        '  RUNS_PATH = "/content/drive/MyDrive/kan_hidden_curse_v5/outputs/runs_raw.csv"')
OUT = OUT_DIR or os.path.join(os.path.dirname(os.path.abspath(RUNS)), "corrected")
os.makedirs(OUT, exist_ok=True)
print("runs file :", RUNS)
print("output dir:", OUT)

# ---- pre-registered constants (Table 3) — do not edit ----------------------------------
SAT_DEFAULT, SAT_GRID = 0.6, [0.5, 0.6, 0.7]
MIN_FIT_POINTS, TOL_RATE, TOL_CE, VIF_MAX = 4, 0.15, 0.25, 10.0
PROBE_KINDS = ("edge_kan", "kanet", "best_spline")
SYNTH_TARGETS = ("additive", "kastruct", "kastruct_np")


def load_runs(path):
    if path.endswith(".jsonl"):
        df = pd.DataFrame([json.loads(l) for l in open(path) if l.strip()])
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
    N = sub["params"].values.astype(float)
    E = sub["edges"].values.astype(float)
    return N, E, N / E - (k + 1), np.log(sub["err"].values)


def fit_pair(sub, k):
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
        out["cG"], out["n_B"] = float(ols(XB, y[ok])[1]), int(ok.sum())
    else:
        out["cG"], out["n_B"] = np.nan, int(ok.sum())
    return out


def aggregate(raw, sat):
    ag = (raw.groupby(["d", "grid", "params", "edges"], dropna=False)["rel_l2"]
             .mean().reset_index().rename(columns={"rel_l2": "err"}))
    return ag[(ag.err < sat) & (ag.err > 0) & (ag.edges > 0) & (ag.params > 0)]


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

            bA, bB, n = [], [], len(ag)
            for _ in range(n_boot):
                s = ag.iloc[rng.integers(0, n, n)]
                try:
                    f = fit_pair(s, k)
                    bA.append(f["cE"])
                    if np.isfinite(f["cG"]):
                        bB.append(f["cG"])
                except Exception:
                    pass
            qA = np.percentile(bA, [2.5, 97.5]) if bA else [np.nan, np.nan]
            qB = np.percentile(bB, [2.5, 97.5]) if bB else [np.nan, np.nan]

            seeds = np.array(sorted(graw["seed"].dropna().unique()))
            by_seed = {s: graw[graw.seed == s] for s in seeds}
            cA_, cB_ = [], []
            if len(seeds) >= 2:
                for _ in range(n_boot):
                    pick = rng.choice(seeds, size=len(seeds), replace=True)
                    agb = aggregate(pd.concat([by_seed[s] for s in pick],
                                              ignore_index=True), sat)
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
                cN=round(point["cN"], 3), cE=round(point["cE"], 3),
                cE_lo=round(qA[0], 3), cE_hi=round(qA[1], 3),
                cE_lo_clu=round(cqA[0], 3), cE_hi_clu=round(cqA[1], 3),
                cE_pred=k + 1,
                cE_pass=bool(abs(point["cE"] - (k + 1)) <= TOL_CE * (k + 1)),
                vif_N=round(point["vif_N"], 2),
                identifiable=bool(point["vif_N"] < VIF_MAX),
                n_cells_B=point["n_B"], cG=round(point["cG"], 3),
                cG_lo=round(qB[0], 3), cG_hi=round(qB[1], 3),
                cG_lo_clu=round(cqB[0], 3), cG_hi_clu=round(cqB[1], 3),
                cG_pred=-(k + 1),
                cG_pass=bool(abs(point["cG"] + (k + 1)) <= TOL_RATE * (k + 1))))

            N, E, Gx, y = design(ag, k)
            ok = Gx > 0
            if ok.sum() >= MIN_FIT_POINTS + 1:
                X1 = np.column_stack([np.ones(ok.sum()), np.log(Gx[ok])])
                X2 = np.column_stack([X1, np.log(E[ok])])
                r1 = y[ok] - X1 @ ols(X1, y[ok])
                r2 = y[ok] - X2 @ ols(X2, y[ok])
                rss1, rss2 = float((r1 ** 2).sum()), float((r2 ** 2).sum())
                dof = ok.sum() - 3
                F = ((rss1 - rss2)) / (rss2 / dof) if dof > 0 and rss2 > 0 else np.nan
                nested.append(dict(family=f"{kind}/{target}", k=int(k), n=int(ok.sum()),
                                   rss_G_only=round(rss1, 4),
                                   rss_G_plus_E=round(rss2, 4), F=round(float(F), 2),
                                   extra_coef_on_logE=round(float(ols(X2, y[ok])[2]), 3)))
    return pd.DataFrame(rows), pd.DataFrame(nested)


def to_booktabs(df, path):
    lines = [r"\begin{tabular}{llccccc}", r"\toprule",
             r"Family & $k$ & $\hat c_{E}$ (A) & 95\% CI & $\hat c_{G}$ (B) & 95\% CI"
             r" & predicted \\", r"\midrule"]
    for _, r in df.iterrows():
        fam = r.family.replace("_", r"\_")
        lines.append(f"{fam} & {r.k} & ${r.cE:.3f}$ & $[{r.cE_lo:.3f}, {r.cE_hi:.3f}]$ & "
                     f"${r.cG:.3f}$ & $[{r.cG_lo:.3f}, {r.cG_hi:.3f}]$ & ${r.cG_pred}$ \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(path, "w").write("\n".join(lines))


# ---- run --------------------------------------------------------------------------------
t0 = time.time()
df = load_runs(RUNS)
print(f"loaded {len(df):,} probe rows in {time.time()-t0:.1f}s")

t1 = time.time()
main, nested = analyse(df, SAT_DEFAULT, N_BOOT)
print(f"main fit + {N_BOOT} cell bootstraps + {N_BOOT} cluster bootstraps: "
      f"{time.time()-t1:.1f}s")

sens = []
for s in SAT_GRID:
    m, _ = analyse(df, s, n_boot=0)
    m["sat"] = s
    sens.append(m[["sat", "family", "k", "cE", "cG"]])
sens = pd.concat(sens)

pd.set_option("display.width", 250)
cols = ["family", "k", "n_cells", "cE", "cE_lo", "cE_hi", "cE_pass",
        "cG", "cG_lo", "cG_hi", "cG_pass", "cG_pred", "vif_N", "identifiable"]
print("\n=== CORRECTED JOINT FIT ===")
print(main[cols].to_string(index=False))
print("\n=== NESTED CHECK: does log E add anything beyond log G_exact? ===")
print(nested.to_string(index=False))
print("\n=== SATURATION SENSITIVITY ===")
print(sens.pivot_table(index=["family", "k"], columns="sat",
                       values=["cE", "cG"]).round(3).to_string())

main.to_csv(f"{OUT}/corrected_joint_fit.csv", index=False)
sens.to_csv(f"{OUT}/corrected_saturation_sens.csv", index=False)
nested.to_csv(f"{OUT}/corrected_nested_test.csv", index=False)
to_booktabs(main, f"{OUT}/corrected_joint_fit.tex")
print(f"\ntotal wall time: {time.time()-t0:.1f}s")
print("written:", sorted(os.listdir(OUT)))
