"""Diagnostic C8: does knot-target commensurability explain the exponent overshoot?

Section 5.2 attributes the oracle's overshoot (c_E = 3.98 / 4.90 / 6.93 against a predicted
3 / 4 / 5) to the non-monotone response of the error to G, caused by the fixed target
psi(t) = sin(pi t) + 0.3 cos(3 pi t) aligning with the uniform knot grid at certain G.

That attribution is currently a hypothesis. This script tests it directly, using the SAME
closed-form probe as the paper (Cox-de Boor basis + SiLU residual column, exact LSQ), on
1-D so that it runs on CPU in minutes:

  fixed    : the paper's target, identical across seeds  -> expect non-monotone error in G
  random   : frequencies and phases drawn per seed       -> expect monotone G^-(k+1)

If the random-phase family removes the non-monotonicity and returns slopes closer to
-(k+1), the attribution is supported and the overshoot is a design artefact rather than a
failure of spline theory.

NOTE: this is a NEW diagnostic, not part of the pre-registered study. It must be re-run in
the author's own environment before any of it enters the manuscript.
"""
import math
import numpy as np
import torch
import torch.nn.functional as F

torch.use_deterministic_algorithms(True, warn_only=True)

N_TRAIN = 16_000
GRIDS = [4, 6, 8, 12, 16, 24, 32, 48, 64]
ORDERS = [2, 3, 4]
SEEDS = list(range(10))
FLOOR = 1e-6           # discard points at/below the fp64 noise floor, as the paper does


def basis(x, grid, order, g0=-1.0, g1=1.0):
    """Cox-de Boor recursion, identical to KANLayer._basis in the pipeline."""
    step = (g1 - g0) / grid
    t = torch.arange(-order, grid + order + 1, dtype=torch.float64) * step + g0
    eps = 1e-6 * (g1 - g0)
    x = x.clamp(g0 + eps, g1 - eps).unsqueeze(-1)
    b = ((x >= t[:-1]) & (x < t[1:])).to(x.dtype)
    for k in range(1, order + 1):
        ld = (t[k:-1] - t[:-(k + 1)]).clamp_min(1e-8)
        rd = (t[k + 1:] - t[1:-k]).clamp_min(1e-8)
        b = (x - t[:-(k + 1)]) / ld * b[..., :-1] + (t[k + 1:] - x) / rd * b[..., 1:]
    return b


def target_fixed(x, seed):
    """The paper's target: identical for every seed."""
    return torch.sin(math.pi * x) + 0.3 * torch.cos(3 * math.pi * x)


def target_random(x, seed):
    """Same functional form, but frequencies and phases drawn per seed.

    Frequencies are drawn from a continuous range so they are generically incommensurate
    with the uniform knot spacing; amplitudes are matched to the fixed target's profile so
    that only commensurability, not smoothness or amplitude, differs.
    """
    g = torch.Generator().manual_seed(9000 + seed)
    f1 = 0.7 + 1.6 * torch.rand(1, generator=g, dtype=torch.float64).item()
    f2 = 2.1 + 1.8 * torch.rand(1, generator=g, dtype=torch.float64).item()
    p1 = 2 * math.pi * torch.rand(1, generator=g, dtype=torch.float64).item()
    p2 = 2 * math.pi * torch.rand(1, generator=g, dtype=torch.float64).item()
    return torch.sin(math.pi * f1 * x + p1) + 0.3 * torch.cos(math.pi * f2 * x + p2)


def probe_error(target_fn, grid, order, seed):
    g = torch.Generator().manual_seed(seed)
    x = (torch.rand(N_TRAIN, 1, generator=g, dtype=torch.float64) * 2 - 1)
    y = target_fn(x, seed)
    y = (y - y.mean()) / y.std().clamp_min(1e-12)
    with torch.no_grad():
        b = basis(x, grid, order).reshape(N_TRAIN, -1)
        A = torch.cat([b, F.silu(x)], 1)
        sol = torch.linalg.lstsq(A, y, rcond=None).solution
        pred = A @ sol
    return (torch.linalg.norm(pred - y) / torch.linalg.norm(y)).item()


def slope(grids, errs):
    pts = [(g, e) for g, e in zip(grids, errs) if e > FLOOR]
    if len(pts) < 2:
        return float("nan"), len(pts)
    s = np.polyfit(np.log([g for g, _ in pts]), np.log([e for _, e in pts]), 1)[0]
    return float(s), len(pts)


def monotonicity(errs):
    """Fraction of consecutive grid steps on which the error decreases."""
    d = np.diff(errs)
    return float((d < 0).mean())


rows = []
for name, fn in [("fixed (paper)", target_fixed), ("phase-randomised", target_random)]:
    for k in ORDERS:
        curves = []
        for s in SEEDS:
            curves.append([probe_error(fn, G, k, s) for G in GRIDS])
        mean_curve = np.mean(curves, axis=0)
        sl, npts = slope(GRIDS, mean_curve)
        per_seed = [slope(GRIDS, c)[0] for c in curves]
        rows.append(dict(target=name, k=k, slope_mean_curve=round(sl, 3),
                         predicted=-(k + 1),
                         abs_err=round(abs(sl + (k + 1)), 3),
                         seed_slope_sd=round(float(np.nanstd(per_seed)), 3),
                         monotone_frac=round(monotonicity(mean_curve), 2),
                         n_points=npts,
                         curve=[float(f"{v:.2e}") for v in mean_curve]))

import pandas as pd
df = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print("=== C8: phase-randomisation control (closed-form 1-D probe, 10 seeds) ===\n")
print(df[["target", "k", "slope_mean_curve", "predicted", "abs_err",
          "seed_slope_sd", "monotone_frac", "n_points"]].to_string(index=False))
print("\nMean error curves over G =", GRIDS)
for r in rows:
    print(f"  {r['target']:17s} k={r['k']}: {r['curve']}")
df.drop(columns=["curve"]).to_csv("/home/claude/kanms/phase_control.csv", index=False)
print("\nwritten: phase_control.csv")
