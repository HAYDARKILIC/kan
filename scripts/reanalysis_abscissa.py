"""Re-analysis: is the oracle's exponent overshoot an abscissa artefact?

The paper derives the edge law by substituting G ~ N/E into the rate bound, i.e. it drops
the (k+1) term in the exact relation

        N = E (G + k + 1)   =>   G = N/E - (k+1).

That substitution is a LARGE-G approximation. In the actual design G ranges over 4..64 and
k+1 over 3..5, so at the small-G end G is not large relative to k+1 at all, and the
approximation is violated across a substantial part of the sweep.

Consequence: the joint regression log(eps) = a + c_N log N + c_E log E estimates an exponent
with respect to (G+k+1), whereas the spline rate theorem is stated with respect to G. This
script quantifies the difference on the ACTUAL probe cells, refitting the same data with the
exact per-edge resolution G = N/E - (k+1) as the regressor.

Fits compared, per (probe family, spline order):
  (A) paper's joint fit              log eps ~ 1 + c_N log N + c_E log E
  (B) exact-resolution single fit    log eps ~ 1 + c_G log G,  G = N/E - (k+1)
  (C) as (B) restricted to G >= 4(k+1), the regime where G >> k+1 actually holds

Pre-registered tolerance for the rate exponent is 0.15 relative (bedrock_rate_rel_tol);
for c_E it is 0.25 relative. Both are applied for reference.

NOTE: this is a POST-HOC re-analysis of the committed run. It must be reproduced inside the
author's own pipeline, with the saturation filter, bootstrap intervals and cluster intervals,
before any of it enters the manuscript.
"""
import numpy as np
import pandas as pd

D = "/home/claude/res/kan_hidden_curse_v5/outputs/"
SAT = 0.6
TOL_RATE, TOL_CE = 0.15, 0.25

df = pd.read_csv(D + "runs_raw.csv", low_memory=False)
probe = df[df.kind.isin(["edge_kan", "kanet", "best_spline"])].copy()

agg = (probe.groupby(["kind", "target", "regime", "k", "d", "grid", "params", "edges"],
                     dropna=False)["rel_l2"].mean().reset_index()
       .rename(columns={"rel_l2": "err"}))

rows = []
for (kind, target), gt in agg.groupby(["kind", "target"]):
    if target not in ("additive", "kastruct", "kastruct_np"):
        continue
    for k, g in gt.groupby("k"):
        g = g[(g.err < SAT) & (g.err > 0) & (g.edges > 0) & (g.params > 0)].copy()
        if len(g) < 4 or g["d"].nunique() < 2:
            continue
        N = g["params"].values.astype(float)
        E = g["edges"].values.astype(float)
        y = np.log(g["err"].values)

        # (A) the paper's two-regressor fit
        XA = np.column_stack([np.ones(len(N)), np.log(N), np.log(E)])
        cA = np.linalg.lstsq(XA, y, rcond=None)[0]

        # (B) exact per-edge resolution as the single regressor
        Gexact = N / E - (k + 1)
        ok = Gexact > 0
        XB = np.column_stack([np.ones(ok.sum()), np.log(Gexact[ok])])
        cB = np.linalg.lstsq(XB, y[ok], rcond=None)[0]

        # (C) restricted to the regime where the large-G approximation is defensible
        sel = ok & (Gexact >= 4 * (k + 1))
        cC = (np.linalg.lstsq(np.column_stack([np.ones(sel.sum()), np.log(Gexact[sel])]),
                              y[sel], rcond=None)[0] if sel.sum() >= 4 else [np.nan, np.nan])

        rows.append(dict(
            family=f"{kind}/{target}", k=int(k), n=len(g),
            cE_paper=round(cA[2], 3),
            cE_pass=bool(abs(cA[2] - (k + 1)) <= TOL_CE * (k + 1)),
            cG_exact=round(cB[1], 3),
            cG_pass=bool(abs(cB[1] + (k + 1)) <= TOL_RATE * (k + 1)),
            n_restricted=int(sel.sum()),
            cG_restricted=round(float(cC[1]), 3),
            cG_r_pass=(bool(abs(cC[1] + (k + 1)) <= TOL_RATE * (k + 1))
                       if np.isfinite(cC[1]) else None),
            predicted=-(k + 1),
        ))

out = pd.DataFrame(rows).sort_values(["family", "k"])
pd.set_option("display.width", 220)
print("=== Re-analysis: exponent with respect to (G+k+1) versus with respect to G ===\n")
print(out.to_string(index=False))
out.to_csv("/home/claude/kanms/reanalysis_abscissa.csv", index=False)

add = out[out.family.str.startswith("best_spline")]
print("\n--- Oracle (best_spline / additive), the family with an exact optimum ---")
for _, r in add.iterrows():
    print(f"  k={r.k}: paper c_E = {r.cE_paper:+.3f} (pass={r.cE_pass})   "
          f"exact-G slope = {r.cG_exact:+.3f} (pass={r.cG_pass})   predicted {r.predicted}")

print("\nHow much of the design violates the large-G approximation:")
for k in (2, 3, 4):
    g = agg[(agg.kind == "best_spline") & (agg.k == k)]
    Gx = g["params"].values / g["edges"].values - (k + 1)
    print(f"  k={k}: {(Gx < 4*(k+1)).mean()*100:5.1f}% of cells have G < 4(k+1); "
          f"min G = {Gx.min():.0f}, max G = {Gx.max():.0f}")
