#!/usr/bin/env python3
"""Between-subject reliability (R) of the residualized food-cued vigor index.

R is what patch_recovery_vigor_v2.py takes as -R. It is NOT lambda (0.33), which is the
within-subject single-probe figure. Run from the analysis_output folder."""

import numpy as np, pandas as pd

# ---------------- USER SETTINGS ----------------
TRIALWISE = "bandit_mid_trial_cleaned.csv"
SUMMARY   = "bandit_mid_subject_summary.csv"
EXCLUDE   = ["grid-013"]          # QC-flagged
FOOD_PREFIX, MIN_PROBES, N_SPLITS, SEED = "food", 8, 2000, 20260723
NUISANCE  = ["bandit_median_choice_rt", "mid_no_response_prop", "mid_premature_prop"]
# ------------------------------------------------

def resid(y, X):
    """OLS residuals of y on X with an intercept; NaN rows pass through as NaN."""
    y, X = np.asarray(y, float), np.asarray(X, float)
    ok = np.isfinite(y) & np.isfinite(X).all(1)
    out = np.full_like(y, np.nan)
    if ok.sum() < X.shape[1] + 2: return out
    A = np.column_stack([np.ones(ok.sum()), X[ok]])
    out[ok] = y[ok] - A @ np.linalg.lstsq(A, y[ok], rcond=None)[0]
    return out

def sb(r, k=2.0):
    """Spearman-Brown step-up; negative half-half correlation means no reliable variance."""
    if not np.isfinite(r) or r <= 0: return 0.0
    return float(np.clip(k * r / (1 + (k - 1) * r), 0, 1))

def z(x):
    s = np.nanstd(x)
    return (x - np.nanmean(x)) / s if s > 1e-12 else np.full_like(x, np.nan)

# Load food probes: usable presses only, log RT, optional staircase window.
d = pd.read_csv(TRIALWISE)
if "trial_type" in d: d = d[d.trial_type.astype(str).str.contains("bonus", case=False, na=False)]
d = d[~d.participant_id.isin(EXCLUDE)]
d = d[d.cue_type.astype(str).str.lower().str.startswith(FOOD_PREFIX)]
for c in ("target_too_fast", "target_no_response"):
    if c in d: d = d[~d[c].fillna(False).astype(bool)]
d = d[d.target_rt_ms.notna() & (d.target_rt_ms > 0)].copy()
d["log_rt"] = np.log(d.target_rt_ms.astype(float))

# Window adjustment: remove staircase-driven variation but restore each subject's level,
# since an intercept-only residual would delete the between-subject variance being measured.
d["log_rt_adj"] = d["log_rt"]
if "adaptive_window_ms" in d:
    d["lw"] = np.log(d.adaptive_window_ms.astype(float))
    for p, g in d.groupby("participant_id"):
        if g.lw.notna().sum() >= 5 and g.lw.std() > 1e-9:
            d.loc[g.index, "log_rt_adj"] = resid(g.log_rt.values, g[["lw"]].values) + g.log_rt.mean()

S = pd.read_csv(SUMMARY).set_index("participant_id")
rng = np.random.default_rng(SEED)

def reliability(col, do_resid):
    """Repeated split-half: two independent vigor indices per iteration, correlated across subjects."""
    by = {p: g[col].to_numpy() for p, g in d.groupby("participant_id")}
    pids = [p for p in sorted(by) if np.isfinite(by[p]).sum() >= MIN_PROBES]
    if len(pids) < 5: return 0.0, 0.0, 0.0, len(pids)
    X = np.column_stack([S.loc[pids, c].to_numpy(float) for c in NUISANCE]) if do_resid else None
    rs = []
    for _ in range(N_SPLITS):
        a, b = [], []
        for p in pids:
            v = by[p][np.isfinite(by[p])]; i = rng.permutation(len(v)); h = len(v) // 2
            a.append(v[i[:h]].mean()); b.append(v[i[h:2*h]].mean())
        a, b = np.array(a), np.array(b)
        if do_resid: a, b = resid(a, X), resid(b, X)
        a, b = z(-a), z(-b)                      # sign flip: faster = more vigor
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() >= 5: rs.append(np.corrcoef(a[ok], b[ok])[0, 1])
    rs = np.array(rs)
    return sb(np.median(rs)), sb(np.percentile(rs, 2.5)), sb(np.percentile(rs, 97.5)), len(pids)

def icc(col):
    """ICC(1) plus the implied reliability of a k-probe mean; independent cross-check."""
    g = d.groupby("participant_id")[col]; n, m = g.count(), g.mean()
    n = n[n >= MIN_PROBES]
    if len(n) < 5: return 0.0, 0.0
    x = d[d.participant_id.isin(n.index)]; k, grand = n.mean(), x[col].mean()
    msb = (n * (m[n.index] - grand) ** 2).sum() / (len(n) - 1)
    msw = sum(((x[x.participant_id == p][col] - m[p]) ** 2).sum() for p in n.index) / (n.sum() - len(n))
    i = max((msb - msw) / (msb + (k - 1) * msw), 0.0)
    return i, sb(i, k)

print("\nFOOD PROBES PER SUBJECT")
print(d.groupby(["participant_id", "cue_type"]).size().unstack(fill_value=0).to_string())

print(f"\n{'variant':<28}{'n':>4}{'R':>8}{'95% CI':>16}{'ICC':>8}{'ICC->R':>9}")
for name, col, rz in [("raw", "log_rt", False), ("residualized", "log_rt", True),
                      ("window-adj + residualized", "log_rt_adj", True)]:
    R, lo, hi, ns = reliability(col, rz); i, ik = icc(col)
    print(f"{name:<28}{ns:>4}{R:>8.3f}{f'{lo:.2f} to {hi:.2f}':>16}{i:>8.3f}{ik:>9.3f}")

R = reliability("log_rt", True)[0]
print(f"\nUse -R {R:.2f} in patch_recovery_vigor_v2.py")
if R > 0:
    print(f"gamma_true needed for 80% power at N=100: {0.177/np.sqrt(R):.2f} "
          f"(0.35 counts as a large effect)")