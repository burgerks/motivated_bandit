#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
precision_r_sigmaphi.py

How many subjects to estimate the two data-dependent unknowns to a target precision,
using no v17.2 subjects. Answers, by simulation:

  A) R      = between-subject reliability of the residualized food-cued vigor index
              (median food-probe log RT residualized on neutral), the summary the
              gamma route regresses on.
  B) sigma_phi = between-subject SD of the trial-wise vigor -> alpha+ coupling phi,
              the quantity that sets required N for the PRIMARY phi test.

Method: simulate cohorts at a plausible truth, apply the SAME estimators the real
pipeline will use, and read the sampling spread of the estimate across many simulated
cohorts at each N. The reported "N needed" is the smallest swept N whose CI half-width
meets the target.

Caveats to keep in mind when reading the output:
  * The sigma_phi estimator here is a method-of-moments variance decomposition
    (Var(phi_hat) minus the pure estimation variance). A full hierarchical/Bayesian
    fit estimates sigma_phi jointly and is usually a little more efficient, so the
    required-N from this script is approximate and mildly conservative.
  * Truth values (true R, true sigma_phi) are set below and are themselves uncertain;
    sweep them. The half-life, lambda, and RT-model constants are pilot/staircase-era
    and should be refreshed once v17.2 data exist.

Dependencies: numpy, pandas, scipy.
Run:  python3 precision_r_sigmaphi.py            (quick defaults)
      python3 precision_r_sigmaphi.py --full     (paper-grade, slower)
"""

import argparse
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# =============================================================================
# USER SETTINGS
# =============================================================================
# ---- shared task / state constants (match block_phi_recovery_v2.py) ----------
HALF_LIFE_S   = 9.0        # pilot; state half-life. REFRESH on v17.2 data.
TRIAL_S       = 3.5        # assumed trial duration
LAMBDA_SINGLE = 0.33       # pilot; single food-probe reliability. REFRESH.
N_TRIALS      = 100        # bandit trials per run
REVERSALS     = (50,)      # representative single-run reversal
P_REWARD      = (0.80, 0.50, 0.30)
ALPHA_POS, ALPHA_NEG, BETA = 0.40, 0.50, 3.0

# ---- part A: R (vigor-index reliability) -------------------------------------
# R is the reliability of the FULL 48-probe residualized index, so the probe count is
# already baked into its value. The precision question is: given a true R, how tightly
# does a sample of N subjects pin it down through the split-half + Spearman-Brown
# estimator the real pipeline uses. R is therefore an explicit input and is swept.
R_TRUE_SWEEP  = (0.3, 0.4, 0.5)   # plausible true reliabilities (pilot suggested ~0.4)
R_TARGET_HALFWIDTH = 0.10          # want the 95% CI half-width on R at or below this
M_SPLITS      = 300                # repeated split-half draws per dataset (vigor_R uses 2000)

# ---- part B: sigma_phi -------------------------------------------------------
K_FOOD            = 48     # v17.2 food probes per run (sets the trial-wise modulator)
MEAN_PHI          = 0.15   # group-mean coupling when present
SIGMA_PHI_TRUE    = 0.15   # assumed true between-subject SD of phi. SWEEP this.
SIGMA_HALFWIDTH   = 0.05   # want the 95% CI half-width on sigma_phi at or below this
V0_POOL           = 400    # homogeneous pool used to estimate pure estimation variance
PHI_QC_ABS        = 1.0    # drop |phi_hat| above this as a failed/boundary fit (QC), so a
                           # few optimizer-bound hits do not inflate the variance estimate

# ---- sweep and run size ------------------------------------------------------
N_SWEEP        = (30, 50, 75, 100, 150)
N_DATASETS     = 150       # cohorts simulated per N (quick default; --full raises it)
N_FIT_STARTS   = 2         # L-BFGS-B multistart per subject fit
SEED           = 20260727

RHO_STATE = 0.5 ** (TRIAL_S / HALF_LIFE_S)   # per-trial state persistence (~0.76)


# =============================================================================
# SAFE HELPERS
# =============================================================================
def safe_corr(a, b):
    """Pearson r that returns nan instead of raising on degenerate input."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or a[ok].std() < 1e-12 or b[ok].std() < 1e-12:
        return np.nan
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def spearman_brown(r, k=2.0):
    """Step a half-length reliability up to full length; clip to [0,1]."""
    if not np.isfinite(r) or r <= 0:
        return 0.0
    return float(np.clip(k * r / (1 + (k - 1) * r), 0.0, 1.0))


# =============================================================================
# PART A: R  (between-subject reliability of the residualized vigor index)
# =============================================================================
def simulate_probes(rng, n, R_true):
    """Per-subject food-probe values whose 48-probe-mean index has reliability R_true.

    Each subject has a true index level plus independent per-probe noise. The noise SD
    is set so the K_FOOD-probe mean reaches the target reliability, matching how the real
    index is built from many probes rather than one."""
    sigma_e = np.sqrt(K_FOOD * (1.0 / R_true - 1.0))     # gives K-probe reliability = R_true
    s = rng.normal(0, 1, n)
    return s[:, None] + rng.normal(0, sigma_e, (n, K_FOOD))


def estimate_R(probes, rng, m_splits):
    """Repeated split-half reliability (median over splits), stepped up by Spearman-Brown.

    Mirrors vigor_R.py: many random 24/24 probe splits, correlate the two half-means
    across subjects, take the median correlation, then step to full length. Averaging
    over splits removes split-selection noise, leaving subject-sampling variability."""
    n, k = probes.shape
    h = k // 2
    rs = []
    for _ in range(m_splits):
        idx = rng.permutation(k)
        a = probes[:, idx[:h]].mean(1)
        b = probes[:, idx[h:2 * h]].mean(1)
        rs.append(safe_corr(a, b))
    return spearman_brown(np.nanmedian(rs))


def run_partA(args, rng):
    """Sweep true R and N; report mean R_hat, its sampling SD, and 95% CI half-width.

    Probe count is not swept because it sets the VALUE of R (through K_FOOD), not the
    sampling error of estimating that R across subjects; that error is driven by N and R."""
    rows = []
    for R_true in R_TRUE_SWEEP:
        for n in args.n_sweep:
            ests = []
            for d in range(args.n_datasets):
                sub = np.random.default_rng([args.seed, int(R_true * 100), n, d])
                probes = simulate_probes(sub, n, R_true)
                ests.append(estimate_R(probes, sub, args.m_splits))
            ests = np.array([e for e in ests if np.isfinite(e)])
            lo, hi = np.percentile(ests, [2.5, 97.5])
            rows.append(dict(quantity="R", true=R_true, n_subjects=n,
                             mean_est=round(float(ests.mean()), 3),
                             sd_est=round(float(ests.std(ddof=1)), 3),
                             ci_lo=round(float(lo), 3), ci_hi=round(float(hi), 3),
                             ci_halfwidth=round(float((hi - lo) / 2), 3),
                             meets_target=int((hi - lo) / 2 <= R_TARGET_HALFWIDTH)))
    return pd.DataFrame(rows)


# =============================================================================
# PART B: sigma_phi  (between-subject SD of the trial-wise coupling)
# =============================================================================
def latent_state(rng):
    """AR(1) motivational state at the measured per-trial persistence, unit variance."""
    s = np.zeros(N_TRIALS); s[0] = rng.normal()
    e = rng.normal(0, np.sqrt(1 - RHO_STATE ** 2), N_TRIALS)
    for t in range(1, N_TRIALS):
        s[t] = RHO_STATE * s[t - 1] + e[t]
    return s


def observed_modulator(probe_trials, state, rng):
    """Noisy carried-forward probe reading: correlates sqrt(lambda) with the state."""
    obs = (np.sqrt(LAMBDA_SINGLE) * state[probe_trials]
           + np.sqrt(1 - LAMBDA_SINGLE) * rng.normal(0, 1, len(probe_trials)))
    mod = np.zeros(N_TRIALS)
    for i, t in enumerate(probe_trials):
        end = probe_trials[i + 1] if i + 1 < len(probe_trials) else N_TRIALS
        mod[t:end] = obs[i]
    return mod


def reward_schedule(rng):
    """Three-arm schedule; rotate the profile at each reversal."""
    order = np.array(P_REWARD)
    perms = [order, order[[2, 0, 1]], order[[1, 2, 0]]]
    bounds = [0] + list(REVERSALS) + [N_TRIALS]
    p = np.zeros((N_TRIALS, 3))
    for i in range(len(bounds) - 1):
        p[bounds[i]:bounds[i + 1]] = perms[i % 3]
    return p


def simulate(rng, state, phi, p_rew):
    """Generate choices with phi modulating alpha+ on the LATENT state (alpha_pos only)."""
    Q = np.zeros(3); ch = np.zeros(N_TRIALS, int); rw = np.zeros(N_TRIALS)
    for t in range(N_TRIALS):
        z = BETA * Q; z -= z.max()
        pr = np.exp(z); pr /= pr.sum()
        c = rng.choice(3, p=pr)
        r = float(rng.random() < p_rew[t, c])
        pe = r - Q[c]
        if pe >= 0:
            a = np.clip(ALPHA_POS + phi * state[t], .01, .99)
        else:
            a = ALPHA_NEG
        Q[c] += a * pe
        ch[t], rw[t] = c, r
    return ch, rw


def nll(params, ch, rw, mod):
    """NLL of the fitted model: alpha+_t = alpha+ + phi * observed modulator."""
    ap, an, beta, phi = params
    Q = np.zeros(3); ll = 0.0
    for t in range(N_TRIALS):
        z = beta * Q; z -= z.max()
        pr = np.exp(z); pr /= pr.sum()
        ll += np.log(max(pr[ch[t]], 1e-12))
        pe = rw[t] - Q[ch[t]]
        a = np.clip(ap + phi * mod[t], .01, .99) if pe >= 0 else an
        Q[ch[t]] += a * pe
    return -ll


def fit_phi(ch, rw, mod, rng, starts):
    """Multistart L-BFGS-B fit; return phi_hat."""
    best = None
    for _ in range(starts):
        x0 = [rng.uniform(.2, .6), rng.uniform(.2, .6),
              rng.uniform(2, 5), rng.uniform(-.2, .2)]
        try:
            r = minimize(nll, x0, args=(ch, rw, mod), method="L-BFGS-B",
                         bounds=[(.01, .99), (.01, .99), (.1, 20), (-2, 2)])
            if best is None or r.fun < best.fun:
                best = r
        except Exception:
            continue
    return best.x[3] if best is not None else np.nan


def sim_one_subject(seed_key, phi_i, starts):
    """Common-random-number subject: fit phi_hat for a given true phi_i."""
    sub = np.random.default_rng(seed_key)
    state = latent_state(sub)
    p_rew = reward_schedule(sub)
    pt = np.linspace(0, N_TRIALS - 1, K_FOOD).astype(int)
    ch, rw = simulate(np.random.default_rng(seed_key + [1]), state, phi_i, p_rew)
    mod = observed_modulator(pt, state, np.random.default_rng(seed_key + [2]))
    return fit_phi(ch, rw, mod, np.random.default_rng(seed_key + [3]), starts)


def estimate_v0(args):
    """Pure estimation variance of phi_hat with NO between-subject spread (sigma_phi=0).

    This is the noise floor subtracted from Var(phi_hat) to recover true spread."""
    hats = []
    for i in range(args.v0_pool):
        hats.append(sim_one_subject([args.seed, 7, i], MEAN_PHI, args.starts))
    hats = np.array([h for h in hats if np.isfinite(h) and abs(h) <= PHI_QC_ABS])
    return float(np.var(hats, ddof=1)), hats.size


def run_partB(args, rng):
    """Sweep N; per cohort estimate sigma_phi by variance decomposition, then read
    the sampling spread of sigma_phi_hat across cohorts."""
    v0, n_v0 = estimate_v0(args)
    rows = []
    for n in args.n_sweep:
        ests = []
        for d in range(args.n_datasets):
            key = [args.seed, 200 + n, d]
            phis_true = np.random.default_rng(key + [0]).normal(MEAN_PHI, SIGMA_PHI_TRUE, n)
            hats = np.array([sim_one_subject([args.seed, 200 + n, d, i], float(phis_true[i]),
                                             args.starts) for i in range(n)])
            hats = hats[np.isfinite(hats) & (np.abs(hats) <= PHI_QC_ABS)]
            var_hat = np.var(hats, ddof=1)
            ests.append(np.sqrt(max(var_hat - v0, 0.0)))   # method-of-moments sigma_phi
        ests = np.array(ests)
        lo, hi = np.percentile(ests, [2.5, 97.5])
        rows.append(dict(quantity="sigma_phi", true=SIGMA_PHI_TRUE, n_subjects=n,
                         mean_est=round(float(ests.mean()), 3),
                         sd_est=round(float(ests.std(ddof=1)), 3),
                         ci_lo=round(float(lo), 3), ci_hi=round(float(hi), 3),
                         ci_halfwidth=round(float((hi - lo) / 2), 3),
                         meets_target=int((hi - lo) / 2 <= SIGMA_HALFWIDTH)))
    return pd.DataFrame(rows), v0, n_v0


# =============================================================================
# MAIN
# =============================================================================
def smallest_n(df):
    """Smallest swept N meeting its precision target, or -1 if none in the sweep."""
    ok = df[df.meets_target == 1]
    return int(ok.n_subjects.min()) if len(ok) else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                    help="paper-grade run size (n_datasets=500, 3 starts, bigger v0 pool)")
    ap.add_argument("--n-datasets", type=int, default=N_DATASETS, dest="n_datasets")
    ap.add_argument("--starts", type=int, default=N_FIT_STARTS)
    ap.add_argument("--v0-pool", type=int, default=V0_POOL, dest="v0_pool")
    ap.add_argument("--m-splits", type=int, default=M_SPLITS, dest="m_splits",
                    help="repeated split-half draws per dataset for the R estimate")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--n-sweep", type=int, nargs="+", default=list(N_SWEEP), dest="n_sweep",
                    help="subject counts to sweep, e.g. --n-sweep 30 50 75 100 150")
    ap.add_argument("--outdir", type=str, default=".")
    args = ap.parse_args()
    if args.full:
        args.n_datasets, args.starts, args.v0_pool = 500, 3, 800

    rng = np.random.default_rng(args.seed)
    print(f"rho_state={RHO_STATE:.3f}  lambda={LAMBDA_SINGLE}  K_food={K_FOOD}  "
          f"n_datasets={args.n_datasets}\n")

    dfA = run_partA(args, rng)
    print(f"PART A  R  (target CI half-width <= {R_TARGET_HALFWIDTH}; swept over true R)")
    print(dfA.to_string(index=False))
    for R_true in R_TRUE_SWEEP:
        print(f"  -> true R={R_true}: smallest N meeting target = "
              f"{smallest_n(dfA[dfA.true == R_true])}")
    print()

    dfB, v0, n_v0 = run_partB(args, rng)
    print(f"PART B  sigma_phi (true {SIGMA_PHI_TRUE}; est-variance floor v0={v0:.5f} "
          f"from {n_v0} subjects; target CI half-width <= {SIGMA_HALFWIDTH})")
    print(dfB.to_string(index=False))
    print(f"  -> smallest N meeting sigma_phi target: {smallest_n(dfB)}")

    out = pd.concat([dfA, dfB], ignore_index=True)
    out.to_csv(f"{args.outdir}/precision_r_sigmaphi.csv", index=False)
    print(f"\nwrote {args.outdir}/precision_r_sigmaphi.csv")
    print("READ: 'meets_target'=1 marks Ns precise enough. R is cheap. sigma_phi is the "
          "subject-hungry one: per-subject phi_hat noise (SD ~0.29 at 100 trials/48 probes) "
          "is roughly 2x the true 0.15, so the method-of-moments estimate is upward-biased "
          "and wide at small N. Read its required N as approximate and demanding; a joint "
          "hierarchical estimate on real data will do somewhat better.")


if __name__ == "__main__":
    main()
