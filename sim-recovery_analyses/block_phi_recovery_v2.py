#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
block_phi_recovery_v2.py

Does the trial-wise phi route survive the three things v1 assumed away?

v1 reported required N ~ 80 for phi = 0.15 with ~32 evenly spread food probes. That
figure rested on three assumptions, all of which this script tests:

  1. NULL SCENARIO. v1 always planted a true phi, so it never produced a false positive
     rate. The between-subject gamma route came back at 0.15 against a nominal 0.05, and
     there is no reason to assume phi is better behaved. If phi's FPR is also inflated,
     required N is optimistic and the fix is permutation inference rather than more
     subjects.
  2. HETEROGENEITY. v1 held phi identical across subjects, so SD(phi_hat) was pure
     estimation noise. Real between-subject spread adds to it. This sweeps SIGMA_PHI.
  3. SPECIFICITY. v1 only ever planted phi on alpha+. Whether a state-modulated beta or
     alpha- leaks into phi_hat is the claim the whole interpretation rests on, and it
     was untested.

Blocking is settled and dropped: v1 showed evenly spread probes match or beat every
blocked layout at matched participant cost. Only spread layouts are swept here.

Method
------
For each (layout, scenario, sigma) a POOL of subjects is simulated and fitted once.
Studies of size N_TARGET are then drawn from the pool by bootstrap and a one-sample
t-test run on each, giving detection rate directly rather than by analytic extrapolation.
For the null scenario that rate is the false positive rate.

phi is planted on the LATENT state; the model is fitted with the OBSERVED carried-forward
probe reading. Using one series for both roles hides the entire effect being studied.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

# =============================================================================
# USER SETTINGS
# =============================================================================
HALF_LIFE_S      = 9.0      # measured, probe_calibration_train
TRIAL_S          = 3.5
LAMBDA_SINGLE    = 0.33     # measured, probe_calibration_train

N_TRIALS         = 100
REVERSALS        = (50, )
P_REWARD         = (0.80, 0.50, 0.30)
ALPHA_POS, ALPHA_NEG, BETA = 0.40, 0.50, 3.0

# Food probe counts to compare. These are FOOD probes only: neutral probes never enter
# the phi estimate, since only food probes carry the modulator.
PROBE_COUNTS     = (25, 32, 48)

TRUE_PHI         = 0.15     # group mean effect when present
SIGMA_PHI_SWEEP  = (0.0, 0.10, 0.20, 0.40)   # between-subject SD of phi
SIGMA_PHI_OTHER  = 0.10     # heterogeneity used for the off-target scenarios

# Pool must be well above N_TARGET: studies are bootstrapped from it, so the pool's own
# sampling error enters the rejection rate. At POOL_SIZE = 6 x N_TARGET the pool mean is
# precise enough that the null rate is not materially inflated.
POOL_SIZE        = 600      # subjects simulated and fitted per cell
N_TARGET         = 100      # planned cohort
N_STUDIES        = 2000     # bootstrap studies drawn from each pool
ALPHA_LEVEL      = 0.05
SEED             = 20260724

RHO = 0.5 ** (TRIAL_S / HALF_LIFE_S)


# =============================================================================
# DESIGN AND STATE
# =============================================================================
def spread_probe_trials(n):
    """Evenly spaced food probes across the task."""
    return np.linspace(0, N_TRIALS - 1, n).astype(int)


def latent_state(rng):
    """AR(1) motivational state at the measured per-trial persistence, unit variance."""
    s = np.zeros(N_TRIALS); s[0] = rng.normal()
    e = rng.normal(0, np.sqrt(1 - RHO ** 2), N_TRIALS)
    for t in range(1, N_TRIALS):
        s[t] = RHO * s[t - 1] + e[t]
    return s


def observed_modulator(probe_trials, state, rng):
    """Noisy probe readings carried forward to the next probe.

    Each reading correlates sqrt(lambda) with the state at the moment it was taken, and
    decays away from it thereafter. That gap is the thing probe density controls."""
    obs = (np.sqrt(LAMBDA_SINGLE) * state[probe_trials]
           + np.sqrt(1 - LAMBDA_SINGLE) * rng.normal(0, 1, len(probe_trials)))
    mod = np.zeros(N_TRIALS)
    for i, t in enumerate(probe_trials):
        end = probe_trials[i + 1] if i + 1 < len(probe_trials) else N_TRIALS
        mod[t:end] = obs[i]
    return mod


def reward_schedule(rng):
    """Three-arm schedule; rotates the profile at each reversal. Works for any
    number of reversal points, so it handles both the 200-trial two-reversal task
    and the 100-trial one-reversal split runs."""
    order = np.array(P_REWARD)
    perms = [order, order[[2, 0, 1]], order[[1, 2, 0]]]
    bounds = [0] + list(REVERSALS) + [N_TRIALS]
    p = np.zeros((N_TRIALS, 3))
    for i in range(len(bounds) - 1):
        p[bounds[i]:bounds[i + 1]] = perms[i % 3]   # cycle profiles across phases
    return p


# =============================================================================
# GENERATION
# =============================================================================
def simulate(rng, state, phi, scenario, p_rew):
    """Generate choices. The scenario decides WHICH parameter the state modulates.

    Only 'alpha_pos' matches the fitted model. The others are the specificity test: if
    a state-modulated beta or alpha- shows up as a nonzero phi_hat, then 'state modulates
    learning rate' cannot be distinguished from 'state modulates choice determinism'."""
    Q = np.zeros(3)
    ch = np.zeros(N_TRIALS, int); rw = np.zeros(N_TRIALS)
    for t in range(N_TRIALS):
        b = BETA * np.exp(phi * state[t]) if scenario == "beta" else BETA
        z = b * Q; z -= z.max()
        pr = np.exp(z); pr /= pr.sum()
        c = rng.choice(3, p=pr)
        r = float(rng.random() < p_rew[t, c])
        pe = r - Q[c]
        if pe >= 0:
            a = np.clip(ALPHA_POS + (phi * state[t] if scenario == "alpha_pos" else 0), .01, .99)
        else:
            a = np.clip(ALPHA_NEG + (phi * state[t] if scenario == "alpha_neg" else 0), .01, .99)
        Q[c] += a * pe
        ch[t], rw[t] = c, r
    return ch, rw


def nll(params, ch, rw, mod):
    """Negative log-likelihood of the fitted model: alpha+_t = alpha+ + phi * modulator."""
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


def fit_phi(ch, rw, mod, rng):
    """Fit the modulated model from two starts; return phi_hat."""
    best = None
    for _ in range(2):
        x0 = [rng.uniform(.2, .6), rng.uniform(.2, .6), rng.uniform(2, 5), rng.uniform(-.2, .2)]
        try:
            r = minimize(nll, x0, args=(ch, rw, mod), method="L-BFGS-B",
                         bounds=[(.01, .99), (.01, .99), (.1, 20), (-2, 2)])
            if best is None or r.fun < best.fun:
                best = r
        except Exception:
            continue
    return best.x[3] if best is not None else np.nan


# =============================================================================
# ONE CELL
# =============================================================================
def run_cell(n_probes, scenario, sigma_phi, mean_phi):
    """Simulate and fit a pool, then bootstrap studies to get a rejection rate.

    Bootstrapping studies out of one fitted pool is far cheaper than simulating each
    study from scratch and gives the same rejection rate, provided the pool is large
    relative to the subsampling."""
    pt = spread_probe_trials(n_probes)
    phis = np.empty(POOL_SIZE)
    for i in range(POOL_SIZE):
        # Common random numbers: subject i gets the same state and schedule in every cell.
        sub = np.random.default_rng([SEED, i])
        state = latent_state(sub)
        p_rew = reward_schedule(sub)
        phi_i = np.random.default_rng([SEED, i, 9]).normal(mean_phi, sigma_phi)
        ch, rw = simulate(np.random.default_rng([SEED, i, 1]), state, phi_i, scenario, p_rew)
        mod = observed_modulator(pt, state, np.random.default_rng([SEED, i, 2]))
        phis[i] = fit_phi(ch, rw, mod, np.random.default_rng([SEED, i, 3]))

    phis = phis[np.isfinite(phis)]
    boot = np.random.default_rng([SEED, n_probes, 7])
    hits = 0
    for _ in range(N_STUDIES):
        s = phis[boot.integers(0, len(phis), N_TARGET)]
        if s.std(ddof=1) > 0 and stats.ttest_1samp(s, 0).pvalue < ALPHA_LEVEL:
            hits += 1
    rate = hits / N_STUDIES

    # Required N is only meaningful where a true effect was planted on the fitted
    # parameter. For null and off-target cells there is nothing to detect, so it is
    # suppressed rather than reported as a huge meaningless number.
    if scenario == "alpha_pos" and mean_phi != 0:
        d = phis.mean() / phis.std(ddof=1) if phis.std(ddof=1) > 0 else np.nan
        n_req = int(round((2.8 / d) ** 2)) if np.isfinite(d) and abs(d) > 1e-6 else -1
        if not (0 < n_req < 10 ** 6):
            n_req = -1
    else:
        n_req = -1
    return dict(probes=n_probes, scenario=scenario, sigma_phi=sigma_phi,
                mean_phi_hat=round(float(phis.mean()), 4),
                sd_phi_hat=round(float(phis.std(ddof=1)), 4),
                rate_at_100=round(rate, 3),
                n_req_80pct=n_req)


# =============================================================================
# MAIN
# =============================================================================
def main():
    if POOL_SIZE < 4 * N_TARGET:
        raise SystemExit(f"POOL_SIZE ({POOL_SIZE}) must be at least 4x N_TARGET "
                         f"({N_TARGET}), or the null rate is inflated by pool noise.")
    print(f"rho = {RHO:.3f}  lambda = {LAMBDA_SINGLE}  pool = {POOL_SIZE}  "
          f"N_target = {N_TARGET}  true phi = {TRUE_PHI}\n")

    rows = []
    ref = PROBE_COUNTS[len(PROBE_COUNTS) // 2]          # reference probe count
    for sig in SIGMA_PHI_SWEEP:                          # heterogeneity, at ref probes
        rows.append(run_cell(ref, "alpha_pos", sig, TRUE_PHI))
        print(f"  done: {ref} probes, alpha_pos, sigma={sig}", flush=True)
    for n in PROBE_COUNTS:                               # probe count, at ref sigma
        if n != ref:
            rows.append(run_cell(n, "alpha_pos", SIGMA_PHI_OTHER, TRUE_PHI))
            print(f"  done: {n} probes, alpha_pos, sigma={SIGMA_PHI_OTHER}", flush=True)
    for sc, mp in [("null", 0.0), ("beta", TRUE_PHI), ("alpha_neg", TRUE_PHI)]:
        rows.append(run_cell(ref, sc, SIGMA_PHI_OTHER if sc != "null" else 0.0, mp))
        print(f"  done: {ref} probes, {sc}", flush=True)

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 160)
    print("\nON-TARGET: phi planted on alpha+, by heterogeneity")
    print(df[df.scenario == "alpha_pos"].to_string(index=False))
    print("\nNULL AND OFF-TARGET: rate here is a FALSE POSITIVE rate")
    print(df[df.scenario != "alpha_pos"].to_string(index=False))
    df.to_csv("block_phi_recovery_v2.csv", index=False)

    print("\nHOW TO READ THIS")
    print("  null rate near 0.05  -> inference is calibrated; on-target rates mean what")
    print("                          they say. Well above 0.05 -> use permutation tests")
    print("                          and treat every on-target rate as optimistic.")
    print("  beta / alpha_neg     -> should also sit near 0.05. If not, a state effect on")
    print("                          another parameter masquerades as a learning-rate")
    print("                          effect, and the specificity claim fails.")
    print("  sigma_phi            -> real between-subject spread in phi. Required N rises")
    print("                          steeply with it and nothing in your data pins it down.")


if __name__ == "__main__":
    main()
