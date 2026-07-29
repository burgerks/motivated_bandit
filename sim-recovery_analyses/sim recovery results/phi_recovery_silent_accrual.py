#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
phi_recovery_silent_accrual.py

Confirms the v17.2 bonus-confound fix and quantifies the bias it avoids, with no
v17.2 subjects. The confound: a fast probe response earns a bonus, and if that bonus
is SHOWN as an outcome it is a reward event sitting between the probe and the next
bandit trial, opening an RT -> bonus -> alpha+ path that is correlated with the
motivational state and can masquerade as phi.

Three data-generating conditions are compared under matched settings:
  silent      (v17.2): the bonus is accrued silently; no reward event enters the
                       bandit update stream. The nuisance path is absent by design.
  shown       (v15/v16 without a trace): the bonus perturbs the adjacent bandit
                       update, in proportion to the hit, which is correlated with the
                       state through RT. This is the confound.
  shown_trace (v16 fallback): 'shown' plus a decaying bonus-trace nuisance regressor
                       in the FIT, to check the regressor absorbs the bias.

Two truth scenarios per condition:
  alpha_pos : true phi = TRUE_PHI on the latent state (on-target recovery).
  null      : true phi = 0 (false-positive check; the pure confound signature).

Honest framing: under 'silent' the clean result is largely BY CONSTRUCTION, since
source removal makes the path structurally absent. The script's value is (1) sizing
the bias 'shown' would have carried, (2) confirming the estimator is clean and the
null false-positive rate is nominal under 'silent', and (3) confirming the trace
fallback works if you ever need it. It is a confirmation, not a discovery.

Dependencies: numpy, pandas, scipy.
Run:  python3 phi_recovery_silent_accrual.py           (quick defaults)
      python3 phi_recovery_silent_accrual.py --full    (slower, tighter estimates)
"""

import argparse
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

# =============================================================================
# USER SETTINGS
# =============================================================================
HALF_LIFE_S   = 9.0
TRIAL_S       = 3.5
LAMBDA_SINGLE = 0.33
N_TRIALS      = 100
REVERSALS     = (50,)
P_REWARD      = (0.80, 0.50, 0.30)
ALPHA_POS, ALPHA_NEG, BETA = 0.40, 0.50, 3.0

N_FOOD_PROBES = 48         # v17.2 food probes per run
TRUE_PHI      = 0.15       # on-target coupling
BONUS_WINDOW_MS = 550.0    # fixed probe window (v17.2)

# Bonus / RT chain. Faster RT when the state is higher; hit if RT < window.
RT_BASE_MS    = 380.0      # median probe RT anchor
RT_STATE_B    = 45.0       # ms faster per +1 state (wanting speeds responses)
RT_NOISE_MS   = 60.0       # within-probe RT noise
# BONUS_LEAK sets how strongly a 'shown' bonus perturbs the ADJACENT bandit update
# (extra positive PE proportional to the mean-centered hit). This is the confound
# strength; 0.15 is a deliberately visible, not extreme, setting. Sweep it.
BONUS_LEAK    = 0.15
TRACE_DECAY   = 0.6        # pre-registered decay for the bonus trace (borrowed from reward trace)

N_SUBJECTS    = 400        # subjects simulated and fitted per (condition, scenario)
N_TARGET      = 100        # planned cohort for the bootstrap detection rate
N_STUDIES     = 2000
N_FIT_STARTS  = 2
ALPHA_LEVEL   = 0.05
SEED          = 20260727

RHO_STATE = 0.5 ** (TRIAL_S / HALF_LIFE_S)
CONDITIONS = ("silent", "shown", "shown_trace")
SCENARIOS  = ("alpha_pos", "null")


# =============================================================================
# DESIGN AND STATE
# =============================================================================
def latent_state(rng):
    """AR(1) motivational state, unit variance, at the measured persistence."""
    s = np.zeros(N_TRIALS); s[0] = rng.normal()
    e = rng.normal(0, np.sqrt(1 - RHO_STATE ** 2), N_TRIALS)
    for t in range(1, N_TRIALS):
        s[t] = RHO_STATE * s[t - 1] + e[t]
    return s


def probe_trials():
    """Evenly spaced food probes across the run."""
    return np.linspace(0, N_TRIALS - 1, N_FOOD_PROBES).astype(int)


def observed_modulator(pt, state, rng):
    """Noisy carried-forward probe reading (correlates sqrt(lambda) with the state)."""
    obs = (np.sqrt(LAMBDA_SINGLE) * state[pt]
           + np.sqrt(1 - LAMBDA_SINGLE) * rng.normal(0, 1, len(pt)))
    mod = np.zeros(N_TRIALS)
    for i, t in enumerate(pt):
        end = pt[i + 1] if i + 1 < len(pt) else N_TRIALS
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


def probe_hits(pt, state, rng):
    """Per-probe hit indicator from the RT->window chain, plus a mean-centered hit
    signal aligned to trials (nonzero only on the trial AFTER each probe)."""
    rt = RT_BASE_MS - RT_STATE_B * state[pt] + rng.normal(0, RT_NOISE_MS, len(pt))
    hit = (rt < BONUS_WINDOW_MS).astype(float)
    hit_c = hit - hit.mean()                       # center so only state-linked part leaks
    bump = np.zeros(N_TRIALS)
    for i, t in enumerate(pt):
        j = t + 1                                  # bonus event sits before the next trial
        if j < N_TRIALS:
            bump[j] = hit_c[i]
    return bump


def bonus_trace(bump):
    """Leaky causal trace of the (centered) bonus events, for the shown_trace fit."""
    tr = np.zeros(N_TRIALS)
    for t in range(1, N_TRIALS):
        tr[t] = TRACE_DECAY * tr[t - 1] + bump[t - 1]
    return tr


# =============================================================================
# GENERATION  (phi on alpha+; optional shown-bonus perturbation of the update)
# =============================================================================
def simulate(rng, state, phi, p_rew, bump, shown):
    """Choices under phi*state on alpha+. When shown=True, a bonus event adds an extra
    positive prediction error to the chosen arm on the trial after a probe, scaled by
    BONUS_LEAK; when shown=False (silent) the bonus never enters the update."""
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
        if shown and bump[t] != 0.0:               # bonus reward event perturbs this update
            Q[c] += BONUS_LEAK * bump[t]
        ch[t], rw[t] = c, r
    return ch, rw


def nll_phi(params, ch, rw, mod):
    """NLL of alpha+_t = alpha+ + phi*modulator (no trace)."""
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


def nll_phi_trace(params, ch, rw, mod, trace):
    """NLL of alpha+_t = alpha+ + phi*modulator + phi_rew*trace (the v16 fallback fit)."""
    ap, an, beta, phi, phir = params
    Q = np.zeros(3); ll = 0.0
    for t in range(N_TRIALS):
        z = beta * Q; z -= z.max()
        pr = np.exp(z); pr /= pr.sum()
        ll += np.log(max(pr[ch[t]], 1e-12))
        pe = rw[t] - Q[ch[t]]
        a = np.clip(ap + phi * mod[t] + phir * trace[t], .01, .99) if pe >= 0 else an
        Q[ch[t]] += a * pe
    return -ll


def fit(ch, rw, mod, rng, starts, trace=None):
    """Multistart fit; returns phi_hat (the alpha+ modulation by the probe signal)."""
    best = None
    for _ in range(starts):
        if trace is None:
            x0 = [rng.uniform(.2, .6), rng.uniform(.2, .6),
                  rng.uniform(2, 5), rng.uniform(-.2, .2)]
            b = [(.01, .99), (.01, .99), (.1, 20), (-2, 2)]
            fn, args = nll_phi, (ch, rw, mod)
        else:
            x0 = [rng.uniform(.2, .6), rng.uniform(.2, .6),
                  rng.uniform(2, 5), rng.uniform(-.2, .2), rng.uniform(-.2, .2)]
            b = [(.01, .99), (.01, .99), (.1, 20), (-2, 2), (-2, 2)]
            fn, args = nll_phi_trace, (ch, rw, mod, trace)
        try:
            r = minimize(fn, x0, args=args, method="L-BFGS-B", bounds=b)
            if best is None or r.fun < best.fun:
                best = r
        except Exception:
            continue
    return best.x[3] if best is not None else np.nan


# =============================================================================
# ONE (condition, scenario) CELL
# =============================================================================
def run_cell(cond, scenario, args):
    """Simulate and fit a pool, then bootstrap N_TARGET studies for a detection rate.
    Under 'null' that rate is the false-positive rate; the pure confound signature."""
    pt = probe_trials()
    mean_phi = 0.0 if scenario == "null" else TRUE_PHI
    shown = cond in ("shown", "shown_trace")
    phis = np.empty(args.n_subjects)
    for i in range(args.n_subjects):
        sub = np.random.default_rng([args.seed, i])
        state = latent_state(sub)
        p_rew = reward_schedule(sub)
        bump = probe_hits(pt, state, np.random.default_rng([args.seed, i, 4]))
        ch, rw = simulate(np.random.default_rng([args.seed, i, 1]),
                          state, mean_phi, p_rew, bump, shown)
        mod = observed_modulator(pt, state, np.random.default_rng([args.seed, i, 2]))
        trace = bonus_trace(bump) if cond == "shown_trace" else None
        phis[i] = fit(ch, rw, mod, np.random.default_rng([args.seed, i, 3]),
                      args.starts, trace=trace)
    phis = phis[np.isfinite(phis)]

    boot = np.random.default_rng([args.seed, 77])
    hits = 0
    for _ in range(args.n_studies):
        s = phis[boot.integers(0, len(phis), N_TARGET)]
        if s.std(ddof=1) > 0 and stats.ttest_1samp(s, 0).pvalue < ALPHA_LEVEL:
            hits += 1
    return dict(condition=cond, scenario=scenario,
                true_phi=mean_phi,
                mean_phi_hat=round(float(phis.mean()), 4),
                bias=round(float(phis.mean() - mean_phi), 4),
                sd_phi_hat=round(float(phis.std(ddof=1)), 4),
                detect_or_fpr_at_100=round(hits / args.n_studies, 3))


# =============================================================================
# MAIN
# =============================================================================
def main():
    global BONUS_LEAK
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                    help="tighter estimates (n_subjects=900, 3 starts)")
    ap.add_argument("--n-subjects", type=int, default=N_SUBJECTS, dest="n_subjects")
    ap.add_argument("--n-studies", type=int, default=N_STUDIES, dest="n_studies")
    ap.add_argument("--starts", type=int, default=N_FIT_STARTS)
    ap.add_argument("--bonus-leak", type=float, default=BONUS_LEAK, dest="bonus_leak")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--outdir", type=str, default=".")
    args = ap.parse_args()
    if args.full:
        args.n_subjects, args.starts = 900, 3
    BONUS_LEAK = args.bonus_leak

    print(f"rho_state={RHO_STATE:.3f}  probes={N_FOOD_PROBES}  bonus_leak={BONUS_LEAK}  "
          f"true_phi={TRUE_PHI}  n_subjects={args.n_subjects}\n")

    rows = [run_cell(c, s, args) for c in CONDITIONS for s in SCENARIOS]
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 160)
    print(df.to_string(index=False))
    df.to_csv(f"{args.outdir}/phi_recovery_silent_accrual.csv", index=False)
    print(f"\nwrote {args.outdir}/phi_recovery_silent_accrual.csv")

    # Plain-language read of the key contrasts.
    def get(c, s, k): return df[(df.condition == c) & (df.scenario == s)][k].iloc[0]
    print("\nHOW TO READ THIS")
    print(f"  silent/null  false-positive rate : {get('silent','null','detect_or_fpr_at_100')}"
          f"   (want ~{ALPHA_LEVEL}; confound absent by design)")
    print(f"  shown/null   false-positive rate : {get('shown','null','detect_or_fpr_at_100')}"
          f"   (elevation here is the bias v17.2 avoids)")
    print(f"  shown_trace/null false-positive  : {get('shown_trace','null','detect_or_fpr_at_100')}"
          f"   (fallback: trace should pull this back toward {ALPHA_LEVEL})")
    print(f"  silent/alpha_pos phi bias        : {get('silent','alpha_pos','bias')}"
          f"   (on-target recovery should match the clean baseline)")
    print("  If silent/null sits near 0.05 and silent/alpha_pos bias matches the baseline "
          "block_phi_recovery run, the source-removal fix is confirmed.")
    print("  CAVEATS: the SILENT result is the load-bearing one and holds by construction. "
          "The SHOWN magnitude depends on how the bonus is assumed to enter the update "
          "(here a direct value bump at --bonus-leak), so treat it as illustrative, not a "
          "calibrated bias. The shown_trace fallback here uses a simplified fit; validate "
          "the real decaying-trace regressor in fit_bandit_mid_rl before relying on it.")


if __name__ == "__main__":
    main()
