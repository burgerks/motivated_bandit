#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gamma_power_v17_2.py

Recomputes power for the SECONDARY (between-subject) gamma route at the v17.2 probe
counts and at realistic effect sizes, with no v17.2 subjects. This is the route where
residual food-cued vigor predicts a bandit learning parameter through a group-level
coefficient gamma, fit by hierarchical EM with rho fixed to 1.

It answers three planning questions in one sweep:
  * probe counts fixed at v17.2 (48 food, 22 neutral), which set the vigor index quality;
  * a sweep over the between-subject vigor reliability R (data-dependent, not yet known);
  * a sweep over the true gamma (the earlier pilot used 0.35, a large effect) and N.

Target parameter is configurable. Default is beta, because with rho fixed the reward-
sensitivity hypothesis lands on beta and beta recovered best in the pilot; alpha_pos is
available too. Each cell also reports the null false-positive rate.

This is a direct descendant of new_bandit_recovery_script_v2.py; the generative model,
the EM fit, and the gamma definition are unchanged. Only the counts, the sweeps, and the
reporting differ.

Caveat: R is set here, not measured. Read power as a function of R and lock the number
once R is estimated on v17.2 data. The alpha_pos coefficient ran anti-conservative in
the pilot (Wald intervals), so treat alpha_pos power/FPR from Wald CIs as optimistic and
confirm with permutation for that parameter.

Dependencies: numpy, pandas.
Run:  python3 gamma_power_v17_2.py                     (quick defaults)
      python3 gamma_power_v17_2.py --full              (paper-grade, slower)
      python3 gamma_power_v17_2.py --target alpha_pos  (switch target parameter)
"""

import argparse
import numpy as np
import pandas as pd

# =============================================================================
# USER SETTINGS
# =============================================================================
N_TRIALS = 200                       # full session (two 100-trial runs)
N_ARMS = 3
REVERSAL_TRIALS = [69, 130]          # representative; per-run reversals differ in-task
PROFILES0 = np.array([[0.80, 0.20], [0.30, 0.70], [0.50, 0.50]], dtype=float)
WIN_REWARD, LOSS_REWARD = 1.0, -1.0

N_FOOD_MID, N_NEUTRAL_MID = 48, 22   # v17.2 counts (was 34/20 in v2)

# Sweeps. Keep these small for a quick look; widen for the final planning table.
R_SWEEP      = (0.4,)       # between-subject vigor reliability (unknown)
GAMMA_SWEEP  = (0.10, 0.20, 0.35)    # true group effect; 0.35 = large (old pilot)
N_SWEEP      = (100, 150)        # cohort sizes

TARGET_DEFAULT = "beta"              # where reward sensitivity lands with rho fixed
GRID_DEFAULT   = 11                  # EM grid points per parameter dimension
EM_ITER        = 50
N_DATASETS     = 200                  # datasets per cell (quick default)
SEED           = 20260727
Z = 1.96


# =============================================================================
# GENERATIVE MODEL  (rho fixed to 1)
# =============================================================================
def softmax_probs(q, beta):
    z = beta * q; z -= np.max(z); e = np.exp(z); return e / np.sum(e)


def sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))
def logit(p): return np.log(p / (1 - p))


def simulate_vigor(rng, n_subjects, reliability):
    """Latent vigor (generates data) plus the noisy index that is actually fitted.
    reliability R is the BETWEEN-SUBJECT reliability of the residualized vigor index."""
    R = float(np.clip(reliability, 1e-6, 1.0))
    v_true = rng.normal(0, 1, n_subjects)
    v_obs = np.sqrt(R) * v_true + np.sqrt(1.0 - R) * rng.normal(0, 1, n_subjects)
    v_obs = (v_obs - v_obs.mean()) / (v_obs.std() + 1e-12)
    return v_true, v_obs


def generate_reward_matrix(rng):
    """Per-trial +1/-1 outcomes per arm, with random-direction reversals."""
    profiles = PROFILES0.copy()
    rmat = np.zeros((N_TRIALS, N_ARMS))
    for t in range(1, N_TRIALS + 1):
        if t in REVERSAL_TRIALS:
            profiles = profiles[[2, 0, 1], :] if rng.random() < 0.5 else profiles[[1, 2, 0], :]
        for arm in range(N_ARMS):
            rmat[t - 1, arm] = WIN_REWARD if rng.random() < profiles[arm, 0] else LOSS_REWARD
    return rmat


def make_params(rng, vigor, scenario, gamma):
    """True subject parameters (rho=1); gamma*vigor added to the target parameter."""
    ap = rng.normal(-1.05, 0.55)
    an = rng.normal(-1.35, 0.55)
    lb = rng.normal(np.log(3.0), 0.45)
    if scenario == "alpha_pos": ap += gamma * vigor
    elif scenario == "alpha_neg": an += gamma * vigor
    elif scenario == "beta": lb += gamma * vigor
    return dict(alpha_pos=float(sigmoid(ap)), alpha_neg=float(sigmoid(an)),
                beta=float(np.clip(np.exp(lb), 0.2, 25.0)),
                ap_logit=float(ap), an_logit=float(an), log_beta=float(lb))


def simulate_choices(rng, params, reward_matrix):
    """One agent's choices under the asymmetric-learning-rate model (rho=1)."""
    ap, an, beta = params["alpha_pos"], params["alpha_neg"], params["beta"]
    q = np.zeros(N_ARMS); ch = np.zeros(N_TRIALS, int); rw = np.zeros(N_TRIALS)
    for t in range(N_TRIALS):
        p = softmax_probs(q, beta)
        c = int(rng.choice(N_ARMS, p=p)); r = float(reward_matrix[t, c])
        pe = r - q[c]
        q[c] += (ap if pe >= 0 else an) * pe
        ch[t] = c; rw[t] = r
    return ch, rw


# =============================================================================
# HIERARCHICAL EM FIT  (partial pooling; gamma = brain-behavior coefficient)
# =============================================================================
def build_grid(n):
    ap = np.linspace(logit(0.04), logit(0.93), n)
    an = np.linspace(logit(0.04), logit(0.93), n)
    lb = np.linspace(np.log(0.3), np.log(20.0), n)
    A, B, C = np.meshgrid(ap, an, lb, indexing="ij")
    grid = np.column_stack([A.ravel(), B.ravel(), C.ravel()])
    nat = np.column_stack([sigmoid(grid[:, 0]), sigmoid(grid[:, 1]), np.exp(grid[:, 2])])
    return grid, nat


def data_nll(nat_ap, nat_an, nat_beta, ch, rw):
    """Vectorized data NLL over all grid cells for one subject (rho=1)."""
    G = nat_beta.size; q = np.zeros((G, 3)); nll = np.zeros(G); idx = np.arange(G)
    for c, r in zip(ch, rw):
        z = nat_beta[:, None] * q; z -= z.max(1, keepdims=True)
        e = np.exp(z); p = e / e.sum(1, keepdims=True)
        nll -= np.log(p[idx, c] + 1e-12)
        pe = r - q[idx, c]
        a = np.where(pe >= 0, nat_ap, nat_an)
        q[idx, c] += a * pe
    return nll


def em_fit(ND, grid, vigor, n_iter):
    """EM over the grid: E-step posterior over cells, M-step group means, gammas, vars.
    Returns gamma coefficients and their Wald SEs (per parameter)."""
    S, G = ND.shape
    m = np.array([-1.05, -1.35, np.log(3.0)])
    gamma = np.zeros(3)
    Sigma = np.array([0.5, 0.5, 0.4]) ** 2
    vz = (vigor - vigor.mean()) / (vigor.std() + 1e-12)
    sx2 = float(np.sum((vz - vz.mean()) ** 2))
    for _ in range(n_iter):
        mu = m[None, :] + np.outer(vz, gamma)
        pen = np.zeros((S, G))
        for d in range(3):
            diff = grid[:, d][None, :] - mu[:, d][:, None]
            pen += 0.5 * diff ** 2 / Sigma[d]
        logw = -(ND + pen); logw -= logw.max(1, keepdims=True)
        w = np.exp(logw); w /= w.sum(1, keepdims=True)
        Etheta = w @ grid
        Vtheta = np.clip(w @ (grid ** 2) - Etheta ** 2, 0, None)
        for d in range(3):
            b = float(np.sum((vz - vz.mean()) * (Etheta[:, d] - Etheta[:, d].mean())) / (sx2 + 1e-12))
            a0 = float(Etheta[:, d].mean() - b * vz.mean())
            gamma[d] = b; m[d] = a0
            resid = Etheta[:, d] - (a0 + b * vz)
            Sigma[d] = float(np.mean(resid ** 2 + Vtheta[:, d]))
    se = np.sqrt(Sigma / (sx2 + 1e-12))
    return gamma, se


# =============================================================================
# ONE CELL  (target power + null FPR)
# =============================================================================
PARAM_IDX = {"alpha_pos": 0, "alpha_neg": 1, "beta": 2}


def run_cell(target, R, gamma_true, n_subjects, args, grid, nat):
    """Detection rate for the target scenario (power) and for null (FPR), at one cell."""
    nat_ap, nat_an, nat_b = nat[:, 0], nat[:, 1], nat[:, 2]
    j = PARAM_IDX[target]
    out = {}
    for scenario, g in [(target, gamma_true), ("null", 0.0)]:
        hits = 0
        for d in range(args.n_datasets):
            rng = np.random.default_rng([args.seed, hash(target) & 0xffff,
                                         int(R * 100), int(gamma_true * 100), n_subjects, d])
            v_true, v_obs = simulate_vigor(rng, n_subjects, R)
            ND = np.zeros((n_subjects, nat_b.size))
            for i in range(n_subjects):
                tp = make_params(rng, float(v_true[i]), scenario, g)
                rmat = generate_reward_matrix(rng)
                ch, rw = simulate_choices(rng, tp, rmat)
                ND[i] = data_nll(nat_ap, nat_an, nat_b, ch, rw)
            gamma_hat, se = em_fit(ND, grid, v_obs, args.em_iter)
            lo, hi = gamma_hat[j] - Z * se[j], gamma_hat[j] + Z * se[j]
            hits += int(lo > 0 or hi < 0)
        out["power" if scenario != "null" else "null_fpr"] = round(hits / args.n_datasets, 3)
    return out


# =============================================================================
# MAIN
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                    help="paper-grade run size (n_datasets=300, grid=13)")
    ap.add_argument("--target", type=str, default=TARGET_DEFAULT,
                    choices=list(PARAM_IDX.keys()))
    ap.add_argument("--n-datasets", type=int, default=N_DATASETS, dest="n_datasets")
    ap.add_argument("--grid", type=int, default=GRID_DEFAULT)
    ap.add_argument("--em-iter", type=int, default=EM_ITER, dest="em_iter")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--outdir", type=str, default=".")
    args = ap.parse_args()
    if args.full:
        args.n_datasets, args.grid = 300, 13

    grid, nat = build_grid(args.grid)
    print(f"target={args.target}  probes={N_FOOD_MID}f/{N_NEUTRAL_MID}n  "
          f"n_datasets={args.n_datasets}  grid={args.grid}\n"
          f"sweeping R={R_SWEEP}  gamma={GAMMA_SWEEP}  N={N_SWEEP}\n")

    rows = []
    for R in R_SWEEP:
        for gt in GAMMA_SWEEP:
            for n in N_SWEEP:
                res = run_cell(args.target, R, gt, n, args, grid, nat)
                rows.append(dict(target=args.target, R=R, true_gamma=gt, n_subjects=n,
                                 power=res["power"], null_fpr=res["null_fpr"]))
                print(f"  R={R}  gamma={gt}  N={n}  ->  power={res['power']}  "
                      f"null_fpr={res['null_fpr']}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(f"{args.outdir}/gamma_power_v17_2.csv", index=False)
    print("\nPOWER TABLE (rows with power >= 0.80 are adequately powered):")
    print(df.to_string(index=False))
    print(f"\nwrote {args.outdir}/gamma_power_v17_2.csv")
    print("READ: pick your row by the R you expect and the gamma you consider realistic. "
          "0.35 is large; if power is only adequate at 0.35, treat the gamma route as "
          "exploratory and lean on the trial-wise phi test.")


if __name__ == "__main__":
    main()
