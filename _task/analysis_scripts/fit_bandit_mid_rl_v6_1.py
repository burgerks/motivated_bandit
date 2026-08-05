#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Per-subject reinforcement-learning fits for the craving-modulated 3-arm bandit
(bandit_mid_task_v17_3.py output). Companion to analyze_bandit_mid_v6.py.

Pipeline version v6.1 (2026-08-04). Adds the joint alpha+/beta state-modulation
model (M5_ab) and per-fit alpha_pos(t) clip-fraction logging. Everything in v6 is
untouched, so all existing summary columns keep their names, meanings, and values.

WHY M5_ab EXISTS. The pre-registered primary analysis is "the state modulates the
learning rate, not choice determinism." Testing that needs a model where the state
loads on alpha_pos AND on beta simultaneously, so the two compete for the same
trials. Recovery simulation put the beta-masquerade false-positive rate at 0.048
under that joint fit at 48 probes, against 0.11-0.16 for a single-knob fit.

NAMING WARNING. The pre-existing "joint" family (M4_joint, M4_joint3) is NOT that
model. Those put two or three modulators on alpha_pos only; they separate vigor
from reward history and from the bonus event, never from beta. The alpha/beta
contrast lives exclusively in the new ab_* columns. Do not read joint_phi_mid_lrt_p
as a beta control.

Models
------
A three-arm Q-learning / Rescorla-Wagner agent. Each arm (screen position) carries
a value Q; only the chosen arm is updated, because participants see feedback only for
the option they picked (counterfactual outcomes are recorded but never shown). Values
carry across the two reversals with no reset, so post-reversal relearning emerges from
the update dynamics rather than being imposed.

    r_t      = +1 if rewarded, -1 if loss          (rho * r; rho fixed at 1, see note)
    PE_t     = rho * r_t - Q_chosen,t
    alpha    = alpha_pos if PE_t >= 0 else alpha_neg
    Q_chosen,t+1 = Q_chosen,t + alpha * PE_t

Choices use a 3-way softmax over arm values, with an optional stickiness/perseveration
term kappa on the previously chosen arm:

    P(choose j) proportional to exp(beta * Q_j + kappa * 1[j == prev_choice])

Nested models are fit and compared by AIC/BIC:
    M1  single alpha, beta                      (k = 2)
    M2  alpha_pos, alpha_neg, beta              (k = 3)   <- primary
    M3  alpha_pos, alpha_neg, beta, kappa       (k = 4)
    M4  M2 + craving-biased positive learning rate (k = 4, adds phi)

M4 is the vigor-modulated-learning model: alpha_pos_t = alpha_pos + phi * craving_t, so
the gain learning rate moves with a trial-wise craving signal. phi > 0 means craving
speeds value updating, and at phi = 0 M4 reduces exactly to M2, so M4-vs-M2 is a clean
1-df test. M4 is fit under two craving signals, reported side by side:
  embedded_mid (PRIMARY): the embedded MID food-cue vigor carried forward across the
     bandit. Each food probe contributes z(-target_rt) (faster = more craving), held
     until the next probe. This is the designed implicit trial-wise craving readout,
     the implicit analog of the explicit craving ratings in Kulkarni et al. 2026.
  reward_trace (COMPETING): a causal leaky trace of past food-win outcomes; included so
     you can show the MID signal predicts learning beyond simple reward history.
Per-subject phi is a diagnostic; with 24 food probes per run (48 per session) it is
well powered only for large effects, so the group-level (hierarchical) phi is the
real test. Do not treat any per-subject LRT as the primary evidence for phi.

A third variant fits BOTH modulators jointly: alpha_pos_t = alpha_pos + phi_mid *
craving_mid_t + phi_rew * reward_trace_t. Its decisive output is joint_phi_mid_lrt_p,
the M4_joint-vs-reward_trace-only test, i.e. whether the embedded-MID vigor predicts
learning-rate modulation ABOVE recent reward history (reward_trace on its own captures
the ordinary reward-driven / associability learning-rate change, not craving).

Note on rho (reward sensitivity). With a fixed reward magnitude, rho and beta are not
jointly identified from choices: only their product is. This script therefore fixes
rho = 1 by default (ESTIMATE_RHO = False) and reports beta on that scale, matching the
hierarchical fit. The between-subject vigor -> parameter links (Table 4) should be
estimated with the hierarchical partial-pooling script, not a two-stage correlation of
these point estimates, which is shrinkage-attenuated; the per-subject values here are
for QC, model comparison, starting values, and the trialwise iEEG regressors.

Outputs
-------
1. bandit_rl_subject_summary.csv     (one row per run; all three models + QC)
2. bandit_rl_trialwise_values.csv    (per-trial Q, PE, alpha_used, choice probs [M2],
                                      both craving signals and the primary dynamic rate)
3. bandit_rl_data_dictionary.csv     (variable definitions)

Nothing in the source data is modified.

Dependencies
------------
    python3 -m pip install pandas numpy scipy

Run
---
    python3 fit_bandit_mid_rl_v6.py
"""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp


# =============================================================================
# USER SETTINGS
# =============================================================================

# Folder containing the task CSV files (searched recursively, like the analyzer).
DATA_DIR = Path(__file__).resolve().parent / "data"

# Where to save results. If None, saves into DATA_DIR / "analysis_output".
OUTPUT_DIR: Optional[Path] = None

# Optional: restrict to specific participant IDs. None fits every file.
INCLUDE_PARTICIPANTS: Optional[List[str]] = None

# Reward coding for the prediction error. Symmetric gain/loss with Q0 = 0 keeps
# alpha_pos / alpha_neg interpretable as gain vs loss learning rates.
REWARD_VALUE = 1.0
LOSS_VALUE = -1.0
Q_INIT = 0.0

# Reward sensitivity rho. Fixed at 1 by default: with fixed reward magnitude rho and
# beta are confounded (only rho*beta is identified), and fixing rho=1 improves beta
# recovery. Set True only if you understand the non-identifiability; a warning is flagged.
ESTIMATE_RHO = False
RHO_FIXED = 1.0

# Optimization bounds and multistart settings.
ALPHA_BOUNDS = (0.001, 0.999)
BETA_BOUNDS = (0.01, 30.0)
KAPPA_BOUNDS = (-5.0, 5.0)
RHO_BOUNDS = (0.05, 20.0)
N_RANDOM_STARTS = 24
RANDOM_SEED = 20260620

# Minimum usable bandit trials before a fit is attempted.
# v16 delivers two 100-trial runs fit jointly (one parameter set, Q reset per run),
# so a complete session is ~200 modelable trials. The threshold guards against a
# badly truncated session; 150 accepts a full session that lost some trials to
# missing data while still rejecting a single near-empty run.
MIN_TRIALS_FOR_FIT = 150

# Boundary-proximity tolerances for the "parameter railed against a bound" flag.
ALPHA_TOL = 0.005
BETA_TOL = 0.2
KAPPA_TOL = 0.05

SAVE_TRIALWISE = True                 # write per-trial Q/PE/choice-prob export (M2)

# --- M4: craving-modulated learning rate (the alpha-bias / vigor model) ------
# M4 extends M2 with alpha_pos_t = alpha_pos + phi * craving_t. phi is the vigor-
# modulated-learning parameter; at phi = 0 the model is exactly M2. Two craving signals
# are fit and reported side by side (the first is primary):
#   'embedded_mid' : embedded MID food-cue vigor, z(-target_rt) per food probe carried
#                    forward across the bandit (the designed implicit modulator).
#   'reward_trace' : causal leaky trace of past food wins (competing control regressor).
FIT_M4 = True
CRAVING_METHODS = ('embedded_mid', 'reward_trace')   # first is primary
CRAVING_DECAY = 0.6                    # leak per trial for the reward_trace signal (0-1)
PHI_BOUNDS = (-2.0, 2.0)              # bias slope bounds (craving is z-scored)
PHI_TOL = 0.02                        # boundary-proximity tolerance for phi
# Per-subject phi significance. The M4-vs-M2 likelihood-ratio test is always reported.
# A craving-shuffling permutation test is more robust; set N_PHI_PERM > 0 (e.g. 500).
N_PHI_PERM = 0
PERM_STARTS = 6                       # multistart count per permutation refit (small for speed)

# --- M5_ab: joint alpha+/beta state modulation (the specificity model) -------
# The state s_t loads on BOTH the gain learning rate and the softmax temperature,
# so a state effect on choice determinism cannot be read as a state effect on
# learning. This is the model the beta-masquerade specificity numbers refer to.
FIT_M5_AB = True

# Form of the beta modulation. 'additive' gives beta_t = beta + phi_beta*s_t,
# matching the additive-plus-clip convention already used for alpha_pos.
# 'multiplicative' gives beta_t = beta * exp(phi_beta*s_t), which cannot go
# non-positive and keeps phi_beta scale-free in beta.
# NOTE: which form block_phi_recovery_v3.py used has NOT been verified. If the
# recovery sims used the other one, phi_beta here is not on the same scale as the
# simulated value, and the 0.048 leak figure does not transfer numerically. The
# alpha-side test (ab_phi_alpha_ctrlbeta_lrt_p) is far less sensitive to this
# choice than the phi_beta point estimate is.
BETA_MOD_FORM = 'additive'            # 'additive' or 'multiplicative'

# Bounds for phi_beta. beta typically lands near 2-6, so an additive slope needs a
# wider range than the alpha-side phi, which is bounded by alpha_pos in (0,1).
PHI_BETA_BOUNDS_ADDITIVE = (-10.0, 10.0)
PHI_BETA_BOUNDS_MULT = (-2.0, 2.0)    # log-scale slope when BETA_MOD_FORM='multiplicative'
PHI_BETA_TOL = 0.1                    # boundary-proximity tolerance for phi_beta

# Column carrying the state for M5_ab. Defaults to the primary vigor modulator.
AB_MOD_COL = 'craving_embedded_mid'


# =============================================================================
# BASIC HELPERS
# =============================================================================


def file_md5(path: Path) -> str:
    """Return the MD5 hash of a file, used to flag exact-duplicate runs."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def first_nonmissing(series: Optional[pd.Series]) -> Optional[str]:
    """Return the first non-missing value of a column as a string, or None."""
    if series is None:
        return None
    s = series.dropna()
    return str(s.iloc[0]) if len(s) else None


def extract_participant_id(df: pd.DataFrame, path: Path) -> str:
    """Resolve participant ID from the participant_id column or the sub-<id> filename token."""
    if "participant_id" in df.columns:
        val = first_nonmissing(df["participant_id"])
        if val and val.lower() != "nan":
            return val
    m = re.search(r"sub-([^_]+)", path.stem, flags=re.IGNORECASE)
    return m.group(1) if m else path.stem


def extract_session(df: pd.DataFrame, path: Path) -> Optional[str]:
    """Resolve session label from the session column or the ses-<id> filename token."""
    if "session" in df.columns:
        val = first_nonmissing(df["session"])
        if val and val.lower() != "nan":
            return val
    m = re.search(r"ses-([^_]+)", path.stem, flags=re.IGNORECASE)
    return m.group(1) if m else None


def safe_mean(x: pd.Series) -> float:
    """Mean of a coerced-numeric series, NaN if empty."""
    v = pd.to_numeric(x, errors="coerce").dropna()
    return float(v.mean()) if len(v) else float("nan")


def safe_median(x: pd.Series) -> float:
    """Median of a coerced-numeric series, NaN if empty."""
    v = pd.to_numeric(x, errors="coerce").dropna()
    return float(v.median()) if len(v) else float("nan")


# =============================================================================
# TRIAL PREPARATION
# =============================================================================


def _craving_reward_trace(rewards: np.ndarray, runs: np.ndarray = None) -> np.ndarray:
    """Competing modulator: causal leaky accumulator of PAST food-win outcomes
    (c_t depends only on trials before t). Zeros if flat.

    v16: the accumulator RESETS at each run boundary, since value and reward history
    do not carry across the break. Pass runs=None for single-run (v15) files.

    KNOWN DEVIATION (documented, not fixed): the accumulator resets per run but the
    z-score below is taken over BOTH runs pooled, not within run. Earlier versions of
    this docstring claimed within-run standardization; the code has never done that.
    Consequence: a run-level shift in reward rate (fatigue, practice) survives as a
    sustained between-run offset rather than being removed. The same pooling applies
    to the vigor modulator in _craving_embedded_mid."""
    n = len(rewards)
    c = np.zeros(n, dtype=float)
    acc = 0.0
    for i in range(n):
        if runs is not None and i > 0 and runs[i] != runs[i - 1]:
            acc = 0.0                      # reset reward history at run boundary
        c[i] = acc                         # state entering trial i (past outcomes only)
        acc = CRAVING_DECAY * acc + float(rewards[i])
    sd = c.std()
    return (c - c.mean()) / sd if sd > 0 else np.zeros(n)


def _craving_embedded_mid(df_full: pd.DataFrame, trials: pd.DataFrame) -> np.ndarray:
    """Primary modulator: the embedded MID food-cue vigor carried forward across the
    bandit. Each food probe contributes z(-target_rt) (faster = more craving); every
    bandit trial takes the value of the most recent PRECEDING food probe (causal), 0
    before the first probe, then z-scored. Zeros if fewer than 2 usable food probes."""
    fp = df_full[df_full['trial_type'] == 'bonus_food'].copy()
    fp['target_rt_ms'] = pd.to_numeric(fp.get('target_rt_ms'), errors='coerce')
    fp['position_in_bandit_stream'] = pd.to_numeric(fp.get('position_in_bandit_stream'), errors='coerce')
    # Run label on probes and trials; default to 1 for pre-v16 files.
    fp['run'] = (pd.to_numeric(fp.get('run'), errors='coerce').fillna(1).astype(int)
                 if 'run' in fp.columns else 1)
    fp = fp.dropna(subset=['target_rt_ms', 'position_in_bandit_stream'])
    T = pd.to_numeric(trials['trial'], errors='coerce').to_numpy()
    R = trials['run'].to_numpy() if 'run' in trials.columns else np.ones(len(T), dtype=int)
    if len(fp) < 2 or fp['target_rt_ms'].std() == 0:
        return np.zeros(len(T))
    # Sign-flipped z(RT) over ALL probes: faster = higher craving, on a common scale.
    rt = fp['target_rt_ms'].to_numpy()
    fp = fp.assign(cp=-(rt - rt.mean()) / rt.std())
    # v16: carry forward WITHIN run, since position and trial index both reset each
    # run. A run-1 probe never attaches to a run-2 trial.
    crav = np.zeros(len(T))
    for i in range(len(T)):
        prior = fp[(fp['run'] == R[i]) & (fp['position_in_bandit_stream'] < T[i])]
        crav[i] = prior['cp'].to_numpy()[-1] if len(prior) else 0.0
    sd = crav.std()
    return (crav - crav.mean()) / sd if sd > 0 else np.zeros(len(T))


def prepare_bandit_trials(df: pd.DataFrame) -> pd.DataFrame:
    """Return usable bandit trials in presentation order, with a 0-indexed arm column.

    A trial is modelable if it has a 1..3 position choice and a reward/loss outcome.
    Bonus probes are dropped; their interleaving does not enter the value updates.
    """
    if "trial_type" not in df.columns:
        return pd.DataFrame()
    b = df[df["trial_type"] == "bandit"].copy()
    if not len(b):
        return pd.DataFrame()
    b["trial"] = pd.to_numeric(b["trial"], errors="coerce")
    b["choice"] = pd.to_numeric(b["choice"], errors="coerce")
    # v16: trial index resets each run, so sort by run THEN trial. Files without a
    # run column (v15 and earlier) get run=1 so they sort and fit exactly as before.
    if "run" in b.columns:
        b["run"] = pd.to_numeric(b["run"], errors="coerce").fillna(1).astype(int)
    else:
        b["run"] = 1
    b = b.sort_values(["run", "trial"])
    b = b[b["choice"].isin([1, 2, 3]) & b["outcome"].astype(str).isin(["reward", "loss"])].copy()
    if not len(b):
        return pd.DataFrame()
    b["arm"] = b["choice"].astype(int) - 1                       # 0,1,2 arm index
    b["reward"] = (b["outcome"].astype(str) == "reward").astype(int)
    b["rt"] = pd.to_numeric(b.get("rt_s"), errors="coerce")
    b["is_optimal"] = pd.to_numeric(b.get("is_optimal"), errors="coerce")
    b["swap_count"] = pd.to_numeric(b.get("swap_count"), errors="coerce")
    return b.reset_index(drop=True)


# =============================================================================
# MODEL CORE (shared negative log-likelihood and replay)
# =============================================================================


def _unpack(params: np.ndarray, model: str) -> Dict[str, float]:
    """Map a parameter vector to named parameters for the requested model.

    rho is appended last when ESTIMATE_RHO is True; otherwise it is held at RHO_FIXED.
    """
    p = list(params)
    out: Dict[str, float] = {}
    if model == "M1":
        out["alpha_pos"] = out["alpha_neg"] = p[0]
        out["beta"] = p[1]
        out["kappa"] = 0.0
        rest = p[2:]
    elif model == "M2":
        out["alpha_pos"], out["alpha_neg"], out["beta"] = p[0], p[1], p[2]
        out["kappa"] = 0.0
        rest = p[3:]
    else:  # M3
        out["alpha_pos"], out["alpha_neg"], out["beta"], out["kappa"] = p[0], p[1], p[2], p[3]
        rest = p[4:]
    out["rho"] = rest[0] if (ESTIMATE_RHO and rest) else RHO_FIXED
    return out


def neg_log_likelihood(params: np.ndarray, data: Tuple[np.ndarray, np.ndarray], model: str) -> float:
    """Three-arm softmax Q-learning NLL on preextracted (arms, rewards) arrays.

    Written as a scalar inner loop with an inline length-3 log-sum-exp, because the
    optimizer evaluates this thousands of times and per-trial numpy/scipy overhead dominates.
    """
    pr = _unpack(params, model)
    apos, aneg, beta, kappa, rho = pr["alpha_pos"], pr["alpha_neg"], pr["beta"], pr["kappa"], pr["rho"]
    if not (0 < apos < 1 and 0 < aneg < 1 and beta > 0 and rho > 0):
        return 1e12                                              # guard outside-bounds probes

    arms, rews, runs = data
    q0 = q1 = q2 = Q_INIT
    prev = -1
    nll = 0.0
    for i in range(arms.shape[0]):
        if i > 0 and runs[i] != runs[i - 1]:
            # v16 two-run split: value estimates and choice history reset at each run
            # boundary (fresh acquisition), while the fitted parameters stay shared.
            q0 = q1 = q2 = Q_INIT
            prev = -1
        c = arms[i]
        u0, u1, u2 = beta * q0, beta * q1, beta * q2
        if prev == 0:
            u0 += kappa
        elif prev == 1:
            u1 += kappa
        elif prev == 2:
            u2 += kappa
        m = u0 if (u0 >= u1 and u0 >= u2) else (u1 if u1 >= u2 else u2)
        denom = math.exp(u0 - m) + math.exp(u1 - m) + math.exp(u2 - m)
        uc = u0 if c == 0 else (u1 if c == 1 else u2)
        nll -= (uc - m - math.log(denom))                       # log P(observed choice)
        r = REWARD_VALUE if rews[i] == 1 else LOSS_VALUE
        if c == 0:
            pe = rho * r - q0
            q0 += (apos if pe >= 0 else aneg) * pe
        elif c == 1:
            pe = rho * r - q1
            q1 += (apos if pe >= 0 else aneg) * pe
        else:
            pe = rho * r - q2
            q2 += (apos if pe >= 0 else aneg) * pe
        prev = c
    return float(nll)


def replay(params: np.ndarray, trials: pd.DataFrame, model: str) -> Tuple[List[dict], np.ndarray]:
    """Replay trials at given parameters; return per-trial rows and final Q-values.

    Per-trial fields (pre-update Q, PE, alpha used, predicted choice probabilities) are
    the regressors for iEEG alignment, e.g. feedback-locked reward prediction error.
    """
    pr = _unpack(params, model)
    apos, aneg, beta, kappa, rho = pr["alpha_pos"], pr["alpha_neg"], pr["beta"], pr["kappa"], pr["rho"]
    q = np.full(3, Q_INIT, dtype=float)
    prev = -1
    rows: List[dict] = []
    eps = 1e-12
    runs = trials["run"].to_numpy() if "run" in trials.columns else None
    prev_run = None
    for _, t in trials.iterrows():
        if runs is not None:
            this_run = t["run"]
            if prev_run is not None and this_run != prev_run:
                q = np.full(3, Q_INIT, dtype=float)      # reset value at run boundary
                prev = -1
            prev_run = this_run
        c = int(t["arm"])
        util = beta * q.copy()
        if prev >= 0:
            util[prev] += kappa
        logp = util - logsumexp(util)
        p = np.exp(logp)
        r = REWARD_VALUE if t["reward"] == 1 else LOSS_VALUE
        pe = rho * r - q[c]
        a_used = apos if pe >= 0 else aneg
        rows.append({
            "trial": int(t["trial"]) if pd.notna(t["trial"]) else np.nan,
            "swap_count": int(t["swap_count"]) if pd.notna(t["swap_count"]) else np.nan,
            "choice_arm": c + 1, "reward": int(t["reward"]), "is_optimal": t.get("is_optimal"),
            "q_arm1_pre": q[0], "q_arm2_pre": q[1], "q_arm3_pre": q[2],
            "q_chosen_pre": q[c], "p_choose_arm1": p[0], "p_choose_arm2": p[1],
            "p_choose_arm3": p[2], "p_chosen": max(min(float(p[c]), 1 - eps), eps),
            "prediction_error": pe, "alpha_used": a_used, "rt": t.get("rt"),
        })
        q[c] += a_used * pe
        prev = c
    return rows, q


# =============================================================================
# FITTING
# =============================================================================


def _start_grid(model: str, rng: np.random.Generator) -> List[np.ndarray]:
    """Build fixed plus random starting points for one model's free parameters."""
    fixed = {
        "M1": [[0.2, 3.0], [0.1, 1.0], [0.5, 5.0]],
        "M2": [[0.2, 0.2, 3.0], [0.1, 0.3, 3.0], [0.3, 0.1, 3.0], [0.05, 0.05, 1.0]],
        "M3": [[0.2, 0.2, 3.0, 0.0], [0.1, 0.3, 3.0, 0.5], [0.3, 0.1, 3.0, -0.5]],
    }[model]
    starts = [np.array(s, dtype=float) for s in fixed]
    for _ in range(N_RANDOM_STARTS):
        if model == "M1":
            s = [rng.uniform(*ALPHA_BOUNDS), rng.uniform(*BETA_BOUNDS)]
        elif model == "M2":
            s = [rng.uniform(*ALPHA_BOUNDS), rng.uniform(*ALPHA_BOUNDS), rng.uniform(*BETA_BOUNDS)]
        else:
            s = [rng.uniform(*ALPHA_BOUNDS), rng.uniform(*ALPHA_BOUNDS),
                 rng.uniform(*BETA_BOUNDS), rng.uniform(*KAPPA_BOUNDS)]
        if ESTIMATE_RHO:
            s.append(rng.uniform(*RHO_BOUNDS))
        starts.append(np.array(s, dtype=float))
    return starts


def _bounds_for(model: str) -> List[Tuple[float, float]]:
    """Optimization bounds for one model's free parameters (rho appended if estimated)."""
    base = {
        "M1": [ALPHA_BOUNDS, BETA_BOUNDS],
        "M2": [ALPHA_BOUNDS, ALPHA_BOUNDS, BETA_BOUNDS],
        "M3": [ALPHA_BOUNDS, ALPHA_BOUNDS, BETA_BOUNDS, KAPPA_BOUNDS],
    }[model]
    return base + ([RHO_BOUNDS] if ESTIMATE_RHO else [])


def fit_model(trials: pd.DataFrame, model: str) -> dict:
    """Fit one model by multistart L-BFGS-B; return parameters and fit indices."""
    n = len(trials)
    k = {"M1": 2, "M2": 3, "M3": 4}[model] + (1 if ESTIMATE_RHO else 0)
    if n < MIN_TRIALS_FOR_FIT:
        return {"model": model, "n_trials": n, "k": k, "optimizer_success": False,
                "optimizer_message": "Too few trials"}

    rng = np.random.default_rng(RANDOM_SEED + hash(model) % 1000)
    bounds = _bounds_for(model)
    data = (trials["arm"].to_numpy(), trials["reward"].to_numpy(),
            trials["run"].to_numpy())   # run array drives per-run Q reset
    best = None
    for x0 in _start_grid(model, rng):
        res = minimize(neg_log_likelihood, x0=x0, args=(data, model),
                       method="L-BFGS-B", bounds=bounds, options={"maxiter": 1000})
        if best is None or res.fun < best.fun:
            best = res

    pr = _unpack(best.x, model)
    nll = float(best.fun)
    aic = 2 * k + 2 * nll
    bic = k * math.log(n) + 2 * nll
    # McFadden pseudo-R^2 against a uniform 3-way guess; predicted-choice accuracy.
    chance = n * math.log(3.0)
    rows, _ = replay(best.x, trials, model)
    ps = np.array([r["p_chosen"] for r in rows])
    argmax_hit = np.mean([
        np.argmax([r["p_choose_arm1"], r["p_choose_arm2"], r["p_choose_arm3"]]) == (r["choice_arm"] - 1)
        for r in rows])
    return {
        "model": model, "n_trials": n, "k": k,
        "alpha_pos": pr["alpha_pos"], "alpha_neg": pr["alpha_neg"],
        "alpha_pos_minus_alpha_neg": pr["alpha_pos"] - pr["alpha_neg"],
        "beta": pr["beta"], "kappa": pr["kappa"], "rho": pr["rho"],
        "neg_log_likelihood": nll, "aic": aic, "bic": bic,
        "pseudo_r2": 1.0 - nll / chance if chance > 0 else np.nan,
        "pred_accuracy": float(argmax_hit), "mean_p_chosen": float(ps.mean()),
        "optimizer_success": bool(best.success), "optimizer_message": str(best.message),
    }


def boundary_flags(fit: dict, model: str) -> List[str]:
    """Flag free parameters sitting within tolerance of an optimization bound."""
    flags = []

    def at(name, val, lo, hi, tol):
        if val is None or pd.isna(val):
            return
        if val <= lo + tol:
            flags.append(f"{model}_{name}_at_lower_bound")
        elif val >= hi - tol:
            flags.append(f"{model}_{name}_at_upper_bound")

    if model == "M1":
        at("alpha", fit.get("alpha_pos"), *ALPHA_BOUNDS, ALPHA_TOL)
    else:
        at("alpha_pos", fit.get("alpha_pos"), *ALPHA_BOUNDS, ALPHA_TOL)
        at("alpha_neg", fit.get("alpha_neg"), *ALPHA_BOUNDS, ALPHA_TOL)
    at("beta", fit.get("beta"), *BETA_BOUNDS, BETA_TOL)
    if model == "M3":
        at("kappa", fit.get("kappa"), *KAPPA_BOUNDS, KAPPA_TOL)
    if model.startswith("M4"):
        at("phi", fit.get("phi"), *PHI_BOUNDS, PHI_TOL)
    if ESTIMATE_RHO:
        at("rho", fit.get("rho"), *RHO_BOUNDS, 0.1)
    return flags


# =============================================================================
# M4: CRAVING-MODULATED LEARNING RATE (alpha-bias model)
# =============================================================================


def neg_log_likelihood_m4(params, data):
    """M4 NLL: the M2 choice model with a craving-biased positive learning rate,
    alpha_pos_t = clip(alpha_pos + phi * craving_t). data = (arms, rewards, craving)."""
    apos0, aneg, beta, phi = params[0], params[1], params[2], params[3]
    rho = params[4] if ESTIMATE_RHO else RHO_FIXED
    if not (0 < apos0 < 1 and 0 < aneg < 1 and beta > 0 and rho > 0):
        return 1e12
    arms, rews, crav, runs = data
    q0 = q1 = q2 = Q_INIT
    nll = 0.0
    for i in range(arms.shape[0]):
        if i > 0 and runs[i] != runs[i - 1]:
            q0 = q1 = q2 = Q_INIT                         # reset value at run boundary
        c = arms[i]
        u0, u1, u2 = beta * q0, beta * q1, beta * q2         # no stickiness in M4 (M2 + phi)
        m = u0 if (u0 >= u1 and u0 >= u2) else (u1 if u1 >= u2 else u2)
        denom = math.exp(u0 - m) + math.exp(u1 - m) + math.exp(u2 - m)
        uc = u0 if c == 0 else (u1 if c == 1 else u2)
        nll -= (uc - m - math.log(denom))
        r = REWARD_VALUE if rews[i] == 1 else LOSS_VALUE
        apos_t = apos0 + phi * crav[i]                       # craving-biased gain learning rate
        if apos_t < 1e-6:
            apos_t = 1e-6
        elif apos_t > 1 - 1e-6:
            apos_t = 1 - 1e-6
        if c == 0:
            pe = rho * r - q0; q0 += (apos_t if pe >= 0 else aneg) * pe
        elif c == 1:
            pe = rho * r - q1; q1 += (apos_t if pe >= 0 else aneg) * pe
        else:
            pe = rho * r - q2; q2 += (apos_t if pe >= 0 else aneg) * pe
    return float(nll)


def replay_m4(params, trials, craving_col='craving_embedded_mid'):
    """Replay M4 to expose the per-trial dynamic gain learning rate for iEEG use."""
    apos0, aneg, beta, phi = params[0], params[1], params[2], params[3]
    rho = params[4] if ESTIMATE_RHO else RHO_FIXED
    q = np.full(3, Q_INIT, dtype=float)
    arms = trials['arm'].to_numpy(); rews = trials['reward'].to_numpy()
    crav = trials[craving_col].to_numpy()
    runs = trials["run"].to_numpy() if "run" in trials.columns else None
    rows = []
    for i in range(len(arms)):
        if runs is not None and i > 0 and runs[i] != runs[i - 1]:
            q = np.full(3, Q_INIT, dtype=float)          # reset value at run boundary
        c = int(arms[i])
        r = REWARD_VALUE if rews[i] == 1 else LOSS_VALUE
        pe = rho * r - q[c]
        apos_t = min(max(apos0 + phi * crav[i], 1e-6), 1 - 1e-6)
        a_used = apos_t if pe >= 0 else aneg
        rows.append({'craving_t': float(crav[i]), 'alpha_pos_dynamic_m4': float(apos_t)})
        q[c] += a_used * pe
    return rows


def fit_model_m4(trials, craving=None, n_random=None):
    """Fit M4 (M2 + craving-biased alpha_pos) by multistart L-BFGS-B."""
    n = len(trials)
    k = 4 + (1 if ESTIMATE_RHO else 0)
    if n < MIN_TRIALS_FOR_FIT:
        return {'model': 'M4', 'n_trials': n, 'k': k, 'optimizer_success': False,
                'optimizer_message': 'Too few trials'}
    if craving is None:
        craving = trials['craving_embedded_mid'].to_numpy()
    if n_random is None:
        n_random = N_RANDOM_STARTS
    rng = np.random.default_rng(RANDOM_SEED + 4004)
    bounds = [ALPHA_BOUNDS, ALPHA_BOUNDS, BETA_BOUNDS, PHI_BOUNDS] + ([RHO_BOUNDS] if ESTIMATE_RHO else [])
    data = (trials['arm'].to_numpy(), trials['reward'].to_numpy(),
            np.asarray(craving, dtype=float), trials['run'].to_numpy())
    starts = [np.array(s, float) for s in
              ([0.2, 0.2, 3.0, 0.0], [0.1, 0.3, 3.0, 0.3], [0.3, 0.1, 3.0, -0.3], [0.2, 0.2, 5.0, 0.0])]
    for _ in range(n_random):
        s = [rng.uniform(*ALPHA_BOUNDS), rng.uniform(*ALPHA_BOUNDS), rng.uniform(*BETA_BOUNDS), rng.uniform(*PHI_BOUNDS)]
        if ESTIMATE_RHO:
            s.append(rng.uniform(*RHO_BOUNDS))
        starts.append(np.array(s, float))
    best = None
    for x0 in starts:
        res = minimize(neg_log_likelihood_m4, x0=x0, args=(data,), method='L-BFGS-B',
                       bounds=bounds, options={'maxiter': 1000})
        if best is None or res.fun < best.fun:
            best = res
    nll = float(best.fun)
    aic = 2 * k + 2 * nll
    bic = k * math.log(n) + 2 * nll
    chance = n * math.log(3.0)
    x = best.x
    return {'model': 'M4', 'n_trials': n, 'k': k,
            'alpha_pos': float(x[0]), 'alpha_neg': float(x[1]), 'beta': float(x[2]), 'phi': float(x[3]),
            'rho': (float(x[4]) if ESTIMATE_RHO else RHO_FIXED),
            'neg_log_likelihood': nll, 'aic': aic, 'bic': bic,
            'pseudo_r2': 1.0 - nll / chance if chance > 0 else np.nan,
            'params': x, 'optimizer_success': bool(best.success), 'optimizer_message': str(best.message)}


def phi_permutation_p(trials, phi_abs_obs, craving):
    """Permutation p-value for phi: shuffle the craving signal (breaking any craving->
    learning coupling), refit M4, and compare |phi_perm| to the observed |phi|."""
    rng = np.random.default_rng(RANDOM_SEED + 909)
    crav = np.asarray(craving, dtype=float)
    hits = 0
    for _ in range(N_PHI_PERM):
        f = fit_model_m4(trials, craving=rng.permutation(crav), n_random=PERM_STARTS)
        if pd.notna(f.get('phi')) and abs(f['phi']) >= phi_abs_obs:
            hits += 1
    return (hits + 1) / (N_PHI_PERM + 1)


def neg_log_likelihood_m4_joint(params, data):
    """M4_joint NLL: two-modulator gain learning rate,
    alpha_pos_t = clip(alpha_pos + phi_mid*cmid_t + phi_rew*crew_t).
    data = (arms, rewards, craving_mid, craving_rew)."""
    apos0, aneg, beta, phi_mid, phi_rew = params[0], params[1], params[2], params[3], params[4]
    rho = params[5] if ESTIMATE_RHO else RHO_FIXED
    if not (0 < apos0 < 1 and 0 < aneg < 1 and beta > 0 and rho > 0):
        return 1e12
    arms, rews, cmid, crew, runs = data
    q0 = q1 = q2 = Q_INIT
    nll = 0.0
    for i in range(arms.shape[0]):
        if i > 0 and runs[i] != runs[i - 1]:
            q0 = q1 = q2 = Q_INIT                         # reset value at run boundary
        c = arms[i]
        u0, u1, u2 = beta * q0, beta * q1, beta * q2
        m = u0 if (u0 >= u1 and u0 >= u2) else (u1 if u1 >= u2 else u2)
        denom = math.exp(u0 - m) + math.exp(u1 - m) + math.exp(u2 - m)
        uc = u0 if c == 0 else (u1 if c == 1 else u2)
        nll -= (uc - m - math.log(denom))
        r = REWARD_VALUE if rews[i] == 1 else LOSS_VALUE
        apos_t = apos0 + phi_mid * cmid[i] + phi_rew * crew[i]
        if apos_t < 1e-6:
            apos_t = 1e-6
        elif apos_t > 1 - 1e-6:
            apos_t = 1 - 1e-6
        if c == 0:
            pe = rho * r - q0; q0 += (apos_t if pe >= 0 else aneg) * pe
        elif c == 1:
            pe = rho * r - q1; q1 += (apos_t if pe >= 0 else aneg) * pe
        else:
            pe = rho * r - q2; q2 += (apos_t if pe >= 0 else aneg) * pe
    return float(nll)


def fit_model_m4_joint(trials, cmid=None, crew=None, n_random=None):
    """Fit M4_joint (M2 + both craving modulators on alpha_pos) by multistart L-BFGS-B."""
    n = len(trials)
    k = 5 + (1 if ESTIMATE_RHO else 0)
    if n < MIN_TRIALS_FOR_FIT:
        return {'model': 'M4_joint', 'n_trials': n, 'k': k, 'optimizer_success': False,
                'optimizer_message': 'Too few trials'}
    if cmid is None:
        cmid = trials['craving_embedded_mid'].to_numpy()
    if crew is None:
        crew = trials['craving_reward_trace'].to_numpy()
    if n_random is None:
        n_random = N_RANDOM_STARTS
    rng = np.random.default_rng(RANDOM_SEED + 4005)
    bounds = [ALPHA_BOUNDS, ALPHA_BOUNDS, BETA_BOUNDS, PHI_BOUNDS, PHI_BOUNDS] + ([RHO_BOUNDS] if ESTIMATE_RHO else [])
    data = (trials['arm'].to_numpy(), trials['reward'].to_numpy(),
            np.asarray(cmid, dtype=float), np.asarray(crew, dtype=float),
            trials['run'].to_numpy())
    starts = [np.array(s, float) for s in
              ([0.2, 0.2, 3.0, 0.0, 0.0], [0.1, 0.3, 3.0, 0.2, 0.0],
               [0.3, 0.1, 3.0, 0.0, 0.2], [0.2, 0.2, 5.0, 0.0, 0.0])]
    for _ in range(n_random):
        s = [rng.uniform(*ALPHA_BOUNDS), rng.uniform(*ALPHA_BOUNDS), rng.uniform(*BETA_BOUNDS),
             rng.uniform(*PHI_BOUNDS), rng.uniform(*PHI_BOUNDS)]
        if ESTIMATE_RHO:
            s.append(rng.uniform(*RHO_BOUNDS))
        starts.append(np.array(s, float))
    best = None
    for x0 in starts:
        res = minimize(neg_log_likelihood_m4_joint, x0=x0, args=(data,), method='L-BFGS-B',
                       bounds=bounds, options={'maxiter': 1000})
        if best is None or res.fun < best.fun:
            best = res
    nll = float(best.fun)
    aic = 2 * k + 2 * nll
    bic = k * math.log(n) + 2 * nll
    chance = n * math.log(3.0)
    x = best.x
    return {'model': 'M4_joint', 'n_trials': n, 'k': k,
            'alpha_pos': float(x[0]), 'alpha_neg': float(x[1]), 'beta': float(x[2]),
            'phi_mid': float(x[3]), 'phi_rew': float(x[4]),
            'rho': (float(x[5]) if ESTIMATE_RHO else RHO_FIXED),
            'neg_log_likelihood': nll, 'aic': aic, 'bic': bic,
            'pseudo_r2': 1.0 - nll / chance if chance > 0 else np.nan,
            'params': x, 'optimizer_success': bool(best.success), 'optimizer_message': str(best.message)}


# =============================================================================
# BONUS-OUTCOME CONTROL (added): control phi_mid for the RT-contingent bonus, not
# just for reward history. The embedded-MID vigor signal is correlated with the +15
# a fast response earns; these pieces add a decaying bonus trace and the decisive
# test of phi_mid above BOTH reward history and the bonus event (the "joint3" test).
# =============================================================================

def _craving_bonus_trace(df_full, trials, decay=None):
    """Competing control for the RT-contingent bonus: a causal leaky accumulator of past
    food-probe HITS (the +15 events), read at each bandit trial and reset at run
    boundaries, then z-scored. Its per-trial decay gives it a different shape from the
    constant carried-forward vigor step (pilot trial-level overlap ~0.04 at decay 0.6)."""
    if decay is None:
        decay = CRAVING_DECAY
    fp = df_full[df_full['trial_type'] == 'bonus_food'].copy()
    fp['target_hit'] = pd.to_numeric(fp.get('target_hit'), errors='coerce')
    fp['position_in_bandit_stream'] = pd.to_numeric(fp.get('position_in_bandit_stream'), errors='coerce')
    fp['run'] = (pd.to_numeric(fp.get('run'), errors='coerce').fillna(1).astype(int)
                 if 'run' in fp.columns else 1)
    fp = fp.dropna(subset=['target_hit', 'position_in_bandit_stream'])
    T = pd.to_numeric(trials['trial'], errors='coerce').to_numpy()
    R = trials['run'].to_numpy() if 'run' in trials.columns else np.ones(len(T), int)
    if len(fp) < 2:
        return np.zeros(len(T))
    pos = fp['position_in_bandit_stream'].to_numpy(float)
    hit = fp['target_hit'].to_numpy(float)
    runs_p = fp['run'].to_numpy()
    out = np.zeros(len(T))
    for run in np.unique(R):
        idx = np.where(R == run)[0]
        order = idx[np.argsort(T[idx])]         # bandit trials in order within run
        acc, prev_t = 0.0, -np.inf
        for j in order:
            # Fold each probe-hit in once, at the first bandit trial after it; >= on the
            # lower edge catches integer positions a strict > would skip.
            new = np.where((runs_p == run) & (pos >= prev_t) & (pos < T[j]))[0]
            for k in new:
                acc = decay * acc + hit[k]
            out[j] = acc                         # state entering this trial (past only)
            acc = decay * acc                    # decay one step toward the next trial
            prev_t = T[j]
    sd = out.std()
    return (out - out.mean()) / sd if sd > 0 else np.zeros(len(T))


def neg_log_likelihood_m4_multi(params, data):
    """NLL for alpha_pos_t = clip(alpha_pos + sum_k phi_k * C[:,k]). Generalizes the
    two-modulator joint to any number of modulators; verified to reproduce
    neg_log_likelihood_m4_joint exactly for the two-modulator case.
    params = [apos0, aneg, beta, phi_1..phi_m, (rho)]; data = (arms, rews, C, runs)."""
    arms, rews, C, runs = data
    m = C.shape[1]
    apos0, aneg, beta = params[0], params[1], params[2]
    phis = np.asarray(params[3:3 + m], dtype=float)
    rho = params[3 + m] if ESTIMATE_RHO else RHO_FIXED
    if not (0 < apos0 < 1 and 0 < aneg < 1 and beta > 0 and rho > 0):
        return 1e12
    q0 = q1 = q2 = Q_INIT
    nll = 0.0
    for i in range(arms.shape[0]):
        if i > 0 and runs[i] != runs[i - 1]:
            q0 = q1 = q2 = Q_INIT                 # reset value at run boundary
        c = arms[i]
        u0, u1, u2 = beta * q0, beta * q1, beta * q2
        mx = u0 if (u0 >= u1 and u0 >= u2) else (u1 if u1 >= u2 else u2)
        denom = math.exp(u0 - mx) + math.exp(u1 - mx) + math.exp(u2 - mx)
        uc = u0 if c == 0 else (u1 if c == 1 else u2)
        nll -= (uc - mx - math.log(denom))
        r = REWARD_VALUE if rews[i] == 1 else LOSS_VALUE
        apos_t = apos0 + float(C[i] @ phis)
        if apos_t < 1e-6:
            apos_t = 1e-6
        elif apos_t > 1 - 1e-6:
            apos_t = 1 - 1e-6
        if c == 0:
            pe = rho * r - q0; q0 += (apos_t if pe >= 0 else aneg) * pe
        elif c == 1:
            pe = rho * r - q1; q1 += (apos_t if pe >= 0 else aneg) * pe
        else:
            pe = rho * r - q2; q2 += (apos_t if pe >= 0 else aneg) * pe
    return float(nll)


def fit_model_m4_multi(trials, mod_cols, n_random=None):
    """Fit the N-modulator joint model by multistart L-BFGS-B. mod_cols is the list of
    trial columns loaded onto alpha_pos (order fixes the phi order in the output)."""
    n, m = len(trials), len(mod_cols)
    k = 3 + m + (1 if ESTIMATE_RHO else 0)
    if n < MIN_TRIALS_FOR_FIT:
        return {'model': f'M4_multi{m}', 'n_trials': n, 'k': k,
                'optimizer_success': False, 'optimizer_message': 'Too few trials'}
    C = np.column_stack([np.asarray(trials[c].to_numpy(), float) for c in mod_cols])
    data = (trials['arm'].to_numpy(), trials['reward'].to_numpy(), C, trials['run'].to_numpy())
    bounds = ([ALPHA_BOUNDS, ALPHA_BOUNDS, BETA_BOUNDS] + [PHI_BOUNDS] * m
              + ([RHO_BOUNDS] if ESTIMATE_RHO else []))
    rng = np.random.default_rng(RANDOM_SEED + 4006)
    n_random = N_RANDOM_STARTS if n_random is None else n_random
    starts = [np.array([0.2, 0.2, 3.0] + [0.0] * m, float),
              np.array([0.1, 0.3, 3.0] + [0.2] + [0.0] * (m - 1), float)]
    for _ in range(n_random):
        s = [rng.uniform(*ALPHA_BOUNDS), rng.uniform(*ALPHA_BOUNDS), rng.uniform(*BETA_BOUNDS)]
        s += [rng.uniform(*PHI_BOUNDS) for _ in range(m)]
        if ESTIMATE_RHO:
            s.append(rng.uniform(*RHO_BOUNDS))
        starts.append(np.array(s, float))
    best = None
    for x0 in starts:
        res = minimize(neg_log_likelihood_m4_multi, x0=x0, args=(data,),
                       method='L-BFGS-B', bounds=bounds, options={'maxiter': 1000})
        if best is None or res.fun < best.fun:
            best = res
    nll = float(best.fun); x = best.x
    return {'model': f'M4_multi{m}', 'n_trials': n, 'k': k,
            'alpha_pos': float(x[0]), 'alpha_neg': float(x[1]), 'beta': float(x[2]),
            'phis': [float(v) for v in x[3:3 + m]], 'mod_cols': list(mod_cols),
            'rho': (float(x[3 + m]) if ESTIMATE_RHO else RHO_FIXED),
            'neg_log_likelihood': nll, 'aic': 2 * k + 2 * nll,
            'bic': k * math.log(n) + 2 * nll,
            'pseudo_r2': 1.0 - nll / (n * math.log(3.0)) if n > 0 else np.nan,
            'params': x, 'optimizer_success': bool(best.success),
            'optimizer_message': str(best.message)}


# =============================================================================
# M5_ab: JOINT ALPHA+/BETA STATE MODULATION (the specificity model)
#
# Everything above modulates alpha_pos only, so a state effect that actually acts
# on choice determinism has nowhere to go except into phi. M5_ab gives it its own
# parameter. The pre-registered claim is the alpha-side LRT below: does the state
# still improve fit through alpha_pos once it is already free to act on beta?
# =============================================================================


def _phi_beta_bounds():
    """Bounds for phi_beta under the configured beta-modulation form."""
    return (PHI_BETA_BOUNDS_ADDITIVE if BETA_MOD_FORM == 'additive'
            else PHI_BETA_BOUNDS_MULT)


def alpha_pos_clip_fraction(apos0, phis, C):
    """Fraction of trials whose UNCLIPPED alpha_pos_t falls outside (0, 1).

    Diagnostic for the additive parameterization: alpha_pos_t = alpha_pos + sum
    phi_k*C_k is clipped into (1e-6, 1-1e-6), and the clip is one-sided in practice
    (only the low-alpha_pos end). A non-trivial fraction means the additive form is
    straining, the effective phi scale has become subject-dependent, and any
    between-subject sigma_phi read off these fits is contaminated by that.
    """
    a = np.asarray(apos0, float) + np.asarray(C, float) @ np.asarray(phis, float)
    return float(np.mean((a <= 1e-6) | (a >= 1 - 1e-6))) if a.size else float('nan')


def neg_log_likelihood_m5_ab(params, data):
    """NLL for simultaneous state modulation of alpha_pos and beta.

        alpha_pos_t = clip(alpha_pos + phi_alpha * s_t,  1e-6, 1-1e-6)
        beta_t      = beta + phi_beta * s_t              ('additive'), floored, or
                      beta * exp(phi_beta * s_t)         ('multiplicative')

    params = [apos0, aneg, beta, phi_alpha, phi_beta, (rho)]. A parameter is held at
    zero by giving it a (0,0) bound in the caller, which is how the reduced models
    are fit without a second likelihood function.
    data   = (arms, rews, s, runs).
    """
    apos0, aneg, beta0, phi_a, phi_b = (params[0], params[1], params[2],
                                        params[3], params[4])
    rho = params[5] if ESTIMATE_RHO else RHO_FIXED
    if not (0 < apos0 < 1 and 0 < aneg < 1 and beta0 > 0 and rho > 0):
        return 1e12
    arms, rews, s, runs = data
    beta_floor = BETA_BOUNDS[0]
    q0 = q1 = q2 = Q_INIT
    nll = 0.0
    for i in range(arms.shape[0]):
        if i > 0 and runs[i] != runs[i - 1]:
            q0 = q1 = q2 = Q_INIT                     # reset value at run boundary
        # State-dependent temperature for this trial.
        if BETA_MOD_FORM == 'multiplicative':
            beta_t = beta0 * math.exp(phi_b * s[i])
        else:
            beta_t = beta0 + phi_b * s[i]
            if beta_t < beta_floor:
                beta_t = beta_floor                   # keep the softmax well defined
        c = arms[i]
        u0, u1, u2 = beta_t * q0, beta_t * q1, beta_t * q2
        mx = u0 if (u0 >= u1 and u0 >= u2) else (u1 if u1 >= u2 else u2)
        denom = math.exp(u0 - mx) + math.exp(u1 - mx) + math.exp(u2 - mx)
        uc = u0 if c == 0 else (u1 if c == 1 else u2)
        nll -= (uc - mx - math.log(denom))
        # State-dependent gain learning rate, same clip convention as M4.
        r = REWARD_VALUE if rews[i] == 1 else LOSS_VALUE
        apos_t = apos0 + phi_a * s[i]
        if apos_t < 1e-6:
            apos_t = 1e-6
        elif apos_t > 1 - 1e-6:
            apos_t = 1 - 1e-6
        if c == 0:
            pe = rho * r - q0; q0 += (apos_t if pe >= 0 else aneg) * pe
        elif c == 1:
            pe = rho * r - q1; q1 += (apos_t if pe >= 0 else aneg) * pe
        else:
            pe = rho * r - q2; q2 += (apos_t if pe >= 0 else aneg) * pe
    return float(nll)


def fit_model_m5_ab(trials, mod_col=None, free=('alpha', 'beta'), n_random=None):
    """Fit M5_ab by multistart L-BFGS-B.

    `free` selects which state slopes are estimated; the others are pinned to zero
    via a degenerate (0,0) bound, which is what makes the three nested fits
    (both / beta-only / alpha-only) exactly comparable by likelihood ratio.
      free=('alpha','beta') -> full model, k = 5
      free=('beta',)        -> state acts on choice determinism only, k = 4
      free=('alpha',)       -> state acts on learning only, k = 4 (equals M4)
    """
    n = len(trials)
    mod_col = AB_MOD_COL if mod_col is None else mod_col
    k = 3 + len(free) + (1 if ESTIMATE_RHO else 0)
    tag = 'M5_ab_' + '_'.join(sorted(free))
    if n < MIN_TRIALS_FOR_FIT:
        return {'model': tag, 'n_trials': n, 'k': k,
                'optimizer_success': False, 'optimizer_message': 'Too few trials'}
    s = np.asarray(trials[mod_col].to_numpy(), dtype=float)
    data = (trials['arm'].to_numpy(), trials['reward'].to_numpy(), s,
            trials['run'].to_numpy())
    pa_b = PHI_BOUNDS if 'alpha' in free else (0.0, 0.0)
    pb_b = _phi_beta_bounds() if 'beta' in free else (0.0, 0.0)
    bounds = [ALPHA_BOUNDS, ALPHA_BOUNDS, BETA_BOUNDS, pa_b, pb_b] + (
        [RHO_BOUNDS] if ESTIMATE_RHO else [])
    # Seed offset differs from every other fitter so the multistart draws are
    # independent of the M4 family and results stay reproducible.
    rng = np.random.default_rng(RANDOM_SEED + 5007)
    n_random = N_RANDOM_STARTS if n_random is None else n_random
    mid = lambda b: 0.0 if b[0] == b[1] else 0.0     # both slopes start at no effect
    starts = [np.array([0.2, 0.2, 3.0, mid(pa_b), mid(pb_b)], float),
              np.array([0.1, 0.3, 3.0, mid(pa_b), mid(pb_b)], float),
              np.array([0.3, 0.1, 5.0, mid(pa_b), mid(pb_b)], float)]
    for _ in range(n_random):
        st = [rng.uniform(*ALPHA_BOUNDS), rng.uniform(*ALPHA_BOUNDS),
              rng.uniform(*BETA_BOUNDS),
              rng.uniform(*pa_b) if pa_b[0] != pa_b[1] else 0.0,
              rng.uniform(*pb_b) if pb_b[0] != pb_b[1] else 0.0]
        if ESTIMATE_RHO:
            st.append(rng.uniform(*RHO_BOUNDS))
        starts.append(np.array(st, float))
    best = None
    for x0 in starts:
        res = minimize(neg_log_likelihood_m5_ab, x0=x0, args=(data,),
                       method='L-BFGS-B', bounds=bounds, options={'maxiter': 1000})
        if best is None or res.fun < best.fun:
            best = res
    nll = float(best.fun); x = best.x
    return {'model': tag, 'n_trials': n, 'k': k, 'free': tuple(free),
            'alpha_pos': float(x[0]), 'alpha_neg': float(x[1]), 'beta': float(x[2]),
            'phi_alpha': float(x[3]), 'phi_beta': float(x[4]),
            'rho': (float(x[5]) if ESTIMATE_RHO else RHO_FIXED),
            'beta_mod_form': BETA_MOD_FORM,
            'alpha_pos_clip_frac': alpha_pos_clip_fraction(
                float(x[0]), [float(x[3])], s.reshape(-1, 1)),
            'neg_log_likelihood': nll, 'aic': 2 * k + 2 * nll,
            'bic': k * math.log(n) + 2 * nll,
            'pseudo_r2': 1.0 - nll / (n * math.log(3.0)) if n > 0 else np.nan,
            'params': x, 'optimizer_success': bool(best.success),
            'optimizer_message': str(best.message)}


def phi_alpha_permutation_p(trials, phi_alpha_abs_obs, mod_col=None, n_perm=None):
    """Permutation p for phi_alpha in the FULL alpha+beta model.

    Permuting the trial-level state vector element-wise destroys its step structure
    (the carried-forward signal is constant between probes), so the null draws are
    less autocorrelated than the observed regressor. A probe-level permutation that
    rebuilds the carry-forward epochs is the correct version and is NOT implemented
    here. Treat this as a placeholder consistent with the existing
    phi_permutation_p, not as the inference the pre-registration should cite.

    DIRECTION OF THE ERROR IS UNSETTLED. An earlier version of this docstring stated
    that the mismatch makes the test anti-conservative. A 200-dataset null simulation
    (phi_true = 0, two 100-trial runs, blocky regressor on 4-trial blocks) gave |phi|
    q95 = 0.133 under the true blocky regressor against 0.156 under the shuffled one,
    i.e. the shuffled null was slightly WIDER, which is mildly conservative. Two
    effects oppose each other: shuffling inflates the effective sample size (narrows
    the null) while flattening the phi likelihood (widens it). That simulation used
    fixed 4-trial blocks and 4 multistarts against the 24 used here, so neither
    direction is established. The case for the probe-level version is correctness,
    not a suspicion that these p-values are inflated.
    """
    n_perm = N_PHI_PERM if n_perm is None else n_perm
    if not n_perm:
        return np.nan
    rng = np.random.default_rng(RANDOM_SEED + 5008)
    mod_col = AB_MOD_COL if mod_col is None else mod_col
    base = trials[mod_col].to_numpy(dtype=float)
    t2 = trials.copy()
    hits = 0
    for _ in range(n_perm):
        t2[mod_col] = rng.permutation(base)
        f = fit_model_m5_ab(t2, mod_col=mod_col, free=('alpha', 'beta'),
                            n_random=PERM_STARTS)
        if pd.notna(f.get('phi_alpha')) and abs(f['phi_alpha']) >= phi_alpha_abs_obs:
            hits += 1
    return (hits + 1) / (n_perm + 1)


# =============================================================================
# PER-FILE ANALYSIS
# =============================================================================


def analyze_one_file(path: Path, md5: str) -> Tuple[dict, List[dict]]:
    """Fit M1/M2/M3 (and optionally M4) for one run; return the summary row and the
    M2 trialwise rows augmented with craving_t and the M4 dynamic learning rate."""
    df = pd.read_csv(path)
    pid = extract_participant_id(df, path)
    if INCLUDE_PARTICIPANTS is not None and pid not in [str(x) for x in INCLUDE_PARTICIPANTS]:
        return {}, []
    session = extract_session(df, path)
    trials = prepare_bandit_trials(df)

    summary: Dict[str, object] = {
        "participant_id": pid, "session": session, "file_name": path.name,
        "file_path": str(path), "file_md5": md5,
        "task_version": first_nonmissing(df["task_version"]) if "task_version" in df.columns else None,
        "seed": first_nonmissing(df["seed"]) if "seed" in df.columns else None,
        "n_model_trials": int(len(trials)),
        "rho_estimated": ESTIMATE_RHO,
    }
    if len(trials) < MIN_TRIALS_FOR_FIT:
        summary["optimizer_success"] = False
        summary["fit_note"] = "too few usable bandit trials"
        return summary, []

    # Trial-wise craving signals for M4 (both fit below; embedded_mid is primary).
    trials = trials.copy()
    trials["craving_embedded_mid"] = _craving_embedded_mid(df, trials)
    trials["craving_reward_trace"] = _craving_reward_trace(
        trials["reward"].to_numpy(), trials["run"].to_numpy())

    fits = {m: fit_model(trials, m) for m in ("M1", "M2", "M3")}
    bflags = sum((boundary_flags(fits[m], m) for m in ("M1", "M2", "M3")), [])

    # Primary parameters come from M2 (dual alpha), the preregistered learning model.
    m2 = fits["M2"]
    for key in ["alpha_pos", "alpha_neg", "alpha_pos_minus_alpha_neg", "beta", "rho",
                "neg_log_likelihood", "aic", "bic", "pseudo_r2", "pred_accuracy",
                "mean_p_chosen", "optimizer_success", "optimizer_message"]:
        summary[key] = m2.get(key)
    # M3 adds stickiness; M1 is the single-alpha baseline. Keep their key fields prefixed.
    for m in ("M1", "M3"):
        for key in ["alpha_pos", "alpha_neg", "beta", "kappa", "neg_log_likelihood", "aic", "bic"]:
            summary[f"{m}_{key}"] = fits[m].get(key)

    # M4: craving-modulated learning rate, fit under each craving signal. phi is the
    # vigor-modulated-learning metric; {method}_phi_lrt_p is the 1-df M4-vs-M2 test.
    craving_cols = {'embedded_mid': 'craving_embedded_mid', 'reward_trace': 'craving_reward_trace'}
    m4_by_method = {}
    if FIT_M4:
        for mth in CRAVING_METHODS:
            cvec = trials[craving_cols[mth]].to_numpy()
            m4 = fit_model_m4(trials, craving=cvec)
            m4_by_method[mth] = m4
            fits[f'M4_{mth}'] = m4
            bflags += boundary_flags(m4, f'M4_{mth}')
            for key in ["alpha_pos", "alpha_neg", "beta", "phi", "pseudo_r2",
                        "neg_log_likelihood", "aic", "bic"]:
                summary[f'{mth}_{key}'] = m4.get(key)
            lrt = (2.0 * (m2['neg_log_likelihood'] - m4['neg_log_likelihood'])
                   if pd.notna(m4.get('neg_log_likelihood')) else np.nan)
            summary[f'{mth}_phi_lrt_stat'] = lrt
            summary[f'{mth}_phi_lrt_p'] = math.erfc(math.sqrt(max(lrt, 0.0) / 2.0)) if pd.notna(lrt) else np.nan
            summary[f'{mth}_phi_perm_p'] = (phi_permutation_p(trials, abs(m4['phi']), cvec)
                                            if (N_PHI_PERM and pd.notna(m4.get('phi'))) else np.nan)
            summary[f'{mth}_delta_aic_vs_M2'] = m4['aic'] - m2['aic'] if pd.notna(m4.get('aic')) else np.nan
            summary[f'{mth}_delta_bic_vs_M2'] = m4['bic'] - m2['bic'] if pd.notna(m4.get('bic')) else np.nan
        primary = CRAVING_METHODS[0]
        summary['craving_primary'] = primary
        summary['phi'] = summary.get(f'{primary}_phi')
        summary['phi_lrt_p'] = summary.get(f'{primary}_phi_lrt_p')
        summary['phi_perm_p'] = summary.get(f'{primary}_phi_perm_p')
        summary['craving_decay'] = CRAVING_DECAY

        # M4_joint: both modulators together. joint_phi_mid_lrt_p (joint vs reward_trace-
        # only) is the decisive test that embedded-MID vigor predicts learning-rate
        # modulation beyond reward history; joint_phi_rew_lrt_p is the reverse.
        if 'embedded_mid' in m4_by_method and 'reward_trace' in m4_by_method:
            mj = fit_model_m4_joint(trials)
            fits['M4_joint'] = mj
            bflags += boundary_flags(mj, 'M4_joint')
            for pname in ('phi_mid', 'phi_rew'):
                v = mj.get(pname)
                if pd.notna(v):
                    if v <= PHI_BOUNDS[0] + PHI_TOL:
                        bflags.append(f'M4_joint_{pname}_at_lower_bound')
                    elif v >= PHI_BOUNDS[1] - PHI_TOL:
                        bflags.append(f'M4_joint_{pname}_at_upper_bound')
            for key in ['alpha_pos', 'alpha_neg', 'beta', 'phi_mid', 'phi_rew',
                        'pseudo_r2', 'neg_log_likelihood', 'aic', 'bic']:
                summary[f'joint_{key}'] = mj.get(key)
            nll_mid = m4_by_method['embedded_mid'].get('neg_log_likelihood')
            nll_rew = m4_by_method['reward_trace'].get('neg_log_likelihood')
            lm = 2.0 * (nll_rew - mj['neg_log_likelihood']) if pd.notna(nll_rew) else np.nan
            lr = 2.0 * (nll_mid - mj['neg_log_likelihood']) if pd.notna(nll_mid) else np.nan
            summary['joint_phi_mid_lrt_stat'] = lm
            summary['joint_phi_mid_lrt_p'] = math.erfc(math.sqrt(max(lm, 0.0) / 2.0)) if pd.notna(lm) else np.nan
            summary['joint_phi_rew_lrt_stat'] = lr
            summary['joint_phi_rew_lrt_p'] = math.erfc(math.sqrt(max(lr, 0.0) / 2.0)) if pd.notna(lr) else np.nan
            summary['joint_delta_aic_vs_M2'] = mj['aic'] - m2['aic'] if pd.notna(mj.get('aic')) else np.nan
            summary['joint_delta_bic_vs_M2'] = mj['bic'] - m2['bic'] if pd.notna(mj.get('bic')) else np.nan

            # M4_joint3 (added): phi_mid controlling for BOTH reward history AND the
            # RT-contingent bonus. Full = mid + reward_trace + bonus_trace; reduced =
            # reward_trace + bonus_trace (drop mid). joint3_phi_mid_ctrl2_lrt_p is the
            # bonus-clean test that vigor modulates alpha+ beyond reward AND the +15.
            trials['craving_bonus_trace'] = _craving_bonus_trace(df, trials)
            full3 = fit_model_m4_multi(
                trials, ['craving_embedded_mid', 'craving_reward_trace', 'craving_bonus_trace'])
            ctrl2 = fit_model_m4_multi(
                trials, ['craving_reward_trace', 'craving_bonus_trace'])
            fits['M4_joint3'] = full3
            bflags += boundary_flags(full3, 'M4_joint3')
            if full3.get('phis') is not None:
                for label, val in zip(('phi_mid', 'phi_rew', 'phi_bonus'), full3['phis']):
                    summary[f'joint3_{label}'] = val
            for key in ('alpha_pos', 'alpha_neg', 'beta', 'neg_log_likelihood',
                        'aic', 'bic', 'pseudo_r2'):
                summary[f'joint3_{key}'] = full3.get(key)
            nll_full = full3.get('neg_log_likelihood')
            nll_ctrl = ctrl2.get('neg_log_likelihood')
            lmb = (2.0 * (nll_ctrl - nll_full)
                   if (pd.notna(nll_full) and pd.notna(nll_ctrl)) else np.nan)
            summary['joint3_phi_mid_ctrl2_lrt_stat'] = lmb
            summary['joint3_phi_mid_ctrl2_lrt_p'] = (
                math.erfc(math.sqrt(max(lmb, 0.0) / 2.0)) if pd.notna(lmb) else np.nan)

        # M5_ab: the specificity model. Three nested fits on the same state vector,
        # differing only in which slopes are free, so both LRTs below are clean 1-df
        # tests. ab_phi_alpha_ctrlbeta_lrt_p is the pre-registered primary: does the
        # state improve fit through the learning rate once it can already act on
        # choice determinism? ab_phi_beta_ctrlalpha_lrt_p is the mirror image and is
        # what a reviewer will ask for.
        if FIT_M5_AB and AB_MOD_COL in trials.columns:
            ab_full = fit_model_m5_ab(trials, free=('alpha', 'beta'))
            ab_bonly = fit_model_m5_ab(trials, free=('beta',))
            ab_aonly = fit_model_m5_ab(trials, free=('alpha',))
            fits['M5_ab'] = ab_full
            bflags += boundary_flags(ab_full, 'M5_ab')
            for pname, bnds, tol in (('phi_alpha', PHI_BOUNDS, PHI_TOL),
                                     ('phi_beta', _phi_beta_bounds(), PHI_BETA_TOL)):
                v = ab_full.get(pname)
                if pd.notna(v):
                    if v <= bnds[0] + tol:
                        bflags.append(f'M5_ab_{pname}_at_lower_bound')
                    elif v >= bnds[1] - tol:
                        bflags.append(f'M5_ab_{pname}_at_upper_bound')
            for key in ('alpha_pos', 'alpha_neg', 'beta', 'phi_alpha', 'phi_beta',
                        'alpha_pos_clip_frac', 'pseudo_r2', 'neg_log_likelihood',
                        'aic', 'bic', 'optimizer_success'):
                summary[f'ab_{key}'] = ab_full.get(key)
            summary['ab_beta_mod_form'] = BETA_MOD_FORM
            summary['ab_mod_col'] = AB_MOD_COL
            nll_full_ab = ab_full.get('neg_log_likelihood')
            nll_bonly = ab_bonly.get('neg_log_likelihood')
            nll_aonly = ab_aonly.get('neg_log_likelihood')
            # Drop phi_alpha from the full model: 1 df.
            la = (2.0 * (nll_bonly - nll_full_ab)
                  if (pd.notna(nll_full_ab) and pd.notna(nll_bonly)) else np.nan)
            # Drop phi_beta from the full model: 1 df.
            lb = (2.0 * (nll_aonly - nll_full_ab)
                  if (pd.notna(nll_full_ab) and pd.notna(nll_aonly)) else np.nan)
            summary['ab_phi_alpha_ctrlbeta_lrt_stat'] = la
            summary['ab_phi_alpha_ctrlbeta_lrt_p'] = (
                math.erfc(math.sqrt(max(la, 0.0) / 2.0)) if pd.notna(la) else np.nan)
            summary['ab_phi_beta_ctrlalpha_lrt_stat'] = lb
            summary['ab_phi_beta_ctrlalpha_lrt_p'] = (
                math.erfc(math.sqrt(max(lb, 0.0) / 2.0)) if pd.notna(lb) else np.nan)
            summary['ab_betaonly_neg_log_likelihood'] = nll_bonly
            summary['ab_alphaonly_neg_log_likelihood'] = nll_aonly
            summary['ab_alphaonly_phi_alpha'] = ab_aonly.get('phi_alpha')
            summary['ab_betaonly_phi_beta'] = ab_bonly.get('phi_beta')
            summary['ab_delta_aic_vs_M2'] = (ab_full['aic'] - m2['aic']
                                             if pd.notna(ab_full.get('aic')) else np.nan)
            summary['ab_phi_alpha_perm_p'] = (
                phi_alpha_permutation_p(trials, abs(ab_full['phi_alpha']))
                if (N_PHI_PERM and pd.notna(ab_full.get('phi_alpha'))) else np.nan)
            # Same clip diagnostic for the primary single-knob M4, so the two are
            # comparable and the additive-form strain is visible in both.
            prim_fit_c = m4_by_method.get(CRAVING_METHODS[0], {})
            if pd.notna(prim_fit_c.get('phi', np.nan)):
                summary['phi_alpha_pos_clip_frac'] = alpha_pos_clip_fraction(
                    prim_fit_c['alpha_pos'], [prim_fit_c['phi']],
                    trials[craving_cols[CRAVING_METHODS[0]]].to_numpy().reshape(-1, 1))

    # Model comparison: lower AIC/BIC wins. Asymmetry (alpha_pos - alpha_neg) is only
    # interpretable where M2 beats M1; stickiness only where M3 beats M2.
    summary["delta_aic_M2_minus_M1"] = m2["aic"] - fits["M1"]["aic"]
    summary["delta_bic_M2_minus_M1"] = m2["bic"] - fits["M1"]["bic"]
    summary["delta_aic_M3_minus_M2"] = fits["M3"]["aic"] - m2["aic"]
    summary["delta_bic_M3_minus_M2"] = fits["M3"]["bic"] - m2["bic"]
    aics = {m: fits[m]["aic"] for m in fits}
    bics = {m: fits[m]["bic"] for m in fits}
    summary["best_model_by_aic"] = min(aics, key=aics.get)
    summary["best_model_by_bic"] = min(bics, key=bics.get)
    summary["boundary_flags"] = ";".join(bflags)
    summary["n_boundary_flags"] = len(bflags)
    summary["rho_identifiability_note"] = (
        "rho estimated despite confounding with beta" if ESTIMATE_RHO else "rho fixed at 1")
    summary["mean_rt"] = safe_mean(trials["rt"])
    summary["median_rt"] = safe_median(trials["rt"])

    trial_rows: List[dict] = []
    if SAVE_TRIALWISE and not pd.isna(m2.get("alpha_pos", np.nan)):
        params = [m2["alpha_pos"], m2["alpha_neg"], m2["beta"]] + ([m2["rho"]] if ESTIMATE_RHO else [])
        rows, _ = replay(np.array(params), trials, "M2")
        # Attach both craving signals and the primary-method dynamic gain rate.
        primary = CRAVING_METHODS[0]
        prim_fit = m4_by_method.get(primary, {})
        prim_rows = (replay_m4(prim_fit['params'], trials, craving_cols[primary])
                     if 'params' in prim_fit else None)
        for i, r in enumerate(rows):
            r['craving_embedded_mid'] = float(trials['craving_embedded_mid'].iloc[i])
            r['craving_reward_trace'] = float(trials['craving_reward_trace'].iloc[i])
            if prim_rows is not None:
                r['alpha_pos_dynamic_primary'] = prim_rows[i]['alpha_pos_dynamic_m4']
            r.update({"participant_id": pid, "session": session, "file_name": path.name})
            trial_rows.append(r)
    return summary, trial_rows


# =============================================================================
# DATA DICTIONARY
# =============================================================================


def make_data_dictionary(summary_cols=None, trialwise_cols=None) -> pd.DataFrame:
    """Return variable definitions for every output column. When column orders are
    supplied, rows are emitted in the same order as the actual output columns (summary
    then trialwise) so the dictionary and the CSVs always agree; undocumented output
    columns are emitted with a blank definition and reported."""
    rows = [
        ("participant_id", "Participant ID from the participant_id column or sub-<id> filename token."),
        ("session", "Session label from the session column or ses-<id> filename token."),
        ("file_name", "Source CSV filename for this run."),
        ("file_path", "Full path to the source CSV."),
        ("file_md5", "MD5 hash of the file (duplicate-run detection)."),
        ("task_version", "TASK_VERSION stamped in the data rows."),
        ("seed", "RNG seed for the run."),
        ("n_model_trials", "Usable bandit trials (1..3 choice and reward/loss outcome) used for fitting."),
        ("rho_estimated", "True if rho was a free parameter; False if fixed at 1 (default)."),
        ("alpha_pos", "M2 learning rate applied when the prediction error was >= 0 (gain)."),
        ("alpha_neg", "M2 learning rate applied when the prediction error was < 0 (loss)."),
        ("alpha_pos_minus_alpha_neg", "M2 asymmetry; positive means faster updating from gains. Interpret only where M2 beats M1."),
        ("beta", "M2 softmax inverse temperature; higher is more value-consistent/exploitative choice."),
        ("rho", "Reward sensitivity used in the update; fixed at 1 unless ESTIMATE_RHO is True (then weakly identified)."),
        ("neg_log_likelihood", "M2 negative log-likelihood of observed choices; lower is better."),
        ("aic", "M2 Akaike information criterion (k=3, or 4 if rho estimated)."),
        ("bic", "M2 Bayesian information criterion."),
        ("pseudo_r2", "M2 McFadden pseudo-R^2 vs a uniform 3-way guess: 1 - NLL/(n*ln3). 0 = chance, 1 = perfect."),
        ("pred_accuracy", "Proportion of trials whose observed choice is the model's argmax-probability arm (M2)."),
        ("mean_p_chosen", "Mean M2 predicted probability of the observed choice."),
        ("optimizer_success", "Whether the M2 optimizer reported convergence."),
        ("optimizer_message", "M2 optimizer message."),
        ("M1_alpha_pos", "M1 single learning rate (alpha_pos == alpha_neg)."),
        ("M1_alpha_neg", "M1 loss learning rate; equals M1_alpha_pos (M1 has one learning rate)."),
        ("M1_beta", "M1 softmax inverse temperature."),
        ("M1_kappa", "M1 stickiness, fixed at 0 (M1 has no perseveration term)."),
        ("M1_neg_log_likelihood", "M1 negative log-likelihood."),
        ("M1_aic", "M1 AIC (k=2)."),
        ("M1_bic", "M1 BIC (k=2)."),
        ("M3_alpha_pos", "M3 positive learning rate (model with stickiness)."),
        ("M3_alpha_neg", "M3 negative learning rate."),
        ("M3_beta", "M3 softmax inverse temperature."),
        ("M3_kappa", "M3 stickiness/perseveration on the previously chosen arm; positive = repeat regardless of value."),
        ("M3_neg_log_likelihood", "M3 negative log-likelihood."),
        ("M3_aic", "M3 AIC (k=4)."),
        ("M3_bic", "M3 BIC (k=4)."),
        ("delta_aic_M2_minus_M1", "M2 AIC minus M1 AIC. Negative => dual-alpha M2 preferred by AIC."),
        ("delta_bic_M2_minus_M1", "M2 BIC minus M1 BIC. Negative => M2 preferred by BIC."),
        ("delta_aic_M3_minus_M2", "M3 AIC minus M2 AIC. Negative => adding stickiness preferred by AIC."),
        ("delta_bic_M3_minus_M2", "M3 BIC minus M2 BIC. Negative => stickiness preferred by BIC."),
        ("best_model_by_aic", "Model with the lowest AIC across EVERY fitted model, including the M4 family, not just M1/M2/M3. Those families answer different questions, so read this as a diagnostic and report family-restricted comparisons instead. Non-converged fits are not excluded."),
        ("best_model_by_bic", "Model with the lowest BIC across EVERY fitted model, including the M4 family, not just M1/M2/M3. Same caveats as best_model_by_aic."),
        ("boundary_flags", "Free parameters sitting at an optimization bound, prefixed by model; railed estimates are weakly identified."),
        ("n_boundary_flags", "Count of boundary flags across M1/M2/M3."),
        ("rho_identifiability_note", "States whether rho was fixed at 1 (recommended) or estimated despite confounding with beta."),
        ("mean_rt", "Mean choice RT (s) across modeled trials."),
        ("median_rt", "Median choice RT (s) across modeled trials."),
        ("duplicate_file_flag", "True if another fitted file shares this file's MD5 hash."),
        # Trialwise export.
        ("trial", "Trialwise: 1-indexed bandit trial number."),
        ("swap_count", "Trialwise: reversals completed before this trial (phase = swap_count + 1)."),
        ("choice_arm", "Trialwise: chosen arm (screen position 1..3)."),
        ("reward", "Trialwise: 1 if rewarded, 0 if loss."),
        ("is_optimal", "Trialwise: 1 if the chosen arm was the current highest-probability arm."),
        ("q_arm1_pre", "Trialwise: pre-update value of arm 1 (left)."),
        ("q_arm2_pre", "Trialwise: pre-update value of arm 2 (middle)."),
        ("q_arm3_pre", "Trialwise: pre-update value of arm 3 (right)."),
        ("q_chosen_pre", "Trialwise: pre-update value of the chosen arm."),
        ("p_choose_arm1", "Trialwise: model probability of choosing arm 1."),
        ("p_choose_arm2", "Trialwise: model probability of choosing arm 2."),
        ("p_choose_arm3", "Trialwise: model probability of choosing arm 3."),
        ("p_chosen", "Trialwise: model probability of the observed choice."),
        ("prediction_error", "Trialwise: rho*r - Q_chosen; the feedback-locked RPE regressor for iEEG."),
        ("alpha_used", "Trialwise: M2 alpha_pos or alpha_neg, by the sign of the prediction error."),
        ("craving_embedded_mid", "Trialwise: z-scored embedded MID food-cue vigor carried forward (PRIMARY craving signal); candidate iEEG regressor."),
        ("craving_reward_trace", "Trialwise: z-scored reward-history craving signal (competing modulator)."),
        ("alpha_pos_dynamic_primary", "Trialwise: primary-method M4 craving-biased gain learning rate, clip(alpha_pos + phi*craving_primary_t)."),
        ("rt", "Trialwise: choice reaction time (s)."),
    ]
    for mth in ("embedded_mid", "reward_trace"):
        role = ("PRIMARY, embedded MID food-cue vigor" if mth == "embedded_mid"
                else "competing, reward-history trace")
        rows += [
            (f"{mth}_alpha_pos", f"M4 [{role}]: baseline gain learning rate at mean craving."),
            (f"{mth}_alpha_neg", f"M4 [{role}]: loss learning rate (static)."),
            (f"{mth}_beta", f"M4 [{role}]: softmax inverse temperature."),
            (f"{mth}_phi", f"M4 [{role}]: craving-modulation slope on the gain rate; phi>0 = craving speeds updating."),
            (f"{mth}_pseudo_r2", f"M4 [{role}]: McFadden pseudo-R^2."),
            (f"{mth}_neg_log_likelihood", f"M4 [{role}]: negative log-likelihood."),
            (f"{mth}_aic", f"M4 [{role}]: AIC (k=4)."),
            (f"{mth}_bic", f"M4 [{role}]: BIC (k=4)."),
            (f"{mth}_phi_lrt_stat", f"M4 [{role}]: LRT statistic 2*(NLL_M2 - NLL_M4)."),
            (f"{mth}_phi_lrt_p", f"M4 [{role}]: 1-df chi-square p for phi != 0 (mildly anti-conservative)."),
            (f"{mth}_phi_perm_p", f"M4 [{role}]: permutation p for phi (NaN unless N_PHI_PERM>0)."),
            (f"{mth}_delta_aic_vs_M2", f"M4 [{role}]: AIC minus M2 AIC. Negative => craving modulation preferred."),
            (f"{mth}_delta_bic_vs_M2", f"M4 [{role}]: BIC minus M2 BIC."),
        ]
    rows += [
        ("craving_primary", "Craving signal mirrored to phi/phi_lrt_p/phi_perm_p (first in CRAVING_METHODS)."),
        ("phi", "Primary vigor-modulated-learning estimate (mirror of <craving_primary>_phi)."),
        ("phi_lrt_p", "Primary 1-df chi-square p for phi (mirror). Prefer phi_perm_p when available."),
        ("phi_perm_p", "Primary permutation p for phi (mirror; NaN unless N_PHI_PERM>0)."),
        ("craving_decay", "Leak-per-trial of the reward_trace craving signal."),
    ]
    rows += [
        ("joint_alpha_pos", "M4_joint: baseline gain learning rate at mean craving (both modulators)."),
        ("joint_alpha_neg", "M4_joint: loss learning rate (static)."),
        ("joint_beta", "M4_joint: softmax inverse temperature."),
        ("joint_phi_mid", "M4_joint: embedded-MID slope controlling for reward history (partial phi_mid)."),
        ("joint_phi_rew", "M4_joint: reward-history slope controlling for embedded-MID (partial phi_rew)."),
        ("joint_pseudo_r2", "M4_joint: McFadden pseudo-R^2."),
        ("joint_neg_log_likelihood", "M4_joint: negative log-likelihood."),
        ("joint_aic", "M4_joint: AIC (k=5)."),
        ("joint_bic", "M4_joint: BIC (k=5)."),
        ("joint_phi_mid_lrt_stat", "LRT statistic for phi_mid | phi_rew: 2*(NLL_reward_trace - NLL_joint)."),
        ("joint_phi_mid_lrt_p", "PER-SUBJECT DIAGNOSTIC: 1-df p that embedded-MID vigor predicts learning beyond reward history (joint vs reward_trace-only). Estimated from ~16-48 state changes, so it is underpowered per subject and is NOT the primary inference; the hierarchical group-level phi is. It also does not control for beta (see the ab_* columns / M5_ab for that)."),
        ("joint_phi_rew_lrt_stat", "LRT statistic for phi_rew | phi_mid: 2*(NLL_embedded_mid - NLL_joint)."),
        ("joint_phi_rew_lrt_p", "1-df p that reward history predicts learning beyond embedded-MID (joint vs embedded_mid-only)."),
        ("joint_delta_aic_vs_M2", "M4_joint AIC minus M2 AIC (2 extra params)."),
        ("joint_delta_bic_vs_M2", "M4_joint BIC minus M2 BIC."),
        ("craving_bonus_trace", "Trialwise: z-scored decaying accumulator of past food-probe hits (the +15 events); control for the RT-contingent bonus."),
        ("joint3_alpha_pos", "M4_joint3: baseline gain learning rate (mid + reward + bonus modulators)."),
        ("joint3_alpha_neg", "M4_joint3: loss learning rate (applied to negative prediction errors)."),
        ("joint3_beta", "M4_joint3: softmax inverse temperature."),
        ("joint3_aic", "M4_joint3: Akaike information criterion (k=6)."),
        ("joint3_bic", "M4_joint3: Bayesian information criterion (k=6)."),
        ("joint3_pseudo_r2", "M4_joint3: McFadden pseudo-R2 vs a uniform-choice baseline."),
        ("joint3_phi_mid", "M4_joint3: embedded-MID slope controlling for BOTH reward history and the bonus event (bonus-clean partial phi_mid)."),
        ("joint3_phi_rew", "M4_joint3: reward-history slope in the three-modulator model."),
        ("joint3_phi_bonus", "M4_joint3: bonus-outcome slope (effect of the +15 events on alpha+)."),
        ("joint3_neg_log_likelihood", "M4_joint3: negative log-likelihood (k=6)."),
        ("joint3_phi_mid_ctrl2_lrt_stat", "LRT statistic for phi_mid | (phi_rew, phi_bonus): 2*(NLL_reward+bonus - NLL_full)."),
        ("joint3_phi_mid_ctrl2_lrt_p", "PER-SUBJECT DIAGNOSTIC, bonus-clean: 1-df p that embedded-MID vigor predicts alpha+ modulation beyond reward history AND the RT-contingent bonus. Same per-subject power caveat as joint_phi_mid_lrt_p. Under v17 silent accrual the bonus trace is a legacy/reviewer control, since the per-trial bonus feedback was removed at source."),
    ]
    # M5_ab: joint alpha+/beta state modulation. NOTE the joint_* and joint3_* rows
    # above load every modulator onto alpha_pos; they contain no beta control. The
    # alpha-versus-beta discrimination is here and only here.
    rows += [
        ("ab_alpha_pos", "M5_ab: baseline gain learning rate at mean state, with the state also free to act on beta."),
        ("ab_alpha_neg", "M5_ab: loss learning rate (static)."),
        ("ab_beta", "M5_ab: baseline softmax inverse temperature at mean state."),
        ("ab_phi_alpha", "M5_ab: state slope on alpha_pos, CONTROLLING for state modulation of beta. This is the specificity-clean phi."),
        ("ab_phi_beta", "M5_ab: state slope on beta (choice determinism). Non-zero here without ab_phi_alpha is the arousal-changes-consistency account."),
        ("ab_beta_mod_form", "Parameterization of the beta modulation: 'additive' (beta + phi_beta*s) or 'multiplicative' (beta*exp(phi_beta*s))."),
        ("ab_mod_col", "Trial column supplying the state vector for M5_ab."),
        ("ab_alpha_pos_clip_frac", "M5_ab: fraction of trials whose unclipped alpha_pos_t left (0,1). Above ~0.05 the additive form is straining and phi is not on a stable scale."),
        ("ab_pseudo_r2", "M5_ab: McFadden pseudo-R^2 vs a uniform 3-way guess."),
        ("ab_neg_log_likelihood", "M5_ab: negative log-likelihood of the full alpha+beta model (k=5)."),
        ("ab_aic", "M5_ab: AIC (k=5)."),
        ("ab_bic", "M5_ab: BIC (k=5)."),
        ("ab_optimizer_success", "M5_ab: optimizer convergence flag for the full model."),
        ("ab_betaonly_neg_log_likelihood", "Reduced fit with phi_alpha pinned to 0 (state acts on beta only), k=4."),
        ("ab_alphaonly_neg_log_likelihood", "Reduced fit with phi_beta pinned to 0 (state acts on alpha_pos only), k=4."),
        ("ab_alphaonly_phi_alpha", "State slope on alpha_pos with no beta modulation available. Compare with ab_phi_alpha to see how much of the single-knob phi was beta leakage."),
        ("ab_betaonly_phi_beta", "State slope on beta with no alpha modulation available."),
        ("ab_phi_alpha_ctrlbeta_lrt_stat", "LRT statistic for phi_alpha | phi_beta: 2*(NLL_betaonly - NLL_full)."),
        ("ab_phi_alpha_ctrlbeta_lrt_p", "PRE-REGISTERED PRIMARY: 1-df p that the state modulates alpha_pos beyond any state modulation of beta."),
        ("ab_phi_beta_ctrlalpha_lrt_stat", "LRT statistic for phi_beta | phi_alpha: 2*(NLL_alphaonly - NLL_full)."),
        ("ab_phi_beta_ctrlalpha_lrt_p", "1-df p that the state modulates beta beyond its effect on alpha_pos. Reported alongside the primary as the specificity mirror."),
        ("ab_phi_alpha_perm_p", "Permutation p for ab_phi_alpha (NaN unless N_PHI_PERM>0). Element-wise shuffling only; see the docstring caveat before citing it."),
        ("ab_delta_aic_vs_M2", "M5_ab AIC minus M2 AIC (2 extra parameters)."),
        ("phi_alpha_pos_clip_frac", "Same clip diagnostic for the primary single-knob M4 fit, for comparison with ab_alpha_pos_clip_frac."),
    ]
    defs = dict(rows)  # variable -> definition (text is the source of truth)
    scols, tcols = list(summary_cols or []), list(trialwise_cols or [])
    if not scols and not tcols:
        return pd.DataFrame([(v, "", d) for v, d in rows],
                            columns=["variable", "output_file", "definition"])
    sset, tset = set(scols), set(tcols)
    out_rows, seen = [], set()

    def _emit(col):
        # One row per output column, tagged with the CSV it appears in, in output order.
        if col in seen:
            return
        seen.add(col)
        if col not in defs:
            print(f"  [data-dictionary] WARNING: output column '{col}' has no definition")
        where = ("subject_summary + trialwise" if (col in sset and col in tset)
                 else "subject_summary" if col in sset else "trialwise_values")
        out_rows.append((col, where, defs.get(col, "")))

    for col in scols:
        _emit(col)
    for col in tcols:
        _emit(col)
    for var, definition in rows:              # any documented-but-unused variables
        if var not in seen:
            seen.add(var)
            out_rows.append((var, "", definition))
    return pd.DataFrame(out_rows, columns=["variable", "output_file", "definition"])


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    data_dir = DATA_DIR.expanduser().resolve()
    output_dir = Path(OUTPUT_DIR).expanduser().resolve() if OUTPUT_DIR else data_dir / "analysis_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(p for p in data_dir.rglob("*.csv")
                       if not p.name.startswith(".")
                       and "analysis_output" not in p.parts
                       and not any(tag in p.name for tag in
                                   ["subject_summary", "phase_summary", "cue_summary",
                                    "data_dictionary", "trial_cleaned", "trialwise", "rl_"]))
    if not csv_files:
        raise FileNotFoundError(f"No .csv files found under: {data_dir}")

    summaries, trialwise = [], []
    for path in csv_files:
        md5 = file_md5(path)
        try:
            summary, rows = analyze_one_file(path, md5)
            if summary:
                summaries.append(summary)
                trialwise.extend(rows)
        except Exception as e:
            summaries.append({"participant_id": extract_participant_id(pd.read_csv(path, nrows=5), path),
                              "file_name": path.name, "file_path": str(path), "file_md5": md5,
                              "optimizer_success": False, "fit_note": f"ERROR: {e}"})

    if not summaries:
        raise RuntimeError("No files fitted. Check INCLUDE_PARTICIPANTS and the data folder.")

    summary_df = pd.DataFrame(summaries)
    dup = summary_df["file_md5"].value_counts().to_dict()
    summary_df["duplicate_file_flag"] = summary_df["file_md5"].map(dup) > 1
    lead = ["participant_id", "session", "file_name", "task_version", "seed", "n_model_trials"]
    summary_df = summary_df[[c for c in lead if c in summary_df.columns] +
                            [c for c in summary_df.columns if c not in lead]]

    summary_path = output_dir / "bandit_rl_subject_summary.csv"
    dict_path = output_dir / "bandit_rl_data_dictionary.csv"
    trial_df = pd.DataFrame(trialwise) if SAVE_TRIALWISE else pd.DataFrame()
    summary_df.to_csv(summary_path, index=False)
    make_data_dictionary(summary_df.columns.tolist(), trial_df.columns.tolist()).to_csv(dict_path, index=False)
    if SAVE_TRIALWISE:
        trial_path = output_dir / "bandit_rl_trialwise_values.csv"
        trial_df.to_csv(trial_path, index=False)

    print("Done.")
    print(f"Fitted files: {len(summary_df)}")
    print(f"Subject summary: {summary_path}")
    print(f"Data dictionary: {dict_path}")
    if SAVE_TRIALWISE:
        print(f"Trialwise values: {output_dir / 'bandit_rl_trialwise_values.csv'}")

    bad = summary_df[summary_df.get("optimizer_success") != True]
    if len(bad):
        print("\nFiles with fit issues:")
        cols = [c for c in ["participant_id", "session", "file_name", "fit_note", "optimizer_message"] if c in bad.columns]
        print(bad[cols].to_string(index=False))
    if "n_boundary_flags" in summary_df.columns:
        railed = summary_df[summary_df["n_boundary_flags"].fillna(0) > 0]
        if len(railed):
            print("\nFiles with boundary parameters (interpret those parameters with caution):")
            cols = [c for c in ["participant_id", "session", "file_name", "boundary_flags"] if c in railed.columns]
            print(railed[cols].to_string(index=False))


if __name__ == "__main__":
    main()
