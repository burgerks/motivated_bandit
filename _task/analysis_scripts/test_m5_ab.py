#!/usr/bin/env python3
"""Functional check on M5_ab. Simulates data under three generative truths and
verifies the two nested LRTs point at the parameter that actually moved."""

import importlib.util
import math
import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location('f', 'fit_bandit_mid_rl_v6_1.py')
F = importlib.util.module_from_spec(spec)
spec.loader.exec_module(F)
F.N_RANDOM_STARTS = 8          # keep the test fast; the real fitter uses 24

N_RUN, N_RUNS = 100, 2
REV = 60                       # one reversal per run
PROF = np.array([0.80, 0.30, 0.50])


def simulate(rng, phi_a, phi_b, apos=0.30, aneg=0.20, beta=4.0):
    """Generate choices under alpha_pos_t = apos + phi_a*s_t and beta_t = beta + phi_b*s_t.
    The state is a blocky carry-forward signal, matching the real modulator's shape."""
    rows = []
    for run in range(1, N_RUNS + 1):
        # Blocky state: a new value every 4 trials, held constant in between.
        blocks = rng.normal(size=N_RUN // 4 + 1)
        s = np.repeat(blocks, 4)[:N_RUN]
        s = (s - s.mean()) / s.std()
        p = PROF.copy()
        q = np.zeros(3)
        for t in range(N_RUN):
            if t == REV:
                p = p[[2, 0, 1]]                      # rotate all three arms
            b_t = max(beta + phi_b * s[t], 0.01)
            u = b_t * q
            u -= u.max()
            pr = np.exp(u); pr /= pr.sum()
            c = rng.choice(3, p=pr)
            rew = 1 if rng.random() < p[c] else 0
            r = 1.0 if rew else -1.0
            pe = r - q[c]
            a_t = min(max(apos + phi_a * s[t], 1e-6), 1 - 1e-6)
            q[c] += (a_t if pe >= 0 else aneg) * pe
            rows.append(dict(run=run, trial=t + 1, arm=c, reward=rew,
                             craving_embedded_mid=s[t]))
    return pd.DataFrame(rows)


def report(label, df):
    full = F.fit_model_m5_ab(df, free=('alpha', 'beta'))
    bonly = F.fit_model_m5_ab(df, free=('beta',))
    aonly = F.fit_model_m5_ab(df, free=('alpha',))
    la = 2 * (bonly['neg_log_likelihood'] - full['neg_log_likelihood'])
    lb = 2 * (aonly['neg_log_likelihood'] - full['neg_log_likelihood'])
    pa = math.erfc(math.sqrt(max(la, 0.0) / 2))
    pb = math.erfc(math.sqrt(max(lb, 0.0) / 2))
    print(f"{label:<28} phi_a={full['phi_alpha']:+.3f} phi_b={full['phi_beta']:+.3f} "
          f"| p(alpha|beta)={pa:.4f} p(beta|alpha)={pb:.4f} "
          f"| clip={full['alpha_pos_clip_frac']:.3f} | naive phi_a={aonly['phi_alpha']:+.3f}")
    return pa, pb


rng = np.random.default_rng(7)
print('Generative truth -> fitted slopes and the two 1-df tests\n')
report('null (no modulation)', simulate(rng, 0.0, 0.0))
report('alpha only (phi_a=0.15)', simulate(rng, 0.15, 0.0))
report('beta only  (phi_b=2.0)', simulate(rng, 0.0, 2.0))
report('both       (0.15, 2.0)', simulate(rng, 0.15, 2.0))
