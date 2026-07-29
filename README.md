# Bandit + mini-MID bonus (PsychoPy; v17.2)

Three-arm probabilistic bandit with an interleaved mini-MID incentive-vigor probe
(48 food + 22 neutral across the session, phase-stratified) on a fixed response deadline.
The session runs as two independent 100-trial runs, each a fresh acquisition with its own
reversal. Total task time including practice is about 15 to 20 minutes.

Sample-size and recovery figures are in the design decision record
(`v17_2_design_decision_record.docx`). The primary test is the trial-wise `phi` coupling of
food-probe vigor to the positive learning rate; the between-subject `gamma` route is secondary.

> **Version-string note (needs reconciling):** the script constant `TASK_VERSION = 'v17'`, so every
> data row is stamped `task_version = v17`, but the file is `bandit_mid_task_v17_2.py` and its docstring
> still reads "Version v16". These three labels disagree. I have not changed them; flag which string is
> canonical before the first recorded session so the stamp is unambiguous.

## What defines v17.2 (vs the v15 single-run version)

- **Two runs, Q reset per run.** The 200 bandit trials are split into two independent 100-trial runs
  with a rest break. Arms are re-drawn and Q-values reset at the break, so run 2 starts from zero.
  Run 2 uses three distinct symbols (`knot`, `rose`, `cinquefoil`) so no learned shape value carries
  over and the model's per-run Q reset is honest. `run` (1 or 2) is logged in every row.
- **One reversal per run**, at trial 54 in run 1 and trial 46 in run 2 (`REVERSAL_TRIAL_BY_RUN`).
- **More probes, single press.** 24 food + 11 neutral per run (48 + 22 total), one press per probe.
  Probe density is the main lever on trial-wise `phi` recovery.
- **Bonus accrued silently.** A hit earns points, but nothing is shown between the probe and the next
  bandit trial: a neutral gap follows the response and the total is revealed only at the end. This
  removes the RT to bonus to learning-rate confound at the source. Food and neutral stakes are equal
  (15 points), so the food-minus-neutral RT contrast carries no magnitude difference. This supersedes
  the earlier v16 plan to model a decaying bonus trace as a nuisance regressor.
- **Distinct trigger codes per event** (see Triggers), replacing the single shared marker.

The fixed 550 ms probe window (staircase removed) predates v17.2; it first landed in v15 and is
unchanged here. The long staircase-era comparison tables have been dropped from this README.

## Run

Place the primary script and the `stimuli/` folder (and contents) in the same path. Requires PsychoPy
(2023.2 or newer recommended). Install via the standalone PsychoPy app or `pip install psychopy`.
Serial triggers also need `pyserial`.

```
python bandit_mid_task_v17_2.py
```

A startup dialog collects participant ID, session, an optional seed (blank draws a random one, logged
in every row), the food set (auto/sweet/savory/sweet+savory), and the iEEG options (photodiode square,
trigger backend, serial port, parallel address). Press SPACE at the instructions to begin. A short,
replayable practice block precedes the recorded task and never logs data; it runs on its own RNG stream,
so the recorded schedule is unaffected. The experimenter can abort at any time with Escape; data
collected up to that point is written and kept.

Reruns never overwrite: each run gets its own numbered output folder (see Output), so the same ID and
session can be entered repeatedly.

## Responses

- **Bandit choice:** LEFT, DOWN, RIGHT arrow keys select the left, middle, and right symbol. A vector
  arrow under each slot shows the mapping. A nudge ("Please answer faster.") appears after 4 s if no key
  is pressed, and the trial still waits for a response.
- **Bonus target:** press ANY key as fast as you can the moment the outlined square appears (PsychoPy
  keyboard, sub-ms RT). Pressing before it appears is logged as "too soon" and costs the round plus a
  short penalty pause.

## Stimuli

All stimuli live under `stimuli/` (any .png/.jpg/.jpeg, auto-discovered):

```
stimuli/shapes/         bandit symbols (run 1: heart, circle, triangle; run 2: knot, rose, cinquefoil)
stimuli/win/sweet/      sweet food images (win feedback + food bonus cue)
stimuli/win/savory/     savory food images (win feedback + food bonus cue)
stimuli/neutral/        neutral / scrambled images (neutral bonus cue)
stimuli/loss/           loss feedback images
```

Images are drawn with aspect ratio preserved: each fits inside its display box without stretching, so
non-square photos and scrambled images are not distorted. With the `sweet+savory` food set, win feedback
and the food cue draw 50/50 per image across both folders, so the two are represented equally regardless
of file counts. Neutral cues always come from `stimuli/neutral/`.

Missing files degrade gracefully: a missing bandit symbol draws a placeholder shape, and empty
food/neutral/loss folders fall back to a labelled box (cues) or a drawn sad face (losses), so the task
runs before your own images are added.

## Task parameters

- **Reward profiles (p_win/p_loss):** 80/20 (EV +6), 30/70 (EV -4), 50/50 (EV 0). Symbol-to-arm and
  arm-to-position mappings are randomized per run, so the best option is not tied to any fixed symbol or
  screen location. Reward and loss are +10 / -10 points.
- **Bandit structure:** 100 trials per run, one reversal per run (trials 54 and 46). The reversal rotates
  all three profiles in a random direction (a 3-cycle), so every arm changes role.
- **Bandit timing:** 400 ms choice animation, then a jittered 400 to 800 ms anticipatory fixation
  (`anticip_ms`), 1500 ms feedback, then a jittered 400 to 700 ms ISI (`isi_ms`). The pre-feedback
  fixation decorrelates choice-locked from feedback-locked responses and gives a clean pre-feedback
  baseline for iEEG.
- **Bonus block:** 24 food + 11 neutral per run, phase-stratified within the run (the single reversal
  splits each run into two phases; each phase gets an equal bonus count with a near-even food/neutral
  split, positions bin-spread within phase and buffered around the reversal, cue type shuffled within
  phase so the upcoming cue stays unpredictable). Sequence per bonus: "Bonus round!" (1 s), cue (1.5 s),
  anticipatory fixation jittered 1200 to 2000 ms, outlined square (550 ms), 500 ms grace, then a 600 ms
  neutral post-response gap (silent accrual). A hit earns 15 points.
- **Probe window:** fixed 550 ms for every probe and participant (`CFG['FIXED_WINDOW_MS']`). A press up
  to 500 ms after the window closes is logged as a miss but keeps its RT, so slow responses still
  contribute a response time.

## Points

The header shows only the task score, right-justified at the top; the trial counter is hidden from the
participant (the progress bar is kept), though the trial number is still recorded. Bonus points accrue
on a separate tally that is not displayed during the task and is revealed with the combined TOTAL on the
end screen. Bonus rows never carry a bandit win/loss outcome, so the bandit reward rate is unaffected.

## Output

Each run creates its own folder under `data/`, named `sub-<pid>_ses-<ses>_<n>`, where `<n>` increments to
the first unused number, so reruns never overwrite. The folder is claimed atomically, so concurrent
starts cannot collide. Inside are a CSV and a matching `.log`, both named after the folder.

The CSV is written one row at a time and flushed (a crash or Escape keeps everything up to the last
completed trial; a try/finally also closes the file on exit). Rows are chronological; `trial_type` is
`bandit`, `bonus_food`, or `bonus_neutral`. Columns include `session`, `task_version`, `run`, `t_onset_s`,
`anticip_ms`, `isi_ms`, `trigger_code`, and the bonus block's `adaptive_window_ms`.

- `run` is the 1-indexed run (1 or 2); Q resets at the run boundary.
- `adaptive_window_ms` is kept for pipeline compatibility and now logs the fixed 550 ms constant (it is
  no longer adaptive). There is no separate `window_ms` column.
- `regret` holds realized counterfactual regret (`optimal_points - points`) and can be negative; this is
  not regret in the decision-theoretic sense, so read the column name with that caveat.

`task_version` is stamped into every row from the `TASK_VERSION` constant (see the version note above).
The `.log` records every event label with its marker for offline alignment.

## Analysis

- **`analyze_bandit_mid_v5.py`** reads a folder of run CSVs and produces subject, phase, and cue summaries
  plus a data dictionary and a pooled trial-level export. It emits QC flags (`qc_flags` / `qc_n_flags`)
  and a recommended-exclusion column.
  - **Known stale item to check:** the design record flags a leftover 0.40 hit-rate floor in the QC logic
    that predates the fixed window and may misfire on fixed-window data. Verify before trusting its QC
    output. I have not confirmed whether this is still present in the v5 file.
- **`fit_bandit_mid_rl_v5_2.py`** fits per-subject RL models (Q-learning / Rescorla-Wagner with separate
  `alpha_pos` / `alpha_neg`, softmax `beta`, optional stickiness). The two 100-trial runs are fit jointly
  with one parameter set and Q reset per run. The craving-modulated model M4 adds `alpha_pos_t =
  alpha_pos + phi * craving_t`; M4_joint fits the embedded-MID vigor signal and a reward trace jointly, so
  `phi_mid` is separated from ordinary reward-driven learning-rate change. Reward sensitivity `rho` is
  fixed to 1, because with a single reward magnitude only the product of `rho` and `beta` is identified.
- **`new_bandit_recovery_script_v2.py`** runs hierarchical-EM recovery and power for the between-subject
  `gamma` route (residual food vigor as a group regressor on the RL parameters). Supporting simulation
  scripts: `block_phi_recovery_v2.py` (trial-wise `phi` recovery vs probe count) and `vigor_R.py`
  (between-subject vigor reliability R).

**Inference caveat (from the design record).** EM Wald intervals for `alpha+` and `phi` are
anti-conservative in the pilot (coverage about 0.78). Confirm any `alpha+` or `phi` effect with a
permutation test or a full posterior, not the Wald interval. Whether the permutation is itself well
calibrated at these N has not been established.

## Triggers

- Every event type now sends a **distinct trigger code (1 to 255)** via `EVENT_CODES`, so iEEG can
  separate event types (choice onset/made, win/loss, cue food/neutral, fixation, anticipation, target,
  response, feedback) from the trigger channel alone. Choose `serial` or `parallel`; with no device
  present the code is logged only. Code-to-name mapping is written to the `.log`.
- **Photodiode:** a white square pulses bottom-right at every event onset. Reposition or resize `pd_stim`
  for your sensor.
- Timing is frame-locked; onset timestamps in the data are flip times.

## Reproducibility

The bandit reward schedule rides on the same mulberry32 generator and draw order as earlier versions
(symbol placement, slot order, then per-trial outcomes), reproduced bit-for-bit from the JavaScript
implementation, so a given seed reproduces the schedule. Cosmetic draws (food set, which pictures, image
tilt, jitters, bonus schedule) run on a separate seed-derived stream.

Because the run is now built twice (two 100-trial runs with re-drawn arms and a Q reset), the call
sequence differs from v15, so v17.2 is **not** byte-identical to v15 for the same seed. This is
intentional: the task structure changed. Within v17.2, a given seed reproduces the same schedule. The
`sweet+savory` food set draws an extra value per image to pick a folder, so single-folder sets reproduce
<<<<<<< HEAD
identically while `sweet+savory` has its own draw pattern.
=======
identically while `sweet+savory` has its own draw pattern.
>>>>>>> 0e3d35c0edef2cf80791b42945252c544277fa22
