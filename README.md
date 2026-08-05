# Bandit + mini-MID bonus (PsychoPy; task v17.3, pipeline v6 / v6.1)

Three-arm probabilistic bandit with an interleaved mini-MID incentive-vigor probe on a fixed
response deadline. Designed by Burger, and it is a bandit, thus the hamburglar. Total task time
including practice is about 25 to 30 minutes.

The primary parameter is phi: the within-subject, trial-wise coupling between food-cue probe vigor
(reaction time to the probe target) and the positive learning rate alpha+. Sample-size estimates
from simulated-data recovery analyses are in the `recovery_power_analyses` folder. sigma_phi, the
between-subject SD of phi, has not been estimated from real data and remains the dominant power
unknown.

## Session structure (v17.3)

- **Two independent runs of 100 bandit trials.** Arm assignments are re-drawn and Q-values reset at
  the break, so run 2 starts naive. Run index is written into every row.
- **One reversal per run**, at trial 54 in run 1 and trial 46 in run 2 (`REVERSAL_TRIAL_BY_RUN`).
  Each reversal rotates all three profiles in a random direction (a 3-cycle, so every arm changes
  role, including the chance arm).
- **Novel symbols in run 2.** Run 1 uses heart / circle / triangle; run 2 uses knot / rose /
  cinquefoil. Distinct symbols make the model's Q-reset at the break honest. The cost is that run
  effects and symbol novelty are not separable.
- **35 probes per run** (24 food + 11 neutral), so 48 food + 22 neutral across the session. Single
  press per round; no bursts.
- **Two VAS** (hunger, then energy/engagement) run once between practice and the recorded task, on a
  continuous -100..100 scale. These are written to a separate `<run>_vas.csv`.

## What changed across versions

### v15: staircase removed

The probe response window is fixed at 550 ms. Earlier versions ran an adaptive staircase that
tightened after each hit and loosened after each miss, converging near a 66.7% hit rate.

The staircase was removed because it is incompatible with the measurement. By design it holds each
participant at their own speed-accuracy threshold, which removes the slow response-time drift that
incentive vigor consists of. Calibration data show the size of the problem:

| | staircase (v13/v14) | fixed 550 ms (v15+) |
| --- | --- | --- |
| Within-participant SD of log probe RT | 0.156 | 0.25 to 0.31 |
| Single-probe reliability (lambda) | about 0.10 | about 0.33 |
| Probe hit rate | about 0.67 | about 0.95 or higher |
| Censored readings | about 34% | 1 to 2% |

Lambda is the share of one probe reading that reflects motivational state rather than measurement
noise. The v15 value replicated across independent calibration participants. 550 ms was chosen from
the observed probe RT distribution; it keeps censoring rare while still imposing time pressure.

### v16: two runs, higher probe density

The 200 bandit trials became two independent 100-trial runs. Probe counts rose to 24 food + 11
neutral per run. Probe density is the lever that most improves trial-wise phi recovery, and a fixed
window plus higher density is what the recovery simulations converged on. Data files from v16 onward
carry a `run` column.

### v17: silent bonus accrual

Bonus points are accrued silently and revealed only at the end. There is no per-trial hit/miss
feedback, only a 600 ms neutral gap. This removes the RT-contingent reward event that used to sit
between the probe and the next bandit trial, eliminating the Vigor -> RT -> Hit -> Bonus -> alpha+
confound at source rather than modeling it out. The premature ("Too soon!") screen is retained,
because it is instructional rather than a reward.

Per-event iEEG trigger codes were also introduced: every event type sends a distinct code (1-255)
rather than every event sharing one marker.

### v17.2: cue stakes

The cue carries a `+15 if fast` stakes label, equal for food and neutral cues, so the food-vs-neutral
RT contrast has no magnitude confound. The cue-to-target jitter was narrowed to 1200-2000 ms.

### v17.3: VAS and blank post-response screen

Two VAS added (above). The post-response fixation cross in the bonus round is now blank; the 600 ms
gap and its iEEG marker and timestamp are unchanged, only the visible cross is removed. No RNG draws
were added, so the bandit and bonus schedules are byte-identical to v17.2 for any seed.

## Run

To run, you need the primary script and the stimuli folder (and contents) in the same path. Requires
PsychoPy (2023.2 or newer recommended). Install via the standalone PsychoPy app or `pip install
psychopy`. Serial triggers also need `pyserial`. VAS input additionally uses `pyglet`'s key state
handler, with a mouse-only fallback if that import fails.

```
python bandit_mid_task_v17_3.py
```

A startup dialog collects participant ID, session, an optional seed (blank draws a random one, logged
in every row), the food set (auto/sweet/savory/sweet+savory), and the iEEG options (photodiode
square, trigger backend, serial port, parallel address). Press SPACE at the instructions to begin.
The experimenter can abort at any time with Escape; data collected is written and kept.

Reruns never overwrite: each run gets its own numbered output folder, so the same ID and session can
be entered repeatedly.

## Responses

- **Bandit choice:** LEFT, DOWN, and RIGHT arrow keys select the left, middle, and right symbol. An
  arrow shape under each slot shows the mapping (drawn as vector shapes, not font glyphs, so they
  render consistently). A 4 s nudge ("Please answer faster.") appears if no key is pressed, and the
  trial still waits for a response.
- **Bonus target:** press ANY key as fast as you can the moment the square appears (PsychoPy
  keyboard, sub-ms RT). The target is a large outlined square with a thick white border. Pressing
  before it appears is logged as "too soon" and costs the round plus a 1500 ms penalty pause.
- **VAS:** click or drag on the track, or use Left/Right arrow keys; SPACE confirms. The marker
  starts centered at 0 and the numeric value is never shown.

## Stimuli

All stimuli live under `stimuli/`. Drop your images into these folders (any .png/.jpg/.jpeg,
auto-discovered):

```
stimuli/shapes/         heart.png, circle.png, triangle.png (run 1 bandit symbols)
                        knot.png, rose.png, cinquefoil.png (run 2 bandit symbols)
stimuli/win/sweet/      sweet food images (win feedback + food bonus cue)
stimuli/win/savory/     savory food images (win feedback + food bonus cue)
stimuli/neutral/        neutral / scrambled images (neutral bonus cue)
stimuli/loss/           loss feedback images
```

Images are drawn with their aspect ratio preserved: each fits inside its display box (the bonus cue
inside a 0.8-of-height box, win and loss feedback inside smaller boxes) without stretching, so
non-square photos and scrambled images are not distorted. Square 1024x1024 sources still work and
simply fill the box.

With the `sweet+savory` food set, win feedback and the food bonus cue draw from both folders,
choosing a folder 50/50 per image and then a picture from it, so the two folders are represented
equally regardless of how many files each holds. Neutral cues always come from `stimuli/neutral/`.

If a bandit symbol file is missing, a plain dark placeholder shape is drawn instead, so the task runs
before you add your own. Empty food/neutral/loss folders fall back to a labeled box (cues) or a drawn
sad face (losses).

## Task parameters

- Reward profiles (p_win/p_loss): 80/20 (EV +6), 50/50 (EV 0), 30/70 (EV -4). Symbol-to-arm and
  arm-to-position mappings are randomized per run.
- Bandit timing: 400 ms choice animation, then a jittered 400-800 ms anticipatory fixation (logged
  per trial as `anticip_ms`), 1500 ms feedback, then a jittered 400-700 ms ISI (logged as `isi_ms`).
  The pre-feedback fixation decorrelates choice-locked from feedback-locked responses and gives a
  clean pre-feedback baseline for the iEEG recordings.
- Bonus block: 24 food + 11 neutral per run, phase-stratified across the two phases each run's
  reversal creates, spread within phase, buffered 3 trials around the reversal, minimum gap 2, first
  probe no earlier than bandit trial 8. Cue type is shuffled within phase rather than alternating, so
  the upcoming cue stays unpredictable. Sequence per bonus: "Bonus round!" (1 s), cue (1.5 s),
  anticipatory fixation jittered 1200-2000 ms, outlined square (550 ms), 500 ms grace, 600 ms neutral
  blank. A hit earns 15 points, tallied silently.
- Response window: fixed at 550 ms for every probe and every participant
  (`CFG['FIXED_WINDOW_MS']`). A press up to 500 ms after the window closes is logged as a miss but
  keeps its target RT, so slow responses still contribute a response time.

## Points

The header shows task points and bonus points, right-justified at the top of the screen. The numeric
trial counter is hidden from the participant (the progress bar is kept); the trial number is still
recorded in the data. Bonus points are tracked on a separate tally and added to the task score for
the combined TOTAL on the end screen only. Bonus rows never carry a bandit win/loss outcome, so the
bandit reward rate is unaffected.

## Output

Each run creates its own folder under `data/`, named `sub-<pid>_ses-<ses>_<n>`, where `<n>` increments
to the first unused number, so reruns with the same ID and session never overwrite. The folder is
claimed atomically, so concurrent starts cannot collide. Inside it are:

- `<folder>.csv` - the main trial-level file, written one row at a time and flushed.
- `<folder>.log` - every event label with its marker, for offline alignment.
- `<folder>_vas.csv` - the two VAS responses. **This file is not yet excluded by the analysis
  scripts; see Known issues.**

Rows in the main CSV are in chronological order; `trial_type` is `bandit`, `bonus_food`, or
`bonus_neutral`. The column set includes `session`, `task_version`, `run`, `t_onset_s`, `anticip_ms`,
`isi_ms`, `adaptive_window_ms` (which now logs the fixed 550 ms constant, kept under the old name for
pipeline compatibility), and `trigger_code`.

`task_version` is written from the `TASK_VERSION` constant near the top of the script and is stamped
into every data row. The `.log` also records the probe deadline once at startup, so the value
actually used can be verified per run.

## Analysis

- **`analyze_bandit_mid_v6.py`** reads a folder of run CSVs and produces subject, phase, and cue
  summaries plus a data dictionary and a pooled trial-level export. Reversal windows, switch rate,
  and reversal QC are computed within run. It accepts legacy v13/v14 files and adds a
  `legacy_staircase_run` QC flag, since their vigor readings are compressed and not comparable.
  Group-level vigor indices (including the residualized food-cued vigor index) require at least 8
  eligible subjects.
- **`fit_bandit_mid_rl_v6.py`** fits per-subject RL models: M1 (single alpha), M2 (dual alpha,
  primary), M3 (+ stickiness kappa), M4 (M2 + phi on alpha+), M4_joint (vigor + reward-history
  trace), M4_joint3 (adds a bonus-hit trace). Reward sensitivity rho is fixed at 1, because with a
  single reward magnitude only the product rho*beta is identified.
- **`fit_bandit_mid_rl_v6_1.py`** adds M5_ab, the joint alpha+/beta state-modulation model, and
  per-fit alpha_pos(t) clip-fraction logging. **v6.1 is a fork of pre-rename code, not a superset of
  v6; see Known issues before using either.**
- **`test_m5_ab.py`** simulates data under three generative truths (alpha only, beta only, both) and
  checks that the two nested 1-df tests point at the parameter that actually moved.
- **`bandit_recovery_script.py`** runs hierarchical parameter-recovery and power simulations.
  Group-level inference on gamma (the between-subject vigor effect) offers `--inference wald`
  (fast, but anti-conservative for alpha+: null rejection 0.10-0.16 against a nominal 0.05, CI
  coverage 0.83-0.90 against a nominal 0.95, and the miscalibration does not shrink as N grows),
  `perm`, or `both`.

Confirm any alpha+ effect with a permutation test rather than a Wald interval.

## Known issues (open, as of task v17.3 / pipeline v6.1)

These are documented rather than fixed. Nothing below has been corrected in the shipped code.

**Data handling**

1. **`_vas.csv` files are read as task files.** Neither `analyze_bandit_mid_v6.py` nor
   `fit_bandit_mid_rl_v6*.py` excludes them from the recursive `*.csv` scan. Verified behavior: the
   VAS file produces a junk summary row carrying the real participant ID, with no error raised. This
   silently inflates N in both output summaries. Highest-priority fix.
2. **v6 and v6.1 are divergent forks.** v6.1 branched before the craving-to-vigor rename and still
   uses `craving_embedded_mid` / `craving_reward_trace` / `craving_bonus_trace`, and its trialwise
   export is missing the `run` column that v6 added. Deciding which is canonical is a prerequisite
   for every other fix, or each fix has to be written twice.

**State regressor construction**

3. **Vigor is standardized across runs, not within run.** Both z-scoring stages in
   `_vigor_embedded_mid` pool the two runs. Carry-forward is correctly within-run, but any run-level
   fatigue or practice shift in probe RT becomes a sustained offset in the state regressor.
4. **The reward-history trace is standardized globally**, although its docstring says within-run. The
   accumulator does reset at run boundaries; only the z-score pools.
5. **Probe-RT selection censors both tails, asymmetrically.** Premature presses log an empty
   `target_rt_ms`, so the fastest responses are dropped; grace-window responses are right-truncated
   at 1050 ms; no-response probes drop the slow tail entirely. This is censoring on exactly the
   variable phi depends on, and needs stating as more than a documentation note.
6. **Trials before the first probe are imputed at 0**, which is the probe-level mean before the
   second standardization. Reasonable, but it is an imputation and belongs in a sensitivity analysis.
7. **Carry-forward relies on file row order.** `_vigor_embedded_mid` takes the last matching probe by
   position in the frame rather than by an explicit sort. Correct for files the task writes, fragile
   for anything reordered downstream.

**Model and inference**

8. **phi_neutral is not implemented.** The 22 neutral probes exist as the arousal specificity control
   (the same estimator run on neutral-probe RTs), but no fitter contains any neutral-probe code path.
   This is the answer to the standing reviewer challenge that phi indexes generic arousal, and it is
   currently missing.
9. **The residualized vigor index has no neutral-RT covariate.** It regresses food RT on bandit
   median RT, MID no-response rate, and premature rate. General speed is controlled; cue-nonspecific
   probe arousal is not.
10. **The phi permutation test shuffles the trial-level vector**, which destroys the blocky
    carry-forward structure, rather than permuting probe values and rebuilding the carried-forward
    epochs. The correct version is the probe-level one. Note that a 200-dataset null simulation gave
    a shuffled-null q95 of 0.156 against a true-null q95 of 0.133, so the shuffle appears mildly
    *conservative* here, not anti-conservative as the v6.1 docstring asserts. That simulation used
    fixed 4-trial blocks and 4 multistarts, so treat the direction as unsettled rather than reversed.
    `N_PHI_PERM` defaults to 0, so nothing currently uses this path.
11. **M4 omits kappa.** Deliberate, since it keeps M4-vs-M2 a clean 1-df test, but state-driven
    perseveration can therefore load onto phi. A kappa robustness variant is not implemented.
12. **Additive alpha+ modulation clips.** `PHI_BOUNDS = (-2, 2)` against a z-scored regressor permits
    a +/-4 excursion on a parameter bounded in (0, 1), so the clip can bind often. v6 logs
    `alpha_pos_clip_frac` for M4 only, not for M4_joint or M4_joint3; v6.1 adds it for M5_ab. Where
    clipping is frequent the effective phi scale becomes participant-dependent, which contaminates
    any sigma_phi read off these fits.
13. **`best_model_by_aic` ranges over mixed model families.** The `fits` dictionary accumulates
    M1-M3 alongside every M4 variant, and the minimum is taken over all of them, so models answering
    different scientific questions are ranked together. The data dictionary still describes this
    column as covering "M1/M2/M3", which is no longer true.
14. **Optimizer starts are not fully reproducible.** `fit_model` seeds with
    `RANDOM_SEED + hash(model) % 1000`, and Python's string hash is randomized per process, so M1,
    M2, and M3 draw different random starts on every run. The M4 family uses fixed offsets and is
    unaffected.
15. **Non-converged fits are retained.** Nothing filters on `optimizer_success` before parameters
    enter the summary or before AIC/BIC model selection.
16. **Per-subject LRTs should not carry the primary inference.** With roughly 16 to 48 state changes
    per subject, per-subject phi is a diagnostic. The intended test is hierarchical. The data
    dictionary currently labels `joint_phi_mid_lrt_p` a "DECISIVE test" at the subject level, which
    invites exactly this misreading.

**Documentation drift**

17. The task script's module docstring still says "Version v16", "same single reversal", "same
    adaptive-window bonus", and instructs the user to run `bandit_mid_task_v14.py`. All four are
    wrong for v17.3.
18. The VAS comment block at line 1266 says "Three continuous VAS"; two are administered.
19. `phi_interpretation_v2.docx` is plain text with a `.docx` extension, so Word and python-docx both
    fail to open it. See `phi_interpretation_v3.docx` for the corrected version.

## Trigger notes

- **Triggers:** each event type sends a distinct code: choice onset 10, choice made 11, bandit win
  20, bandit loss 21, bonus intro 30, food cue 31, neutral cue 32, fixation 33, anticipation 34,
  target 35, response 36, bonus feedback 37. Choose `serial` or `parallel` in the startup dialog;
  with no device present the marker is logged only. Event identity is also preserved in the `.log`
  labels and in the data file via `trial_type`, `trigger_code`, and the onset-time columns.
- **Photodiode:** a white square pulses bottom-right at every event onset. Reposition or resize
  `pd_stim` for your sensor.
- Timing is frame-locked; onset timestamps in the data are flip times.

## Reproducibility

The bandit reward schedule uses the same mulberry32 generator and draw order as the web version
(symbol placement, slot order, then per-trial outcomes), so a given seed reproduces the schedule. The
lab spec differs from the web spec in profiles and reversals, so the schedule matches the web version
only where those settings match. Because the run is now built twice, the call sequence differs from
v15 and earlier; within v16 onward it is stable.

Cosmetic draws (food set, which pictures, image tilt, anticipation jitter, ISI jitter, and the bonus
schedule) run on a separate seed-derived stream. The bonus schedule is phase-stratified and
lab-specific, so it does not follow the web deck order; the `sweet+savory` food set draws an extra
value per image to pick a folder.

Neither the VAS nor the blank-screen change in v17.3 draws from either RNG stream, so v17.3 and
v17.2 produce byte-identical schedules for any given seed.
