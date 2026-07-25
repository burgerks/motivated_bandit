# Bandit + mini-MID bonus (PsychoPy; v15)

Three-arm probabilistic bandit with two reversals and an interleaved mini-MID incentive-vigor probe
(16 food + 14 neutral, phase-stratified) on a fixed response deadline. Designed by Burger, and it is a
bandit, thus the hamburglar. Total task time including practice is about 15 to 20 minutes.

Sample-size estimates from simulated-data recovery analyses are in the `recovery_power_analyses`
folder. The realistic value of gamma (the between-subject vigor effect on a learning parameter) has
not been estimated from real data. The v13/v14 pilot cannot provide it, because those runs used an
adaptive staircase that compressed the probe response-time variance the vigor measure depends on.

## What changed in v15, and why

The probe response window is now **fixed at 550 ms**. Earlier versions ran an adaptive staircase that
tightened after each hit and loosened after each miss, converging near a 66.7% hit rate.

The staircase was removed because it is incompatible with the measurement. By design it holds each
participant at their own speed-accuracy threshold, which removes the slow response-time drift that
incentive vigor consists of. Calibration data show the size of the problem:

| | staircase (v13/v14) | fixed 550 ms (v15) |
| --- | --- | --- |
| Within-participant SD of log probe RT | 0.156 | 0.25 to 0.31 |
| Single-probe reliability (lambda) | about 0.10 | about 0.33 |
| Probe hit rate | about 0.67 | about 0.95 or higher |
| Censored readings | about 34% | 1 to 2% |

Lambda is the share of one probe reading that reflects motivational state rather than measurement
noise. The v15 value replicated across independent calibration participants.

550 ms was chosen from the observed probe RT distribution. It keeps censoring rare (pooled hit rate
about 0.99, worst participant about 0.96, holding above 0.92 even if responses slow by 20% once the
deadline stops chasing them). Censoring matters more than it sounds: slow probes are exactly the ones
that get lost, so a tight deadline biases RT variance downward, and RT variance is the measurement the
probe exists to provide. The deadline still imposes time pressure, so the probe remains a speeded
incentive measure.

Data files from v15 carry a `window_ms` column holding the constant deadline. Files from v13/v14 carry
`adaptive_window_ms` instead. The analyzer reads both and flags legacy runs.

## Run

To run, you need the primary script and the stimuli folder (and contents) in the same path. Requires
PsychoPy (2023.2 or newer recommended). Install via the standalone PsychoPy app or `pip install
psychopy`. Serial triggers also need `pyserial`.

```
python bandit_mid_task_v15.py
```

A startup dialog collects participant ID, session, an optional seed (blank draws a random one, logged
in every row), the food set (auto/sweet/savory/sweet+savory), and the iEEG options (photodiode square,
trigger backend, serial port, parallel address). Press SPACE at the instructions to begin. The
experimenter can abort at any time with Escape; data collected is written and kept.

Reruns never overwrite: each run gets its own numbered output folder (see Output below), so the same
ID and session can be entered repeatedly.

## Responses

- Bandit choice: the LEFT, DOWN, and RIGHT arrow keys select the left, middle, and right symbol. An
  arrow shape under each slot shows the mapping (the arrows are drawn as vector shapes, not font
  glyphs, so they render consistently). A 4 s nudge ("Please answer faster.") appears if no key is
  pressed, and the trial still waits for a response.
- Bonus target: press ANY key as fast as you can the moment the square appears (PsychoPy keyboard,
  sub-ms RT). The target is a large outlined square with a thick white border. Pressing before it
  appears is logged as "too soon" and costs the round plus a short penalty pause.

## Stimuli

All stimuli live under `stimuli/`. Drop your images into these folders (any .png/.jpg/.jpeg,
auto-discovered):

```
stimuli/shapes/         heart.png, circle.png, triangle.png (the bandit symbols)
stimuli/win/sweet/      sweet food images (win feedback + food bonus cue)
stimuli/win/savory/     savory food images (win feedback + food bonus cue)
stimuli/neutral/        neutral / scrambled images (neutral bonus cue)
stimuli/loss/           loss feedback images
```

Images are drawn with their aspect ratio preserved: each fits inside its display box (the bonus cue
inside a 0.8-of-height box, win and loss feedback inside smaller boxes) without stretching, so
non-square photos and scrambled images are not distorted. Square 1024x1024 sources still work and
simply fill the box.

With the `sweet+savory` food set, win feedback and the food bonus cue draw from both folders, choosing
a folder 50/50 per image and then a picture from it, so the two folders are represented equally
regardless of how many files each holds. Neutral cues always come from `stimuli/neutral/`.

If a bandit symbol file is missing, a plain dark placeholder shape is drawn instead, so the task runs
before you add your own. Empty food/neutral/loss folders fall back to a labeled box (cues) or a drawn
sad face (losses).

## Task parameters

- Reward profiles (p_win/p_loss): 80/20 (EV +6), 50/50 (EV 0), 30/70 (EV -4). Symbol-to-arm and
  arm-to-position mappings are randomized per session, so the best option is not tied to any fixed
  symbol or screen location.
- 200 bandit trials, two reversals at trials 69 and 130. Each reversal rotates all three profiles in a
  random direction (a 3-cycle, so every arm changes role, including the chance arm).
- Bandit timing: 400 ms choice animation, then a jittered 400-800 ms anticipatory fixation (logged per
  trial as `anticip_ms`), 1500 ms feedback, then a jittered 400-700 ms ISI (logged as `isi_ms`). The
  pre-feedback fixation decorrelates choice-locked from feedback-locked responses and gives a clean
  pre-feedback baseline for the iEEG recordings.
- Bonus block: 16 food + 14 neutral, phase-stratified across the three task phases set by the
  reversals. Each phase gets an equal count of bonus trials (about 10) with a balanced food/neutral
  split (6/4, 5/5, 5/5), positions are spread within each phase and buffered around both reversals,
  and cue type is shuffled within phase rather than alternating, so the upcoming cue stays
  unpredictable. Sequence per bonus: "Bonus round!" (1 s), cue (1.5 s), anticipatory fixation jittered
  1500-3000 ms, outlined square (550 ms), 500 ms grace, feedback (1.5 s). A hit earns 15 points.
- Response window: fixed at 550 ms for every probe and every participant (`CFG['FIXED_WINDOW_MS']`).
  A press up to 500 ms after the window closes is logged as a miss but keeps its target RT, so slow
  responses still contribute a response time.

## Points

The header shows task points and bonus points, right-justified at the top of the screen. The numeric
trial counter is hidden from the participant (the progress bar is kept); the trial number is still
recorded in the data. Bonus points are tracked on a separate tally and added to the task score for the
combined TOTAL on the end screen. Bonus rows never carry a bandit win/loss outcome, so the bandit
reward rate is unaffected.

## Output

Each run creates its own folder under `data/`, named `sub-<pid>_ses-<ses>_<n>`, where `<n>` increments
to the first unused number, so reruns with the same ID and session never overwrite. The folder is
claimed atomically, so concurrent starts cannot collide. Inside it are a CSV and a matching `.log`,
both named after the folder.

The CSV is written one row at a time and flushed (a crash or Escape quit keeps everything up to the
last completed trial; a try/finally also closes the file on any exit). Rows are in chronological
order; `trial_type` is `bandit`, `bonus_food`, or `bonus_neutral`. The column set includes `session`,
`task_version`, `t_onset_s`, `anticip_ms`, `isi_ms`, `window_ms`, and `trigger_code`. The `.log`
records every event label with its marker for offline alignment.

`task_version` is written from the `TASK_VERSION` constant near the top of the script and is stamped
into every data row. The `.log` also records the probe deadline once at startup, so the value actually
used can be verified per run.

## Analysis

- `analyze_bandit_mid_v4.py` reads a folder of run CSVs and produces subject, phase, and cue
  summaries plus a data dictionary and a pooled trial-level export. It accepts both v15 (`window_ms`)
  and legacy v13/v14 (`adaptive_window_ms`) files, and adds a `legacy_staircase_run` QC flag to the
  latter, since their vigor readings are compressed and not comparable to v15.
- `fit_bandit_mid_rl_v4.py` fits per-subject reinforcement-learning models, including the
  craving-modulated learning-rate model whose phi parameter carries the trial-wise vigor effect.
  Reward sensitivity rho is fixed at 1, because with a single reward magnitude only the product of rho
  and beta is identified.
- `bandit_recovery_script.py` runs hierarchical parameter-recovery and power simulations. Group-level
  inference on the vigor coefficients (gamma) offers two options via `--inference`:
  - `wald` (default): the EM Wald interval. Fast, but anti-conservative for alpha+. Across the
    simulation grid its null rejection rate ran 0.10 to 0.16 against a nominal 0.05, and its CI
    coverage ran 0.83 to 0.90 against a nominal 0.95, at every sample size tested. The miscalibration
    does not shrink as N grows.
  - `perm`: a permutation test that shuffles the vigor vector across subjects. Use `--n-perm` to set
    the number of permutations and `--perm-mode full` to re-run the whole EM per permutation rather
    than only the M-step regression.
  - `both`: reports each side by side, which is the way to see how much the Wald interval inflates.

  Confirm any alpha+ effect with the permutation option rather than the Wald interval. Whether the
  permutation is itself well calibrated at these sample sizes has not been established, and needs a
  run with several hundred datasets under the null before it is relied on.

## Trigger notes

- Triggers: every event sends the same marker, a comma. Choose `serial` (writes the byte `,` = 0x2C to
  the configured port) or `parallel` (writes the comma byte 44). With no device present the marker is
  logged only. Event identity is preserved in the `.log` labels and in the data file via `trial_type`
  and the onset-time columns. The event set includes an `anticipation` marker for the pre-feedback
  fixation. If your recording system needs a different transport or a distinct code per event, that is
  a small change in the `Triggers` class and `EVENT_CODES`.
- Photodiode: a white square pulses bottom-right at every event onset (choice, anticipatory fixation,
  outcome, cue, target square, feedback). Reposition or resize `pd_stim` for your sensor.
- Timing is frame-locked; onset timestamps in the data are flip times.

## Reproducibility

The bandit reward schedule uses the same mulberry32 generator and draw order as the web version
(symbol placement, slot order, then per-trial outcomes), so a given seed reproduces the schedule. The
lab spec differs from the web spec in profiles and reversals (two reversals at 69 and 130, 80/20 best
arm), so the schedule matches the web version only where those settings match.

Cosmetic draws (food set, which pictures, image tilt, anticipation jitter, ISI jitter, and the bonus
schedule) run on a separate seed-derived stream and are reproducible within PsychoPy. The bonus
schedule is phase-stratified and lab-specific, so it does not follow the web deck order; and the
`sweet+savory` food set draws an extra value per image to pick a folder, so single-folder sets
reproduce exactly as before while `sweet+savory` has its own draw pattern.

Removing the staircase did not touch the bandit schedule RNG, so a given seed reproduces the same
bandit schedule as v13/v14. The probe deadline is now a constant rather than a per-probe draw, so no
RNG draw was removed either.
