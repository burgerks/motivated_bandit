#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Probabilistic 3-arm bandit with an interleaved mini-MID "bonus round".

PsychoPy (Coder) port of the jsPsych task. Same reward schedule, same single
reversal, same adaptive-window bonus, same data fields. A staged instruction
walkthrough and a short, replayable practice block (two rigged bandit trials, a
scripted too-soon demonstration, and three bonus rounds) precede the recorded
task; practice never logs data and runs
on its own RNG stream, so the recorded schedule is byte-identical to earlier
versions (v10 onward) for any given seed. The mulberry32 RNG is
reproduced bit-for-bit from JavaScript, so a given seed yields the same bandit
schedule as the web version (verified against the JS implementation).

Tested against the PsychoPy 2023.2+/2024.x API (visual, core, gui,
hardware.keyboard, parallel). Run from the PsychoPy Coder or `python bandit_mid_task_v14.py`.
Version v16 (see TASK_VERSION), logged in every data row.

Changes from v15 (these deliberately break byte-identical reproducibility with
v15, because the task structure itself has changed):

  1. TWO RUNS. The 200 bandit trials are delivered as two independent 100-trial
     runs with a rest break between them. Each run is a fresh acquisition: arm
     assignments are re-drawn and Q-values reset at the break, so run 2 starts
     from zero rather than continuing run 1. Run index is logged in every row.
     One reversal per run: trial 54 in run 1, trial 46 in run 2.

  2. STAIRCASE REMOVED. Every probe uses one fixed response window
     (FIXED_WINDOW_MS) for every participant and every round. The up/down
     stepping is gone. The output column 'adaptive_window_ms' is kept for
     pipeline compatibility and now logs that fixed constant.

  3. MORE PROBES, SINGLE PRESS. 24 food + 11 neutral probes per run (48 food +
     22 neutral across the session), single press per round (no bursts). Probe
     density is the lever that most improves trial-wise phi recovery; a fixed
     window plus higher density is what the recovery simulations converged on.

The bandit schedule still rides on the MAIN mulberry32 stream and cosmetic
randomness on the SECOND stream, exactly as before; only the call sequence has
changed because the run is now built twice.

Folder layout expected next to this file:
    stimuli/shapes/         heart.png, circle.png, triangle.png (bandit symbols)
    stimuli/win/sweet/      <your sweet food images>
    stimuli/win/savory/     <your savory food images>
    stimuli/neutral/        <your neutral / scrambled images>
    stimuli/loss/           <your loss images>
Images are auto-discovered (any .png/.jpg/.jpeg) and drawn with aspect ratio
preserved. Missing symbol files fall back to a drawn shape; empty food/neutral/
loss folders fall back to a labelled box or a drawn sad face, so the task still
runs for piloting.
"""

import os
import csv
import glob
import math
import datetime

from PIL import Image            # read pixel dimensions to preserve aspect ratio
from psychopy import visual, core, gui, logging, event   # event: mouse for the VAS
from psychopy.hardware import keyboard

# ════════════════════════════════════════════════════════════════════════════
#  CONFIG  (timing in ms, mirrors the web CFG)
# ════════════════════════════════════════════════════════════════════════════
TASK_VERSION = 'v17.3'            # cue-stakes + silent-accrual variant of v16
# Changes from v17.2 (no RNG draws added, so the bandit/bonus schedule is
# byte-identical for any seed):
#   1. Two continuous -100..100 VAS (hunger, then an energy/engagement scale) are
#      shown once, after practice and before the recorded task. The marker starts
#      centered, is set with the mouse or the Left/Right arrow keys, and SPACE
#      confirms. Responses are written to a separate <run>_vas.csv, so the main
#      task CSV schema is unchanged.
#   2. The post-response fixation "+" in the bonus round is now blank. The 600 ms
#      neutral gap (and its iEEG marker/timestamp) is kept; only the visible cross
#      is removed, since it flashed between the press and the next bandit trial.

CFG = dict(
    N_ARMS=3,
    # ---- Two-run split -----------------------------------------------------
    # The session is two independent runs of N_TRIALS bandit trials each, with a
    # rest break, arm re-draw, and Q-reset between them.
    N_RUNS=2,
    N_TRIALS=100,               # bandit trials PER RUN (200 total across two runs)
    # One reversal per run. *** CHANGE THE SPLIT REVERSAL TRIAL HERE ***
    # One 1-indexed trial per run; list length must equal N_RUNS.
    REVERSAL_TRIAL_BY_RUN=[54, 46],
    REWARD_PTS=10,
    LOSS_PTS=-10,
    # Arm profiles [p_reward, p_loss]. EVs below assume symmetric REWARD_PTS=10, LOSS_PTS=-10.
    PROFILE_A=[0.80, 0.20],   # best   (EV +6)
    PROFILE_B=[0.30, 0.70],   # worst  (EV -4)
    PROFILE_C=[0.50, 0.50],   # chance (EV  0)

    # Bandit timing
    FEEDBACK_MS=1500,
    ISI_MIN_MS=400,           # post-feedback blank, jittered per trial (cosmetic stream)
    ISI_MAX_MS=700,
    ANIM_MS=400,
    ANTICIP_MIN_MS=400,       # jittered fixation between the pull and feedback (iEEG)
    ANTICIP_MAX_MS=800,
    CHOICE_DEADLINE_MS=4000,  # nudge prompt after 4 s; trial still waits for a key

    # Mini-MID bonus block. Counts are PER RUN (so 48 food + 22 neutral total).
    N_BONUS_FOOD=24,
    N_BONUS_NEUTRAL=11,
    BONUS_INTRO_MS=1000,
    CUE_MS=1500,
    # Jitter between the cue picture and the target square. Narrowed from
    # 1500-3000 so the anticipation gap is shorter and less variable (the mean
    # and spread both drop), which tightens the RT read; keep some jitter so the
    # target onset stays unpredictable and anticipatory presses are deterred.
    DELAY_MIN_MS=1200,
    DELAY_MAX_MS=2000,
    GRACE_MS=500,             # window stays open past the deadline: late press logged with RT, not no_response
    BONUS_FEEDBACK_MS=1500,   # used only for the premature "too soon" deterrent now
    # Neutral gap after a hit/miss response. No points, no hit/miss valence: the
    # bonus is accrued silently and revealed only at the end, so no reward event
    # sits between the probe and the next bandit trial (removes the RT->bonus->
    # learning confound at the source).
    POST_TARGET_MS=600,
    BONUS_PREMATURE_PENALTY_MS=1500,   # blank pause after a too-early press, so anticipating costs time
    # Stakes are shown on the cue and are EQUAL for food and neutral, so the
    # food-vs-neutral RT difference stays clean (points buy food either way, so
    # equal points is still food vigor without a magnitude confound).
    BONUS_PTS=15,
    # Fixed response window (staircase removed in v16). Every probe, every
    # participant, uses this one deadline so probe RT variance is preserved.
    FIXED_WINDOW_MS=550,
    # Bonus placement across the bandit stream
    BONUS_FIRST_AFTER=8,
    BONUS_REV_BUFFER=3,
    BONUS_MIN_GAP=2,
)

# Display / hardware (overridable in the startup dialog)
FULLSCREEN = True
BG_COLOR = [-0.5, -0.5, -0.5]   # dark grey, PsychoPy [-1..1] RGB
WIN_SIZE = [1280, 800]          # used only when FULLSCREEN is False

# Self-contained layout: everything is anchored to the folder this script lives in
# (name it _task), so stimuli, the script, and the data output all sit together and
# the task runs from one spot regardless of the working directory.
HERE = os.path.dirname(os.path.abspath(__file__))

SYMBOL_DIR = os.path.join(HERE, 'stimuli', 'shapes')
IMG_DIRS = dict(
    sweet=os.path.join(HERE, 'stimuli', 'win', 'sweet'),
    savory=os.path.join(HERE, 'stimuli', 'win', 'savory'),
    neutral=os.path.join(HERE, 'stimuli', 'neutral'),
    loss=os.path.join(HERE, 'stimuli', 'loss'),
)
SYMBOL_NAMES = ['heart', 'circle', 'triangle']   # run 1 symbols; files in stimuli/shapes/
# Run 2 uses three DISTINCT symbols so the model's Q-reset at the run break is honest
# (value learned about a shape does not carry into run 2). Same folder, matched style.
SYMBOL_NAMES_RUN2 = ['knot', 'rose', 'cinquefoil']

# Each event type sends a DISTINCT trigger code (1-255), so iEEG can tell event types
# apart (cue vs target vs response vs win/loss/feedback) from the trigger channel
# alone, instead of every event sharing one marker. The optional photodiode carries
# precise onset timing; these codes carry event identity. Both are logged.
EVENT_CODES = dict(
    choice_onset=10, choice_made=11,
    bandit_win=20, bandit_loss=21,
    bonus_intro=30, cue_food=31, cue_neutral=32,
    fixation=33, anticipation=34, target=35, response=36,
    bonus_feedback=37,
)
CODE_NAMES = {v: k for k, v in EVENT_CODES.items()}   # code -> name, for the log


# ════════════════════════════════════════════════════════════════════════════
#  mulberry32 RNG  (two seed-derived streams)
# ════════════════════════════════════════════════════════════════════════════
# The bandit schedule rides on the MAIN stream; its call order must not change.
# All cosmetic randomness (food set, which pictures, corner tilt, bonus deck and
# placement) rides on a SECOND stream so adding stimuli never shifts the schedule.
def _imul(a, b):
    """32-bit integer multiply matching JS Math.imul (low 32 bits)."""
    return ((a & 0xFFFFFFFF) * (b & 0xFFFFFFFF)) & 0xFFFFFFFF


def make_rng(seed):
    """Return a mulberry32 generator (float in [0,1)) seeded with a uint32."""
    a = seed & 0xFFFFFFFF

    def rng():
        nonlocal a
        a = (a + 0x6D2B79F5) & 0xFFFFFFFF
        t = _imul(a ^ (a >> 15), 1 | a)
        t = ((t + _imul(t ^ (t >> 7), 61 | t)) & 0xFFFFFFFF) ^ t
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296.0
    return rng


# ════════════════════════════════════════════════════════════════════════════
#  Startup dialog
# ════════════════════════════════════════════════════════════════════════════
# Collect participant info and session options. A blank seed draws a fresh random
# uint32 (logged in every row); a fixed seed reproduces a schedule exactly.
def run_dialog():
    info = {
        'participant': '',
        'session': '001',
        'seed (blank = random)': '',
        'food_set': ['auto', 'sweet', 'savory', 'sweet+savory'],
        'photodiode': False,
        'triggers': ['none', 'serial', 'parallel'],
        'serial_port': 'COM3',
        'parallel_address (hex)': '0x378',
    }
    ok = gui.DlgFromDict(info, title='Bandit + Bonus',
                         order=['participant', 'session', 'seed (blank = random)',
                                'food_set', 'photodiode', 'triggers',
                                'serial_port', 'parallel_address (hex)'])
    if not ok.OK:
        core.quit()
    seed_txt = str(info['seed (blank = random)']).strip()
    seed = (int(seed_txt, 0) & 0xFFFFFFFF) if seed_txt else \
        (int.from_bytes(os.urandom(4), 'little'))
    return dict(
        pid=str(info['participant']).strip() or 'test',
        session=str(info['session']).strip() or '001',
        seed=seed,
        food_override=(None if info['food_set'] == 'auto' else info['food_set']),
        photodiode=bool(info['photodiode']),
        trig_mode=info['triggers'],
        serial_port=str(info['serial_port']).strip(),
        parallel_addr=int(str(info['parallel_address (hex)']), 0),
    )


# ════════════════════════════════════════════════════════════════════════════
#  Triggers: each event sends a distinct code
# ════════════════════════════════════════════════════════════════════════════
# send(code) emits the event-specific value to the recording system (one byte on
# serial, or a TTL value on a parallel port) and logs the code with its name. With no
# device present the code is logged only, so offline alignment still works from the log.
class Triggers:
    def __init__(self, mode='none', serial_port='COM3', address=0x378):
        self.mode = mode
        self.port = None
        if mode == 'serial':
            try:
                import serial
                self.port = serial.Serial(serial_port, baudrate=115200, timeout=0)
            except Exception as e:
                logging.warn('Serial port unavailable (%s); triggers logged only.' % e)
                self.mode = 'none'
        elif mode == 'parallel':
            try:
                from psychopy import parallel
                self.port = parallel.ParallelPort(address=address)
                self.port.setData(0)
            except Exception as e:
                logging.warn('Parallel port unavailable (%s); triggers logged only.' % e)
                self.mode = 'none'

    def send(self, code):
        # One byte on serial, or a TTL value on parallel. Clamped to a single byte.
        code = int(code) & 0xFF
        if self.mode == 'serial' and self.port is not None:
            self.port.write(bytes([code]))
        elif self.mode == 'parallel' and self.port is not None:
            self.port.setData(code)
        logging.exp('TRIGGER %d (%s)' % (code, CODE_NAMES.get(code, '?')))

    def clear(self):
        if self.mode == 'parallel' and self.port is not None:
            self.port.setData(0)                             # serial needs no line clear


# ════════════════════════════════════════════════════════════════════════════
#  Reward schedule  (reproduces the web logic and rand() call order)
# ════════════════════════════════════════════════════════════════════════════
def shuffle3(rand):
    """Fisher-Yates permutation of [0,1,2] using the MAIN stream (2 rand calls)."""
    a = [0, 1, 2]
    for i in range(len(a) - 1, 0, -1):
        j = int(rand() * (i + 1))
        a[i], a[j] = a[j], a[i]
    return a


def sample_outcome(rand, profile):
    """Draw 'reward' or 'loss' from a [p_reward, p_loss] profile (1 rand call)."""
    return 'reward' if rand() < profile[0] else 'loss'


def apply_reversal(rand, profiles, zero_idx, swap_idx_set):
    """If this trial is a reversal trial, rotate all three profiles in a random
    direction (1 rand call). A 3-cycle has no fixed point, so every arm changes
    role. swap_idx_set is the set of 0-indexed reversal trials."""
    if zero_idx in swap_idx_set:
        p = profiles
        if rand() < 0.5:
            return [p[2], p[0], p[1]], True   # rotate right
        return [p[1], p[2], p[0]], True       # rotate left
    return profiles, False


# ════════════════════════════════════════════════════════════════════════════
#  Cosmetic helpers (SECOND stream): food set, image draws, bonus schedule
# ════════════════════════════════════════════════════════════════════════════
def discover_images(folder):
    """Return sorted image paths in a folder (png/jpg/jpeg), or [] if none."""
    if not os.path.isdir(folder):
        return []
    files = []
    for ext in ('*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG'):
        files.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(files)


_ASPECT_CACHE = {}


def fit_size(path, box):
    """Size (w, h) in height units that fits the image inside a box-by-box square
    while preserving its aspect ratio. The pixel dimensions are read once (header
    only) and cached, so repeats are free."""
    ar = _ASPECT_CACHE.get(path)
    if ar is None:
        try:
            with Image.open(path) as im:
                w, h = im.size
            ar = w / float(h)
        except Exception:
            ar = 1.0
        _ASPECT_CACHE[path] = ar
    return (box, box / ar) if ar >= 1.0 else (box * ar, box)


def draw_images(srand, files, n):
    """Pick n image paths without replacement where possible, else with
    replacement, using the cosmetic stream. Returns [] if the set is empty."""
    if not files:
        return []
    pool, out = list(files), []
    for _ in range(n):
        if not pool:
            pool = list(files)
        idx = int(srand() * len(pool))
        out.append(pool.pop(idx))
    return out


def build_bonus_schedule(srand, n_food, n_neutral, n_trials, rev_trials):
    """Phase-stratified bonus schedule (cosmetic stream). Splits the bandit
    stream into the three inter-reversal phases, gives each phase an equal count
    of bonus trials and a near-even food/neutral split (so the small neutral cell
    is represented in every phase), bin-spreads positions within each phase with
    a reversal buffer and a minimum gap, and shuffles cue order within each phase
    so cue type stays unpredictable (deliberately not alternating). Returns
    {bandit_trial_index: is_food}."""
    lo  = CFG['BONUS_FIRST_AFTER']
    buf = CFG['BONUS_REV_BUFFER']
    gap = CFG['BONUS_MIN_GAP']
    n_bonus = n_food + n_neutral
    n_ph = len(rev_trials) + 1

    # Per-phase eligible ranges [a, b] inclusive, each clearing the reversal buffer.
    edges = [lo] + list(rev_trials) + [n_trials]
    phases = []
    for k in range(n_ph):
        a = edges[k] + (buf if k > 0 else 0)
        b = edges[k + 1] - 1 - (buf if k < len(rev_trials) else 0)
        phases.append((a, b))

    # Equal bonus count per phase; food split near-even, neutral takes the rest.
    base, rem = divmod(n_bonus, n_ph)
    per_phase = [base + (1 if i < rem else 0) for i in range(n_ph)]
    fb, fr = divmod(n_food, n_ph)
    food_per = [fb + (1 if i < fr else 0) for i in range(n_ph)]

    insert_at, used = {}, []
    def too_close(p):
        return any(abs(u - p) < gap for u in used)

    for (a, b), n_p, n_f in zip(phases, per_phase, food_per):
        # One buffered position per equal-width bin across the phase.
        bin_w = (b - a + 1) / float(n_p)
        pos = []
        for i in range(n_p):
            s = int(a + i * bin_w)
            e = max(s, int(a + (i + 1) * bin_w) - 1)
            p, tries = s, 0
            while True:
                p = s + int(srand() * (e - s + 1))
                tries += 1
                if not too_close(p) or tries >= 40:
                    break
            used.append(p); pos.append(p)
        pos.sort()
        # Shuffle this phase's cue types and map them onto the sorted positions.
        cues = [True] * n_f + [False] * (n_p - n_f)
        for i in range(len(cues) - 1, 0, -1):
            j = int(srand() * (i + 1))
            cues[i], cues[j] = cues[j], cues[i]
        for p, c in zip(pos, cues):
            insert_at[p] = c
    return insert_at


# ════════════════════════════════════════════════════════════════════════════
#  Window, stimuli, and small drawing utilities
# ════════════════════════════════════════════════════════════════════════════
def build_window():
    """Open the experiment window in height units (resolution independent)."""
    win = visual.Window(size=WIN_SIZE, fullscr=FULLSCREEN, color=BG_COLOR,
                        units='height', allowGUI=False, winType='pyglet')
    win.mouseVisible = True
    return win


def make_sad_face(win, pos, size, ori=0.0):
    """A font-independent sad face (circle, two eyes, frown) as a stim list, so
    rendering does not depend on emoji-capable fonts."""
    r = size / 2.0
    head = visual.Circle(win, radius=r, pos=pos, fillColor=[1, 0.85, -0.6],
                        lineColor=[0.2, 0.0, -0.6], lineWidth=2, ori=ori)
    eye_dx, eye_dy, eye_r = r * 0.38, r * 0.30, r * 0.12
    eyes = [visual.Circle(win, radius=eye_r, pos=(pos[0] + sx * eye_dx, pos[1] + eye_dy),
                          fillColor=[-1, -1, -1], lineColor=[-1, -1, -1])
            for sx in (-1, 1)]
    # Frown: lower arc, drawn as a downward-curving polyline.
    pts = []
    for k in range(9):
        ang = math.radians(200 + 140 * k / 8.0)   # spans the lower mouth region
        pts.append((pos[0] + 0.45 * r * math.cos(ang),
                    pos[1] - 0.30 * r + 0.45 * r * math.sin(ang)))
    frown = visual.ShapeStim(win, vertices=pts, closeShape=False,
                            lineColor=[-1, -1, -1], lineWidth=4)
    return [head] + eyes + [frown]


def make_arrow(win, pos, ori, size, color=(1, 1, 1)):
    """A filled arrow as a ShapeStim, font-independent so it renders the same on
    any system. The base shape points right; ori rotates it clockwise in degrees
    (0 right, 90 down, 180 left). Used for the choice-screen key labels because
    the Unicode arrow glyphs rendered inconsistently on this setup."""
    base = [(-0.5, 0.12), (0.1, 0.12), (0.1, 0.28), (0.5, 0.0),
            (0.1, -0.28), (0.1, -0.12), (-0.5, -0.12)]
    verts = [(x * size, y * size) for x, y in base]
    return visual.ShapeStim(win, vertices=verts, pos=pos, ori=ori,
                            fillColor=color, lineColor=color, closeShape=True)


# ════════════════════════════════════════════════════════════════════════════
#  Incremental CSV writer (crash-safe: each row is flushed as it completes)
# ════════════════════════════════════════════════════════════════════════════
# A fixed superset of columns (bandit columns first, then bonus-only) keeps the
# schema identical to the web export while allowing per-row writing.
FIELDNAMES = [
    'participant_id', 'session', 'task_version', 'trial_type', 't_onset_s', 'seed',
    'run',                       # v16: 1-indexed run (1..N_RUNS); Q resets each run
    # bandit
    'trial', 'swap_count', 'position1', 'position2', 'position3',
    'p_reward_pos1', 'p_reward_pos2', 'p_reward_pos3',
    'choice', 'chosen_logo', 'rt_s', 'choice_late', 'outcome', 'points',
    'optimal_position', 'is_optimal', 'optimal_points', 'regret', 'cumulative_score',  # regret = realized counterfactual (optimal_points - points), can be negative
    'anticip_ms', 'isi_ms',
    'cf_outcome_pos1', 'cf_outcome_pos2', 'cf_outcome_pos3',
    'cf_points_pos1', 'cf_points_pos2', 'cf_points_pos3',
    # bonus-only
    'bonus_trial_index', 'position_in_bandit_stream', 'food_set', 'cue_type',
    'food_bonus_cue', 'cue_image', 'cue_duration_ms', 'anticipatory_delay_ms',
    'adaptive_window_ms', 'target_response_key', 'target_rt_ms', 'premature_rt_ms',
    'target_hit', 'target_miss', 'target_too_fast', 'target_no_response',
    'bonus_points_earned', 'bonus_cumulative', 'bonus_hit_rate',
    'cue_onset_ms', 'delay_onset_ms', 'target_onset_ms', 'response_ms',
    'feedback_onset_ms', 'trigger_code',
]


class DataLog:
    def __init__(self, path):
        self.path = path
        self.f = open(path, 'w', newline='', encoding='utf-8')
        self.w = csv.DictWriter(self.f, fieldnames=FIELDNAMES,
                                restval='', extrasaction='ignore')
        self.w.writeheader()
        self.f.flush()

    def write(self, row):
        self.w.writerow(row)
        self.f.flush()          # survive a crash mid-session (important for patients)
        os.fsync(self.f.fileno())

    def close(self):
        try:
            self.f.close()
        except Exception:
            pass


def make_run_dir(pid, ses):
    """Create and return (path, name) for a fresh run folder under _task/data/. Reruns
    with the same id/session never overwrite: the copy number increments to the
    first unused value, giving data/sub-<pid>_ses-<ses>_<n>. os.makedirs fails if
    the folder already exists, so the number is claimed atomically."""
    data_root = os.path.join(HERE, 'data')
    os.makedirs(data_root, exist_ok=True)
    base = 'sub-%s_ses-%s' % (pid, ses)
    n = 1
    while True:
        name = '%s_%d' % (base, n)
        path = os.path.join(data_root, name)
        try:
            os.makedirs(path)
            return path, name
        except FileExistsError:
            n += 1


# ════════════════════════════════════════════════════════════════════════════
#  Main experiment
# ════════════════════════════════════════════════════════════════════════════
def main():
    settings = run_dialog()
    seed = settings['seed']

    # Two independent streams from the one seed (main = schedule, srand = cosmetic).
    rand = make_rng(seed)
    srand = make_rng((seed ^ 0x9E3779B9) & 0xFFFFFFFF)

    # ---- Output folder (never overwrites; numbered copy per run) -----------
    run_dir, run_name = make_run_dir(settings['pid'], settings['session'])
    log = DataLog(os.path.join(run_dir, run_name + '.csv'))
    logging.LogFile(os.path.join(run_dir, run_name + '.log'), level=logging.EXP)
    # Record the fixed response window once (staircase removed in v16), so the
    # analyzer can confirm the deadline actually used instead of assuming it.
    logging.exp('FIXED_WINDOW_MS=%d N_RUNS=%d N_TRIALS_PER_RUN=%d' %
                (CFG['FIXED_WINDOW_MS'], CFG['N_RUNS'], CFG['N_TRIALS']))

    # ---- Window, triggers, photodiode --------------------------------------
    win = build_window()
    win.mouseVisible = False
    trig = Triggers(settings['trig_mode'], serial_port=settings['serial_port'],
                    address=settings['parallel_addr'])
    pd_on = settings['photodiode']
    aspect = win.size[0] / float(win.size[1])
    pd_stim = visual.Rect(win, width=0.07, height=0.07,
                          pos=(aspect * 0.5 - 0.05, -0.5 + 0.05),
                          fillColor=[-1, -1, -1], lineColor=None) if pd_on else None

    # ---- Stimuli ------------------------------------------------------------
    card_xs = [-0.42, 0.0, 0.42]
    cards = [visual.Rect(win, width=0.34, height=0.42, pos=(x, 0.0),
                         fillColor=[0.85, 0.85, 0.85], lineColor=[0.4, 0.4, 0.4],
                         lineWidth=2) for x in card_xs]

    def make_symbol_stim(name):
        """One stim per symbol: the PNG in stimuli/shapes if present (aspect
        preserved), otherwise a dark primitive so the task still runs."""
        path = os.path.join(SYMBOL_DIR, name + '.png')
        if os.path.exists(path):
            return visual.ImageStim(win, image=path, size=fit_size(path, 0.24))
        dark = [-0.7, -0.7, -0.7]                    # visible on the light card
        if name == 'circle':
            return visual.Circle(win, radius=0.11, fillColor=dark, lineColor=None)
        if name == 'triangle':
            return visual.Polygon(win, edges=3, radius=0.13, fillColor=dark, lineColor=None)
        if name == 'knot':
            return visual.Polygon(win, edges=5, radius=0.13, fillColor=dark, lineColor=None)
        if name == 'rose':
            return visual.Polygon(win, edges=6, radius=0.13, fillColor=dark, lineColor=None)
        if name == 'cinquefoil':
            return visual.Polygon(win, edges=8, radius=0.13, fillColor=dark, lineColor=None)
        return visual.Polygon(win, edges=4, radius=0.13, fillColor=dark, lineColor=None)  # heart placeholder (diamond)

    shape_stims = {n: make_symbol_stim(n) for n in SYMBOL_NAMES + SYMBOL_NAMES_RUN2}
    # Arrow-key labels under each slot (left, down, right map to slots 0, 1, 2).
    CHOICE_KEYS = ['left', 'down', 'right']
    arrow_oris = [180, 90, 0]                # left, down, right (clockwise degrees)
    # Drawn as shapes, not font glyphs: the Unicode arrows rendered inconsistently
    # here (a dash, then a hash), so vector arrows guarantee correct display.
    key_labels = [make_arrow(win, (x, -0.30), o, 0.12)
                  for x, o in zip(card_xs, arrow_oris)]

    header_score = visual.TextStim(win, text='', pos=(aspect * 0.5 - 0.04, 0.45),
                                   height=0.035, color='white', anchorHoriz='right')
    header_trial = visual.TextStim(win, text='', pos=(aspect * 0.5 - 0.04, 0.45),
                                   height=0.035, color='white', anchorHoriz='right')
    prog_bg = visual.Rect(win, width=aspect, height=0.012, pos=(0, 0.49),
                          fillColor=[0.2, 0.2, 0.2], lineColor=None)
    # Rect uses `anchor`, not the TextStim-only `anchorHoriz`. 'left' pins the
    # bar's left edge to pos so it grows rightward as width is updated.
    prog_fg = visual.Rect(win, width=0.0001, height=0.012, pos=(0, 0.49),
                          fillColor=[0.1, 0.6, 0.4], lineColor=None, anchor='left')
    prompt = visual.TextStim(win, text='', pos=(0, -0.36), height=0.045, color='white')

    # font='Arial' gives the bold flag a real bold face to draw (the default
    # font does not always carry a bold weight).
    fb_center = visual.TextStim(win, text='', pos=(0, 0), height=0.10, bold=True, font='Arial')
    cue_stim = visual.ImageStim(win, pos=(0, 0))     # size set per image (aspect preserved)
    # Stakes shown on the cue, below the picture. Equal magnitude for food and
    # neutral, so it cues the incentive (MID-style) without a food/neutral value gap.
    stakes_stim = visual.TextStim(win, text='', pos=(0, -0.40), height=0.05,
                                  color=[1.0, 0.85, 0.2], bold=True, font='Arial')
    fix_stim = visual.TextStim(win, text='+', pos=(0, 0), height=0.08, color='white')
    square = visual.Rect(win, width=0.30, height=0.30, pos=(0, 0),
                         fillColor=None, lineColor=[1, 1, 1], lineWidth=15)
    big = visual.TextStim(win, text='', pos=(0, 0.05), height=0.07, color='white', bold=True)
    sub = visual.TextStim(win, text='', pos=(0, -0.08), height=0.04, color='white')
    kb = keyboard.Keyboard()

    # Corner geometry for bandit feedback images / sad faces (size set per image).
    # Pulled in off the corners (was 0.34/0.30) so the images sit a little
    # closer to the centered points readout.
    corner_pos = [(-aspect * 0.30, 0.28), (aspect * 0.30, 0.28),
                  (-aspect * 0.30, -0.28), (aspect * 0.30, -0.28)]
    corner_imgs = [visual.ImageStim(win, pos=p) for p in corner_pos]

    # ---- Image sets + session food set -------------------------------------
    images = {k: discover_images(v) for k, v in IMG_DIRS.items()}
    for _paths in images.values():               # warm aspect cache (header reads only)
        for _p in _paths:
            fit_size(_p, 1.0)

    def food_set():
        """Assign sweet/savory once. 'auto' draws from the cosmetic stream; an
        explicit override is honoured without consuming a draw."""
        if state['food_set'] is None:
            if settings['food_override'] is not None:
                state['food_set'] = settings['food_override']
            else:
                state['food_set'] = 'sweet' if srand() < 0.5 else 'savory'
        return state['food_set']

    def win_pool():
        """Win-image folder(s) for the session food set. 'sweet+savory' returns
        both lists so draws can sample each folder equally."""
        fs = food_set()
        if fs == 'sweet':
            return [images['sweet']]
        if fs == 'savory':
            return [images['savory']]
        return [images['sweet'], images['savory']]   # sweet+savory

    def draw_win(srand_, n):
        """Draw n win-image paths. Single-folder sets defer to draw_images (no
        extra RNG draw, so existing seeds reproduce). For sweet+savory, each image
        picks a folder 50/50 then an image from it, equalizing folder share."""
        pools = win_pool()
        if len(pools) == 1:
            return draw_images(srand_, pools[0], n)
        out = []
        for _ in range(n):
            both = pools[0] and pools[1]
            pool = (pools[0] if srand_() < 0.5 else pools[1]) if both else (pools[0] or pools[1])
            out.extend(draw_images(srand_, pool, 1))
        return out

    # ---- Task state ---------------------------------------------------------
    state = dict(
        run=1,                  # 1-indexed current run (1..N_RUNS)
        trial=0,                # 0-indexed completed bandit trials WITHIN the current run
        score=0,                # cumulative across runs (points carry; Q-values do not)
        profiles=[list(CFG['PROFILE_A']), list(CFG['PROFILE_B']), list(CFG['PROFILE_C'])],
        swap_count=0,
        food_set=None,
        bonus_count=0, bonus_score=0, bonus_hits=0,
        bonus_window=CFG['FIXED_WINDOW_MS'],   # constant; kept in state so logging is unchanged
        task_start=None,
    )
    # 0-indexed reversal trial(s) for the CURRENT run; reset at the top of each run.
    swap_idx_set = set()

    # Per-run arm placement, kept in state so it can be re-drawn each run and the
    # closures below always read the current run's mapping. Draw order matches the
    # web (symbolMap then slotOrder). start_run() re-draws these at each run start.
    state['symbol_map'] = shuffle3(rand)   # logical arm -> symbol index
    state['slot_order'] = shuffle3(rand)   # screen slot  -> logical arm

    def sym_name(arm):
        # Run 2 shows the novel symbol set; practice and run 1 use the original.
        names = SYMBOL_NAMES_RUN2 if state.get('run') == 2 else SYMBOL_NAMES
        return names[state['symbol_map'][arm]]

    def save_and_close():
        log.close()

    # ---- Timed-hold helper --------------------------------------------------
    def hold(drawables, dur_ms, trig_code=None):
        """Draw a static screen, flip (onset), pulse trigger + photodiode, hold."""
        for d in drawables:
            d.draw()
        if pd_on and trig_code is not None:
            pd_stim.fillColor = [1, 1, 1]
            pd_stim.draw()
        win.flip()                                  # onset
        if trig_code is not None:
            trig.send(trig_code)
        clk = core.Clock()
        while clk.getTime() < dur_ms / 1000.0:
            for k in kb.getKeys(keyList=['escape'], clear=True):
                if k.name == 'escape':
                    save_and_close(); win.close(); core.quit()
            for d in drawables:                     # redraw, then flip (no blank frame)
                d.draw()
            if pd_on:
                pd_stim.fillColor = [-1, -1, -1]
                pd_stim.draw()
            win.flip()
            trig.clear()                            # marker line returns to baseline
        return clk

    def update_header():
        # Bonus total is NOT shown live (silent accrual); only the bandit score.
        header_score.text = 'Score: %d pts' % state['score']
        header_trial.text = ''                       # trial counter hidden from participant
        # Progress across the whole session (both runs), not just the current run.
        done = (state['run'] - 1) * CFG['N_TRIALS'] + state['trial']
        frac = done / float(CFG['N_RUNS'] * CFG['N_TRIALS'])
        prog_fg.width = max(0.0001, aspect * frac)
        prog_fg.pos = (-aspect * 0.5, 0.49)

    # Shared bandit feedback renderer (real trial + practice). Draws four tilt
    # values then four image picks from `rng` regardless of outcome, so the call
    # count is identical for wins and losses and the cosmetic stream stays aligned.
    def show_bandit_feedback(points, rng, send_trig=True):
        is_win = points > 0
        fb_center.text = ('+%d Points' % points) if points > 0 else ('%d Points' % points)
        fb_center.color = [0.1, 0.8, 0.3] if is_win else [0.9, 0.2, 0.2]
        tilts = [int(rng() * 11) - 5 for _ in range(4)]   # srandInt(-5,5); 4 draws kept so wins/losses consume the stream equally
        pics = draw_win(rng, 4) if is_win else draw_images(rng, images['loss'], 4)
        drawables = [fb_center]
        if is_win:
            # Win: four tilted food images framing the centered points readout.
            fb_center.pos = (0, 0)
            for i, ci in enumerate(corner_imgs):
                ci.ori = tilts[i]
                if i < len(pics) and pics[i]:
                    ci.image = pics[i]
                    ci.size = fit_size(pics[i], 0.50)    # 2.5x box, aspect preserved
                    ci.pos = corner_pos[i]
                    drawables.append(ci)
        else:
            # Loss: one image (or a drawn sad face) to the right of the points; the
            # text+gap+image group is centered as a unit. boundingBox is px -> height units.
            gap = 0.13
            minor_tilt = 0.25 * tilts[0]                 # very slight tilt (about +/-3 deg)
            tw = fb_center.boundingBox[0] / float(win.size[1])
            if not tw or tw <= 0:                        # fallback if boundingBox is unset
                tw = len(fb_center.text) * 0.10 * 0.55
            if pics and pics[0]:
                ci = corner_imgs[0]
                ci.ori = minor_tilt
                ci.image = pics[0]
                ci.size = fit_size(pics[0], 0.30)        # smaller than the win box
                iw = ci.size[0]
                total = tw + gap + iw
                fb_center.pos = (-total / 2.0 + tw / 2.0, 0)
                ci.pos = (total / 2.0 - iw / 2.0, 0)
                drawables.append(ci)
            else:                                        # loss folder empty -> one sad face, same centering
                iw = 0.30
                total = tw + gap + iw
                fb_center.pos = (-total / 2.0 + tw / 2.0, 0)
                drawables.extend(make_sad_face(win, (total / 2.0 - iw / 2.0, 0), 0.30, ori=minor_tilt))
        tc = (EVENT_CODES['bandit_win'] if is_win else EVENT_CODES['bandit_loss']) if send_trig else None
        hold(drawables, CFG['FEEDBACK_MS'], trig_code=tc)

    # ── Bandit trial ─────────────────────────────────────────────────────────
    def bandit_trial():
        z = state['trial']                          # 0-indexed index of THIS trial (within run)
        slot_order = state['slot_order']            # current run's slot->arm mapping
        t_onset = core.getTime() - state['task_start']

        # Reversal check happens before sampling, exactly as in the web version.
        state['profiles'], swapped = apply_reversal(rand, state['profiles'], z, swap_idx_set)
        if swapped:
            state['swap_count'] += 1

        # Predetermined per-arm outcomes this trial (3 draws, arm order 0,1,2).
        cf_out = [sample_outcome(rand, p) for p in state['profiles']]
        cf_pts = [CFG['REWARD_PTS'] if o == 'reward' else CFG['LOSS_PTS'] for o in cf_out]

        # Draw choice screen; collect an arrow-key press (4 s -> nudge, then keep waiting).
        for c in cards:
            c.fillColor = [0.85, 0.85, 0.85]
        # Symbol shown at each slot this trial (symbol_map + slot_order are per-session).
        slot_syms = [shape_stims[sym_name(slot)] for slot in slot_order]
        prompt.text = ''                            # press-arrows text removed; drawn arrows convey the keys
        prompt.color = 'white'
        update_header()

        def draw_choice():
            prog_bg.draw(); prog_fg.draw(); header_score.draw(); header_trial.draw()
            for c in cards:
                c.draw()
            for i, st in enumerate(slot_syms):
                st.pos = (card_xs[i], 0.0)
                st.opacity = 1.0
                st.draw()
            for kl in key_labels:
                kl.draw()
            prompt.draw()

        draw_choice()
        if pd_on:
            pd_stim.fillColor = [1, 1, 1]; pd_stim.draw()
        win.flip()                                  # choice onset
        trig.send(EVENT_CODES['choice_onset'])
        kb.clearEvents()
        kb.clock.reset()                            # key .rt measured from choice onset
        chosen_slot = None
        rt = None
        late = False
        while chosen_slot is None:
            if not late and kb.clock.getTime() >= CFG['CHOICE_DEADLINE_MS'] / 1000.0:
                late = True
                prompt.text = 'Please answer faster.'
                prompt.color = 'red'
            for k in kb.getKeys(keyList=CHOICE_KEYS + ['escape'], waitRelease=False, clear=True):
                if k.name == 'escape':
                    save_and_close(); win.close(); core.quit()
                if chosen_slot is None:
                    chosen_slot = CHOICE_KEYS.index(k.name)   # left/down/right -> slot 0/1/2
                    rt = k.rt
            draw_choice()
            if pd_on:
                pd_stim.fillColor = [-1, -1, -1]; pd_stim.draw()
            win.flip()
            trig.clear()

        arm = slot_order[chosen_slot]               # logical arm at the chosen slot
        trig.send(EVENT_CODES['choice_made'])

        # Brief "pull" animation: highlight the chosen card, dim the others.
        anim = core.Clock()
        while anim.getTime() < CFG['ANIM_MS'] / 1000.0:
            for i, c in enumerate(cards):
                c.opacity = 1.0 if i == chosen_slot else 0.35
                c.draw()
            for i, st in enumerate(slot_syms):
                st.pos = (card_xs[i], 0.0)
                st.opacity = 1.0 if i == chosen_slot else 0.35
                st.draw()
            prog_bg.draw(); prog_fg.draw(); header_score.draw(); header_trial.draw()
            win.flip()
            trig.clear()
        for c in cards:
            c.opacity = 1.0

        # Jittered anticipation fixation: decorrelates choice- and feedback-locked
        # iEEG responses and gives feedback epochs a clean pre-stimulus baseline.
        anticip_ms = CFG['ANTICIP_MIN_MS'] + int(srand() * (CFG['ANTICIP_MAX_MS'] - CFG['ANTICIP_MIN_MS'] + 1))
        hold([fix_stim], anticip_ms, trig_code=EVENT_CODES['anticipation'])

        # Resolve outcome and score (chosen arm's predetermined outcome).
        outcome = cf_out[arm]
        points = cf_pts[arm]
        state['score'] += points

        # Position-frame bookkeeping (1 = left .. 3 = right), matching the web export.
        def L(pos):
            return slot_order[pos - 1]
        choice_pos = slot_order.index(arm) + 1
        p_rewards = [pr[0] for pr in state['profiles']]
        opt_arm = p_rewards.index(max(p_rewards))
        optimal_pos = slot_order.index(opt_arm) + 1
        optimal_points = cf_pts[opt_arm]
        isi_ms = CFG['ISI_MIN_MS'] + int(srand() * (CFG['ISI_MAX_MS'] - CFG['ISI_MIN_MS'] + 1))

        row = {
            'participant_id': settings['pid'], 'session': settings['session'],
            'task_version': TASK_VERSION,
            'run': state['run'],
            'trial_type': 'bandit', 't_onset_s': round(t_onset, 4), 'seed': seed,
            'trial': z + 1, 'swap_count': state['swap_count'],
            'position1': sym_name(L(1)), 'position2': sym_name(L(2)), 'position3': sym_name(L(3)),
            'p_reward_pos1': round(state['profiles'][L(1)][0], 4),
            'p_reward_pos2': round(state['profiles'][L(2)][0], 4),
            'p_reward_pos3': round(state['profiles'][L(3)][0], 4),
            'choice': choice_pos, 'chosen_logo': sym_name(arm),
            'rt_s': round(rt, 4), 'choice_late': 1 if late else 0,
            'outcome': outcome, 'points': points,
            'optimal_position': optimal_pos, 'is_optimal': 1 if choice_pos == optimal_pos else 0,
            'optimal_points': optimal_points, 'regret': optimal_points - points,
            'cumulative_score': state['score'], 'anticip_ms': anticip_ms, 'isi_ms': isi_ms,
            'cf_outcome_pos1': cf_out[L(1)], 'cf_outcome_pos2': cf_out[L(2)], 'cf_outcome_pos3': cf_out[L(3)],
            'cf_points_pos1': cf_pts[L(1)], 'cf_points_pos2': cf_pts[L(2)], 'cf_points_pos3': cf_pts[L(3)],
            'trigger_code': EVENT_CODES['bandit_win'] if outcome == 'reward' else EVENT_CODES['bandit_loss'],
        }
        log.write(row)
        state['trial'] += 1
        update_header()

        # Feedback then jittered blank ISI. show_bandit_feedback draws the same
        # four tilts + four picks from srand that were inlined here in v10, so the
        # cosmetic stream is consumed in the identical order.
        show_bandit_feedback(points, srand)
        hold([], isi_ms)                             # jittered blank ISI (400-700 ms)

    # ── Bonus (mini-MID) trial ───────────────────────────────────────────────
    def bonus_trial(is_food):
        state['bonus_count'] += 1
        t0 = core.getTime()
        t_onset = t0 - state['task_start']

        cue_type = ('food_%s' % food_set()) if is_food else 'neutral'
        picked = draw_win(srand, 1) if is_food else draw_images(srand, images['neutral'], 1)
        cue_path = picked[0] if picked else None
        delay_ms = CFG['DELAY_MIN_MS'] + int(srand() * (CFG['DELAY_MAX_MS'] - CFG['DELAY_MIN_MS'] + 1))
        window_ms = int(round(state['bonus_window']))

        # Phase onset timestamps (ms since t0), filled from real flip times.
        marks = dict(cue=0, delay=0, target=0, feedback=0)
        kb.clearEvents()
        kb.clock.reset()                            # key .rt measured from intro onset (t0)

        # 1) "Bonus round!" intro
        big.text = 'Bonus round!'; big.color = 'white'
        hold([big], CFG['BONUS_INTRO_MS'], trig_code=EVENT_CODES['bonus_intro'])
        kb.clearEvents()                            # ignore any press made during the intro

        press = {'key': '', 'rt_ms': None}

        def listen():
            """Quit on Escape; otherwise record the first key press. Returns True
            on the frame a response key is caught."""
            for k in kb.getKeys(waitRelease=False, clear=True):
                if k.name == 'escape':
                    save_and_close(); win.close(); core.quit()
                if press['rt_ms'] is None:
                    press['key'] = k.name
                    press['rt_ms'] = k.rt * 1000.0      # ms since t0 (kb clock)
                    return True
            return False

        def phase(drawables, dur_ms, trig_code):
            """Hold a screen while listening; break early on the first key press."""
            for d in drawables:
                d.draw()
            if pd_on and trig_code is not None:
                pd_stim.fillColor = [1, 1, 1]; pd_stim.draw()
            on = win.flip()
            if trig_code is not None:
                trig.send(trig_code)
            clk = core.Clock()
            caught = False
            while clk.getTime() < dur_ms / 1000.0:
                if press['rt_ms'] is None and listen():
                    caught = True
                    break
                for d in drawables:
                    d.draw()
                if pd_on:
                    pd_stim.fillColor = [-1, -1, -1]; pd_stim.draw()
                win.flip()
                trig.clear()
            return on, caught

        # 2) cue (with equal stakes shown for food and neutral)
        cue_trig = EVENT_CODES['cue_food'] if is_food else EVENT_CODES['cue_neutral']
        stakes_stim.text = '+%d if fast' % CFG['BONUS_PTS']
        if cue_path:
            cue_stim.image = cue_path
            cue_stim.size = fit_size(cue_path, 0.8)   # larger cue (was 0.65)
            cue_draw = [cue_stim, stakes_stim]
        else:
            sub.text = ('FOOD CUE' if is_food else 'NEUTRAL CUE') + '\n(placeholder)'
            sub.pos = (0, 0); sub.color = 'white'
            cue_draw = [sub, stakes_stim]
        on, caught = phase(cue_draw, CFG['CUE_MS'], cue_trig)
        marks['cue'] = round((on - t0) * 1000.0)
        sub.pos = (0, -0.08)
        outcome = None
        if caught:                                   # press during cue -> premature
            outcome = 'too_fast'

        # 3) anticipatory fixation
        if outcome is None:
            on, caught = phase([fix_stim], delay_ms, EVENT_CODES['fixation'])
            marks['delay'] = round((on - t0) * 1000.0)
            if caught:
                outcome = 'too_fast'

        # 4) target square (response window)
        if outcome is None:
            on, caught = phase([square], window_ms, EVENT_CODES['target'])
            marks['target'] = round((on - t0) * 1000.0)
            if caught:
                outcome = 'hit'
                trig.send(EVENT_CODES['response'])

        # 5) grace window (late press still logged as miss-with-RT)
        if outcome is None:
            on, caught = phase([], CFG['GRACE_MS'], None)
            if caught:
                outcome = 'miss'
                trig.send(EVENT_CODES['response'])   # late press still marks the motor response for iEEG
            else:
                outcome = 'no_response'

        # Classify RT fields relative to the right phase onset.
        prem_rt = rt_ms = ''
        resp_ms = ''
        if press['rt_ms'] is not None:
            resp_ms = round(press['rt_ms'])
            if outcome == 'too_fast':
                prem_rt = round(press['rt_ms'] - marks['cue'])
            else:
                rt_ms = round(press['rt_ms'] - marks['target'])

        # v16: fixed window, so no staircase step. Tallies only.
        # (too_fast still leaves the window unchanged, trivially, since it is constant.)
        hit = outcome == 'hit'
        pts = CFG['BONUS_PTS'] if hit else 0
        state['bonus_score'] += pts
        state['bonus_hits'] += 1 if hit else 0

        # 6) NO reward feedback (silent accrual). A hit still earns points and a
        # miss earns none (tallied above), but nothing about the outcome or the
        # points is shown. This removes the RT-contingent reward event that used to
        # sit between the probe and the next bandit trial. The only post-response
        # screen is the premature deterrent, which is instructional, not a reward.
        if outcome == 'too_fast':
            big.text = 'Too soon!'; big.color = [0.95, 0.6, 0.1]
            sub.text = 'Wait for the white square before you press.'
            fb_on = core.getTime()
            hold([big, sub], CFG['BONUS_FEEDBACK_MS'], trig_code=EVENT_CODES['bonus_feedback'])
        else:
            # Neutral gap, identical after a hit or a miss (no valence). v17.3:
            # blank instead of a fixation "+", which flashed between the press and
            # the next bandit trial. Duration and the iEEG marker/timestamp are
            # unchanged, so only the on-screen cross is removed.
            fb_on = core.getTime()
            hold([], CFG['POST_TARGET_MS'], trig_code=EVENT_CODES['bonus_feedback'])
        marks['feedback'] = round((fb_on - t0) * 1000.0)

        # Extra blank pause after a premature press so anticipating is not free.
        # No trigger and no RNG draw here, so the recorded schedule is untouched.
        if outcome == 'too_fast':
            hold([], CFG['BONUS_PREMATURE_PENALTY_MS'])

        row = {
            'participant_id': settings['pid'], 'session': settings['session'],
            'task_version': TASK_VERSION,
            'run': state['run'],
            'trial_type': 'bonus_food' if is_food else 'bonus_neutral',
            't_onset_s': round(t_onset, 4), 'seed': seed,
            'bonus_trial_index': state['bonus_count'],
            'position_in_bandit_stream': state['trial'],
            'food_set': food_set(), 'cue_type': cue_type,
            'food_bonus_cue': 1 if is_food else 0,
            'cue_image': cue_path or '',
            'cue_duration_ms': CFG['CUE_MS'], 'anticipatory_delay_ms': delay_ms,
            'adaptive_window_ms': window_ms,
            'target_response_key': press['key'], 'target_rt_ms': rt_ms, 'premature_rt_ms': prem_rt,
            'target_hit': 1 if hit else 0,
            'target_miss': 1 if outcome == 'miss' else 0,
            'target_too_fast': 1 if outcome == 'too_fast' else 0,
            'target_no_response': 1 if outcome == 'no_response' else 0,
            'bonus_points_earned': pts, 'bonus_cumulative': state['bonus_score'],
            'bonus_hit_rate': round(state['bonus_hits'] / state['bonus_count'], 3),
            'cue_onset_ms': marks['cue'], 'delay_onset_ms': marks['delay'],
            'target_onset_ms': marks['target'], 'response_ms': resp_ms,
            'feedback_onset_ms': marks['feedback'],
            'trigger_code': EVENT_CODES['response'] if hit else EVENT_CODES['bonus_feedback'],
        }
        log.write(row)

    # ---- Instructions + practice (staged pages, replayable on request) ------
    # The whole walkthrough runs on its own RNG stream (prand) and writes nothing
    # to the data log, so it cannot shift the recorded schedule or scores. The only
    # shared draw it triggers is food_set() (one srand value), which v10 also drew
    # here before the schedule build, so srand stays aligned either way.
    def run_intro_and_practice():
        # Practice randomness only: image picks, jitter, and trial order. Distinct
        # constant from srand, so practice and the recorded task never collide.
        prand = make_rng((seed ^ 0x2545F491) & 0xFFFFFFFF)
        page = visual.TextStim(win, color='white', height=0.04, wrapWidth=1.3, pos=(0, 0))
        practice_tag = visual.TextStim(win, text='Practice', pos=(0, 0.45),
                                       height=0.03, color=[0.6, 0.6, 0.6])

        def wait_page(text, keys):
            """Show a static page until one of `keys` is pressed; Escape quits.
            Returns the key pressed."""
            page.text = text
            page.draw(); win.flip()
            while True:
                for k in kb.getKeys(keyList=list(keys) + ['escape'], clear=True):
                    if k.name == 'escape':
                        save_and_close(); win.close(); core.quit()
                    if k.name in keys:
                        return k.name
                page.draw(); win.flip()

        def practice_bandit_trial(force_win):
            """One forgiving bandit trial with a forced win/loss, using the real
            cards, symbols, arrows, and feedback renderer. Outcome is fixed (not
            tied to the choice) and nothing is scored or logged."""
            for c in cards:
                c.fillColor = [0.85, 0.85, 0.85]; c.opacity = 1.0
            slot_syms = [shape_stims[sym_name(slot)] for slot in state['slot_order']]
            prompt.text = ''; prompt.color = 'white'

            def draw_choice():
                for c in cards:
                    c.draw()
                for i, st in enumerate(slot_syms):
                    st.pos = (card_xs[i], 0.0); st.opacity = 1.0; st.draw()
                for kl in key_labels:
                    kl.draw()
                practice_tag.draw(); prompt.draw()

            draw_choice(); win.flip()
            kb.clearEvents(); kb.clock.reset()
            chosen_slot, late = None, False
            while chosen_slot is None:
                if not late and kb.clock.getTime() >= CFG['CHOICE_DEADLINE_MS'] / 1000.0:
                    late = True; prompt.text = 'Please answer faster.'; prompt.color = 'red'
                for k in kb.getKeys(keyList=CHOICE_KEYS + ['escape'], clear=True):
                    if k.name == 'escape':
                        save_and_close(); win.close(); core.quit()
                    if chosen_slot is None:
                        chosen_slot = CHOICE_KEYS.index(k.name)
                draw_choice(); win.flip()

            # Same brief pull animation as the recorded trial.
            anim = core.Clock()
            while anim.getTime() < CFG['ANIM_MS'] / 1000.0:
                for i, c in enumerate(cards):
                    c.opacity = 1.0 if i == chosen_slot else 0.35; c.draw()
                for i, st in enumerate(slot_syms):
                    st.pos = (card_xs[i], 0.0)
                    st.opacity = 1.0 if i == chosen_slot else 0.35; st.draw()
                practice_tag.draw(); win.flip()
            for c in cards:
                c.opacity = 1.0

            hold([fix_stim], 500)                        # short fixed anticipation (no jitter)
            show_bandit_feedback(CFG['REWARD_PTS'] if force_win else CFG['LOSS_PTS'],
                                 prand, send_trig=False)  # practice sends no markers
            hold([], 400)

        def practice_bonus_trial(is_food, force_early=False):
            """One forgiving bonus round on a fixed (non-adaptive) window. Teaches
            wait-for-the-square timing, including the too-soon message, without
            touching the staircase, the bonus score, or the data log. When
            force_early is set the round scripts a premature after the cue, so the
            participant is guaranteed to see the "too soon" outcome once."""
            picked = draw_win(prand, 1) if is_food else draw_images(prand, images['neutral'], 1)
            cue_path = picked[0] if picked else None
            delay_ms = 1500 + int(prand() * 1500)        # 1500-3000 ms anticipatory delay
            window_ms = 600                              # generous fixed practice window

            big.text = 'Bonus round!'; big.color = 'white'
            hold([big], CFG['BONUS_INTRO_MS'])           # no marker during practice
            kb.clearEvents(); kb.clock.reset()
            press = {'hit': False, 'early': False}

            def phase(drawables, dur_ms, is_target):
                """Hold a screen while listening. A press marks a hit during the
                target square and a premature 'early' otherwise; Escape quits."""
                for d in drawables:
                    d.draw()
                win.flip()
                clk = core.Clock()
                while clk.getTime() < dur_ms / 1000.0:
                    for k in kb.getKeys(clear=True):
                        if k.name == 'escape':
                            save_and_close(); win.close(); core.quit()
                        press['hit' if is_target else 'early'] = True
                        return
                    for d in drawables:
                        d.draw()
                    win.flip()

            # Cue (food or neutral) with the same equal stakes as the task; fixation; target.
            stakes_stim.text = '+%d if fast' % CFG['BONUS_PTS']
            if cue_path:
                cue_stim.image = cue_path; cue_stim.size = fit_size(cue_path, 0.8)
                cue_draw = [cue_stim, stakes_stim]
            else:
                sub.text = ('FOOD CUE' if is_food else 'NEUTRAL CUE') + '\n(placeholder)'
                sub.pos = (0, 0); sub.color = 'white'; cue_draw = [sub, stakes_stim]
            phase(cue_draw, CFG['CUE_MS'], is_target=False)
            sub.pos = (0, -0.08)
            if force_early:
                press['early'] = True                    # scripted demo: jump to the too-soon outcome
            if not press['early']:
                phase([fix_stim], delay_ms, is_target=False)
            if not press['early']:
                phase([square], window_ms, is_target=True)

            # Practice gives TIMING guidance only (no points shown), matching the
            # recorded round's silent accrual. Practice is not analyzed.
            if press['hit']:
                big.text = 'Good, you pressed in time.'; big.color = [0.1, 0.8, 0.3]
                sub.text = ''
            elif press['early']:
                big.text = 'Too soon!'; big.color = [0.95, 0.6, 0.1]
                sub.text = 'Wait for the square.'
            else:
                big.text = 'A little faster next time.'; big.color = [0.95, 0.6, 0.1]
                sub.text = ''
            hold([big, sub] if sub.text else [big], CFG['BONUS_FEEDBACK_MS'])
            if press['early']:
                hold([], CFG['BONUS_PREMATURE_PENALTY_MS'])   # mirror the recorded penalty pause

        # Page text follows the approved instruction sheet verbatim.
        INTRO = ("In this game you are going to earn points.\n"
                 "You can earn points two different ways: a regular round and a bonus "
                 "round.\n\n"
                 "Press SPACE for next.")
        BANDIT = ("You'll see three different shapes on the screen. Select one using "
                  "the LEFT, DOWN, or RIGHT arrow key. If you select correctly, you'll "
                  "receive 10 points. If you miss, you'll lose 10 points.\n\n"
                  "Some symbols win more often than others. None of them win or "
                  "lose every time. The best one can change during the task, so keep "
                  "paying attention and use what you learn.\n\n"
                  "The goal is to get as many points as possible.\n\n"
                  "Let's practice 2 times.\n\nPress SPACE for next.")
        MID = ("Great!\n\nEvery so often a short Bonus round will happen. You will see "
               "a picture with the points on offer (the same amount for every "
               "picture), then a cross, and then a white square.\n\n"
               "The moment the square appears, press ANY KEY as fast as you can. The "
               "faster you are, the more of these bonus points you keep.\n\n"
               "You will not see these points during the game; they are added up and "
               "shown at the very end. Wait for the square; pressing too early does "
               "not count.\n\nLet's practice a couple of times.\n\nPress SPACE for next.")
        MID_DEMO = ("First, here is what happens if you press too early.\n\n"
                    "Just watch; do not press any key during the example.\n\n"
                    "Press SPACE to see it.")
        MID_DEMO_DONE = ("Ok, that's what it looks like when you press too soon.\n\n"
                         "Remember, wait for the square.\n\n"
                         "Now let's have you practice.\n\nPress SPACE to continue.")
        CONFIRM = ("Nice work!\n\nDo you understand the task, or would you like to do "
                   "more practice rounds?\n\n"
                   "Press RETURN to practice again        Press SPACE to begin the task\n\n"
                   "The whole task takes about 15-20 minutes.")

        # Loop the entire walkthrough until the participant chooses SPACE to begin.
        while True:
            wait_page(INTRO, ['space'])
            wait_page(BANDIT, ['space'])
            outcomes = [True, False]                     # exactly one win, one loss
            if prand() < 0.5:
                outcomes.reverse()                       # randomize which comes first
            for w in outcomes:
                practice_bandit_trial(w)
            wait_page(MID, ['space'])
            wait_page(MID_DEMO, ['space'])
            practice_bonus_trial(True, force_early=True)  # guaranteed too-soon demonstration
            wait_page(MID_DEMO_DONE, ['space'])
            cues = [True, False, prand() < 0.5]          # one food, one neutral, one either
            for i in range(len(cues) - 1, 0, -1):        # Fisher-Yates on prand so order is unpredictable
                j = int(prand() * (i + 1))
                cues[i], cues[j] = cues[j], cues[i]
            for f in cues:
                practice_bonus_trial(f)
            if wait_page(CONFIRM, ['space', 'return']) == 'space':
                break

    logging.exp('PRACTICE start')
    run_intro_and_practice()
    logging.exp('PRACTICE end')

    # ---- Visual analog scales (post-practice, pre-task) --------------------
    # Three continuous VAS on a -100..100 scale, shown once between practice and
    # the recorded task. The marker starts centered (0) and the numeric value is
    # never shown; the participant clicks the track to set it and presses SPACE.
    # No RNG is drawn here, so the bandit/bonus schedule is untouched. Responses
    # are written to a separate <run>_vas.csv to keep the main task schema stable.
    mouse = event.Mouse(win=win)

    def run_vas(name, question, left_anchor, right_anchor):
        """Show one continuous -100..100 VAS and return its result dict. The
        marker starts at center (0), the participant clicks/drags on the track to
        set a value (left = -100, right = +100), and SPACE or RETURN confirms.
        The number is not shown; 'moved' flags whether the marker was ever set."""
        win.mouseVisible = True
        half = 0.5                                   # track spans x in [-half, half]
        y = 0.0                                      # track vertical center
        # Static parts: question above, the track with end ticks, anchor labels
        # below each end, and a one-line instruction. Anchor centers sit at the
        # track ends; on a ~16:9/16:10 window they clear the screen edges, nudge
        # them inward if a long label clips on an unusual aspect ratio.
        q = visual.TextStim(win, text=question, pos=(0, 0.25), height=0.05,
                            color='white', wrapWidth=1.3)
        track = visual.Line(win, start=(-half, y), end=(half, y),
                            lineColor='white', lineWidth=3)
        tick_l = visual.Line(win, start=(-half, y - 0.03), end=(-half, y + 0.03),
                             lineColor='white', lineWidth=3)
        tick_r = visual.Line(win, start=(half, y - 0.03), end=(half, y + 0.03),
                             lineColor='white', lineWidth=3)
        lab_l = visual.TextStim(win, text=left_anchor, pos=(-half, y - 0.12),
                                height=0.032, color='white', wrapWidth=0.55)
        lab_r = visual.TextStim(win, text=right_anchor, pos=(half, y - 0.12),
                                height=0.032, color='white', wrapWidth=0.55)
        hint = visual.TextStim(win, text='Use the mouse or arrow keys to answer, then press SPACE.',
                               pos=(0, -0.38), height=0.03, color=[0.7, 0.7, 0.7])
        marker = visual.Rect(win, width=0.012, height=0.06, pos=(0, y),
                             fillColor=[1.0, 0.85, 0.2], lineColor=None)

        rating = 0.0                                 # center start on -100..100
        moved = False
        clk = core.Clock()
        kb.clearEvents()

        # Left/Right arrows move the marker in addition to the mouse. A pyglet
        # key-state handler gives smooth motion while a key is held, and the speed
        # is scaled by frame time so it is refresh-rate independent. Wrapped in
        # try/except: if the handler cannot attach, the mouse still works and the
        # arrows are simply inactive.
        arrow_speed = 90.0                           # units/sec on the -100..100 scale
        key_state = None
        try:
            from pyglet.window import key as _pk
            key_state = _pk.KeyStateHandler()
            win.winHandle.push_handlers(key_state)
        except Exception as _e:
            logging.warn('VAS arrow keys unavailable (%s); mouse only.' % _e)

        last_t = clk.getTime()
        # Each frame: apply held-arrow motion, then a mouse click on the track,
        # redraw, and check SPACE/RETURN to confirm or Escape to quit.
        while True:
            now = clk.getTime(); dt = now - last_t; last_t = now
            if key_state is not None:
                dx = 0.0
                if key_state[_pk.LEFT]:
                    dx -= arrow_speed * dt
                if key_state[_pk.RIGHT]:
                    dx += arrow_speed * dt
                if dx:
                    rating = max(-100.0, min(100.0, rating + dx)); moved = True
            if mouse.getPressed()[0]:
                mx, my = mouse.getPos()
                if -half - 0.05 <= mx <= half + 0.05 and abs(my - y) <= 0.12:
                    rating = max(-100.0, min(100.0, (max(-half, min(half, mx)) / half) * 100.0))
                    moved = True
            marker.pos = ((rating / 100.0) * half, y)
            for d in (q, track, tick_l, tick_r, lab_l, lab_r, hint, marker):
                d.draw()
            win.flip()
            for k in kb.getKeys(keyList=['space', 'return', 'escape'], clear=True):
                if k.name == 'escape':
                    save_and_close(); win.close(); core.quit()
                if k.name in ('space', 'return'):
                    if key_state is not None:                # detach before leaving
                        try:
                            win.winHandle.remove_handlers(key_state)
                        except Exception:
                            pass
                    win.mouseVisible = False
                    return dict(name=name, question=question,
                                left_anchor=left_anchor, right_anchor=right_anchor,
                                rating=round(rating, 2), rt_s=round(clk.getTime(), 4),
                                moved=1 if moved else 0)

    # Fixed order: hunger, then the energy/engagement scale. Left anchor maps to
    # -100 and the right to +100, so higher always means more of the right label.
    logging.exp('VAS start')
    vas_results = [
        run_vas('hunger', 'How hungry do you feel right now?',
                "I'm not hungry at all", "I'm the most hungry I can be"),
        run_vas('energy', 'How do you feel right now?',
                'Feeling extremely bored/low energy',
                'Feeling extremely engaged/energized'),
    ]
    # Write the three ratings to their own CSV, flushed and fsynced like the main
    # log so a crash after this point cannot lose them.
    vas_path = os.path.join(run_dir, run_name + '_vas.csv')
    vas_cols = ['participant_id', 'session', 'task_version', 'seed',
                'vas_name', 'question', 'left_anchor', 'right_anchor',
                'rating', 'rt_s', 'moved']
    with open(vas_path, 'w', newline='', encoding='utf-8') as vf:
        vw = csv.DictWriter(vf, fieldnames=vas_cols, extrasaction='ignore')
        vw.writeheader()
        for r in vas_results:
            vw.writerow(dict(participant_id=settings['pid'], session=settings['session'],
                             task_version=TASK_VERSION, seed=seed,
                             vas_name=r['name'], question=r['question'],
                             left_anchor=r['left_anchor'], right_anchor=r['right_anchor'],
                             rating=r['rating'], rt_s=r['rt_s'], moved=r['moved']))
        vf.flush(); os.fsync(vf.fileno())
    logging.exp('VAS end (%s)' % ', '.join('%s=%.1f' % (r['name'], r['rating'])
                                           for r in vas_results))

    food_set()                                       # fix the session food set once, session-wide

    def start_run(run_idx):
        """Prepare a fresh run: reset Q-driven state, re-draw the arm mapping, set
        this run's single reversal, and build this run's bonus schedule. Points and
        bonus tallies carry across runs; the learned values (profiles + on-screen
        arm layout) do not, so run 2 is a genuine new acquisition."""
        state['run'] = run_idx
        state['trial'] = 0
        state['swap_count'] = 0                       # reversal counter is per-run; reset at run start
        # Reset the reward profiles to their canonical start; the reversal below
        # will rotate them mid-run exactly as in a standalone 100-trial task.
        state['profiles'] = [list(CFG['PROFILE_A']), list(CFG['PROFILE_B']),
                             list(CFG['PROFILE_C'])]
        # Re-draw the arm placement on the MAIN stream (symbolMap then slotOrder),
        # so the participant cannot carry a learned symbol/slot value into run 2.
        state['symbol_map'] = shuffle3(rand)
        state['slot_order'] = shuffle3(rand)
        # This run's single reversal (1-indexed trial -> 0-indexed set).
        rev = CFG['REVERSAL_TRIAL_BY_RUN'][run_idx - 1]
        # Bonus schedule for this run, on the cosmetic stream, phase-stratified
        # around this run's one reversal.
        sched = build_bonus_schedule(srand, CFG['N_BONUS_FOOD'], CFG['N_BONUS_NEUTRAL'],
                                     CFG['N_TRIALS'], [rev])
        return {rev - 1}, sched

    def break_screen(next_run):
        """Rest break between runs. No RNG draws, nothing logged, so streams stay
        aligned. Waits for an explicit keypress so the RA controls the restart."""
        msg = visual.TextStim(
            win, color='white', height=0.045, wrapWidth=1.2,
            text=("End of part %d of %d.\n\n"
                  "Take a short break.\n\n"
                  "The next part works exactly the same way, but the shapes and "
                  "their values start fresh.\n\n"
                  "When you are ready, press SPACE to begin part %d."
                  % (next_run - 1, CFG['N_RUNS'], next_run)))
        msg.draw(); win.flip()
        while True:
            for k in kb.getKeys(keyList=['space', 'escape'], clear=True):
                if k.name == 'escape':
                    save_and_close(); win.close(); core.quit()
                if k.name == 'space':
                    return

    # ---- Run loop: N_RUNS runs of N_TRIALS bandit trials each ----------------
    # try/finally guarantees the data log is flushed and closed on any exit path;
    # completed rows are already fsynced as written, so nothing collected is lost.
    # bandit_trial and bonus_trial read swap_idx_set as a nonlocal, so it is bound
    # in the enclosing scope and rebound once per run.
    state['task_start'] = core.getTime()
    try:
        for run_idx in range(1, CFG['N_RUNS'] + 1):
            if run_idx > 1:
                break_screen(run_idx)
            # Exactly one start_run call per run (it draws RNG); capture both parts.
            swap_idx_set, insert_at = start_run(run_idx)
            for t in range(1, CFG['N_TRIALS'] + 1):
                bandit_trial()
                if t in insert_at:
                    bonus_trial(insert_at[t])

        # ---- End screen -----------------------------------------------------
        dur_min = (core.getTime() - state['task_start']) / 60.0
        total = state['score'] + state['bonus_score']
        end = visual.TextStim(
            win, color='white', height=0.04, wrapWidth=1.2,
            text=("Great work, you finished the task.\n\n"
                  "Task points: %d pts\n"
                  "Bonus points: %d pts  (%d / %d hits)\n"
                  "TOTAL: %d pts\n\n"
                  "Total time: %.1f min\n\n"
                  "Your responses have been saved.\nPress SPACE to exit."
                  % (state['score'], state['bonus_score'], state['bonus_hits'],
                     state['bonus_count'], total, dur_min)))
        end.draw()
        win.flip()
        kb.waitKeys(keyList=['space'])
    finally:
        save_and_close()
    win.close()
    core.quit()


if __name__ == '__main__':
    main()
