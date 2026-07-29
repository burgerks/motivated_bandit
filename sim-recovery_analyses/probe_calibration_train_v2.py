#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Standalone probe-calibration train for the Hamburglar mini-MID.

WHAT THIS IS
------------
Not an experiment. A calibration. It runs the mini-MID probe from
bandit_mid_task_v15.py with NO bandit, NO learning, and NO points that matter,
and it exists to measure two numbers that the v15 pilot could not separate:

  lambda  how much of ONE probe RT is state rather than measurement noise
  rho     how fast the state fades (quoted per second here, not per trial)

WHY THE PILOT COULD NOT DO IT
-----------------------------
Agreement between two probe readings at time gap g equals rho^g * lambda. That
is an identity under the model (slow state + white noise), not a hypothesis. In
the 11-subject pilot the agreement curve sat at the floor from the closest bin
(3 bandit trials, ~15-20 s) outward. A widened calibration grid showed that a
near-perfect instrument reading a FAST state (rho=0.50, lambda=0.80) fits the
pilot exactly as well as a noisy instrument reading a slow one (rho=0.95,
lambda=0.10). Nothing in the pilot separates them, because its shortest gap is
already past the point where a fast state has turned over.

HOW BURSTS SEPARATE THEM
------------------------
Three full probes back to back put readings ~4.6 s and ~9.2 s apart, which is the
window the pilot is blind to. The three worlds predict different agreement
between rep 1 and rep 2:

    noisy instrument, slow state   ->  ~0.10
    good instrument, fast state    ->  ~0.50
    no state at all                ->  ~0.00

The burst does not ASSUME the state is constant across 4.6 s. It measures whether
it is, which is the stronger claim and the reason this works.

Between-burst pairs then span ~15 s to ~4 min and give the rest of the fade curve.
Combining both scales yields rho and lambda from one dataset.

READ THIS BEFORE INTERPRETING THE OUTPUT
----------------------------------------
1. REP POSITION IS A REAL EFFECT, NOT NOISE. Reps 2 and 3 are anticipated, so RT
   should fall and premature presses should rise across a burst. Model rep_index
   as a fixed effect and remove it BEFORE reading within-burst variance as
   measurement noise. Failing to do this inflates the noise estimate and
   understates lambda. The column is logged for exactly this reason.
2. REPS SHARE ERROR. Posture, hand position, and a lapse spanning 14 s are common
   to all three reps and will not average away. This is the F term in the power
   simulations. The within-burst decomposition estimates it rather than assuming
   it, which is the point.
3. NO STAIRCASE. The deadline is fixed at 550 ms for every rep, every burst,
   every subject. The staircase was dropped because it parks each person on
   their own speed-accuracy threshold and removes the slow RT drift the vigor
   construct consists of; the pilot showed within-subject SD of log probe RT at
   0.156 against 0.637 for self-paced bandit choice, and an ICC of 0.055 after
   residualizing on the window. A fixed deadline keeps urgency without chasing
   anyone onto their floor, makes all subjects the same instrument, and cuts
   censoring from ~34% of readings to ~1-2%. That last point is not cosmetic:
   censoring is non-random, since slow trials are exactly the ones that vanish,
   and it biases variance estimates downward, which is fatal in a task whose
   only output is a variance decomposition.
4. THIS IS A GO / NO-GO. If rho comes back fast, the burst will have measured it
   cleanly and will then tell you that carrying a probe value forward over ~6
   bandit trials (~25 s) retains almost none of it. That kills the modulator, and
   no amount of probe density or repetition revives it. Two of the three possible
   answers are bad news for the design. Run this before touching v15 again.

WHO RUNS IT
-----------
Lab members. This is a psychophysics measurement, not a hypothesis test, so it
needs no patients and no iEEG. Five people is enough.

FOLDER LAYOUT (same as v15, run this file next to the same stimuli/ tree)
    stimuli/win/sweet/      sweet food images
    stimuli/win/savory/     savory food images
    stimuli/neutral/        neutral / scrambled images
Missing folders fall back to a labelled placeholder box so the script still runs.

Tested against the PsychoPy 2023.2+/2024.x API. Run from the Coder or
`python probe_calibration_train_v1.py`.
"""

import os
import csv
import glob
import math
import datetime

from PIL import Image
from psychopy import visual, core, gui, logging
from psychopy.hardware import keyboard

# ════════════════════════════════════════════════════════════════════════════
#  USER SETTINGS
# ════════════════════════════════════════════════════════════════════════════
TASK_VERSION = 'cal-v2'           # stamped into every data row for provenance

CFG = dict(
    # ---- Probe timing. COPIED FROM v15 CFG. Do not tune these. The whole point
    # is to calibrate the instrument v15 actually uses; changing the cue or delay
    # makes this a measurement of some other probe.
    BONUS_INTRO_MS=1000,          # "Bonus round!" screen, once per BURST
    CUE_MS=1500,                  # food/neutral picture
    DELAY_MIN_MS=1500,            # anticipatory fixation, jittered per REP
    DELAY_MAX_MS=3000,
    GRACE_MS=500,                 # late press logged as miss-with-RT
    BONUS_FEEDBACK_MS=1500,       # once per BURST, not per rep
    BONUS_PREMATURE_PENALTY_MS=1500,
    BONUS_PTS=15,                 # per hit
    REP_GAP_MS=500,               # blank between reps inside a burst

    # ---- Burst structure
    N_BURSTS=16,                  # measured bursts
    REPS_PER_BURST=3,             # 3 gives lag-1 (~4.6 s) and lag-2 (~9.2 s)
    CUE_DECK='food',              # 'food' (matches the proposed design) or 'mixed'

    # ---- Inter-burst interval, jittered log-uniform. Between-burst gaps are
    # already ~15 s minimum because a burst takes that long, so the jitter is
    # only there to decorrelate gap length from burst index (and hence from any
    # fatigue drift). Cheap insurance.
    IBI_MIN_MS=1000,
    IBI_MAX_MS=6000,

    # ---- FIXED DEADLINE. No staircase, anywhere, ever.
    # The staircase was dropped because it holds each person at their own
    # speed-accuracy threshold and in doing so strips out the slow RT drift the
    # vigor construct is made of. The 11-subject pilot showed the damage: probe
    # RT sat at 237-323 ms directly on top of windows the staircase had driven
    # to 230-390, within-subject SD of log RT was 0.156 against 0.637 for
    # self-paced bandit choice, and ICC after residualizing on the window was
    # 0.055. A modulator with no variance cannot modulate anything.
    #
    # A fixed deadline keeps urgency (RT stays a motivational readout) without
    # chasing anyone onto their floor, and it makes every subject the same
    # instrument, which matters when the entire output is a variance
    # decomposition compared across people.
    #
    # 550 ms chosen from the pilot: at 550 the pooled hit rate is ~0.99 and the
    # worst subject ~0.96, and even if every RT inflated 20% once the deadline
    # stops chasing them (it should slow and spread), that holds at 0.98 pooled
    # and 0.92 worst. Censoring goes from a third of readings to a rounding
    # error, which matters because censoring is non-random (slow trials are the
    # ones that vanish) and biases variance estimates downward.
    FIXED_WINDOW_MS=550,

    # ---- Familiarization. Not a staircase and not analysed. Three rounds so
    # nobody meets burst 1 cold. Logged as phase='familiar' and dropped at
    # analysis.
    N_FAMILIAR=3,
)

FULLSCREEN = True
BG_COLOR = [-0.5, -0.5, -0.5]
WIN_SIZE = [1280, 800]

IMG_DIRS = dict(
    sweet=os.path.join('stimuli', 'win', 'sweet'),
    savory=os.path.join('stimuli', 'win', 'savory'),
    neutral=os.path.join('stimuli', 'neutral'),
)

TRIGGER_CHAR = ','
EVENT_CODES = dict(
    burst_intro='burst_intro', cue_food='cue_food', cue_neutral='cue_neutral',
    fixation='fixation', target='target', response='response',
    burst_feedback='burst_feedback',
)


# ════════════════════════════════════════════════════════════════════════════
#  mulberry32 RNG
# ════════════════════════════════════════════════════════════════════════════
# Same generator as v15 so a seed means the same thing across both files. Only
# one stream is needed here: this task has no schedule to protect.
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
# Photodiode and triggers default off: this is a behavioural calibration and
# needs no iEEG alignment. They are kept available so the same rig can run it.
def run_dialog():
    info = {
        'participant': '',
        'session': '001',
        'seed (blank = random)': '',
        'food_set': ['sweet', 'savory', 'sweet+savory'],
        'cue_deck': ['food', 'mixed'],
        'photodiode': False,
        'triggers': ['none', 'serial', 'parallel'],
        'serial_port': 'COM3',
        'parallel_address (hex)': '0x378',
    }
    ok = gui.DlgFromDict(info, title='Probe calibration train',
                         order=['participant', 'session', 'seed (blank = random)',
                                'food_set', 'cue_deck', 'photodiode', 'triggers',
                                'serial_port', 'parallel_address (hex)'])
    if not ok.OK:
        core.quit()
    seed_txt = str(info['seed (blank = random)']).strip()
    seed = (int(seed_txt, 0) & 0xFFFFFFFF) if seed_txt else \
        int.from_bytes(os.urandom(4), 'little')
    return dict(
        pid=str(info['participant']).strip() or 'test',
        session=str(info['session']).strip() or '001',
        seed=seed,
        food_set=info['food_set'],
        cue_deck=info['cue_deck'],
        photodiode=bool(info['photodiode']),
        trig_mode=info['triggers'],
        serial_port=str(info['serial_port']).strip(),
        parallel_addr=int(str(info['parallel_address (hex)']), 0),
    )


# ════════════════════════════════════════════════════════════════════════════
#  Triggers
# ════════════════════════════════════════════════════════════════════════════
# Identical to v15: every event emits the same comma marker; the label goes to
# the .log so events are recoverable offline from the label sequence.
class Triggers:
    def __init__(self, mode='none', serial_port='COM3', address=0x378):
        self.mode, self.port = mode, None
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

    def send(self, label):
        if self.mode == 'serial' and self.port is not None:
            self.port.write(TRIGGER_CHAR.encode('ascii'))
        elif self.mode == 'parallel' and self.port is not None:
            self.port.setData(ord(TRIGGER_CHAR))
        logging.exp('TRIGGER %s (%s)' % (TRIGGER_CHAR, label))

    def clear(self):
        if self.mode == 'parallel' and self.port is not None:
            self.port.setData(0)


# ════════════════════════════════════════════════════════════════════════════
#  Image helpers
# ════════════════════════════════════════════════════════════════════════════
# Lifted from v15 so the cues are drawn at the same size and aspect ratio. Fresh
# exemplars are drawn per REP, which matches v15 (it draws a new picture each
# round) and avoids repetition priming from showing one image three times.
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
    """Size (w, h) in height units fitting the image in a box-square, aspect
    preserved. Pixel dims read once from the header and cached."""
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


def draw_images(rand, files, n):
    """Pick n paths without replacement where possible, else with replacement."""
    if not files:
        return []
    pool, out = list(files), []
    for _ in range(n):
        if not pool:
            pool = list(files)
        idx = int(rand() * len(pool))
        out.append(pool.pop(idx))
    return out


# ════════════════════════════════════════════════════════════════════════════
#  Data log
# ════════════════════════════════════════════════════════════════════════════
# One row per REP (not per burst), because the rep is the unit of measurement.
# burst_index + rep_index identify the pair structure; t_target_onset_s is the
# wall clock the fade curve is computed against. Trial indices are meaningless
# here since there is no bandit, so seconds are the native unit, which also
# sidesteps the trial-duration variation that makes a per-trial rho ambiguous.
FIELDNAMES = [
    'participant_id', 'session', 'task_version', 'seed', 'timestamp',
    'phase',                      # 'familiar' (drop) or 'measure' (analyse)
    'burst_index', 'rep_index', 'reps_per_burst',
    't_burst_onset_s', 't_target_onset_s',   # seconds from task start (fade-curve axis)
    'cue_type', 'food_set', 'cue_image',
    'anticipatory_delay_ms', 'window_ms',
    'target_response_key', 'target_rt_ms', 'premature_rt_ms',
    'target_hit', 'target_miss', 'target_too_fast', 'target_no_response',
    'burst_hits', 'burst_points', 'cumulative_points',
    'cue_onset_ms', 'delay_onset_ms', 'target_onset_ms', 'response_ms',
]


class DataLog:
    """Row-at-a-time CSV writer, flushed and fsynced per row so a crash costs
    at most the current rep."""

    def __init__(self, path):
        self.f = open(path, 'w', newline='', encoding='utf-8')
        self.w = csv.DictWriter(self.f, fieldnames=FIELDNAMES,
                                restval='', extrasaction='ignore')
        self.w.writeheader()
        self.f.flush()

    def write(self, row):
        self.w.writerow(row)
        self.f.flush()
        os.fsync(self.f.fileno())

    def close(self):
        try:
            self.f.close()
        except Exception:
            pass


def make_run_dir(pid, ses):
    """Fresh numbered run folder under data/; never overwrites a previous run."""
    os.makedirs('data', exist_ok=True)
    base = 'sub-%s_ses-%s_cal' % (pid, ses)
    n = 1
    while True:
        name = '%s_%d' % (base, n)
        path = os.path.join('data', name)
        try:
            os.makedirs(path)
            return path, name
        except FileExistsError:
            n += 1


def estimate_runtime_s():
    """Honest runtime before the window opens. Uses mean jitter and a typical
    ~280 ms RT (the target phase breaks early on the press, so the deadline is
    an upper bound on its duration, not its duration)."""
    mean_delay = (CFG['DELAY_MIN_MS'] + CFG['DELAY_MAX_MS']) / 2.0
    typical_rt = 300.0
    rep = CFG['CUE_MS'] + mean_delay + typical_rt + CFG['REP_GAP_MS']
    burst = CFG['BONUS_INTRO_MS'] + CFG['REPS_PER_BURST'] * rep + CFG['BONUS_FEEDBACK_MS']
    fam = CFG['N_FAMILIAR'] * (CFG['BONUS_INTRO_MS'] + rep + CFG['BONUS_FEEDBACK_MS'])
    # Log-uniform mean is (b-a)/ln(b/a), not the arithmetic mean.
    a, b = CFG['IBI_MIN_MS'], CFG['IBI_MAX_MS']
    ibi = (CFG['N_BURSTS'] - 1) * (b - a) / math.log(b / float(a))
    return (fam + CFG['N_BURSTS'] * burst + ibi) / 1000.0


# ════════════════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════════════════
def main():
    settings = run_dialog()
    seed = settings['seed']
    CFG['CUE_DECK'] = settings['cue_deck']
    rand = make_rng(seed)

    est = estimate_runtime_s()
    print('Estimated runtime: %.1f min (excludes self-paced instruction pages).'
          % (est / 60.0))

    run_path, run_name = make_run_dir(settings['pid'], settings['session'])
    log = DataLog(os.path.join(run_path, run_name + '_calibration.csv'))
    logging.LogFile(os.path.join(run_path, run_name + '.log'), level=logging.EXP)

    win = visual.Window(size=WIN_SIZE, fullscr=FULLSCREEN, color=BG_COLOR,
                        units='height', allowGUI=False, winType='pyglet')
    win.mouseVisible = False
    kb = keyboard.Keyboard()
    trig = Triggers(settings['trig_mode'], settings['serial_port'], settings['parallel_addr'])
    pd_on = settings['photodiode']

    # ---- Stimuli. Same geometry as v15's bonus round.
    big = visual.TextStim(win, text='', color='white', height=0.075, pos=(0, 0.06))
    sub = visual.TextStim(win, text='', color='white', height=0.042, pos=(0, -0.08),
                          wrapWidth=1.2)
    page = visual.TextStim(win, text='', color='white', height=0.04, wrapWidth=1.3)
    cue_stim = visual.ImageStim(win, image=None, pos=(0, 0))
    fix_stim = visual.TextStim(win, text='+', color='white', height=0.09)
    square = visual.Rect(win, width=0.18, height=0.18, fillColor=[1, 1, 1],
                         lineColor=[1, 1, 1])
    pd_stim = visual.Rect(win, width=0.09, height=0.09, pos=(-0.86, -0.44),
                          fillColor=[-1, -1, -1], lineColor=None)

    # Food images. cue_deck='food' uses food on every burst (matches the proposed
    # 25-probe food-only design); 'mixed' alternates food and neutral.
    fs = settings['food_set']
    food_pool = []
    for k in (['sweet', 'savory'] if fs == 'sweet+savory' else [fs]):
        food_pool += discover_images(IMG_DIRS[k])
    neutral_pool = discover_images(IMG_DIRS['neutral'])

    state = dict(task_start=0.0, points=0)

    def save_and_close():
        log.close()

    def hold(drawables, dur_ms, trig_code=None):
        """Draw a static screen for dur_ms. Photodiode square goes white on the
        first frame only, so its onset marks the event."""
        for d in drawables:
            d.draw()
        if pd_on and trig_code is not None:
            pd_stim.fillColor = [1, 1, 1]; pd_stim.draw()
        on = win.flip()
        if trig_code is not None:
            trig.send(trig_code)
        clk = core.Clock()
        while clk.getTime() < dur_ms / 1000.0:
            for d in drawables:
                d.draw()
            if pd_on:
                pd_stim.fillColor = [-1, -1, -1]; pd_stim.draw()
            win.flip()
            trig.clear()
            if kb.getKeys(['escape']):
                save_and_close(); win.close(); core.quit()
        return on

    def show_page(text):
        """Instruction page; SPACE advances, ESC quits."""
        page.text = text
        while True:
            page.draw(); win.flip()
            keys = kb.getKeys(['space', 'escape'])
            if any(k.name == 'escape' for k in keys):
                save_and_close(); win.close(); core.quit()
            if any(k.name == 'space' for k in keys):
                return

    # ────────────────────────────────────────────────────────────────────────
    #  One rep = one full v15 probe (cue, jittered fixation, target, grace).
    #  Deliberately NO feedback inside a burst: a win or a miss between reps is a
    #  reward event that could move the very state we are trying to hold still.
    # ────────────────────────────────────────────────────────────────────────
    def run_rep(is_food, window_ms, t_burst, burst_i, rep_i, phase_name):
        t0 = core.getTime()
        cue_type = ('food_%s' % fs) if is_food else 'neutral'
        pool = food_pool if is_food else neutral_pool
        picked = draw_images(rand, pool, 1)
        cue_path = picked[0] if picked else None
        delay_ms = CFG['DELAY_MIN_MS'] + int(rand() * (CFG['DELAY_MAX_MS'] - CFG['DELAY_MIN_MS'] + 1))

        marks = dict(cue=0, delay=0, target=0)
        kb.clearEvents(); kb.clock.reset()
        press = {'key': '', 'rt_ms': None}

        def listen():
            for k in kb.getKeys(waitRelease=False, clear=True):
                if k.name == 'escape':
                    save_and_close(); win.close(); core.quit()
                if press['rt_ms'] is None:
                    press['key'] = k.name
                    press['rt_ms'] = k.rt * 1000.0
                    return True
            return False

        def phase(drawables, dur_ms, trig_code):
            for d in drawables:
                d.draw()
            if pd_on and trig_code is not None:
                pd_stim.fillColor = [1, 1, 1]; pd_stim.draw()
            on = win.flip()
            if trig_code is not None:
                trig.send(trig_code)
            clk = core.Clock(); caught = False
            while clk.getTime() < dur_ms / 1000.0:
                if press['rt_ms'] is None and listen():
                    caught = True; break
                for d in drawables:
                    d.draw()
                if pd_on:
                    pd_stim.fillColor = [-1, -1, -1]; pd_stim.draw()
                win.flip(); trig.clear()
            return on, caught

        # 1) cue
        cue_trig = EVENT_CODES['cue_food'] if is_food else EVENT_CODES['cue_neutral']
        if cue_path:
            cue_stim.image = cue_path
            cue_stim.size = fit_size(cue_path, 0.8)      # 0.8 matches v15
            cue_draw = [cue_stim]
        else:
            sub.text = ('FOOD CUE' if is_food else 'NEUTRAL CUE') + '\n(placeholder)'
            sub.pos = (0, 0); cue_draw = [sub]
        on, caught = phase(cue_draw, CFG['CUE_MS'], cue_trig)
        marks['cue'] = round((on - t0) * 1000.0)
        sub.pos = (0, -0.08)
        outcome = 'too_fast' if caught else None

        # 2) anticipatory fixation
        if outcome is None:
            on, caught = phase([fix_stim], delay_ms, EVENT_CODES['fixation'])
            marks['delay'] = round((on - t0) * 1000.0)
            if caught:
                outcome = 'too_fast'

        # 3) target
        t_target_abs = None
        if outcome is None:
            on, caught = phase([square], window_ms, EVENT_CODES['target'])
            marks['target'] = round((on - t0) * 1000.0)
            t_target_abs = on - state['task_start']       # fade-curve axis, seconds
            if caught:
                outcome = 'hit'; trig.send(EVENT_CODES['response'])

        # 4) grace
        if outcome is None:
            on, caught = phase([], CFG['GRACE_MS'], None)
            outcome = 'miss' if caught else 'no_response'

        prem_rt = rt_ms = resp_ms = ''
        if press['rt_ms'] is not None:
            resp_ms = round(press['rt_ms'])
            if outcome == 'too_fast':
                prem_rt = round(press['rt_ms'] - marks['cue'])
            else:
                rt_ms = round(press['rt_ms'] - marks['target'])

        if outcome == 'too_fast':
            hold([], CFG['BONUS_PREMATURE_PENALTY_MS'])
        else:
            hold([], CFG['REP_GAP_MS'])

        hit = outcome == 'hit'
        row = {
            'participant_id': settings['pid'], 'session': settings['session'],
            'task_version': TASK_VERSION, 'seed': seed,
            'timestamp': datetime.datetime.now().isoformat(timespec='seconds'),
            'phase': phase_name,
            'burst_index': burst_i, 'rep_index': rep_i,
            'reps_per_burst': CFG['REPS_PER_BURST'] if phase_name == 'measure' else 1,
            't_burst_onset_s': round(t_burst, 4),
            't_target_onset_s': (round(t_target_abs, 4) if t_target_abs is not None else ''),
            'cue_type': cue_type, 'food_set': fs, 'cue_image': cue_path or '',
            'anticipatory_delay_ms': delay_ms, 'window_ms': window_ms,
            'target_response_key': press['key'], 'target_rt_ms': rt_ms,
            'premature_rt_ms': prem_rt,
            'target_hit': 1 if hit else 0,
            'target_miss': 1 if outcome == 'miss' else 0,
            'target_too_fast': 1 if outcome == 'too_fast' else 0,
            'target_no_response': 1 if outcome == 'no_response' else 0,
            'cue_onset_ms': marks['cue'], 'delay_onset_ms': marks['delay'],
            'target_onset_ms': marks['target'], 'response_ms': resp_ms,
        }
        return hit, outcome, row

    # ────────────────────────────────────────────────────────────────────────
    #  Instructions
    # ────────────────────────────────────────────────────────────────────────
    show_page('Bonus rounds.\n\n'
              'You will see a picture, then a plus sign, then a WHITE SQUARE.\n\n'
              'Press SPACE as fast as you can when the square appears.\n\n'
              'Press SPACE before the square and you lose the round.\n\n'
              'Press SPACE to continue.')
    show_page('A few single rounds first, then the rounds come in sets of\n'
              'three in a row, with one score at the end of each set.\n\n'
              'Press SPACE to begin.')

    state['task_start'] = core.getTime()

    # ────────────────────────────────────────────────────────────────────────
    #  Familiarization: three single rounds at the same fixed deadline as the
    #  measurement. Nothing adapts and nothing here is analysed; it exists so
    #  nobody meets burst 1 cold. Rows carry phase='familiar', drop them.
    # ────────────────────────────────────────────────────────────────────────
    for i in range(1, CFG['N_FAMILIAR'] + 1):
        big.text = 'Bonus round!'; big.color = 'white'
        t_b = core.getTime() - state['task_start']
        hold([big], CFG['BONUS_INTRO_MS'], trig_code=EVENT_CODES['burst_intro'])
        is_food = True if CFG['CUE_DECK'] == 'food' else (i % 2 == 1)
        hit, outcome, row = run_rep(is_food, CFG['FIXED_WINDOW_MS'], t_b, 0, i, 'familiar')

        state['points'] += CFG['BONUS_PTS'] if hit else 0
        row.update(burst_hits=1 if hit else 0,
                   burst_points=CFG['BONUS_PTS'] if hit else 0,
                   cumulative_points=state['points'])
        log.write(row)

        big.text = 'Congrats!' if hit else ('Too soon!' if outcome == 'too_fast'
                                            else 'Next time respond faster.')
        big.color = [0.1, 0.8, 0.3] if hit else [0.95, 0.6, 0.1]
        sub.text = '+%d points' % CFG['BONUS_PTS'] if hit else ''
        hold([big, sub] if sub.text else [big], CFG['BONUS_FEEDBACK_MS'],
             trig_code=EVENT_CODES['burst_feedback'])

    show_page('From here the rounds come in sets of three.\n\n'
              'Same job: press SPACE when the white square appears.\n\n'
              'Press SPACE to start.')

    # ────────────────────────────────────────────────────────────────────────
    #  Measurement bursts. Window is held constant across every rep and every
    #  burst, so nothing about the deadline can vary with time or rep position.
    # ────────────────────────────────────────────────────────────────────────
    for bi in range(1, CFG['N_BURSTS'] + 1):
        big.text = 'Bonus round!'; big.color = 'white'
        t_b = core.getTime() - state['task_start']
        hold([big], CFG['BONUS_INTRO_MS'], trig_code=EVENT_CODES['burst_intro'])

        w = CFG['FIXED_WINDOW_MS']          # identical on every rep and every burst
        is_food = True if CFG['CUE_DECK'] == 'food' else (bi % 2 == 1)
        hits, rows = 0, []
        for ri in range(1, CFG['REPS_PER_BURST'] + 1):
            hit, outcome, row = run_rep(is_food, w, t_b, bi, ri, 'measure')
            hits += 1 if hit else 0
            rows.append(row)

        pts = CFG['BONUS_PTS'] * hits
        state['points'] += pts
        for r in rows:
            r.update(burst_hits=hits, burst_points=pts, cumulative_points=state['points'])
            log.write(r)

        # One feedback screen per burst, reporting the set.
        big.text = '+%d points' % pts if pts else 'No points that time.'
        big.color = [0.1, 0.8, 0.3] if pts else [0.95, 0.6, 0.1]
        sub.text = '%d of %d' % (hits, CFG['REPS_PER_BURST'])
        hold([big, sub], CFG['BONUS_FEEDBACK_MS'], trig_code=EVENT_CODES['burst_feedback'])

        # Jittered inter-burst interval (log-uniform), decorrelating gap from index.
        if bi < CFG['N_BURSTS']:
            lo, hi = math.log(CFG['IBI_MIN_MS']), math.log(CFG['IBI_MAX_MS'])
            hold([], int(round(math.exp(lo + rand() * (hi - lo)))))

    # ────────────────────────────────────────────────────────────────────────
    #  End
    # ────────────────────────────────────────────────────────────────────────
    big.text = 'Done. %d points.' % state['points']; big.color = 'white'
    sub.text = 'Thank you.'
    hold([big, sub], 3000)
    save_and_close()
    win.close()
    core.quit()


if __name__ == '__main__':
    main()
