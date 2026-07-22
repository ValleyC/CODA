import math
import numpy as np
import os
import random
import soundfile as sf

from coda.utils.data_utils import SAMPLE_RATE
from multiprocessing import get_context
from pathlib import Path
from scipy.signal import fftconvolve, resample_poly


# filter some impulse responses that produce a weird distorted sound
FILTER_LIST = [
    "1a_marble_hall.wav",
    "3a_hats_cloaks_the_lord.wav",
    "4a_hats_cloaks_visitors.wav",
    "18a_smoking_room.wav",
    "ir_-_location_1_s1_-_r1.wav",
    "ir_-_location_2_s1_-_r2.wav",
    "ir_-_location_3_s1_-_r3.wav",
    "ir_-_location_4_s2_-_r3.wav",
    "ir_-_location_5_s2_-_r1.wav",
    "ir_-_location_6_s2_-_r4.wav",
    "ir_p1.wav",
    "ir_p2.wav",
    "ir_p3.wav",
    "ir_p4.wav",
    "ir_p5.wav",
    "ir_p6.wav",
    "ir_posic2-pb.wav",
    "ir_posic2-pb_-_sfdc.wav",
    "ir_posic3-1erpiso_-_sfdc.wav",
    "ir_posic3-pb_-_sfdc.wav",
    "ir_posic4-1erpiso_-_sfdc.wav",
    "ir_posic5-1erpiso_-_sfdc.wav",
    "ir_posic5-pb_-_sfdc.wav",
    "ir_stage_-_sfdc.wav",
    "phase1_bformat.wav",
    "phase1_bformat_catt.wav",
    "phase1_stereo.wav",
    "phase2_bformat.wav",
    "phase2_stereo.wav",
    "phase3_bformat.wav",
    "phase3_stereo.wav",
    "slinky_ir.wav",
    "sp1_mp4_ir_stereo_trimmed.wav",
    "sp2_mp2_ir_stereo_trimmed.wav",
    "sp2_mp4_ir_bformat_trimmed.wav",
    "sp2_mp4_ir_stereo_trimmed.wav",
    "spokane_womans_club_ir.wav",
    "stairwell_ir_bformat_trimmed.wav",
    "stairwell_ir_stereo_trimmed.wav",
    "tsfthr.wav"
]

# OPENAIR contains much more than impulse responses under its IRs tree:
# rendered music examples, sine sweeps, anechoic recordings, and raw
# measurement captures are stored beside the actual room responses. Feeding
# those files to convolution is both semantically wrong and numerically unsafe.
NON_IR_PATH_MARKERS = (
    "convolution_example",
    "auralization_result",
    "sweep",
    "measurement",
    "anechoic",
    "examples",
)
MAX_IR_DURATION_SECONDS = 30.0
FILTER_NAMES = frozenset(name.lower() for name in FILTER_LIST)


def _canonical_path_text(path):
    return str(path).lower().replace("-", "_").replace(" ", "_")


def is_candidate_ir(path, max_duration=MAX_IR_DURATION_SECONDS):
    """Return whether a WAV is a plausible room impulse response."""
    path = Path(path)
    if path.name.lower() in FILTER_NAMES:
        return False
    if any(marker in _canonical_path_text(path) for marker in NON_IR_PATH_MARKERS):
        return False
    try:
        info = sf.info(str(path))
    except (RuntimeError, OSError):
        return False
    duration = info.frames / float(info.samplerate) if info.samplerate else 0.0
    return 0.0 < duration <= max_duration


def load_signal(path):
    """Load and gain-condition one IR, returning diagnostics on failure."""
    try:
        signal, sample_rate = sf.read(
            path, dtype='float32', always_2d=True
        )
        signal = signal.mean(axis=1, dtype=np.float32)
        if sample_rate != SAMPLE_RATE:
            divisor = math.gcd(int(sample_rate), int(SAMPLE_RATE))
            signal = resample_poly(
                signal,
                SAMPLE_RATE // divisor,
                int(sample_rate) // divisor,
            )
        signal = np.asarray(signal, dtype=np.float32)
        if signal.size == 0 or not np.isfinite(signal).all():
            return str(path), None, "empty or non-finite signal"

        # Unit-L1 conditioning bounds convolution by the dry-signal peak. It
        # does not change the eventual wet waveform because __call__ restores
        # the dry RMS, but it prevents intermediate overflow.
        l1 = float(np.sum(np.abs(signal), dtype=np.float64))
        if not np.isfinite(l1) or l1 <= 1e-8:
            return str(path), None, "silent or invalid signal"
        signal = np.ascontiguousarray(signal / l1, dtype=np.float32)
        return str(path), signal, None
    except Exception as exc:
        return str(path), None, f"{type(exc).__name__}: {exc}"


def load_irs(ir_paths):
    print("Loading IRs from", ir_paths)
    all_paths = []
    for ir_path in ir_paths:
        all_paths.extend([path for path in Path(os.path.expanduser(ir_path)).rglob('*.wav')])

    # Deterministic discovery and explicit curation. Virtual-membranes are
    # synthetic filters rather than measured room responses.
    all_paths = sorted(set(all_paths), key=lambda path: str(path).lower())
    arguments = [
        str(path) for path in all_paths
        if "virtual-membranes" not in str(path)
        and is_candidate_ir(path)
    ]

    # Use spawn instead of fork to avoid deadlocks with CUDA/audio libs
    mp_start = os.getenv("CODA_DATASET_START_METHOD", "spawn")
    mp_workers = int(os.getenv("CODA_DATASET_WORKERS", "8"))

    if mp_workers <= 0:
        loaded = [load_signal(arg) for arg in arguments]
    else:
        with get_context(mp_start).Pool(mp_workers) as pool:
            loaded = pool.map(load_signal, arguments)

    irs = []
    accepted_paths = []
    rejected = []
    for path, signal, error in loaded:
        if signal is None:
            rejected.append((path, error))
        else:
            accepted_paths.append(path)
            irs.append(signal)

    print(
        f"Accepted {len(irs)} genuine IR(s); excluded "
        f"{len(all_paths) - len(arguments)} non-IR asset(s) and "
        f"{len(rejected)} invalid IR file(s)."
    )
    for path, error in rejected[:10]:
        print(f"Warning: skipping IR {path}: {error}")
    if not irs:
        raise RuntimeError(f"No valid impulse responses found in {ir_paths}")

    return irs, accepted_paths


class ImpulseResponse(object):

    def __init__(self, ir_paths, ir_prob):
        self.ir_prob = ir_prob

        self.irs, self.ir_paths = load_irs(ir_paths)

        print(f'Loaded {len(self.irs)} impulse responses.')

    def __call__(self, sample):

        performance = np.asarray(sample['performance'], dtype=np.float32)

        if performance.size and random.random() < self.ir_prob:
            ir = random.choices(self.irs, k=1)[0]
            wet = fftconvolve(performance, ir, mode='full')[:performance.shape[0]]
            wet = np.asarray(wet, dtype=np.float32)

            # Preserve dry RMS so IR gain is not an unintended loudness
            # augmentation, then peak-limit to the dry signal.
            if np.isfinite(wet).all():
                dry_rms = float(np.sqrt(np.mean(
                    performance.astype(np.float64) ** 2
                )))
                wet_rms = float(np.sqrt(np.mean(wet.astype(np.float64) ** 2)))
                if dry_rms > 1e-8 and wet_rms > 1e-8:
                    wet *= dry_rms / wet_rms

                dry_peak = float(np.max(np.abs(performance)))
                wet_peak = float(np.max(np.abs(wet)))
                if wet_peak > max(dry_peak, 1e-8):
                    wet *= dry_peak / wet_peak
                sample['performance'] = np.ascontiguousarray(wet, dtype=np.float32)

        return sample
