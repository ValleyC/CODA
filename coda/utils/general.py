from contextlib import contextmanager
import hashlib
import math
import random
from functools import lru_cache

import librosa
import soundfile as sf
import torch
import yaml

import numpy as np


def stable_named_seed(base_seed, name):
    """Derive a process-independent uint32 seed for one named artifact."""
    payload = f"{int(base_seed)}\0{name}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="big") % (2 ** 32)


@contextmanager
def isolated_python_numpy_seed(seed):
    """Use deterministic local RNG streams without perturbing the caller."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    random.seed(int(seed))
    np.random.seed(int(seed))
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


@contextmanager
def isolated_python_numpy_torch_seed(seed):
    """Isolate all CPU RNGs commonly used by dataset augmentations."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)


def load_yaml(config_file):
    """Load config from YAML file."""
    with open(config_file, 'rb') as fp:
        config = yaml.load(fp, Loader=yaml.FullLoader)
    return config


def load_wav(audio_path, sr):
    signal, _ = librosa.load(audio_path, sr=sr)

    return signal


@lru_cache(maxsize=1024)
def _soundfile_layout(audio_path):
    info = sf.info(audio_path)
    return info.samplerate, info.channels


def load_wav_segment(audio_path, start, stop, sr):
    """Read only a requested mono waveform interval.

    MSMD audio is already mono at the model sample rate, so SoundFile can seek
    directly instead of decoding the entire performance for every randomly
    sampled training frame. A librosa fallback preserves support for other
    sample rates and channel layouts.
    """
    start = max(0, int(start))
    stop = max(start, int(stop))
    sample_rate, channels = _soundfile_layout(audio_path)
    if sample_rate == sr and channels == 1:
        signal, _ = sf.read(
            audio_path, start=start, stop=stop,
            dtype='float32', always_2d=False,
        )
        return signal

    offset = start / float(sr)
    duration = (stop - start) / float(sr)
    signal, _ = librosa.load(audio_path, sr=sr, mono=True,
                             offset=offset, duration=duration)
    return signal


def make_divisible(x, divisor):
    """Returns x evenly divisible by divisor."""
    return math.ceil(x / divisor) * divisor


def xyxy2xywh(x):
    """Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right."""
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    """Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right."""
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def box_iou(box1, box2):
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def bbox_iou(box1, box2, eps=1e-9):
    """Returns the IoU of box1 to box2. box1 is 4xn, box2 is nx4."""
    box2 = box2.T

    # transform from xywh to xyxy
    b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
    b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
    b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
    b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    return iou


def get_max_box(prediction, class_id=0):
    """
    Returns:
         most confident detection with shape: 1x4 (x, y, w, h)
    """

    output = []
    for xi, x in enumerate(prediction):  # image index, image inference
        class_filtered = x[:, -1] == class_id

        x = x[class_filtered]  # confidence

        _, idx = x[..., 4].max(-1)
        max_per_sample = x[idx][:4]
        output.append(max_per_sample)

    output = torch.stack(output)

    return output


class AverageMeter(object):
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if not math.isfinite(float(val)):
            raise ValueError(f"AverageMeter received non-finite value: {val}")
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
