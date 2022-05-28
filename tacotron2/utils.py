import numpy as np
from scipy.io.wavfile import read
import soundfile as sf
import librosa
import torch
import csv


def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))
    mask = (ids < lengths.unsqueeze(1)).bool()
    return mask


def load_wav_to_torch(full_path):
    data, sampling_rate = sf.read(full_path)
    data, _ = librosa.effects.trim(data, top_db=15)
    #sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(filename, split="|"):
    """with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split) for line in f]"""
    with open(filename) as split_f:
        csv_reader = csv.reader(split_f, delimiter=split)
        next(csv_reader, None)
        filepaths_and_text = [line for line in csv_reader]
    return filepaths_and_text


def to_gpu(x):
    x = x.contiguous()

    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return torch.autograd.Variable(x)
