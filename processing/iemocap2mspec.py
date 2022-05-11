import numpy as np
import csv
import sys
import torch
import os
import sys

sys.path.append('.')
sys.path.append('tacotron2/')
from tacotron2.layers import TacotronSTFT
from tacotron2.utils import load_wav_to_torch

PATH_TO_DATA="data"
OUT_DIR = f"{PATH_TO_DATA}/melspec"

MAX_WAV_VALUE=32768.0
SAMPLING_RATE=22050
FILTER_LENGTH=1024
HOP_LENGTH=256
WIN_LENGTH=1024
N_MEL_CHANNELS=80
MEL_FMIN=0.0
MEL_FMAX=8000.0

# TODO: Adjust parameters for melspec extraction (sampling_rate)
# TODO: Make sure melspec_extraction works
# TODO: Complete dataset class and data loaders

def extract_melspec(path_to_wav):
    stft = TacotronSTFT(FILTER_LENGTH, HOP_LENGTH, WIN_LENGTH, N_MEL_CHANNELS, SAMPLING_RATE, MEL_FMIN, MEL_FMAX)
    audio, sampling_rate = load_wav_to_torch(path_to_wav)
    if sampling_rate != stft.sampling_rate:
        raise ValueError(f"{sampling_rate} != {stft.sampling_rate} SR doesn't match target SR")
    audio_norm = audio / MAX_WAV_VALUE
    audio_norm = audio_norm.unsqueeze(0)
    audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
    melspec = stft.mel_spectrogram(audio_norm)
    melspec = torch.squeeze(melspec, 0)
    return melspec

def main():
    if not os.path.exists(OUT_DIR):
        os.mkdir(OUT_DIR)

    for split in ["train", "test", "val"]:
        if not os.path.exists(f"{OUT_DIR}/{split}"):
            os.mkdir(f"{OUT_DIR}/{split}")

        with open(f"{PATH_TO_DATA}/splits/{split}.csv") as split_f:
            processed_count = 0
            csv_reader = csv.reader(split_f, delimiter="|")
            next(csv_reader, None)
            for row in csv_reader:
                print(processed_count)
                path_to_wav = row[0]
                melspec = extract_melspec(path_to_wav)
                processed_count += 1
                melspec_name = path_to_wav.split(sep="/")[-1]
                np.save(f"{OUT_DIR}/{split}/{melspec_name}.npy", melspec)

if __name__ == "__main__":
    main()