import csv
import sys
import torch
import os
import sys
import librosa

sys.path.append('.')
sys.path.append('tacotron2/')
from tacotron2.layers import TacotronSTFT
from tacotron2.utils import load_wav_to_torch
import soundfile as sf
import numpy as np

PATH_TO_DATA="data"
OUT_DIR = f"{PATH_TO_DATA}/melspec_no_silence"

MAX_WAV_VALUE=32768.0
SAMPLING_RATE=22050
FILTER_LENGTH=1024
HOP_LENGTH=256
WIN_LENGTH=1024
N_MEL_CHANNELS=80
MEL_FMIN=0.0
MEL_FMAX=8000.0


def extract_melspec(path_to_wav):
    stft = TacotronSTFT(FILTER_LENGTH, HOP_LENGTH, WIN_LENGTH, N_MEL_CHANNELS, SAMPLING_RATE, MEL_FMIN, MEL_FMAX)
    audio, sampling_rate = load_wav_to_torch(path_to_wav)
    #audio = torch.from_numpy(np.clip(librosa.resample(audio.numpy(), orig_sr=sampling_rate, target_sr=SAMPLING_RATE), -MAX_WAV_VALUE, MAX_WAV_VALUE))
    #sf.write(path_to_wav.split(".")[0] + '_22kHz.wav', audio, SAMPLING_RATE)
    if sampling_rate != stft.sampling_rate:
        raise ValueError(f"{sampling_rate} != {stft.sampling_rate} SR doesn't match target SR")
    audio_norm = audio / MAX_WAV_VALUE
    audio_norm = audio_norm.unsqueeze(0)
    audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
    melspec = stft.mel_spectrogram(audio_norm)
    melspec = torch.squeeze(melspec, 0)
    return melspec

def main():
    if not os.path.exists(PATH_TO_DATA):
        os.mkdir(PATH_TO_DATA)

    if not os.path.exists(OUT_DIR):
        os.mkdir(OUT_DIR)

    for split in ["val", "train", "test"]:
        print("##############################################################")
        print(f"- Extracting Mel Spectogram features into {OUT_DIR}/{split} -")
        print("##############################################################")
        if not os.path.exists(f"{OUT_DIR}/{split}"):
            os.mkdir(f"{OUT_DIR}/{split}")

        total_count = 0
        with open(f"{PATH_TO_DATA}/splits/{split}.csv") as split_f:
            total_count = sum(1 for line in split_f) - 1

        with open(f"{PATH_TO_DATA}/splits/{split}.csv") as split_f:
            csv_reader = csv.reader(split_f, delimiter="|")
            file_count = 0
            next(csv_reader, None)
            for row in csv_reader:
                path_to_wav = row[0].split(".")[0] + "_no_silence.wav"
                # path_to_wav = row[0]
                melspec = extract_melspec(path_to_wav)

                filename = path_to_wav.split(sep="/")[-1]
                filename = filename.split(sep=".")[0]
                filename = f"{OUT_DIR}/{split}/{filename}_16k.pt"
                torch.save(melspec, filename)

                file_count += 1
                print(f"\t[{file_count}/{total_count}]: {filename}")

if __name__ == "__main__":
    main()