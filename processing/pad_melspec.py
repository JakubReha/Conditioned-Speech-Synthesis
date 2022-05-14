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
import soundfile
import numpy as np
import torch.nn.functional as F

PATH_TO_DATA="data"
OUT_DIR = f"{PATH_TO_DATA}/padded_melspec_no_silence"

MAX_WAV_VALUE=32768.0
SAMPLING_RATE=22050
FILTER_LENGTH=1024
HOP_LENGTH=256
WIN_LENGTH=1024
N_MEL_CHANNELS=80
MEL_FMIN=0.0
MEL_FMAX=8000.0


def main():
    if not os.path.exists(PATH_TO_DATA):
        os.mkdir(PATH_TO_DATA)

    if not os.path.exists(OUT_DIR):
        os.mkdir(OUT_DIR)

    print("############################")
    print(f"- Finding MAX_MELSPEC_LEN -")
    print("############################")
    max_melspec_len = 0
    for split in ["train", "test", "val"]:
        with open(f"{PATH_TO_DATA}/splits/{split}.csv") as split_f:
            csv_reader = csv.reader(split_f, delimiter="|")
            next(csv_reader, None)
            for row in csv_reader:
                path_to_melspec = row[0].split("/")[-1]
				# melspecs with silence removed
                path_to_melspec = path_to_melspec.split(".")[0] + "_no_silence.pt"
                path_to_melspec = f"{PATH_TO_DATA}/melspec_no_silence/{split}/{path_to_melspec}"
				# melspecs including silence
				# path_to_melspec = path_to_melspec.split(".")[0] + ".pt"
				# path_to_melspec = f"{PATH_TO_DATA}/melspec/{split}/{path_to_melspec}"
                melspec = torch.load(path_to_melspec)
                melspec_len = melspec.shape[1]

                if melspec_len >= max_melspec_len:
                    max_melspec_len = melspec_len
                    print(f"{path_to_melspec}: New MAX_MELSPEC_LEN = {max_melspec_len}")

    for split in ["train", "test", "val"]:
        if not os.path.exists(f"{OUT_DIR}/{split}"):
            os.mkdir(f"{OUT_DIR}/{split}")

        print("#################################################################################################")
        print(f"- Padding Mel Spectogram features to MAX_SEQUENCE_LEN={max_melspec_len} into {OUT_DIR}/{split} -")
        print("#################################################################################################")

        with open(f"{PATH_TO_DATA}/splits/{split}.csv") as split_f:
            csv_reader = csv.reader(split_f, delimiter="|")
            next(csv_reader, None)
            for row in csv_reader:
                filename = row[0].split("/")[-1]
				# melspecs with silence removed
                filename = filename.split(".")[0] + "_no_silence.pt"
                path_to_melspec = f"{PATH_TO_DATA}/melspec_no_silence/{split}/{filename}"
				# melspecs including silence
				# filename = path_to_melspec.split(".")[0] + ".pt"
				# path_to_melspec = f"{PATH_TO_DATA}/melspec/{split}/{filename}"

                melspec = torch.load(path_to_melspec)
                melspec = F.pad(input=melspec, pad=(0, max_melspec_len - melspec.shape[1]), mode="constant", value=0.0)
                filename = f"{OUT_DIR}/{split}/{filename}"
                torch.save(melspec, filename)

if __name__ == "__main__":
    main()