import soundfile as sf
import librosa
import csv
import sys
import os

sys.path.append('.')
PATH_TO_DATA="data"

def remove_silence(audio_file, dur, orig):
    audio, sr = librosa.load(audio_file, sr=16000, mono=True)
    clips = librosa.effects.split(audio, top_db=20)
    wav_data = []
    for c in clips:
        print(c)
        data = audio[c[0]: c[1]]
        wav_data.extend(data)
    sf.write(audio_file.split(".")[0] + '_no_silence.wav', wav_data, sr)
    dur += len(wav_data)/sr
    orig += len(audio) / sr
    return dur, orig

def main():
    dur = 0
    orig = 0
    if not os.path.exists(PATH_TO_DATA):
        os.mkdir(PATH_TO_DATA)

    for split in ["train", "test", "val"]:
        total_count = 0
        with open(f"{PATH_TO_DATA}/splits/{split}.csv") as split_f:
            total_count = sum(1 for line in split_f) - 1

        with open(f"{PATH_TO_DATA}/splits/{split}.csv") as split_f:
            csv_reader = csv.reader(split_f, delimiter="|")
            file_count = 0
            next(csv_reader, None)
            for row in csv_reader:
                path_to_wav = row[0]
                dur, orig = remove_silence(path_to_wav, dur, orig)
                file_count += 1
                print(f"\t[{file_count}/{total_count}]: {path_to_wav}")
    print(orig)
    print(dur)

if __name__ == "__main__":
    main()