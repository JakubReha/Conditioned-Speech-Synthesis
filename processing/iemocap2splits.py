import random
import csv
import os
import numpy as np
import matplotlib.pyplot as plt

PATH_TO_DATA = "data"
OUT_DIR = f"{PATH_TO_DATA}/splits"
TEST_SPLIT = 0
VAL_SPLIT = 0.1
TRAIN_SPLIT = 1 - (TEST_SPLIT+VAL_SPLIT)
EMOTION_DICT = {
    'neu': 0,
    'sur': 1,
    'fru': 2,
    'sad': 3,
    'exc': 4,
    'ang': 5,
    'hap': 6}

SPEAKER_DICT = {
    'Ses01F': 0,
    'Ses01M': 1,
    'Ses02F': 2,
    'Ses02M': 3,
    'Ses03F': 4,
    'Ses03M': 5,
    'Ses04F': 6,
    'Ses04M': 7,
    'Ses05F': 8,
    'Ses05M': 9,
}


def main():
    random.seed(1)

    if not os.path.exists(PATH_TO_DATA):
        os.mkdir(PATH_TO_DATA)

    if not os.path.exists(OUT_DIR):
        os.mkdir(OUT_DIR)

    with open(f"{PATH_TO_DATA}/iemocap/metadata.csv") as metadata_f:
        total_count = sum(1 for line in metadata_f) - 1

    with open(f"{PATH_TO_DATA}/iemocap/metadata.csv") as metadata_f:
        print("#####################################################################")
        print(f"- Splitting IEMOCAP data into train/val/test splits into {OUT_DIR} -")
        print("#####################################################################")
        csv_reader = csv.reader(metadata_f, delimiter=",")
        next(csv_reader, None)
        file_count = 0
        items = {}
        for row in csv_reader:
            path_to_wav, speaker_id = get_path(row[0])
            emotion, valence, arousal, dominance, text = row[3], row[4], row[5], row[6], row[7]
            if speaker_id not in items:
                items[speaker_id] = {}
            if emotion not in EMOTION_DICT:
                continue
            if emotion not in items[speaker_id]:
                items[speaker_id][emotion] = []
            items[speaker_id][emotion] += [[path_to_wav, EMOTION_DICT[emotion], valence, arousal, dominance, text, SPEAKER_DICT[speaker_id]]]
            file_count += 1
            print(f"\t[{file_count}/{total_count}]: {path_to_wav}")
    train_items = []
    val_items = []
    test_items = []
    weights = {}
    for speaker in items:
        for emotion in items[speaker]:
            random.shuffle(items[speaker][emotion])
            if emotion not in weights:
                weights[emotion] = 0
            file_count = len(items[speaker][emotion])
            weights[emotion] += int(np.ceil(file_count*TRAIN_SPLIT))
            train_items += items[speaker][emotion][0:int(np.ceil(file_count*TRAIN_SPLIT))]
            val_items += items[speaker][emotion][int(np.ceil(file_count*TRAIN_SPLIT)):int(np.ceil(file_count*TRAIN_SPLIT)+np.ceil(file_count*VAL_SPLIT))]
            test_items += items[speaker][emotion][int(np.ceil(file_count*TRAIN_SPLIT)+np.ceil(file_count*VAL_SPLIT)):]

    plt.bar(np.arange(len(weights.values())), weights.values())
    plt.show()
    writeToFile("train.csv", train_items)
    writeToFile("val.csv", val_items)
    writeToFile("test.csv", test_items)
    weights = 1/np.array(list(weights.values()))
    weights = weights/weights.sum()
    np.save("data/weights.npy", weights)

def get_path(filename):
    sessions = {
        '1': 'Session1',
        '2': 'Session2',
        '3': 'Session3',
        '4': 'Session4',
        '5': 'Session5',
    }
    # Getting session
    sess_id = filename[4]  # Get the session id number (1, 2, 3, 4 or 5)
    session_name = sessions[sess_id]
    # Getting sentence folder
    sentence_folder = filename[:14]
    # Getting sentence wav name
    wav_name = filename[:19]

    if filename[7:18] == 'script01_1b':
        sentence_folder = filename[:18]
        wav_name = filename[:23]

    elif sentence_folder[7:13] == 'script':
        sentence_folder = filename[:17]
        wav_name = filename[:22]

    elif str(filename[7:15]) in ['impro05a', 'impro05b', 'impro08a', 'impro08b']:
        sentence_folder = filename[:15]
        wav_name = filename[:20]

    path_to_wav = f'{PATH_TO_DATA}/iemocap/{session_name}/sentences/wav/{sentence_folder}/{wav_name}.wav'
    speaker_id = wav_name[:6]
    return path_to_wav, speaker_id

def writeToFile(filename, items):
    with open(os.path.join(OUT_DIR, filename), "w") as f:
        writer = csv.writer(f, delimiter="|")
        writer.writerow(["filename", "emotion", "valence", "arousal", "dominance", "text", "speaker_id"], )
        writer.writerows(items)

if __name__ == "__main__":
	main()