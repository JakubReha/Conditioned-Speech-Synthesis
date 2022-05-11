import random
import csv
import os
import numpy as np

PATH_TO_DATA="data"
OUT_DIR = f"{PATH_TO_DATA}/splits"
TEST_SPLIT = 0.05
VAL_SPLIT = 0.15
TRAIN_SPLIT = 1 - (TEST_SPLIT+VAL_SPLIT)

def main():
    items = []
    n_items = 0
    random.seed(1)

    if not os.path.exists(OUT_DIR):
        os.mkdir(OUT_DIR)

    with open(f"{PATH_TO_DATA}/iemocap/metadata.csv") as metadata_f:
        csv_reader = csv.reader(metadata_f, delimiter=",")
        next(csv_reader, None)
        for row in csv_reader:
            print(n_items)
            path_to_wav = get_path(row[0])
            emotion, valence, arousal, dominance, text = row[3], row[4], row[5], row[6], row[7]
            items += [[path_to_wav, emotion, valence, arousal, dominance, text]]
            n_items += 1
			
    random.shuffle(items)
    
    train_items = items[0:int(np.round(n_items*TRAIN_SPLIT))]
    test_items = items[int(np.round(n_items*TRAIN_SPLIT)):int(np.round(n_items*TRAIN_SPLIT)+np.round(n_items*VAL_SPLIT))]
    val_items = items[int(np.round(n_items*TRAIN_SPLIT)+np.round(n_items*VAL_SPLIT)):]

    writeToFile("train.csv", train_items)
    writeToFile("val.csv", val_items)
    writeToFile("test.csv", test_items)
	
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
    return path_to_wav

def writeToFile(filename, items):
    with open(os.path.join(OUT_DIR, filename), "w") as f:
        writer = csv.writer(f, delimiter="|")
        writer.writerow(["filename", "emotion", "valence", "arousal", "dominance", "text"], )
        writer.writerows(items)

if __name__ == "__main__":
	main()