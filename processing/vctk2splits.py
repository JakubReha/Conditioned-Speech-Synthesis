import random
import csv
import os
import numpy as np
import glob

PATH_TO_DATA = "data/VCTK-Corpus-0.92"
OUT_DIR = f"{PATH_TO_DATA}/splits"
TEST_SPLIT = 0
VAL_SPLIT = 0.05
TRAIN_SPLIT = 1 - (TEST_SPLIT+VAL_SPLIT)

SPEAKERS = ['p364', 'p256', 'p299', 'p313', 'p306', 'p261', 'p244', 'p304', 'p240', 'p305', 'p250', 'p278', 'p255',
            'p239', 'p293', 'p288', 'p225', 'p316', 'p270', 'p286', 'p334', 'p271', 'p279', 'p336', 'p343', 'p295',
            'p285', 'p312', 'p265', 's5', 'p326', 'p262', 'p361', 'p253', 'p302', 'p226', 'p269', 'p341', 'p300',
            'p259', 'p252', 'p317', 'p376', 'p268', 'p247', 'p318', 'p287', 'p251', 'p330', 'p301', 'p283', 'p335',
            'p329', 'p275', 'p314', 'p238', 'p297', 'p311', 'p345', 'p254', 'p363', 'p236', 'p284', 'p282', 'p245',
            'p294', 'p276', 'p281', 'p234', 'p258', 'p351', 'p241', 'p340', 'p232', 'p237', 'p228', 'p339', 'p310',
            'p267', 'p264', 'p323', 'p233', 'p298', 'p227', 'p230', 'p229', 'p333', 'p260', 'p308', 'p231', 'p263',
            'p292', 'p257', 'p360', 'p249', 'p246', 'p248', 'p374', 'p266', 'p274', 'p272', 'p307', 'p277',
            'p273', 'p243', 'p303', 'p347']

def main():
    random.seed(1)

    if not os.path.exists(PATH_TO_DATA):
        os.mkdir(PATH_TO_DATA)

    if not os.path.exists(OUT_DIR):
        os.mkdir(OUT_DIR)

    print("#####################################################################")
    print(f"- Splitting IEMOCAP data into train/val/test splits into {OUT_DIR} -")
    print("#####################################################################")
    train_items = []
    val_items = []
    for speaker in SPEAKERS:
        files = glob.glob(os.path.join(os.path.join(PATH_TO_DATA, "txt/"+speaker), "*.txt"))
        file_count = len(files)
        random.shuffle(files)
        for file in files[0:int(np.ceil(file_count*TRAIN_SPLIT))]:
            with open(file) as f:
                text = f.readlines()[0][:-1].rstrip()
            train_items += [[file.replace(".txt", "_mic2.flac").replace("txt", "wav48_silence_trimmed"), text, SPEAKERS.index(speaker)]]
        for file in files[int(np.ceil(file_count*TRAIN_SPLIT)):int(np.ceil(file_count*TRAIN_SPLIT)+np.ceil(file_count*VAL_SPLIT))]:
            with open(file) as f:
                text = f.readlines()[0][:-2].rstrip()
            val_items += [[file.replace(".txt", "_mic2.flac").replace("txt", "wav48_silence_trimmed"), text, SPEAKERS.index(speaker)]]
    random.shuffle(train_items)
    random.shuffle(val_items)
    writeToFile("train.txt", train_items)
    writeToFile("val.txt", val_items)



def writeToFile(filename, items):
    with open(os.path.join(OUT_DIR, filename), "w") as f:
        writer = csv.writer(f, delimiter="|")
        writer.writerow(["filename", "text", "speaker_id"], )
        writer.writerows(items)

if __name__ == "__main__":
	main()