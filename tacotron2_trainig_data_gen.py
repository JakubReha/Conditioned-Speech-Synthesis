import numpy as np
import csv
from tqdm import tqdm
import os
import random

def getPath(iemocap_main_dir, name):
    session = 'Session' + name[4]
    folder_name = name[0:-5]
    return os.path.join(iemocap_main_dir, session, 'sentences', 'wav', folder_name, name + '.wav')
    
def writeToFile(outdir, filename, items):
    with open(os.path.join(outdir,filename),"w") as f:
        f.writelines(items)

def main():
    iemocap_main_dir = '..\\iemocap\\iemocap\\'
    items = []
    test_split = 0.05
    val_split = 0.1
    train_split = 1-(test_split+val_split)
    n_items = 0
    with open(os.path.join(iemocap_main_dir, 'metadata.csv'), "r") as f:
        metadata = csv.reader(f)
        next(metadata, None)
        for lines in metadata:
            n_items += 1
            text = lines[-1]+'\n'
            filepath = getPath(iemocap_main_dir, lines[0])
            items.append("|".join([filepath, text]))
    
    random.shuffle(items)
    
    train_items = items[0:int(np.round(n_items*train_split))]
    test_items = items[int(np.round(n_items*train_split)):int(np.round(n_items*train_split)+np.round(n_items*val_split))]
    val_items = items[int(np.round(n_items*train_split)+np.round(n_items*val_split)):]
    
    outdir = './filelist/'
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    writeToFile(outdir, 'train_filelist.txt', train_items)
    writeToFile(outdir, 'test_filelist.txt', test_items)
    writeToFile(outdir, 'val_filelist.txt', val_items)
    

if __name__ == '__main__':
    main()
