import torch
import csv
from torchvision import datasets  # type: ignore
import torchvision.transforms as T  # type: ignore
import torch.utils.data as tud
import numpy as np
import matplotlib.pyplot as plt  # type: ignore
from torchvision.utils import make_grid  # type: ignore
import librosa.display

class IEMOCAPDataset(torch.utils.data.Dataset):
    def __init__(self, path_to_csv: str, ):
        self.data = {"melspec": [], "labels":[]}

        self.path_to_melspec = path_to_csv.split(sep=".")[:-1][0]
        self.path_to_melspec = self.path_to_melspec.replace('splits', 'melspec')

        with open(path_to_csv) as f:
            csv_reader = csv.reader(f, delimiter="|")
            next(csv_reader, None)
            count=0
            for row in csv_reader:
                melspec_file = row[0].split(sep="/")[-1]
                melspec_file = f"{melspec_file.split('.')[0]}.pt"
                melspec_file = f"{self.path_to_melspec}/{melspec_file}"
                melspec = torch.load(melspec_file)

                emotion = row[1]

                self.data["melspec"] += [melspec]
                self.data["labels"] += [emotion]
                count += 1
                if count > 4:
                    break


    def __getitem__(self, index):
        return self.data["melspec"][index], self.data["labels"][index]
    
    def __len__(self):
        return len(self.data["melspec"])

    def show_batch(self, dataloader):
        for melspec, labels in dataloader:
            fig, ax = plt.subplots(figsize=(10,10))
            librosa.display.specshow(melspec)
            plt.colorbar()
            plt.show()
            break

if __name__ == "__main__":
    train_data = IEMOCAPDataset(path_to_csv="data/splits/train.csv")
    train_dataloader = tud.DataLoader(train_data, num_workers=4, prefetch_factor=2, batch_size=4, shuffle=False)
    train_data.show_batch(train_dataloader)