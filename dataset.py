import torch
from torchvision import datasets  # type: ignore
import torchvision.transforms as T  # type: ignore
import torch.utils.data as tud
import numpy as np
import matplotlib.pyplot as plt  # type: ignore
from torchvision.utils import make_grid  # type: ignore

class IEMOCAPDataset():
    def __init__(self, num_workers: int = 4, prefetch_factor: int = 2):
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor

    def get_dataloaders(self, val_prop: float = 0.2 , batch_size: int = 200 , shuffle: bool = True):
        pass

    def show_batch(self, dataloader):
        for images, labels in dataloader:
            fig, ax = plt.subplots(figsize=(10,10))
            ax.imshow(make_grid(images.to(torch.device("cpu")), 10).permute(1,2,0))
            plt.show()
            break

if __name__ == "__main__":
    data = IEMOCAPDataset(num_workers=4, prefetch_factor=2)
    dataloaders = data.get_dataloaders(val_prop=0.2, batch_size=8, shuffle=False)
    data.show_batch(dataloaders["train_labeled"])