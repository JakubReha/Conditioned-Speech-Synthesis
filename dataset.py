import torch
import csv
from torchvision import datasets  # type: ignore
import torchvision.transforms as T  # type: ignore
import torch.utils.data as tud
import matplotlib.pyplot as plt  # type: ignore
from torchvision.utils import make_grid  # type: ignore
import librosa.display
import torch.nn.functional as F
import sys

sys.path.append('tacotron2/')
from tacotron2.text import text_to_sequence
from tacotron2.text import sequence_to_text

class IEMOCAPDataset(torch.utils.data.Dataset):
    def __init__(self, path_to_csv: str, ):
        self.path_to_melspec = path_to_csv.split(sep=".")[:-1][0]
        self.path_to_melspec = self.path_to_melspec.replace('splits', 'melspec')
        self.melspec_paths = []
        self.emotions = []
        self.speakers = []
        self.transciptions = []

        with open(path_to_csv) as f:
            csv_reader = csv.reader(f, delimiter="|")
            next(csv_reader, None)
            count = 0
            for row in csv_reader:
                melspec_file = row[0].split(sep="/")[-1]
                melspec_file = f"{melspec_file.split('.')[0]}.pt"
                melspec_file = f"{self.path_to_melspec}/{melspec_file}"
                emotion = row[1]
                speaker = row[-1]
                transcription = row[5]
                self.melspec_paths += [melspec_file]
                self.emotions += [emotion]
                self.speakers += [speaker]
                self.transciptions += [torch.IntTensor(text_to_sequence(text=transcription, cleaner_names=['english_cleaners']))]
                count += 1

    def __getitem__(self, index):
        return torch.load(self.melspec_paths[index]), int(self.emotions[index]), self.transciptions[index], int(self.speakers[index])
    
    def __len__(self):
        return len(self.melspec_paths)

    def collate(self, batch_data):
        melspec_lens = torch.LongTensor([melspec.shape[1] for melspec, _, _ in batch_data])
        transcription_lens = torch.LongTensor([len(transcription) for _, _, transcription in batch_data])
        max_melspec_len = torch.max(melspec_lens)
        max_transcription_len = torch.max(transcription_lens)

        padded_melspec = torch.zeros((len(batch_data), batch_data[0][0].shape[0], max_melspec_len))
        emotions = torch.zeros((len(batch_data)), dtype=torch.int)
        speakers = torch.zeros((len(batch_data)), dtype=torch.int)
        padded_transcription = torch.zeros((len(batch_data), max_transcription_len), dtype=torch.int)
        for index, (melspec, emotion, transcription, speaker) in enumerate(batch_data):
            melspec = F.pad(input=melspec, pad=(0, max_melspec_len - melspec.shape[1]), mode="constant", value=0.0)
            transcription = F.pad(input=transcription, pad=(0, max_transcription_len - len(transcription)), mode="constant", value=0)
            padded_melspec[index] = melspec
            padded_transcription[index] = transcription
            emotions[index] = emotion
            speakers[index] = speaker

        return padded_melspec, emotions, padded_transcription, speakers, melspec_lens, transcription_lens


def show_batch(dataloader):
    for melspecs, emotions, transcriptions, speakers, melspec_lens, transcription_lens in dataloader:
        fig, axes = plt.subplots(nrows=len(emotions), figsize=(15, 10))
        for i in range(len(axes)):
            melspec = melspecs[i][:, :]
            transcription = transcriptions[i][:]
            axes[i].set_title(sequence_to_text(transcription.tolist()))
            img = librosa.display.specshow(melspec.numpy(), ax=axes[i])
            fig.colorbar(img, ax=axes[i])
        plt.show()
        break



if __name__ == "__main__":
    train_data = IEMOCAPDataset(path_to_csv="data/splits/train.csv")
    train_dataloader = tud.DataLoader(train_data, collate_fn=train_data.collate, num_workers=4, prefetch_factor=2, batch_size=4, shuffle=False)
    show_batch(train_dataloader)