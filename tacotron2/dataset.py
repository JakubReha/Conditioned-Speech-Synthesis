import torch
import csv
from torchvision import datasets  # type: ignore
import torchvision.transforms as T  # type: ignore
import torch.utils.data as tud
import matplotlib.pyplot as plt  # type: ignore
from torchvision.utils import make_grid  # type: ignore
import librosa.display
import torch.nn.functional as F
import argparse
import sys
sys.path.append('tacotron2/')
from hparams import create_hparams
from data_utils import TextMelLoader, TextMelCollate

sys.path.append('tacotron2/')
from text import text_to_sequence
from text import sequence_to_text

class IEMOCAPDataset(torch.utils.data.Dataset):
    def __init__(self, path_to_csv: str, silence: bool, padded: bool):
        folder_name = "melspec"
        if padded:
            folder_name = f"padded_{folder_name}"
        if not silence:
            folder_name = f"{folder_name}_no_silence"
        self.silence = silence
        self.path_to_melspec = path_to_csv.split(sep=".")[:-1][0]
        self.path_to_melspec = self.path_to_melspec.replace('splits', folder_name)
        self.melspec_paths = []
        self.emotions = []
        self.speakers = []
        self.transcriptions = []

        with open(path_to_csv) as f:
            csv_reader = csv.reader(f, delimiter="|")
            next(csv_reader, None)
            count = 0
            for row in csv_reader:
                melspec_file = row[0].split(sep="/")[-1]
                if not self.silence:
                    melspec_file = f"{melspec_file.split('.')[0]}_no_silence.pt"
                else:
                    melspec_file = f"{melspec_file.split('.')[0]}.pt"
                melspec_file = f"{self.path_to_melspec}/{melspec_file}"
                emotion = row[1]
                speaker = row[-1]
                transcription = row[5]
                self.melspec_paths += [melspec_file]
                self.emotions += [emotion]
                self.speakers += [speaker]
                self.transcriptions += [torch.IntTensor(text_to_sequence(text=transcription, cleaner_names=['english_cleaners']))]
                count += 1

    def __getitem__(self, index):
        return self.transcriptions[index], torch.load(self.melspec_paths[index]), int(self.speakers[index])
    
    def __len__(self):
        return len(self.melspec_paths)

class EmotionEmbeddingNetworkCollate():
    def __call__(self, batch_data):
        melspecs = torch.zeros((len(batch_data), batch_data[0][0].shape[0], batch_data[0][0].shape[1]))
        emotions = torch.zeros((len(batch_data)), dtype=torch.int)
        speakers = torch.zeros((len(batch_data)), dtype=torch.int)
        for index, (melspec, emotion, transcription, speaker) in enumerate(batch_data):
            melspecs[index] = melspec
            emotions[index] = emotion
            speakers[index] = speaker
        return melspecs, emotions, speakers 

class TacotronCollate():
    def __call__(self, batch_data):
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
    for transcriptions, input_lengths, melspecs, gate_padded, output_lengths, speakers in dataloader:
        fig, axes = plt.subplots(nrows=len(speakers), figsize=(15, 10))
        for i in range(len(axes)):
            melspec = melspecs[i][:, :]
            transcription = transcriptions[i][:]
            axes[i].set_title(sequence_to_text(transcription.tolist()))
            img = librosa.display.specshow(melspec.numpy(), ax=axes[i])
            fig.colorbar(img, ax=axes[i])
        plt.show()
        break



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_directory', type=str,
                        help='directory to save checkpoints', default="tacotron_output_vctk")
    parser.add_argument('-l', '--log_directory', type=str,
                        help='directory to save tensorboard logs', default="vctk_tacotron_logs")
    parser.add_argument('-c', '--checkpoint_path', type=str, default="tacotron2/tacotron2_statedict.pt",
                        required=False, help='checkpoint path')
    parser.add_argument('--warm_start', action='store_false',
                        help='load model weights only, ignore specified layers')
    parser.add_argument('--n_gpus', type=int, default=1,
                        required=False, help='number of gpus')
    parser.add_argument('--rank', type=int, default=0,
                        required=False, help='rank of current gpu')
    parser.add_argument('--group_name', type=str, default='group_name',
                        required=False, help='Distributed group name')
    parser.add_argument('--hparams', type=str,
                        required=False, help='comma separated name=value pairs')

    args = parser.parse_args()
    hparams = create_hparams(args.hparams)
    train_data = IEMOCAPDataset(path_to_csv="data/splits/train.csv", silence=False, padded=False)


    trainset = TextMelLoader("data/VCTK-Corpus-0.92/splits/train.txt", hparams)
    valset = TextMelLoader("data/VCTK-Corpus-0.92/splits/val.txt", hparams)
    collate_fn = TextMelCollate(hparams.n_frames_per_step)

    """train_loader = tud.DataLoader(train_data, collate_fn=collate_fn, num_workers=2, prefetch_factor=2, batch_size=4,
                                  shuffle=False)"""
    train_loader = tud.DataLoader(trainset, num_workers=2, shuffle=False,
                              sampler=None,
                              batch_size=6, pin_memory=False,
                              drop_last=True, collate_fn=collate_fn)

    show_batch(train_loader)
