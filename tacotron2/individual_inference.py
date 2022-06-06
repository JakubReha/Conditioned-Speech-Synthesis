import matplotlib.pylab as plt
import sys
sys.path.append('tacotron2/waveglow/')
sys.path.append('tacotron2/')
import numpy as np
import torch
import os
from tqdm import tqdm
from hparams import create_hparams
from train import load_model
from text import text_to_sequence
import soundfile as sf
from data_utils import TextMelLoader
import librosa
from resemblyzer import preprocess_wav, VoiceEncoder

def plot_data(data, OUT_DIR, k, figsize=(16, 4)):
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto',
                       interpolation='none')
    plt.savefig(os.path.join(OUT_DIR, "allignments_"+str(k)+".png"))
    plt.show()

if __name__ == "__main__":
    hparams = create_hparams()
    hparams.sampling_rate = 22050
    checkpoint_path = "tacotron2/tacotron2_statedict.pt"
    #checkpoint_path = "tacotron_output_vctk_"+hparams.encoding_type+"/checkpoint_31000"
    OUT_DIR = "original_inference_vctk_"+hparams.encoding_type+"_"+checkpoint_path.split("_")[-1]
    if not os.path.exists(OUT_DIR):
        os.mkdir(OUT_DIR)


    model = load_model(hparams)
    model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    model = model.cuda().eval().half()

    waveglow_path = 'tacotron2/waveglow_256channels_new.pt'
    waveglow = torch.load(waveglow_path)['model']
    waveglow = waveglow.remove_weightnorm(waveglow)

    waveglow.cuda().eval().half()
    for k in waveglow.convinv:
        k.float()

    text = "Hello, does it work?"
    sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
    sequence = torch.autograd.Variable(
        torch.from_numpy(sequence)).cuda().long()
    trainset = TextMelLoader("data/VCTK-Corpus-0.92/splits/train.txt", hparams)
    for filename in os.listdir("own"):
        mel = trainset.get_mel(os.path.join("own", filename)).unsqueeze(0).cuda().half()
        fig, axes = plt.subplots(nrows=1, figsize=(10, 10))
        img = librosa.display.specshow(mel.squeeze().detach().cpu().numpy(), ax=axes)
        fig.colorbar(img, ax=axes)
        plt.savefig(os.path.join(OUT_DIR, filename + "_mel.png"))
        plt.show()
        encoder = VoiceEncoder().cuda()
        wav = preprocess_wav(os.path.join("own", filename))
        embedding = torch.from_numpy(encoder.embed_utterance(wav)).cuda().half().unsqueeze(0)
        (mel_outputs, mel_outputs_postnet, _, alignments), embedded_speaker = model.inference(sequence, embedding=embedding, mel=mel)

        plot_data((mel_outputs.float().data.cpu().numpy()[0],
                   mel_outputs_postnet.float().data.cpu().numpy()[0],
                   alignments.float().data.cpu().numpy()[0].T), OUT_DIR, filename)

        with torch.no_grad():
            audio = waveglow.infer(mel_outputs_postnet, sigma=0.666)
        sf.write(os.path.join(OUT_DIR, "fake_" + filename), audio[0].data.cpu().numpy().astype(np.float32), hparams.sampling_rate)
