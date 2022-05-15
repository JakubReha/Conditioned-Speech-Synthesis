import matplotlib.pylab as plt
import sys
sys.path.append('tacotron2/waveglow/')
sys.path.append('tacotron2/')
import numpy as np
import torch

from hparams import create_hparams
from train import load_model
from text import text_to_sequence
from denoiser import Denoiser
import librosa
import soundfile as sf

def plot_data(data, k, figsize=(16, 4)):
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto',
                       interpolation='none')
    plt.savefig("inference/allignments_"+str(k)+".png")
    plt.show()

hparams = create_hparams()
hparams.sampling_rate = 22050

#checkpoint_path = "tacotron2/tacotron2_statedict.pt"
checkpoint_path = "tacotron_output/checkpoint_15000"
model = load_model(hparams)
model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
_ = model.cuda().eval().half()

waveglow_path = 'tacotron2/waveglow_256channels_new.pt'
waveglow = torch.load(waveglow_path)['model']
waveglow = waveglow.remove_weightnorm(waveglow)

waveglow.cuda().eval().half()
for k in waveglow.convinv:
    k.float()
denoiser = Denoiser(waveglow)


text = "Fuck yeah. It finally works."
sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
sequence = torch.autograd.Variable(
    torch.from_numpy(sequence)).cuda().long()
embeddings = []
for i in range(10):
    (mel_outputs, mel_outputs_postnet, _, alignments), embedded_speaker = model.inference(sequence, torch.LongTensor([[i]]).cuda())
    embeddings.append(embedded_speaker)
    plot_data((mel_outputs.float().data.cpu().numpy()[0],
               mel_outputs_postnet.float().data.cpu().numpy()[0],
               alignments.float().data.cpu().numpy()[0].T), i)

    with torch.no_grad():
        audio = waveglow.infer(mel_outputs_postnet, sigma=0.666)
    sf.write("inference/demo_speaker_"+str(i)+".wav", audio[0].data.cpu().numpy().astype(np.float32), hparams.sampling_rate)

    audio_denoised = denoiser(audio, strength=0.01)[:, 0]
    sf.write("inference/demo_denoised_speaker_"+str(i)+".wav", audio_denoised.squeeze().cpu().numpy().astype(np.float32), hparams.sampling_rate)

