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
from text import text_to_sequence, sequence_to_text
from denoiser import Denoiser
import soundfile as sf
from train import prepare_dataloaders
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

SPEAKERS = ['p364', 'p256', 'p299', 'p313', 'p306', 'p261', 'p244', 'p304', 'p240', 'p305', 'p250', 'p278', 'p255',
            'p239', 'p293', 'p288', 'p225', 'p316', 'p270', 'p286', 'p334', 'p271', 'p279', 'p336', 'p343', 'p295',
            'p285', 'p312', 'p265', 's5', 'p326', 'p262', 'p361', 'p253', 'p302', 'p226', 'p269', 'p341', 'p300',
            'p259', 'p252', 'p317', 'p376', 'p268', 'p247', 'p318', 'p287', 'p251', 'p330', 'p301', 'p283', 'p335',
            'p329', 'p275', 'p314', 'p238', 'p297', 'p311', 'p345', 'p254', 'p363', 'p236', 'p284', 'p282', 'p245',
            'p294', 'p276', 'p281', 'p234', 'p258', 'p351', 'p241', 'p340', 'p232', 'p237', 'p228', 'p339', 'p310',
            'p267', 'p264', 'p323', 'p233', 'p298', 'p227', 'p230', 'p229', 'p333', 'p260', 'p308', 'p231', 'p263',
            'p362', 'p292', 'p257', 'p360', 'p249', 'p246', 'p248', 'p374', 'p266', 'p274', 'p272', 'p307', 'p277',
            'p273', 'p243', 'p303', 'p347']

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
    tokens = True
    #checkpoint_path = "tacotron2/tacotron2_statedict.pt"
    checkpoint_path = "tacotron_output_vctk_gst/checkpoint_31000"
    OUT_DIR = "negative_half_tokens_inference_vctk_"+hparams.encoding_type+"_"+checkpoint_path.split("_")[-1]
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
    denoiser = Denoiser(waveglow)


    text = "Hello, does it work?"
    sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
    sequence = torch.autograd.Variable(
        torch.from_numpy(sequence)).cuda().long()
    embeddings = []
    ref_mel = {}
    if hparams.encoding_type == "gst":
        hparams.batch_size = 1
        train_loader, valset, collate_fn = prepare_dataloaders(hparams)
        val_sampler = None
        val_loader = DataLoader(valset, sampler=val_sampler, num_workers=2,
                                shuffle=False, batch_size=1,
                                pin_memory=False, collate_fn=collate_fn)
        dur = 0
        for i, batch in enumerate(tqdm(val_loader)):
            x, y = model.parse_batch(batch)
            mel = x[2]
            sp = x[-1]
            dur += mel.shape[-1]
            """
            id = int(sp.squeeze().detach().cpu().numpy())
            if id not in ref_mel: #or mel.shape[-1] > ref_mel[id].shape[-1]:
                ref_mel[id] = mel.half()
                if SPEAKERS[id] == "p258":
                    print(id)
                    print(sequence_to_text(x[0].squeeze().detach().cpu().numpy()))"""
    print(dur)
    if tokens == True:
        for i in tqdm(range(10)):
            weights = torch.zeros(10).cuda().half()
            weights[i] = - 0.5
            (mel_outputs, mel_outputs_postnet, _, alignments), embedded_speaker = model.inference(sequence, weights=weights)

            embeddings.append(embedded_speaker.detach().cpu().numpy().squeeze())
            plot_data((mel_outputs.float().data.cpu().numpy()[0],
                       mel_outputs_postnet.float().data.cpu().numpy()[0],
                       alignments.float().data.cpu().numpy()[0].T), OUT_DIR, i)

            with torch.no_grad():
                audio = waveglow.infer(mel_outputs_postnet, sigma=0.666)
            sf.write(os.path.join(OUT_DIR, "demo_token_" + str(i) + ".wav"),
                     audio[0].data.cpu().numpy().astype(np.float32), hparams.sampling_rate)
    else:
        for i in tqdm(SPEAKERS):
            if hparams.encoding_type == "gst":
                (mel_outputs, mel_outputs_postnet, _, alignments), embedded_speaker = model.inference(sequence,
                                                                                                      torch.LongTensor([[SPEAKERS.index(i)]]).cuda(), ref_mel[SPEAKERS.index(i)])
            else:
                (mel_outputs, mel_outputs_postnet, _, alignments), embedded_speaker = model.inference(sequence, torch.LongTensor([[SPEAKERS.index(i)]]).cuda())
            embeddings.append(embedded_speaker.detach().cpu().numpy().squeeze())
            plot_data((mel_outputs.float().data.cpu().numpy()[0],
                       mel_outputs_postnet.float().data.cpu().numpy()[0],
                       alignments.float().data.cpu().numpy()[0].T), OUT_DIR, i)

            with torch.no_grad():
                audio = waveglow.infer(mel_outputs_postnet, sigma=0.666)
            sf.write(os.path.join(OUT_DIR, "demo_speaker_"+str(i)+".wav"), audio[0].data.cpu().numpy().astype(np.float32), hparams.sampling_rate)

            #audio_denoised = denoiser(audio, strength=0.01)[:, 0]
            #sf.write(os.path.join(OUT_DIR, "demo_denoised_speaker_"+str(i)+".wav"), audio_denoised.squeeze().cpu().numpy().astype(np.float32), hparams.sampling_rate)


    """pca = TSNE(n_components=2, perplexity=5)
    pca2 = PCA(n_components=2)
    y = pca.fit_transform(embeddings)
    y2 = pca2.fit_transform(embeddings)
    fig, ax = plt.subplots()
    ax.scatter([i for i in y[:, 0]], [i for i in y[:, 1]])
    for i in range(10):
        ax.annotate(list(SPEAKER_DICT.keys())[i], y[i])
    plt.savefig(os.path.join(OUT_DIR, "tsne.png"))
    plt.show()
    fig, ax = plt.subplots()
    ax.scatter([i for i in y2[:, 0]], [i for i in y2[:, 1]])
    for i in range(10):
        ax.annotate(list(SPEAKER_DICT.keys())[i], y2[i])
    plt.savefig(os.path.join(OUT_DIR, "pca.png"))
    plt.show()"""
