import torch
from dataset import IEMOCAPDataset, EmotionEmbeddingNetworkCollate
import torch.utils.data as tud
from tqdm import tqdm
import pickle
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

SPEAKER_DICT = {
    'Ses01F': 0,
    'Ses01M': 1,
    'Ses02F': 2,
    'Ses02M': 3,
    'Ses03F': 4,
    'Ses03M': 5,
    'Ses04F': 6,
    'Ses04M': 7,
    'Ses05F': 8,
    'Ses05M': 9,
}
SPEAKERS_LIST = list(SPEAKER_DICT.keys())

model = torch.hub.load('RF5/simple-speaker-embedding', 'gru_embedder').cuda()
model.eval()



"""train_data = IEMOCAPDataset(path_to_csv="data/splits/train.csv", silence=False, padded=False)
train_dataloader = tud.DataLoader(train_data, collate_fn=EmotionEmbeddingNetworkCollate(), num_workers=1, batch_size=1, shuffle=False)
embed_dict = {s: [] for s in SPEAKERS_LIST}
for mel, _, speaker in tqdm(train_dataloader):
    speaker = int(speaker.numpy().squeeze())
    mel = mel.transpose(1, 2).cuda()
    embedding = model(mel).detach().cpu().numpy()
    embed_dict[SPEAKERS_LIST[speaker]].append(embedding)

with open('embed_dict.pickle', 'wb') as handle:
    pickle.dump(embed_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)"""

with open('embed_dict.pickle', 'rb') as handle:
    embed_dict = pickle.load(handle)

tsne = TSNE(n_components=2)
pca = PCA(n_components=2)

fig, ax = plt.subplots()
for speaker in SPEAKERS_LIST:
    y = pca.fit_transform(np.array(embed_dict[speaker]).squeeze())
    ax.scatter([i for i in y[:, 0]], [i for i in y[:, 1]])
plt.legend(SPEAKERS_LIST)
plt.savefig("fixed_encoding_pca.png")
plt.show()

fig, ax = plt.subplots()
for speaker in SPEAKERS_LIST:
    y = tsne.fit_transform(np.array(embed_dict[speaker]).squeeze())
    ax.scatter([i for i in y[:, 0]], [i for i in y[:, 1]])
plt.legend(SPEAKERS_LIST)
plt.savefig("fixed_encoding_tsne.png")
plt.show()

fig, ax = plt.subplots()
y = pca.fit_transform([np.mean(np.array(embed_dict[speaker]).squeeze(), axis=0) for speaker in SPEAKERS_LIST])
for speaker in SPEAKERS_LIST:
    ax.scatter([y[SPEAKER_DICT[speaker], 0]], [y[SPEAKER_DICT[speaker], 1]])
plt.legend(SPEAKERS_LIST)
plt.savefig("fixed_encoding_pca_means.png")
plt.show()

fig, ax = plt.subplots()
y = tsne.fit_transform([np.mean(np.array(embed_dict[speaker]).squeeze(), axis=0) for speaker in SPEAKERS_LIST])
for speaker in SPEAKERS_LIST:
    ax.scatter([y[SPEAKER_DICT[speaker], 0]], [y[SPEAKER_DICT[speaker], 1]])
plt.legend(SPEAKERS_LIST)
plt.savefig("fixed_encoding_tsne_means.png")
plt.show()



