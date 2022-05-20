from resemblyzer import preprocess_wav, VoiceEncoder
import numpy as np
import pickle
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import csv

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


## Compute the embeddings
encoder = VoiceEncoder().cuda()

embed_list = []
speaker_list = []
"""with open("data/splits/train.csv") as split_f:
    csv_reader = csv.reader(split_f, delimiter="|")
    file_count = 0
    next(csv_reader, None)
    for row in csv_reader:
        path_to_wav = row[0].split(".")[0] + "_no_silence.wav"
        speaker = row[-1]
        speaker = int(speaker)
        wav = preprocess_wav(path_to_wav)
        embedding = encoder.embed_utterance(wav)
        embed_list.append(embedding)
        speaker_list.append(speaker)

with open('embed_list.pickle', 'wb') as handle:
    pickle.dump(embed_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('speaker_list.pickle', 'wb') as handle:
    pickle.dump(speaker_list, handle, protocol=pickle.HIGHEST_PROTOCOL)"""

with open('embed_list.pickle', 'rb') as handle:
    embed_list = pickle.load(handle)

with open('speaker_list.pickle', 'rb') as handle:
    speaker_list = pickle.load(handle)

## Project the embeddings in 2D space
"""plot_projections(utterance_embeds, speakers, title="Embedding projections")
plt.show()"""

tsne = TSNE(n_components=2)
pca = PCA(n_components=2)

fig, ax = plt.subplots()
y = pca.fit_transform(np.array(embed_list))
x_pl = []
y_pl = []
for speaker in range(10):
    x_pl.append([i for (i, s) in zip(y[:, 0], speaker_list) if s==speaker])
    y_pl.append([i for (i, s) in zip(y[:, 1], speaker_list) if s==speaker])
    ax.scatter(x_pl[speaker], y_pl[speaker])
plt.legend(SPEAKERS_LIST)
plt.savefig("fixed_encoding_pca.png")
plt.show()

fig, ax = plt.subplots()
for speaker in range(10):
    ax.scatter(np.mean(x_pl[speaker]), np.mean(y_pl[speaker]))
plt.legend(SPEAKERS_LIST)
plt.savefig("fixed_encoding_pca.png")
plt.show()

fig, ax = plt.subplots()
y = tsne.fit_transform(np.array(embed_list))
x_pl = []
y_pl = []
for speaker in range(10):
    x_pl.append([i for (i, s) in zip(y[:, 0], speaker_list) if s==speaker])
    y_pl.append([i for (i, s) in zip(y[:, 1], speaker_list) if s==speaker])
    ax.scatter(x_pl[speaker], y_pl[speaker])
plt.legend(SPEAKERS_LIST)
plt.savefig("fixed_encoding_tsne.png")
plt.show()

fig, ax = plt.subplots()
for speaker in range(10):
    ax.scatter(np.mean(x_pl[speaker]), np.mean(y_pl[speaker]))
plt.legend(SPEAKERS_LIST)
plt.savefig("fixed_encoding_tsne.png")
plt.show()

