from resemblyzer import preprocess_wav, VoiceEncoder
import numpy as np
import pickle
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import csv
from tqdm import tqdm
SPEAKERS = ['p364', 'p256', 'p299', 'p313', 'p306', 'p261', 'p244', 'p304', 'p240', 'p305', 'p250', 'p278', 'p255',
            'p239', 'p293', 'p288', 'p225', 'p316', 'p270', 'p286', 'p334', 'p271', 'p279', 'p336', 'p343', 'p295',
            'p285', 'p312', 'p265', 's5', 'p326', 'p262', 'p361', 'p253', 'p302', 'p226', 'p269', 'p341', 'p300',
            'p259', 'p252', 'p317', 'p376', 'p268', 'p247', 'p318', 'p287', 'p251', 'p330', 'p301', 'p283', 'p335',
            'p329', 'p275', 'p314', 'p238', 'p297', 'p311', 'p345', 'p254', 'p363', 'p236', 'p284', 'p282', 'p245',
            'p294', 'p276', 'p281', 'p234', 'p258', 'p351', 'p241', 'p340', 'p232', 'p237', 'p228', 'p339', 'p310',
            'p267', 'p264', 'p323', 'p233', 'p298', 'p227', 'p230', 'p229', 'p333', 'p260', 'p308', 'p231', 'p263',
            'p362', 'p292', 'p257', 'p360', 'p249', 'p246', 'p248', 'p374', 'p266', 'p274', 'p272', 'p307', 'p277',
            'p273', 'p243', 'p303', 'p347']


"""## Compute the embeddings
encoder = VoiceEncoder().cuda()

embed_list = []
speaker_list = []
with open("data/VCTK-Corpus-0.92/splits/train.txt") as split_f:
    csv_reader = csv.reader(split_f, delimiter="|")
    file_count = 0
    next(csv_reader, None)
    for row in tqdm(csv_reader):
        path_to_wav = row[0]
        speaker = row[-1]
        speaker = int(speaker)
        wav = preprocess_wav(path_to_wav)
        embedding = encoder.embed_utterance(wav)
        embed_list.append(embedding)
        speaker_list.append(speaker)

with open('embed_list.pickle', 'wb') as handle:
    pickle.dump(embed_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('speaker_list.pickle', 'wb') as handle:
    pickle.dump(speaker_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
"""
with open('embed_list.pickle', 'rb') as handle:
    embed_list = pickle.load(handle)

with open('speaker_list.pickle', 'rb') as handle:
    speaker_list = pickle.load(handle)

gender_dict = {}
with open("data/VCTK-Corpus-0.92/speaker-info.txt") as f:
    lines = f.readlines()
    for text in tqdm(lines[1:]):
        try:
            row = text.split()
            speaker = SPEAKERS.index(row[0])
            gender = row[2]
            accent = row[3]
            gender_dict[speaker] = [gender, accent]
        except:
            continue



## Project the embeddings in 2D space
"""plot_projections(utterance_embeds, speakers, title="Embedding projections")
plt.show()"""

tsne = TSNE(n_components=2)
pca = PCA(n_components=2)

fig, ax = plt.subplots()
y = pca.fit_transform(np.array(embed_list))
x_pl = []
y_pl = []
for speaker in range(len(SPEAKERS)):
    x_pl.append([i for (i, s) in zip(y[:, 0], speaker_list) if s==speaker])
    y_pl.append([i for (i, s) in zip(y[:, 1], speaker_list) if s==speaker])
    if gender_dict[speaker][0] == "F":
        marker = "."
    else:
        marker = "x"
    if gender_dict[speaker][1] == "American":
        c = "blue"
    elif gender_dict[speaker][1] == "English":
        c = "red"
    elif gender_dict[speaker][1] == "Canadian":
        c = "purple"
    elif gender_dict[speaker][1] == "Indian":
        c = "yellow"
    elif gender_dict[speaker][1] == "Scottish":
        c = "green"
    elif "Irish" in gender_dict[speaker][1]:
        c = "pink"
    else:
        c = "black"


    ax.scatter(x_pl[speaker], y_pl[speaker], marker=marker, c=c)
#plt.legend(SPEAKERS)
plt.savefig("fixed_encoding_pca_vctk.png")
plt.show()

fig, ax = plt.subplots()
for speaker in range(len(SPEAKERS)):
    if gender_dict[speaker][0] == "F":
        marker = "."
    else:
        marker = "x"
    if gender_dict[speaker][1] == "American":
        c = "blue"
    elif gender_dict[speaker][1] == "English":
        c = "red"
    elif gender_dict[speaker][1] == "Canadian":
        c = "purple"
    elif gender_dict[speaker][1] == "Indian":
        c = "yellow"
    elif gender_dict[speaker][1] == "Scottish":
        c = "green"
    elif "Irish" in gender_dict[speaker][1]:
        c = "pink"
    else:
        c = "black"
    ax.scatter(np.mean(x_pl[speaker]), np.mean(y_pl[speaker]), marker=marker, c=c)
#plt.legend(SPEAKERS)
plt.savefig("fixed_encoding_pca_vctk.png")
plt.show()

fig, ax = plt.subplots()
y = tsne.fit_transform(np.array(embed_list))
x_pl = []
y_pl = []
for speaker in range(len(SPEAKERS)):
    x_pl.append([i for (i, s) in zip(y[:, 0], speaker_list) if s==speaker])
    y_pl.append([i for (i, s) in zip(y[:, 1], speaker_list) if s==speaker])
    ax.scatter(x_pl[speaker], y_pl[speaker])
#plt.legend(SPEAKERS)
plt.savefig("fixed_encoding_tsne_vctk.png")
plt.show()

fig, ax = plt.subplots()
for speaker in range(len(SPEAKERS)):
    if gender_dict[speaker][0] == "F":
        marker = "."
    else:
        marker = "x"
    if gender_dict[speaker][1] == "American":
        c = "blue"
    elif gender_dict[speaker][1] == "English":
        c = "red"
    elif gender_dict[speaker][1] == "Canadian":
        c = "purple"
    elif gender_dict[speaker][1] == "Indian":
        c = "yellow"
    elif gender_dict[speaker][1] == "Scottish":
        c = "green"
    else:
        c = "black"
    ax.scatter(np.mean(x_pl[speaker]), np.mean(y_pl[speaker]), marker=marker, c=c)
#plt.legend(SPEAKERS)
plt.savefig("fixed_encoding_tsne_vctk.png")
plt.show()

