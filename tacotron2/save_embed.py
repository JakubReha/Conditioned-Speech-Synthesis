import pickle
import numpy as np

with open('embed_list.pickle', 'rb') as handle:
    embed_list = pickle.load(handle)

with open('speaker_list.pickle', 'rb') as handle:
    speaker_list = pickle.load(handle)

embed_dict = {s: [] for s in speaker_list}
for s, e in zip(speaker_list, embed_list):
    embed_dict[s].append(e)

for s in np.unique(speaker_list):
    embed_dict[s] = np.mean(embed_dict[s], axis=0)

with open('speaker_embeddings.pickle', 'wb') as handle:
    pickle.dump(embed_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("Done")