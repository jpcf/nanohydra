from sklearn.manifold import TSNE
import tensorflow_datasets as tfds
from nanohydra.hydra import NanoHydra
import matplotlib.pyplot as plt
import numpy as np
from nanohydra.utils import vs_creator, get_idx_of_class

#Xval = model.load_transform(f"SpeechCommands_300", "./work", "val")
model  = NanoHydra(input_length=140, num_channels=8, k=8, g=32, max_dilations=1, dist="binomial", classifier="Logistic", scaler="Sparse", seed=19930111)    
(__, __), (__, Ytest), (__, Yval) = tfds.as_numpy(tfds.load('speech_commands', split=['train', 'test', 'validation'], batch_size=-1, as_supervised=True))


#print("Loading the validation set")
#Xval = model.load_transform(f"SpeechCommands_300", "./work", "val")
print("Loading the test set")
Xtest = model.load_transform(f"SpeechCommands_300", "./work", "test")

mapping = TSNE(n_components=2)
data = mapping.fit_transform(Xtest)

print(f"Shape: {data.shape}")
classes = np.unique(Ytest)
colors  = ['aqua', 'indigo', 'blue', 'olive', 'orange', 'tomato', 'green', 'lavender', 'yellow', 'sienna', 'black', 'fuchsia']
plt.figure(1)
for c in classes:
    idxs = get_idx_of_class(Ytest, c)
    plt.scatter(data[idxs,0], data[idxs,1], label=f"Class {c}", color=colors[c])
plt.legend()
plt.show()

