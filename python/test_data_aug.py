from nanohydra.utils import get_idx_of_class, augment_data_of_class, transform_mfcc, show_mfcc, flatten_mfcc
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gc 
import time

SHOW_HISTOGRAMS = False

(Xtrain, Ytrain), (__,__), (__,__) = tfds.as_numpy(tfds.load('speech_commands', split=['train', 'test', 'validation'], batch_size=-1, as_supervised=True))

del __
gc.collect()

class_loc = {i: get_idx_of_class(Ytrain, i) for i in range(12)}

print(f"Shape of X: {Xtrain.shape}")
# Verify correctness of results
total = 0
for i in range(12):
    assert np.all(Ytrain[class_loc[i]] == i)
    print(f"Samples of class {i}: {len(Ytrain[class_loc[i]])}")
    total += len(Ytrain[class_loc[i]])

assert total == Xtrain.shape[0]
print(f"Total Number of Samples: {total}")

if(SHOW_HISTOGRAMS):
    plt.figure(1)
    plt.hist(Ytrain, bins=12)
    plt.show()

X = {}

alpha = 0.97
print(f"Performing pre-emphasis (alpha={alpha})...")
Xtrain = Xtrain[:,1:] - alpha*Xtrain[:,:-1]

for n in range(10):
    for i in range(12):
        print(f"Class {i}")
        X[i] = Xtrain[class_loc[i], :].astype(np.float32)#/(2**15-1)
        transform_mfcc(X[i][n*10:n*10+1])
        plt.show()

#Xbackground = Xtrain[class_loc[11]]
#print(f"Shape of Xbackground: {Xbackground.shape}")
#
#for i in range(11):
#    print(f"Augmenting class {i} by Factor={3}")
#    Xaug   = augment_data_of_class(Xtrain[class_loc[i]], Xbackground, 4)
#    Xmfcc  = transform_mfcc(Xaug)

gc.collect()
