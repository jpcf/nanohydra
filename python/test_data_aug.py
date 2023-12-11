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

X0 = Xtrain[class_loc[0], :].astype(np.float32)/(2**15-1)
X10 = Xtrain[class_loc[10], :].astype(np.float32)/(2**15-1)

X0_mfcc = transform_mfcc(X0[np.random.choice(len(X0), 10),:])
X10_mfcc = transform_mfcc(X10[np.random.choice(len(X10), 10),:])

#print(f"O {X0_mfcc[0].shape}")
#print(f"M {np.mean(X0_mfcc[0], axis=1).shape}")
#X0_mfcc[1] = X0_mfcc[0]-np.mean(X0_mfcc[0], axis=1)[:,np.newaxis]
#X0_mfcc[1] /= np.std(X0_mfcc[0], axis=1)[:,np.newaxis]
#
#show_mfcc(X0_mfcc[0], "Class 0", 0)
#show_mfcc(X0_mfcc[1], "Class 0", 1)
#show_mfcc(X10_mfcc[0], "Class 10", 2)
#show_mfcc(X10_mfcc[1], "Class 10", 3)
plt.show()

#Xbackground = Xtrain[class_loc[11]]
#print(f"Shape of Xbackground: {Xbackground.shape}")
#
#for i in range(11):
#    print(f"Augmenting class {i} by Factor={3}")
#    Xaug   = augment_data_of_class(Xtrain[class_loc[i]], Xbackground, 4)
#    Xmfcc  = transform_mfcc(Xaug)

gc.collect()
