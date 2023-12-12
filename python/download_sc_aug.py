import tensorflow_datasets as tfds
import numpy as np
from nanohydra.hydra import NanoHydra, SparseScaler
from nanohydra.utils import transform_mfcc, show_mfcc, get_idx_of_class, augment_data_of_class
import gc
import time
import matplotlib.pyplot as plt

start_t = time.time()


# Defines
CLASS_SILENCE       = 10
CLASS_UNKNOWN       = 11
NUM_CLASSES         = 12
NUM_SAMPLES_SCALING = 4000
NUM_SAMPLES_UNKNOWN = 20000
NUM_MFCC_CHANS      = 8
K=8
G=32
D=1
FEATURE_VEC_SZ      = 2*K*G*(D+1)*NUM_MFCC_CHANS
#FEATURE_VEC_SZ      = K*G*(D+1)*NUM_MFCC_CHANS
(Xtrain, Ytrain), (Xtest, Ytest), (Xval, Yval) = tfds.as_numpy(tfds.load('speech_commands', split=['train', 'test', 'validation'], batch_size=-1, as_supervised=True))

print(f"Shape of Xtrain: {Xtrain.shape}")
print(f"Shape of Ytrain: {Ytrain.shape}")
print(f"Shape of Xtest:  {Xtest.shape}")
print(f"Shape of Ytest:  {Ytest.shape}")
print(f"Shape of Xval:   {Xval.shape}")
print(f"Shape of Yval:   {Yval.shape}")


Xtrain = Xtrain.astype(np.float32)
Xval   = Xval.astype(np.float32)
Xtest  = Xtest.astype(np.float32)
Ytrain = Ytrain

plt.figure(1)
plt.plot(Xtest[0,:])
plt.show()
# Call garbage collector on useless arrays
del Ytest 
del Yval  
gc.collect()

# Collection of X samples to perform batch statistics
# We use NUM_SAMPLES_SCALING of each of the 12 classes
Xsample_stats = np.empty((NUM_SAMPLES_SCALING*12, FEATURE_VEC_SZ))


# Build dictionary with locations of samples of each class
class_loc = {i: get_idx_of_class(Ytrain, i) for i in range(NUM_CLASSES)}

# Isolate the background (class==10) samples
Xbackground = Xtrain[class_loc[CLASS_SILENCE]]

# Augment, MFCC transform and Hydra transform samples of each class, except Background
factors = {k:15 for k in range(NUM_CLASSES)}
factors[CLASS_SILENCE] = 80   # Background class is much smaller than others 
factors[CLASS_UNKNOWN] =  2   # Unknown class is much bigger than others 

for cl in range(NUM_CLASSES):
    Xraw = Xtrain[class_loc[cl]]

    if(cl != CLASS_UNKNOWN):
        print(f"Augmenting Class {cl} (Size={len(Xraw)}) by Factor={factors[cl]}")
        Xaug   = augment_data_of_class(Xraw, Xbackground, factors[cl], add_noise=factors[cl]!=CLASS_SILENCE)
        X = np.concatenate([Xraw, Xaug])
    else:
        print(f"Subsample Class {cl} 'Unknown' (Size={len(Xraw)}, New Size={NUM_SAMPLES_UNKNOWN})")
        X = Xraw[np.random.choice(len(Xraw), NUM_SAMPLES_UNKNOWN, replace=False),:]
        Xaug   = augment_data_of_class(X, Xbackground, factors[cl], add_noise=factors[cl]!=CLASS_SILENCE)
        X = np.concatenate([X, Xaug])
        print(f"Shape = {X.shape}")

    print(f"MFCC-Transforming Class {cl}...")
    Xmfcc  = transform_mfcc(X)

    if(cl == 0):
        # Model instantiation, only used to perform the transforms
        model  = NanoHydra(input_length=Xmfcc.shape[2], num_channels=NUM_MFCC_CHANS, k=K, g=G, max_dilations=D, dist="binomial", classifier="Logistic", scaler="Sparse", seed=1002, dtype=np.float32)    

    print(f"Hydra-Transforming Class {cl}...")
    Xtr  = model.forward_batch(Xmfcc.astype(np.float32), 200, do_fit=False, do_scale=False)

    print(Xtr.shape)
    assert Xtr.shape[1] == FEATURE_VEC_SZ, f"The feature vector has unexpected size={Xtr.shape[1]}, expected {FEATURE_VEC_SZ}"
    Xsample_stats[cl*NUM_SAMPLES_SCALING:(cl+1)*NUM_SAMPLES_SCALING,:] = Xtr[np.random.choice(len(Xtr), NUM_SAMPLES_SCALING, replace=False),:]

    model.save_transform(Xtr, f"SpeechCommands_300_cl_{cl}_notscaled", "./work", "train")
    gc.collect()

# Save the sample matrix for the scaler fitting (if eventually needed).
model.save_transform(Xsample_stats, "SpeechCommands_300_sample_stats", "./work", "train")

# Fit the scaler
print(f"Fitting the Scaler on a sample of the transformed features.")
scaler = SparseScaler()
scaler.fit(Xsample_stats)

# Scale the features
for cl in range(NUM_CLASSES):
    print(f"Scaling Class {cl}")
    X = model.load_transform(f"SpeechCommands_300_cl_{cl}_notscaled", "./work", "train")
    X = scaler.transform(X)
    model.save_transform(X.astype(np.float32), f"SpeechCommands_300_cl_{cl}", "./work", "train")

del Xtrain
del Ytrain
gc.collect()

# Transform and Scale Test Set
print(f"Transforming Test Set")
Xmfcc = transform_mfcc(Xtest)
Xtr = model.forward_batch(Xmfcc.astype(np.float32), 200, do_fit=False, do_scale=False)
Xsc = scaler.transform(Xtr)
model.save_transform(Xsc, f"SpeechCommands_300", "./work", "test")
del Xtest
gc.collect()

# Transforma and Scale Validation Set
print(f"Transforming Validation Set")
Xmfcc = transform_mfcc(Xval)
Xtr = model.forward_batch(Xmfcc.astype(np.float32), 200, do_fit=False, do_scale=False)
Xsc = scaler.transform(Xtr)
model.save_transform(Xsc, f"SpeechCommands_300", "./work", "val")
del Xval
gc.collect()

print(f"Successfully transformed in {(time.time()-start_t)/60:.1f} minutes")