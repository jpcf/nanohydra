import tensorflow_datasets as tfds
import numpy as np
from nanohydra.hydra import NanoHydra, SparseScaler
from nanohydra.utils import transform_mfcc, show_mfcc, get_idx_of_class, augment_data_of_class
import gc
import time

start_t = time.time()


# Defines
CLASS_SILENCE       = 11
NUM_SAMPLES_SCALING = 2000
NUM_MFCC_CHANS      = 6
K=8
G=32
D=1
FEATURE_VEC_SZ      = 2*K*G*(D+1)*NUM_MFCC_CHANS
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

# Call garbage collector on useless arrays
del Ytest 
del Yval  
gc.collect()

# Collection of X samples to perform batch statistics
# We use NUM_SAMPLES_SCALING of each of the 12 classes
Xsample_stats = np.empty((NUM_SAMPLES_SCALING*12, 2*K*G*(D+1)*NUM_MFCC_CHANS))


# Build dictionary with locations of samples of each class
class_loc = {i: get_idx_of_class(Ytrain, i) for i in range(12)}

# Isolate the background (class==11) samples
Xbackground = Xtrain[class_loc[11]]

# Augment, MFCC transform and Hydra transform samples of each class, except Background
factors = {k:15, for k in range(10)}
factors[11] = 50   # Unknown word class is much smaller than others 
for cl in range(11):
    Xraw = Xtrain[class_loc[cl]]

    print(f"Augmenting Class {cl} by Factor={15}")
    Xaug   = augment_data_of_class(Xraw, Xbackground, 15)
    X = np.concatenate([Xraw, Xaug])

    print(f"MFCC-Transforming Class {cl}...")
    Xmfcc  = transform_mfcc(X)

    if(cl == 0):
        # Model instantiation, only used to perform the transforms
        model  = NanoHydra(input_length=Xmfcc.shape[2], num_channels=NUM_MFCC_CHANS, k=8, g=32, max_dilations=1, dist="binomial", classifier="Logistic", scaler="Sparse", seed=23981, dtype=np.float32)    

    print(f"Hydra-Transforming Class {cl}...")
    Xtr  = model.forward_batch(Xmfcc.astype(np.float32), 200, do_fit=False, do_scale=False)

    print(Xtr.shape)
    assert Xtr.shape[1] == FEATURE_VEC_SZ, f"The feature vector has unexpected size={Xtr.shape[1]}, expected {FEATURE_VEC_SZ}"
    Xsample_stats[cl*NUM_SAMPLES_SCALING:(cl+1)*NUM_SAMPLES_SCALING,:] = Xtr[np.random.choice(len(Xtr), NUM_SAMPLES_SCALING, replace=False),:]

    model.save_transform(Xtr, f"SpeechCommands_300_cl_{cl}_notscaled", "./work", "train")
    gc.collect()

# MFCC transfrom and Hydra transform Background class
num_examples = len(Xbackground)
print(f"MFCC-Transforming Class 'Background'...")
Xmfcc = transform_mfcc(Xbackground)
print(f"Hydra-Transforming Class 'Background'...")
Xtr  = model.forward_batch(Xmfcc.astype(np.float32), 200, do_fit=False, do_scale=False)
Xsample_stats[11*NUM_SAMPLES_SCALING:12*NUM_SAMPLES_SCALING,:] = Xtr[np.random.choice(num_examples, NUM_SAMPLES_SCALING, replace=False),:]
model.save_transform(Xtr, "SpeechCommands_300_cl_11_notscaled", "./work", "train")

# Save the sample matrix for the scaler fitting (if eventually needed).
model.save_transform(Xsample_stats, "SpeechCommands_300_sample_stats", "./work", "train")

# Fit the scaler
print(f"Fitting the Scaler on a sample of the transformed features.")
scaler = SparseScaler()
scaler.fit(Xsample_stats)

# Scale the features
for cl in range(12):
    print(f"Scaling Class {cl}")
    X = model.load_transform(f"SpeechCommands_300_cl_{cl}_notscaled", "./work", "train")
    X = scaler.transform(X)
    model.save_transform(X, f"SpeechCommands_300_cl_{cl}", "./work", "train")

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