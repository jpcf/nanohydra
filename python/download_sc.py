import tensorflow_datasets as tfds
import numpy as np
from nanohydra.hydra import NanoHydra
from nanohydra.utils import transform_mfcc, show_mfcc
import gc

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

del Ytrain
del Ytest 
del Yval  
del tfds
gc.collect()
input_length = Xtrain.shape[1]

print(f"MFCC-Transforming Training Set...")
Xtrain_mfcc = transform_mfcc(Xtrain)
del Xtrain
gc.collect()
print(f"MFCC-Transforming Validation Set...")
Xval_mfcc = transform_mfcc(Xval)
del Xval
gc.collect()
print(f"MFCC-Transforming Testing Set...")
Xtest_mfcc = transform_mfcc(Xtest)
del Xtest
gc.collect()

#print(Xtrain_mfcc.dtype)
#print(Ytrain[0:20])
#show_mfcc(Xtrain_mfcc[0:10,:,:], Ytrain[0:10])
#show_mfcc(Xtrain_mfcc[10:20,:,:], Ytrain[10:20])

input_len    = Xtrain_mfcc.shape[2]
num_channels = Xtrain_mfcc.shape[1]

print(f"Input Len: {input_len} and Num Channels: {num_channels}")

model  = NanoHydra(input_length=input_len, num_channels=num_channels, k=8, g=32, max_dilations=1, dist="binomial", classifier="Logistic", scaler="Sparse", seed=23981, dtype=np.float32)    

print(f"Hydra-Transforming Train Fold...")
Xtr  = model.forward_batch(Xtrain_mfcc.astype(np.float32), 200, do_fit=True)
del Xtrain_mfcc
gc.collect()
model.save_transform(Xtr, "SpeechCommands_300", "./work", "train")
del Xtr
gc.collect()

print(f"Transforming Test Fold...")
Xts  = model.forward_batch(Xtest_mfcc.astype(np.float32), 200, do_fit=False)
del Xtest_mfcc
gc.collect()
model.save_transform(Xts, "SpeechCommands_300", "./work", "test")
del Xts
gc.collect()

print(f"Transforming Val Fold...")
Xva  = model.forward_batch(Xval_mfcc.astype(np.float32), 200, do_fit=False)
del Xval_mfcc
gc.collect()
model.save_transform(Xva, "SpeechCommands_300", "./work", "val")
del Xva
gc.collect()