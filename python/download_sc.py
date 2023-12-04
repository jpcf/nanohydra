import tensorflow_datasets as tfds
import numpy as np
from nanohydra.hydra import NanoHydra

(Xtrain, Ytrain), (Xtest, Ytest), (Xval, Yval) = tfds.as_numpy(tfds.load('speech_commands', split=['train', 'test', 'validation'], batch_size=-1, as_supervised=True))

print(f"Shape of Xtrain: {Xtrain.shape}")
print(f"Shape of Ytrain: {Ytrain.shape}")
print(f"Shape of Xtest:  {Xtest.shape}")
print(f"Shape of Ytest:  {Ytest.shape}")
print(f"Shape of Xval:   {Xval.shape}")
print(f"Shape of Yval:   {Yval.shape}")

#Xtrain = Xtrain.astype(np.float32)/(2**15)
#Xtest  = Xtest.astype(np.float32)/(2**15)
#Xval   = Xval.astype(np.float32)/(2**15)
Ytrain = Ytrain.astype(np.float32)
Ytest  = Ytest.astype(np.float32)
Yval   = Yval.astype(np.float32)
input_length = Xtrain.shape[1]

# Initialize the kernel transformer, scaler and classifier
model  = NanoHydra(input_length=input_length, k=8, g=64, max_dilations=10, dist="binomial", classifier="Logistic", scaler="Sparse", seed=23981)    

# Transforming Train Fold
print(f"Transforming Train Fold...")
Xtr  = model.forward_batch(Xtrain, 250, do_fit=True)
model.save_transform(Xtr, "SpeechCommands_300", "./work", "train")
print(f"Transforming Test Fold...")
Xts  = model.forward_batch(Xtest, 250, do_fit=False)
model.save_transform(Xts, "SpeechCommands_300", "./work", "test")
print(f"Transforming Val Fold...")
Xva  = model.forward_batch(Xval, 250, do_fit=False)
model.save_transform(Xva, "SpeechCommands_300", "./work", "val")