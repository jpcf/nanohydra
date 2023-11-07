import tensorflow_datasets as tfds

(Xtrain, Ytrain), (Xtest, Ytest), (Xval, Yval) = tfds.as_numpy(tfds.load('speech_commands', split=['train', 'test', 'validation'], batch_size=-1, as_supervised=True))

print(f"Shape of Xtrain: {Xtrain.shape}")
print(f"Shape of Ytrain: {Ytrain.shape}")
print(f"Shape of Xtest:  {Xtest.shape}")
print(f"Shape of Ytest:  {Ytest.shape}")
print(f"Shape of Xval:   {Xval.shape}")
print(f"Shape of Yval:   {Yval.shape}")