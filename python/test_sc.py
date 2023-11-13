from sktime.datasets                     import load_UCR_UEA_dataset as load_ucr_ds
import tensorflow_datasets as tfds
import numpy as np
import sys
import time
from nanohydra.hydra import NanoHydra

BATCH_TRAIN = True

if __name__ == "__main__":

    num_ex = int(sys.argv[1])

    start = time.perf_counter()


    (Xtrain, Ytrain), (Xtest, Ytest), (Xval, Yval) = tfds.as_numpy(tfds.load('speech_commands', split=['train', 'test', 'validation'], batch_size=-1, as_supervised=True))

    # The split argument splits into the opposite fold. Therefore, we here cross them back
    # together into the correct one.

    Xtrain = Xtrain.astype(np.float32)
    Xtest  = Xtest.astype(np.float32)
    Ytrain = Ytrain.astype(np.float32)
    Ytest  = Ytest.astype(np.float32)
    print(np.unique(Ytrain))
    print(f"Training fold: {Xtrain.shape}")
    print(f"Testing  fold: {Xtest.shape}")
    input_length = Xtrain.shape[1]

    # Initialize the kernel transformer, scaler and classifier
    model  = NanoHydra(input_length=input_length, k=8, g=32, dist="binomial", classifier="Logistic", scaler="Sparse")    

    # Transform and scale
    print(f"Transforming {Xtrain.shape[0]} training examples...")
    if(not BATCH_TRAIN):
        Xt  = model.forward(Xtrain)
        print(f"Transform size: {Xt.shape}")
        model.fit_scaler(Xt)
        Xts = model.forward_scaler(Xt)
        print(f"Scaled-Transform size: {Xts.shape}")

        # Fit the transformed features
        model.fit_classifier(Xts, Ytrain)
        print(f"Fitting the classifier")
    else:
        Xt  = model.forward_batch(Xtrain, 256, do_fit=False)
        model.fit_classifier(Xt, Ytrain)

    # Test the classifier
    print(f"Transforming Test Fold...")
    Xr = model.forward_batch(Xtest, 256, do_fit=False)
    Ypred = model.predict_batch(Xr, 256)
    print(f"Ypred shape: {Ypred.shape}")
    score_man = model.score_manual(Ypred, Ytest, "subset")
    print(f"Score for 'Speech Commands v0.0.3': {100*score_man:0.02f} %") 	

    print(f"Execution of examples took {time.perf_counter()-start} seconds")
