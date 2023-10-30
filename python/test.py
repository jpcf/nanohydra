from sktime.datasets                     import load_UCR_UEA_dataset as load_ucr_ds
from sktime.transformations.panel.rocket import MiniRocket
from sklearn.linear_model                 import RidgeClassifierCV
from sklearn.preprocessing                 import StandardScaler
import numpy as np
import sys
import time
from tsc.hydra import Hydra, SparseScaler

DATASETS = ["ECG5000"]

X  = {'test': {}, 'train': {}}
y  = {'test': {}, 'train': {}}


if __name__ == "__main__":

    num_ex = int(sys.argv[1])

    start = time.perf_counter()

    for ds in DATASETS:
        
        # Fetch the dataset
        for sp in ["test", "train"]:
            X[sp][ds],y[sp][ds]  = load_ucr_ds(ds, split=sp, return_type="numpy2d")

        Ns = min(X['test'][ds].shape[0], num_ex)

        # The split argument splits into the opposite fold. Therefore, we here cross them back
        # together into the correct one.
        Xtrain = X['test'][ds][:Ns,:].astype(np.float32)
        Xtest  = X['train'][ds][:Ns,:].astype(np.float32)
        Ytrain = y['test'][ds][:Ns].astype(np.float32)
        Ytest  = y['train'][ds][:Ns].astype(np.float32)
        print(Xtrain)
        print(f"Training fold: {Xtrain.shape}")
        print(f"Testing  fold: {Xtest.shape}")

        input_length = Xtrain.shape[1]

        # Initialize the kernel transformer, scaler and classifier
        cl = RidgeClassifierCV(alphas=np.logspace(-3,3,10))
        model  = Hydra(input_length=input_length)    
        scaler = SparseScaler()

        # Transform and scale
        print(f"Transforming {Xtrain.shape[0]} training examples...")
        Xt  = model.forward(Xtrain)
        print(f"Transform size: {Xt.shape}")
        Xts = scaler.fit_transform(Xt) 
        print(f"Scaled-Transform size: {Xt.shape}")

        # Fit the transformed features
        cl.fit(Xts, Ytrain)
        print(f"Fitting the classifier")

        # Test the classifier
        print(f"Transforming Test Fold...")
        Xr = model.forward(Xtest)
        scaler = SparseScaler()
        print(f"Scaling Test Fold...")
        Xr = scaler.fit_transform(Xr)
        print(f"Scoring Test Fold...")
        score = cl.score(Xr, Ytest)
        print(f"Score for '{ds}': {100*score:0.02f} %") 	

    print(f"Execution of {Ns} examples took {time.perf_counter()-start} seconds")