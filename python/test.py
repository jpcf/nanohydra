from sktime.datasets                     import load_UCR_UEA_dataset as load_ucr_ds
import numpy as np
import sys
import time
from nanohydra.hydra import NanoHydra

DATASETS    = ["ECG5000"]
BATCH_TRAIN = True

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

        print(f"Type of X: {X['train'][ds].dtype}")

        # The split argument splits into the opposite fold. Therefore, we here cross them back
        # together into the correct one.
        Xtrain = X['test'][ds][:Ns,:].astype(np.float32)
        Xtest  = X['train'][ds][:Ns,:].astype(np.float32)
        Ytrain = y['test'][ds][:Ns].astype(np.float32)
        Ytest  = y['train'][ds][:Ns].astype(np.float32)
        print(np.unique(Ytrain))
        print(f"Training fold: {Xtrain.shape}")
        print(f"Testing  fold: {Xtest.shape}")

        input_length = Xtrain.shape[1]

        # Initialize the kernel transformer, scaler and classifier
        model  = NanoHydra(input_length=input_length, k=8, g=8, max_dilations=8, dist="binomial", classifier="Logistic", scaler="Sparse", seed=109)    

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
            Xt = model.load_transform(ds, "./work", "train") 
            if(Xt is None):
                Xt  = model.forward_batch(Xtrain, 100, do_fit=False)
                model.save_transform(Xt, ds, "./work", "train")
            else:
                print("Using cached transform...")
            model.fit_classifier(Xt, Ytrain)

        # Test the classifier
        print(f"Transforming Test Fold...")
        Xt = model.load_transform(ds, "./work", "test") 
        if(Xt is None):
            Xt  = model.forward_batch(Xtest, 100, do_fit=False)
            model.save_transform(Xt, ds, "./work", "test")
        else:
            print("Using cached transform...")
        Ypred = model.predict_batch(Xt, 100)
        print(f"Ypred shape: {Ypred.shape}")
        score_man = model.score_manual(Ypred, Ytest, "subset")
        score = model.score(Xt, Ytest)
        print(f"Score (Aut) for '{ds}': {100*score:0.02f} %") 	
        print(f"Score (Man) for '{ds}': {100*score:0.02f} %") 	
        #print(model.cfg.get_classf().coef_)
        #print(np.array(model.cfg.get_classf().coef_).shape)
    print(f"Execution of {Ns} examples took {time.perf_counter()-start} seconds")
