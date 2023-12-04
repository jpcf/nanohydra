from sktime.datasets                     import load_UCR_UEA_dataset as load_ucr_ds
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import sys
import time
from nanohydra.hydra import NanoHydra

DATASETS        = ["Phoneme"]
SHOW_HISTOGRAMS = True
SHOW_CONFMATRIX = True
SHOW_EXAMPLES   = False
SHOW_TRANSFORM  = True
SHOW_COLI_MEAS  = False
USE_CACHED      = False
SHOW_ALPHAS_RR  = False
DO_PCA          = False

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
        Ytrain = y['test'][ds][:Ns]
        Ytest  = y['train'][ds][:Ns]
        print(np.unique(Ytrain))
        print(f"Training fold: {Xtrain.shape}")
        print(f"Testing  fold: {Xtest.shape}")

        input_length = Xtrain.shape[1]

        # Display Histograms
        if(SHOW_HISTOGRAMS):
            plt.figure(1)
            plt.hist(Ytrain)
            plt.title(f"Dataset '{ds}' Training Classes Histogram.")
            plt.figure(2)
            plt.hist(Ytest)
            plt.title(f"Dataset '{ds}' Testing Classes Histogram.")

            print(f"Sample   Size: {len(Xtrain[0,:])}")
            print(f"Training Size: {Ytrain.shape[0]}")
            print(f"Testing  Size: {Ytest.shape[0]}")
            print(f"Num   Classes: {len(np.unique(Ytest))}")

        if(SHOW_EXAMPLES):
            plt.figure(3)
            plt.plot(Xtrain[0])
            #for c in np.unique(Ytest):
            #    idx = 0
            #    while(Ytrain[idx] != c):
            #        idx += 1
            #    plt.plot(Xtrain[idx], label=f"Class {c}")
            plt.legend()
                

        # Initialize the kernel transformer, scaler and classifier
        model  = NanoHydra(input_length=input_length, k=8, g=64, max_dilations=10, dist="binomial", classifier="Logistic", scaler="Sparse", seed=int(time.time()), dtype=np.float32, verbose=False)    

        # Transform and scale
        print(f"Transforming {Xtrain.shape[0]} training examples...")
        Xt = model.load_transform(ds, "./work", "train") 
        if(Xt is None or not USE_CACHED):
            Xt  = model.forward_batch(Xtrain, Xtrain.shape[1], do_fit=True)
            model.save_transform(Xt, ds, "./work", "train")
        else:
            print("Using cached transform...")


        if(SHOW_TRANSFORM):
            plt.figure(5)
            plt.imshow(Xt, vmin=np.min(Xt), vmax=np.max(Xt))
            plt.title(f"Transformed Training Set (Full, Not Shuffled)")

            # Display Training 10 Examples per class
            idxs = Ytrain.astype(np.float32).argsort()
            Xt_sorted = Xt[idxs]

            change_idxs,  = np.nonzero(np.diff(sorted(Ytrain.astype(np.float32))))
            print(change_idxs)

            plt.figure(9)
            ax = plt.subplot()
            ax.imshow(Xt_sorted)
            for i,y in enumerate(change_idxs):
                ax.text(Xt.shape[1], y, f"Class {i+1}")
                ax.axhline(y, color='r', linestyle='-')
            plt.title(f"Transformed Training Set (Ordered by classes)")

        if(SHOW_COLI_MEAS):
            print(f"Condition Number for the Design Matrix (Transf Features): k={np.linalg.cond(np.hstack([np.ones((Xt.shape[0],1)), Xt]), p=2)}")

        if(DO_PCA):
            prev_dim = Xt.shape[1]
            PCATransf = PCA(n_components='mle')
            PCATransf.fit(Xt)
            Xt = PCATransf.transform(Xt)
            assert Xt.shape[1] < prev_dim, f"PCA did not reduce dimensionality ({Xt.shape[1]} vs. {prev_dim})"
            print(f"New Dimensionality = {Xt.shape[1]} (Reduc. Ratio = { Xt.shape[1]/prev_dim:.2f}")

        # Fit the classifier
        model.fit_classifier(Xt, Ytrain)

        # Test the classifier
        print(f"Transforming Test Fold...")
        Xt = model.load_transform(ds, "./work", "test") 
        if(Xt is None) or not USE_CACHED:
            Xt  = model.forward_batch(Xtest, 100, do_fit=False)
            model.save_transform(Xt, ds, "./work", "test")
        else:
            print("Using cached transform...")

        if(DO_PCA):
            Xt = PCATransf.transform(Xt)
        Ypred = model.predict_batch(Xt, 100)
        print(f"Ypred shape: {Ypred.shape}")
        score_man = model.score_manual(Ypred, Ytest, "subset")
        score = model.score(Xt, Ytest)
        print(f"Score (Aut) for '{ds}': {100*score:0.02f} %") 	
        print(f"Score (Man) for '{ds}': {100*score:0.02f} %") 	

        # Display Confusion Matrix
        if(SHOW_CONFMATRIX):
            cm = confusion_matrix(Ytest, Ypred, labels=model.cfg.get_classf().classes_)
            cmd = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.cfg.get_classf().classes_)
            cmd.plot()

        # Display Alphas
        if(SHOW_ALPHAS_RR):
            #plt.figure(4)
            #plt.plot(np.logspace(-6,4,20), model.cfg.get_classf().attrs.cv_values_)
            print(f"Best alpha: {model.cfg.get_classf().alpha_}")
            
        if(SHOW_CONFMATRIX or SHOW_HISTOGRAMS or SHOW_EXAMPLES):
            plt.show()

    print(f"Execution of {Ns} examples took {time.perf_counter()-start} seconds")
