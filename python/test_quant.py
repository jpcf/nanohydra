from sktime.datasets                     import load_UCR_UEA_dataset as load_ucr_ds
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import sys
import time
from nanohydra.hydra import NanoHydra
from mlutils.quantizer import LayerQuantizer

DATASETS        = ["ECG5000"]
DO_QUANTIZE     = True

X  = {'test': {}, 'train': {}}
y  = {'test': {}, 'train': {}}


if __name__ == "__main__":

    num_ex = int(sys.argv[1])

    start = time.perf_counter()

    for ds in DATASETS:
        
        # Fetch the dataset
        for sp in ["test", "train"]:
            X[sp][ds],y[sp][ds]  = load_ucr_ds(ds, split=sp, return_type="numpy3d")

        Ns = min(X['test'][ds].shape[0], num_ex)

        # The split argument splits into the opposite fold. Therefore, we here cross them back
        # together into the correct one.
        Xtrain = X['train'][ds][:Ns,:].astype(np.float32)
        Xtest  = X['test'][ds][:Ns,:].astype(np.float32)
        Ytrain = y['train'][ds][:Ns]
        Ytest  = y['test'][ds][:Ns]
        print(np.unique(Ytrain))
        print(f"Training fold: {Xtrain.shape}")
        print(f"Testing  fold: {Xtest.shape}")

        input_length = Xtrain.shape[0]

        if(DO_QUANTIZE):                
            lq_input = LayerQuantizer(Xtrain, 16)
            Xtrain = lq_input.quantize(Xtrain)
            Xtest  = lq_input.quantize(Xtest)
            print(f"Input Vector Quant.: {lq_input}")
            accum_bits_shift = lq_input.get_fract_bits()

        # Initialize the kernel transformer, scaler and classifier
        model  = NanoHydra(input_length=input_length, num_channels=1, k=8, g=16, max_dilations=8, dist="binomial", classifier="Logistic", scaler="Sparse", seed=int(time.time()), dtype=np.int16, verbose=False)    

        # Transform and scale
        print(f"Transforming {Xtrain.shape[0]} training examples...")
        Xt  = model.forward_batch(Xtrain, 500, do_fit=True, do_scale=True, quantize_scaler=True, frac_bit_shift=accum_bits_shift)
        
        print(f"Feature Vect Train: {np.min(Xt)} -- {np.max(Xt)}")
        plt.figure(1)
        plt.plot(model.cfg.get_scaler().muq / accum_bits_shift)
        plt.title("Mu")
        
        plt.figure(2)
        plt.plot(model.cfg.get_scaler().sigmaq)
        plt.title("Sigma")
        plt.show()

        # Fit the classifier
        model.fit_classifier(Xt, Ytrain)
        model.quantize_classifier(8)

        # Test the classifier
        print(f"Transforming Test Fold...")
        Xtq  = model.forward_batch(Xtest, 1000, do_scale=True, quantize_scaler=False,  frac_bit_shift=accum_bits_shift)

        print(f"Feature Vect Test: {np.min(Xt)} -- {np.max(Xt)}")
        Ypred = model.predict_batch(Xtq, 100).astype(np.uint8)
        Yquan = model.predict_quantized(Xtq)

        scoreq = model.score_manual(Yquan, Ytest.astype(np.uint8), "subset")
        score  = model.score(Xtq, Ytest)

        print(f"Score (FP)    for '{ds}': {100*score :0.02f} %") 	
        print(f"Score (Quant) for '{ds}': {100*scoreq:0.02f} %") 	

    print(f"Execution of {Ns} examples took {time.perf_counter()-start} seconds")
