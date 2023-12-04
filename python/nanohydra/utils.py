import numpy as np

from librosa import display
from librosa.feature import mfcc
from scipy.signal.windows import hann
from tqdm import tqdm
import matplotlib.pyplot as plt

def vs_creator(X, Y, class_to_elim, method="onevsall"):
    if(method.lower() == "onevsall"):
        for y in Y:
            # Move class 
            np.putmask(y, y > class_to_elim-1, class_to_elim+1)
            np.putmask(y, y < class_to_elim,   0)
            np.putmask(y, y > 0,               1)
    elif(method.lower() == "allvsall_butone"):
        for y in Y:
            #X[ds] = np.delete(X[ds], y==class_to_elim, axis=0)
            y = np.delete(y, y==class_to_elim)
    else:
        assert False, f"Unknown Method Chosen: '{method}'"

def transform_mfcc(X, fs=16000):
    BATCH_SZ = 500
    N_MFCC = 10
    for i in tqdm(range(0, int(X.shape[0]/BATCH_SZ)+1, 1)):
        _X = mfcc(y=X[i*BATCH_SZ: (i+1)*BATCH_SZ,:], sr=fs, n_mfcc=N_MFCC, n_fft=512, hop_length=128, win_length=512, window=hann)
    
        if i == 0:
            X_mfcc = np.empty((X.shape[0], N_MFCC, _X.shape[2]))
        
        X_mfcc[i*BATCH_SZ: (i+1)*BATCH_SZ,:,:] = _X


    # Make sure all sizes are consistent
    assert X.shape[0] == X_mfcc.shape[0], f"The output dataset {X_mfcc.shape[0]} does not have the same number of examples as the original dataset{X.shape[0]}"
    assert X_mfcc.shape[1] == N_MFCC, f"The output dataset has only {X_mfcc.shape[1]} MFCC coefficients, when it should have {N_MFCC}"

    return X_mfcc

def show_mfcc(X_mfcc, classes):
    assert X_mfcc.shape[0] == len(classes), f"X and Y must have the same dimensions"

    fig, ax = plt.subplots(nrows=len(classes), sharex=True, sharey=True)

    for i in range(len(classes)):
        img = display.specshow(X_mfcc[i], x_axis='time', ax=ax[i])
        ax[i].set(title=f"Class {classes[i]}")
        fig.colorbar(img, ax=[ax[i]])
    plt.show()
