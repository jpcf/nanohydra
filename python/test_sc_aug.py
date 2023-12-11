from sktime.datasets                     import load_UCR_UEA_dataset as load_ucr_ds
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
from sklearn.model_selection import PredefinedSplit
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import sys
import time
import pickle
from tqdm import tqdm
import gc

from datetime import datetime as dt
from nanohydra.hydra import NanoHydra
from nanohydra.utils import vs_creator

BATCH_TRAIN = True
BATCH_SIZE  = 32

if __name__ == "__main__":

    start = time.perf_counter()

    # Initialize the kernel transformer, scaler and classifier
    model  = NanoHydra(input_length=100, num_channels=8, k=8, g=32, max_dilations=1, dist="normal", classifier="Logistic", scaler="Sparse", seed=1002)    

    # For debugging, visualize the feature image
    Ximg = np.empty((12000, 8192))

    for cl in tqdm(range(12)):
        Xcl = model.load_transform(f"SpeechCommands_300_cl_{cl}", "./work", "train")
        if(cl == 0):
            Xtrain = Xcl
            Ytrain = cl * np.ones(len(Xcl))
        else:
            Xtrain = np.concatenate([Xtrain, Xcl])
            Ytrain = np.concatenate([Ytrain, cl * np.ones(len(Xcl))])
        Ximg[cl*1000:(cl+1)*1000, :] = Xtrain[:1000,:]

    print(f"Shape of Xtrain: {Xtrain.shape}")
    print(f"Shape of Ytrain: {Ytrain.shape}")

    print(np.min(Ximg))
    print(np.max(Ximg))

    plt.figure(1)
    ax = plt.subplot()
    pos = ax.imshow(Ximg)
    plt.colorbar(pos, ax=ax)
    for i in range(12):
        ax.axhline(i*1000, color='r', linestyle='-')
    plt.title(f"Transformed Training Set")
    plt.show()

    # Initialize the kernel transformer, scaler and classifier
    model  = NanoHydra(input_length=Xtrain.shape[1], num_channels=8, k=8, g=32, max_dilations=1, dist="normal", classifier="Logistic", scaler="Sparse", seed=1002)    

    # Prepare the Train+Val split
    #val_split_idx = [-1]*len(Xtrain) + [0]*len(Xval)
    #X = np.concatenate((Xtrain, Xval), axis=0)
    #Y = np.concatenate((Ytrain, Yval), axis=0)
    #val_split = PredefinedSplit(test_fold=val_split_idx)
    #print(f"Shape of Train+Val Fold: X={X.shape}, Y={Y.shape}")
    #classifier_args={'cv': val_split}

    # Train the classifier
    #Xval = model.load_transform(f"SpeechCommands_300", "./work", "val")
    (__, __), (__, Ytest), (__, Yval) = tfds.as_numpy(tfds.load('speech_commands', split=['train', 'test', 'validation'], batch_size=-1, as_supervised=True))
    #model.fit_tf_classifier(Xtrain, Ytrain, Xval, Yval)

    # Add validation set
    #Xval = model.load_transform(f"SpeechCommands_300", "./work", "val")
    #Xtrain = np.concatenate([Xtrain, Xval])
    #Ytrain = np.concatenate([Ytrain, Yval])

    model.fit_classifier(Xtrain, Ytrain)
    del Xtrain
    del Ytrain
    gc.collect()


    # Load the Test set
    print("Loading the test set")
    Xtest = model.load_transform(f"SpeechCommands_300", "./work", "test")

    # Perform Predictions  
    #Ypred = model.predict_tf(Xtest)
    Ypred = model.predict_batch(Xtest, 256)
    
    # Score the classifier
    score_man = model.score_manual(Ypred, Ytest, "subset")
    print(f"Score    for 'Speech Commands v0.0.3': {100*score_man:0.02f} %") 	

    cm = confusion_matrix(Ytest, Ypred, labels=model.cfg.get_classf().classes_)
    # Show accuracy instead of abs count of samples
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cmd = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.cfg.get_classf().classes_)
    cmd.plot()
    plt.show()

    score_f1  = f1_score(Ytest, Ypred, average='weighted')
    print(f"F1-Score for 'Speech Commands v0.0.3': {score_f1} ") 	

    with open(f"exec_{int(time.time())}.txt", "w") as f:
        f.write(f"[{dt.now()}] Score for 'Speech Commands v0.0.3': {100*score_man:0.02f} %")

    with open(f"model_checkpoint_{int(time.time())}.bin", "wb") as f:
        pickle.dump(model.cfg.get_classf(), f) 

    print(f"Execution of examples took {time.perf_counter()-start} seconds")
