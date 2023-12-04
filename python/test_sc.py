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

from datetime import datetime as dt
from nanohydra.hydra import NanoHydra
from nanohydra.utils import vs_creator

BATCH_TRAIN = True
BATCH_SIZE  = 32

if __name__ == "__main__":

    num_ex = int(sys.argv[1])

    start = time.perf_counter()


    (Xtrain, Ytrain), (Xtest, Ytest), (Xval, Yval) = tfds.as_numpy(tfds.load('speech_commands', split=['train', 'test', 'validation'], batch_size=-1, as_supervised=True))

    Ytrain = Ytrain.astype(np.int8)
    Ytest  = Ytest.astype(np.int8)
    Yval   = Yval.astype(np.int8)

    # Initialize the kernel transformer, scaler and classifier
    model  = NanoHydra(input_length=Xtrain.shape[1], k=8, g=64, max_dilations=10, dist="binomial", classifier="Logistic", scaler="Sparse", seed=23981, classifier_args={'cv': None})    

    # Load the Dataset
    print(f"Loading the dataset")
    Xtrain = model.load_transform("SpeechCommands_300", "./work", "train")[:,:5000]
    Xval = model.load_transform("SpeechCommands_300", "./work", "val")[:,:5000]
    Xtest = model.load_transform("SpeechCommands_300", "./work", "test")[:,:5000]

    print(f"Ytrain={Ytrain[0]}")
    for i in range(30):
       print(f"test[{i}]={Yval[i]}")

    plt.figure()
    plt.plot(Xtrain[6], label="Xtrain")
    plt.plot(Xval[22],  label="Xtest")
    plt.legend()
    plt.show()

    print(f"Shape of Training   Fold: {Xtrain.shape}")
    print(f"Shape of Validation Fold: {Xval.shape}")
    print(f"Shape of Testing    Fold: {Xtest.shape}")
    print(f"Shape of Training Y Fold: {Ytrain.shape}")
    print(f"Shape of Validatio Y Fold: {Yval.shape}")
    #print(f"Class labels : {np.unique(Ytrain)}")

    # Prepare the Train+Val split
    #val_split_idx = [-1]*len(Xtrain) + [0]*len(Xval)
    #X = np.concatenate((Xtrain, Xval), axis=0)
    #Y = np.concatenate((Ytrain, Yval), axis=0)
    #val_split = PredefinedSplit(test_fold=val_split_idx)
    #print(f"Shape of Train+Val Fold: X={X.shape}, Y={Y.shape}")

    # Initialize the kernel transformer, scaler and classifier
    model  = NanoHydra(input_length=Xtrain.shape[1], k=8, g=64, max_dilations=10, dist="binomial", classifier="Logistic", scaler="Sparse", seed=23981)    

    # Train the classifier
    #model.fit_tf_classifier(Xtrain, Ytrain, Xval, Yval)
    model.fit_classifier(Xtrain, Ytrain)

    # Perform Predictions  
    #Ypred = model.predict_tf(Xtest)
    Ypred = model.predict(Xtest)
    
    # Score the classifier
    score_man = model.score_manual(Ypred, Ytest, "prob")
    print(f"Score    for 'Speech Commands v0.0.3': {100*score_man:0.02f} %") 	

    cm = confusion_matrix(Ytest, Ypred, labels=model.cfg.get_classf().classes_)
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
