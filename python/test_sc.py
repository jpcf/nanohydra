from sktime.datasets                     import load_UCR_UEA_dataset as load_ucr_ds
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import numpy as np
import sys
import time
from datetime import datetime as dt
from nanohydra.hydra import NanoHydra

BATCH_TRAIN = True
BATCH_SIZE  = 32

if __name__ == "__main__":

    num_ex = int(sys.argv[1])

    start = time.perf_counter()


    (Xtrain, Ytrain), (Xtest, Ytest), (Xval, Yval) = tfds.as_numpy(tfds.load('speech_commands', split=['train', 'test', 'validation'], batch_size=-1, as_supervised=True))

    # The split argument splits into the opposite fold. Therefore, we here cross them back
    # together into the correct one.

    Xtrain = Xtrain.astype(np.float32)/(2**15)
    Xtest  = Xtest.astype(np.float32)/(2**15)
    Ytrain = Ytrain.astype(np.int8)
    Ytest  = Ytest.astype(np.int8)

    # Try One vs All training
    #np.putmask(Ytrain, Ytrain > 10,  12)
    #np.putmask(Ytrain, Ytrain < 11,   0)
    #np.putmask(Ytrain, Ytrain > 0,    1)
    #np.putmask(Ytest,   Ytest > 10,  12)
    #np.putmask(Ytest,   Ytest < 11,   0)
    #np.putmask(Ytest,   Ytest > 0,    1)
    #np.putmask(Yval,     Yval > 10,  12)
    #np.putmask(Yval,     Yval < 11,   0)
    #np.putmask(Yval,     Yval > 0,    1)

    #Ytrain = np.hstack([Ytrain, Yval])

    # Plot Histograms for Class Membership of splits
    #plt.figure(1)
    #plt.hist(Ytrain)
    #plt.title("Histogram of Class Distrs in TRAIN")
    #plt.figure(3)
    #plt.hist(Ytest)
    #plt.title("Histogram of Class Distrs in TEST (AFTER)")
    #plt.figure(3)
    #plt.hist(Yval)
    #plt.title("Histogram of Class Distrs in VAL")
    plt.show()

    print(np.unique(Ytrain))
    print(f"Training fold: {Xtrain.shape}")
    print(f"Testing  fold: {Xtest.shape}")
    input_length = Xtrain.shape[1]

    # Initialize the kernel transformer, scaler and classifier
    model  = NanoHydra(input_length=input_length, k=8, g=64, max_dilations=8, dist="binomial", classifier="Logistic", scaler="Sparse", seed=23981)    

    # Transform and scale (load from cached)
    print(f"Transforming Train Fold...")
    Xtrain = model.load_transform("SpeechCommands_300", "./work", "train")
    Xval = model.load_transform("SpeechCommands_300", "./work", "val")

    #Xtrain = np.vstack([Xtrain, Xval])

    print(f"Shape of Training Fold: {Xtrain.shape}")

    # Train the classifier
    model.fit_classifier(Xtrain, Ytrain)

    # Test the classifier
    print(f"Transforming Test Fold...")
    Xtest = model.load_transform("SpeechCommands_300", "./work", "test")
    Ypred = model.predict_batch(Xtest, 256, "predict")
    print(np.array(Ypred).shape)
    
    # Score the classifier
    score_man = model.score_manual(Ypred, Ytest, "subset")
    score_f1  = f1_score(Ytest, Ypred)
    print(f"Score    for 'Speech Commands v0.0.3': {100*score_man:0.02f} %") 	
    print(f"F1-Score for 'Speech Commands v0.0.3': {score_f1} ") 	

    #score = model.score(Ypred, Ytest)
    #print(f"Score for 'Speech Commands v0.0.3': {100*score:0.02f} %") 	
    # Display confusion matrix
    cm = confusion_matrix(Ytest, Ypred, labels=model.cfg.get_classf().classes_)
    cmd = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.cfg.get_classf().classes_)
    cmd.plot()
    plt.show()


    with open(f"exec_{int(time.time())}.txt", "w") as f:
        f.write(f"[{dt.now()}] Score for 'Speech Commands v0.0.3': {100*score_man:0.02f} %")

    print(f"Execution of examples took {time.perf_counter()-start} seconds")
