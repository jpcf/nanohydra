from sktime.datasets                     import load_UCR_UEA_dataset as load_ucr_ds
from sktime.transformations.panel.rocket import MiniRocket
from sklearn.linear_model                 import RidgeClassifierCV
from sklearn.preprocessing                 import StandardScaler
import numpy as np
from tsc.hydra import Hydra, SparseScaler

DATASETS = ["ECG5000", "OliveOil"]

X  = {'test': {}, 'train': {}}
Xr = {'test': {}, 'train': {}}
y  = {'test': {}, 'train': {}}

for ds in DATASETS:
    
    # Fetch the dataset
    for sp in ["test", "train"]:
        X[sp][ds],y[sp][ds]  = load_ucr_ds(ds, split=sp, return_type="numpy2d")
    X_s = X['test'][ds]
    X_s1 = X['train'][ds]
    print(f"Training fold: {X_s.shape}")
    print(f"Testing  fold: {X_s1.shape}")
    input_length = X_s.shape[1]

    # Initialize the kernel transformer, scaler and classifier
    cl = RidgeClassifierCV(alphas=np.logspace(-3,3,10))
    model  = Hydra(input_length=input_length)
    scaler = SparseScaler()

    # Transform and scale
    print(f"Transforming {X_s.shape[0]} training examples...")
    Xt  = model.forward(X_s)
    print(f"Transform size: {Xt.shape}")
    Xts = scaler.fit_transform(Xt) 
    print(f"Scaled-Transform size: {Xt.shape}")

    # Fit the transformed features
    cl.fit(Xts, y["test"][ds])
    print(f"Fitting the classifier")

    # Test the classifier
    print(f"Transforming Test Fold...")
    Xr = model.forward(X["train"][ds])
    scaler = SparseScaler()
    print(f"Scaling Test Fold...")
    Xr = scaler.fit_transform(Xr)
    print(f"Scoring Test Fold...")
    score = cl.score(Xr, y["train"][ds])
    print(f"Score for '{ds}': {100*score:0.02f} %") 	