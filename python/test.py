from sktime.datasets                     import load_UCR_UEA_dataset as load_ucr_ds
from sktime.transformations.panel.rocket import MiniRocket
from sklearn.linear_model                 import RidgeClassifierCV
from sklearn.preprocessing                 import StandardScaler
import numpy as np
from tsc.hydra import Hydra
import torch

DATASETS = ["ECG5000", "OliveOil"]

X  = {'test': {}, 'train': {}}
Xr = {'test': {}, 'train': {}}
y  = {'test': {}, 'train': {}}

for ds in DATASETS:
    
    # Fetch the dataset
    for sp in ["test", "train"]:
        X[sp][ds],y[sp][ds]  = load_ucr_ds(ds, split=sp, return_type="numpy2d")
        print(X[sp][ds].shape)

    X_s = X['train'][ds][:10,:]
    input_length = X_s.shape[1]

    model = Hydra(input_length=input_length)
    Xt = model.forward(X_s)
    print(f"Transform size: {Xt}")