from sktime.datasets import load_UCR_UEA_dataset as load_ucr_ds
from tsc.hydra import Hydra, SparseScaler, hard_counting
from line_profiler import  LineProfiler
import numpy as np

DATASETS = ["ECG5000"]

X  = {'test': {}, 'train': {}}
Xr = {'test': {}, 'train': {}}
y  = {'test': {}, 'train': {}}


for ds in DATASETS:
    
    # Fetch the dataset
    for sp in ["test", "train"]:
        X[sp][ds],y[sp][ds]  = load_ucr_ds(ds, split=sp, return_type="numpy2d")

    X_s = X['test'][ds][:50,:].astype(np.float32)

    model  = Hydra(input_length=X_s.shape[1])    
    scaler = SparseScaler()

    # When profiling runs are neede
    lp         = LineProfiler()
    lp_forward = lp(model.forward)

    # Transform and scale
    print(f"Transforming {X_s.shape[0]} training examples...")
    Xt  = lp_forward(X_s)
    lp.print_stats()