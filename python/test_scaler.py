from nanohydra.hydra import SparseScaler
import numpy as np

NUM_EXAMPLES = 5
NUM_DIMS     = 10

X = np.random.uniform(-100, 100, (NUM_EXAMPLES, NUM_DIMS))
X[0,5] = 0
X[0,7] = 0
X[4,5] = 0
X[4,7] = 0
print(X)
scaler = SparseScaler()
scaler.fit(X)
Xs  = scaler.transform(X)
print(Xs)