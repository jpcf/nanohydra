import sys
import numpy as np
# Adding Hydra Model Path
sys.path.append('/home/josefonseca/Documents/embrocket/python/')
from nanohydra.hydra import NanoHydra 

# Test Data and Model Parameters
INPUT_LEN = 140
NUM_CHAN  = 1
K=8
G=16
MAX_DILATIONS = 2

# Import Model
model = NanoHydra(input_length=INPUT_LEN, 
                  num_channels=NUM_CHAN, 
                  k=K, g=G, 
                  max_dilations=MAX_DILATIONS, 
                  dist="binomial", 
                  classifier="Logistic", 
                  scaler="Sparse", 
                  seed=3213)

# Randomize input data
rng = np.random.default_rng(seed=42)
X = rng.integers(low=-2**10, high=2**10, size=(1, INPUT_LEN)).astype(np.int16)

# Convolution weights
W = model.dump_weights()

# ToDo
