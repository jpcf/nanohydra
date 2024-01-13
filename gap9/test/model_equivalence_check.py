import sys, os
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

# Randomize input data, and dump it to file
rng = np.random.default_rng(seed=42)
X = rng.integers(low=-2**10, high=2**10, size=(1,1,INPUT_LEN)).astype(np.int16)

with open("dist/input.txt", "w") as f:
    f.write(",".join([str(x) for x in X[0]])+"\n")

# Dump Convolution weights to file
W = model.dump_weights()
Y = model.forward(X).astype(np.int16)

with open("dist/weights.txt", "w") as f:
    for wline in W:
        f.write(",".join([str(w) for w in wline])+"\n")
        
os.system("./dist/forward_equivalence_check")

print(f"Expected outputs: {Y[0][:16]}")