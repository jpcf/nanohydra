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
MAX_DILATIONS = 5
CONV_FRAC_BITS = 8

def check_rck_output(Y, Yc):
    for s in range(Y.shape[0]):
        Y_ex  = Y[s]
        Yc_ex = Yc[s]

        # First, check both have the same length
        if(len(Y_ex) != len(Yc_ex)):
            print(f"ERROR: Feature Vector Length -- Python Model: {len(Y_ex)} vs C Model: {len(Yc_ex)}")
        else:
            print(f"PASS: Feature Vector Length -- Python Model: {len(Y_ex)} vs C Model: {len(Yc_ex)}")

        # Check for errors
        err  = Y_ex != Yc_ex
        nerr = np.sum(err)

        if(nerr):
            print(f"ERROR: Number of errors={nerr}")
            print(f"Positions: {np.arange(len(Y_ex))[err]}")
        else:
            print(f"PASS: No Errors!")

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
    f.write(",".join([str(x) for x in X[0][0]])+"\n")

# Dump Convolution weights to file, and produce the expected output
W = model.dump_weights()
Y = model.forward(X, CONV_FRAC_BITS).astype(np.int16)

with open("dist/weights.txt", "w") as f:
    for wline in W:
        f.write(",".join([str(w) for w in wline])+"\n")

# Run C model, which will dump the output feature vector 
os.system("./dist/forward_equivalence_check")

# Read the output feature vector produced by the C model
Yc = []
with open("dist/output.txt", "r") as f:
    for line in f.readlines():
        Yc.append(np.array(line.split(",")[:-1]).astype(np.int16))

print(Y[0,-20:])
print(Yc[0][-20:])

check_rck_output(Y, Yc)

