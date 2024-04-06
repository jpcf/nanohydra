import sys, os
import numpy as np
# Adding Hydra Model Path
sys.path.append('/home/josefonseca/Documents/embrocket/python/')
from nanohydra.hydra import NanoHydra 

# Test Data and Model Parameters
FEAT_VEC_LEN = 128
NUM_CLASSES  = 5

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


# Randomize input data, and dump it to file
rng = np.random.default_rng(seed=23123)
X = rng.integers(low=-2**9, high=2**9, size=(FEAT_VEC_LEN)).astype(np.int16)

with open("dist/input_featvec.txt", "w") as f:
    f.write(",".join([str(x) for x in X])+"\n")

# Dump Convolution weights to file, and produce the expected output
W = rng.integers(low=-2**4, high=2**4, size=(NUM_CLASSES, FEAT_VEC_LEN)).astype(np.int16)
b = rng.integers(low=-100, high=100, size=(NUM_CLASSES)).astype(np.int16)
Y = np.dot(W,X) + b
print(Y)

with open("dist/weights_classf.txt", "w") as f:
    for wline in W:
        f.write(",".join([str(w) for w in wline])+",\n")

with open("dist/bias_classf.txt", "w") as f:
    f.write(",".join([str(bv) for bv in b])+",\n")

# Run C model, which will dump the output feature vector 
os.system("./dist/classf_equivalence_check")
#
# Read the output feature vector produced by the C model
Yc = []
with open("./dist/output.txt", "r") as f:
    for line in f.readlines():
        Yc.append(np.array(line.split(",")[:-1]).astype(np.int16))

print(Yc)
check_rck_output(Y.reshape((1,NUM_CLASSES)), Yc)

