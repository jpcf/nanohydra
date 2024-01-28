import sys, os
import numpy as np
import time
from sktime.datasets                     import load_UCR_UEA_dataset as load_ucr_ds
import matplotlib.pyplot as plt

# Adding Hydra Model Path
sys.path.append('/home/josefonseca/Documents/embrocket/python/')
from nanohydra.hydra import NanoHydra 
from mlutils.quantizer import LayerQuantizer

# Test Data and Model Parameters
INPUT_LEN = 140
NUM_CHAN  = 1
K=8
G=16
MAX_DILATIONS = 5
DO_QUANTIZE=True

def check_rck_output(Y, Yc):
    for s in range(len(Yc)):
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
            print(f"Values: {Y_ex[err]} vs {Yc_ex[err]}")
        else:
            print(f"PASS: No Errors!")

# Fetch the dataset
X  = {'test': {}, 'train': {}}
y  = {'test': {}, 'train': {}}

ds = "ECG5000"
for sp in ["test", "train"]:
    X[sp][ds],y[sp][ds]  = load_ucr_ds(ds, split=sp, return_type="numpy3d")

# The split argument splits into the opposite fold. Therefore, we here cross them back
# together into the correct one.
Xtrain = X['train'][ds].astype(np.float32)
Xtest  = X['test'][ds].astype(np.float32)
Ytrain = y['train'][ds]
Ytest  = y['test'][ds]

input_length = Xtrain.shape[2]

lq_input = LayerQuantizer(Xtrain, 16)
Xtrain = lq_input.quantize(Xtrain)
Xtest  = lq_input.quantize(Xtest)
print(f"Input Vector Quant.: {lq_input}")
accum_bits_shift = lq_input.get_fract_bits()-1

# Initialize the kernel transformer, scaler and classifier
model  = NanoHydra(input_length=input_length, num_channels=NUM_CHAN, k=K, g=G, max_dilations=MAX_DILATIONS, dist="binomial", classifier="Logistic", scaler="Sparse", seed=int(time.time()), dtype=np.int16, verbose=False)    

# Transform and scale
print(f"Transforming {Xtrain.shape[0]} training examples...")
Xt  = model.forward_batch(Xtrain, 500, do_fit=True, do_scale=True, quantize_scaler=True, frac_bit_shift=accum_bits_shift)

# Fit
print(f"Fitting {Xtrain.shape[0]} training examples...")
model.fit_classifier(Xt, Ytrain)
model.quantize_classifier(8)

# Classify (model)
model.predict_quantized(Xt)
Y = model.activ

# Dump model params
print(f"Feature Vector Length: {len(Xt[0])}")
Wq, bq =model.dump_classifier_weights()
W = model.dump_weights()

with open("dist/input.txt", "w") as f:
    f.write(",".join([str(x) for x in Xtrain[356][0].astype(np.int16)])+"\n")

with open("dist/weights.txt", "w") as f:
    for wline in W:
        f.write(",".join([str(w) for w in wline])+"\n")

with open("dist/means.txt", "w") as f:
    f.write(",".join([str(x) for x in model.cfg.get_scaler().muq])+"\n")

with open("dist/stds.txt", "w") as f:
    f.write(",".join([str(x) for x in model.cfg.get_scaler().sigmaq])+"\n")

with open("dist/weights_classf.txt", "w") as f:
    for wline in Wq:
        f.write(",".join([str(w) for w in wline])+",\n")

with open("dist/bias_classf.txt", "w") as f:
    f.write(",".join([str(bv) for bv in bq])+",\n")

# Run C model, which will dump the output feature vector 
os.system("./dist/model_equivalence_check")

# Read the output feature vector produced by the C model
Yc = []
with open("dist/output.txt", "r") as f:
    for line in f.readlines():
        Yc.append(np.array(line.split(",")[:-1]).astype(np.int16))
print(Y[356])
print(Yc[0])
check_rck_output([Y[356]], Yc)

