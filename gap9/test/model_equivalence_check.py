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
MAX_DILATIONS = 3
NUM_DIFFS     = 1
DO_QUANTIZE   = True

DIST_FOLDER = "./dist/"
SPLITS = ["train"] #"test"]
RUN_ON_TARGET_GAP9 = True

def check_rck_output(Y, Yc):

    nerr_ex = 0

    for s in range(len(Yc)):
        Y_ex  = Y[s]
        Yc_ex = Yc[s]

        # First, check both have the same length
        if(len(Y_ex) != len(Yc_ex)):
            print(f"ERROR: Feature Vector Length -- Python Model: {len(Y_ex)} vs C Model: {len(Yc_ex)}")
        else:
            pass
            #print(f"PASS: Feature Vector Length -- Python Model: {len(Y_ex)} vs C Model: {len(Yc_ex)}")

        # Check for errors
        err  = Y_ex != Yc_ex
        nerr = np.sum(err)

        if(nerr):
            print(f"ERROR: Number of errors={nerr}")
            print(f"Positions: {np.arange(len(Y_ex))[err]}")
            print(f"Values: {Y_ex[err]} vs {Yc_ex[err]}")
            nerr_ex += 1
        else:
            pass
            #print(f"PASS: No Errors!")

    return nerr_ex    

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

# Dump input vectors
fp = np.memmap(f"dist/input_train.dat", dtype='int16', mode="w+", shape=(Xtrain.shape[0], Xtrain.shape[2]))
fp[:] = Xtrain[:,0,:]
fp.flush()
print(fp)
del fp
fp = np.memmap(f"dist/input_test.dat", dtype='int16', mode="w+", shape=(Xtest.shape[0], Xtest.shape[2]))
fp[:] = Xtest[:,0,:]
fp.flush()
print(fp)
del fp

# Initialize the kernel transformer, scaler and classifier
model  = NanoHydra(input_length=input_length, num_channels=NUM_CHAN, k=K, g=G, max_dilations=MAX_DILATIONS, num_diffs=NUM_DIFFS, dist="binomial", classifier="Logistic", scaler="Sparse", seed=int(time.time()), dtype=np.int16, verbose=False)    

# Transform and scale
print(f"Transforming {Xtrain.shape[0]} training examples...")
Xt = {}
Xt['train']  = model.forward_batch(Xtrain, 500, do_fit=True,  do_scale=True, quantize_scaler=True, frac_bit_shift=accum_bits_shift)
Xt['test']   = model.forward_batch(Xtest,  500, do_fit=False, do_scale=True, quantize_scaler=True, frac_bit_shift=accum_bits_shift)

# Fit
print(f"Fitting {Xtrain.shape[0]} training examples...")
model.fit_classifier(Xt['train'], Ytrain)
model.quantize_classifier(8)

# Dump model params
Wq, bq =model.dump_classifier_weights()
W = model.dump_weights()
model.dump_defines("./include")

fp = np.memmap(f"dist/weights.dat", dtype='int16', mode="w+", shape=(W.shape[0], W.shape[1]))
fp[:] = W[:,:]
fp.flush()
del fp

means = model.cfg.get_scaler().muq
fp = np.memmap(f"dist/means.dat", dtype='int16', mode="w+", shape=(len(means)))
fp[:] = means[:]
fp.flush()
del fp

sigmas = model.cfg.get_scaler().sigmaq
fp = np.memmap(f"dist/stds.dat", dtype='uint8', mode="w+", shape=(len(sigmas)))
fp[:] = sigmas[:]
fp.flush()
del fp

fp = np.memmap(f"dist/weights_classf.dat", dtype='int8', mode="w+", shape=(Wq.shape[0], Wq.shape[1]))
fp[:] = Wq[:,:]
fp.flush()
del fp

fp = np.memmap(f"dist/weights_bias.dat", dtype='int8', mode="w+", shape=(bq.shape[0]))
fp[:] = bq[:]
fp.flush()
del fp

# Compile C model with generated .h file defines
os.system(f"make model_equivalence_check")

# Run C model, which will dump the output feature vector 
for split in SPLITS:
    model.predict_quantized(Xt[split])
    
    t_start = time.perf_counter()
    if(RUN_ON_TARGET_GAP9):
        os.system(f"cd ./gap_apps/first_app_full_omp && cmake --build build --target run && cp build/output.dat ../../dist && cd ../..")
    else:
        os.system(f"./dist/model_equivalence_check ./dist/input_{split}.dat {len(Xt[split])}")
    t_end= time.perf_counter()

    # Read the output feature vector produced by the C model
    Yc = np.memmap(f"./dist/output.dat", dtype='int32', mode="r", shape=(len(Xt[split]), 5))
    #Yc = np.memmap(f"./dist/output.dat", dtype='int32', mode="r", shape=(50, 5))

    nerr = check_rck_output(model.activ, Yc)

    print(f"Tested {len(model.activ)} vectors. Found {nerr} wrong outputs vs Model. Duration: {t_end-t_start: .6f}")
    
    # Y has one-based indexing, so we subtract 1
    if(split == "test"):
        Yground_truth = Ytest.astype(np.uint8)-1
    else:
        Yground_truth = Ytrain.astype(np.uint8)-1

    #acc = model.score_manual(Yc, Yground_truth[:50], method='prob')
    acc = model.score_manual(Yc, Yground_truth, method='prob')
    print(f"Prediction accuracy for '{split}': {acc*100:.2f} %")

