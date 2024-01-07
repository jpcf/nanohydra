import time
import numpy as np
from sklearn.linear_model             import RidgeClassifierCV, SGDClassifier, LogisticRegressionCV
from sklearn.tree                     import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.neural_network           import MLPClassifier
import h5py
import os
import gc
from tqdm import tqdm

import tensorflow.keras as tf
from   tensorflow.keras           import Sequential, regularizers
from   tensorflow.keras.layers    import Dense, Flatten
from   tensorflow.keras.callbacks import EarlyStopping
from   tensorflow.keras.losses    import SparseCategoricalCrossentropy

from .optimized_fns.conv1d_opt_x_f32_w_f32        import conv1d_opt_x_f32_w_f32
from .optimized_fns.conv1d_opt_x_int16_w_b1       import conv1d_opt_x_int16_w_b1
from .optimized_fns.hard_counting_opt import hard_counting_opt
from .optimized_fns.soft_counting_opt import soft_counting_opt

WORK_FOLDER = "./work/"

class NanoHydraCfg():
    def __init__(self, seed, scalertype, classifiertype, classifier_args=None):
        # Define scaler
        if(scalertype.lower() == "sparse"):
            self.scaler = SparseScaler()
        else:
            print("Unknown Scaler specified")

        # Define Classifier
        if(classifiertype.lower() == "logistic"):
            # Alternative 1: SGD Classifier
            self.classf = SGDClassifier(
                loss='log_loss', 
                alpha=0.0001, 
                penalty='l1', 
                class_weight="balanced", 
                shuffle=True, 
                n_jobs=22, 
                verbose=0, 
                tol=1e-4, 
                learning_rate='adaptive', 
                eta0=1e-3, 
                n_iter_no_change=20
            )
        elif(classifiertype.lower() == "ridge"):
            self.classf = RidgeClassifierCV(alphas=np.logspace(0,1,50), store_cv_values=True)
        elif(classifiertype.lower() == "perceptron"):
            self.classf = SGDClassifier(
                loss='perceptron', 
                alpha=0.001, 
                penalty='elasticnet', 
                class_weight="balanced", 
                shuffle=True, 
                n_jobs=22, 
                verbose=1, 
                learning_rate='adaptive', 
                eta0=1e-3, 
                early_stopping=True, 
                n_iter_no_change=10
            )
        elif(classifiertype.lower() == "tree"):
            self.classf = ExtraTreeClassifier()
        elif(classifiertype.lower() == "nn"):
            self.classf = MLPClassifier(
                hidden_layer_sizes=[50], 
                activation="relu", 
                batch_size=64, 
                learning_rate_init=0.001,
                verbose=1, 
                n_iter_no_change=10, 
                max_iter=500)

        self.seed = seed

    def get_scaler(self):
        return self.scaler

    def get_classf(self):
        return self.classf

    def get_seed(self):
        return self.seed

    def set_seed(self, seed):
        self.seed = seed

class NanoHydra():

    __KERNEL_LEN = 9

    def __init__(self, input_length, num_channels=1,  k = 8, g = 64, max_dilations=8, seed = None, dist = "normal", classifier="Logistic", scaler="Sparse", dtype=np.int16, verbose=True, classifier_args=None):

        self.cfg = NanoHydraCfg(seed= seed, scalertype=scaler, classifiertype=classifier, classifier_args=classifier_args)
        self.dist = dist
        self.input_length = input_length
        self.classifier = classifier
        self.dtype = dtype
        self.verbose = verbose

        self.__set_seed(self.cfg.get_seed())

        self.k = k # num kernels per group
        self.g = g # num groups
        self.num_channels = num_channels

        max_exponent = np.log2((input_length - 1) / (self.__KERNEL_LEN - 1))

        self.dilations = np.array(2 ** np.arange(int(max_exponent)), dtype=np.int32)[:max_dilations]
        self.dilations = np.insert(self.dilations, 0, 0)
        self.num_dilations = len(self.dilations)

        self.paddings = np.round(np.divide((self.__KERNEL_LEN - 1) * self.dilations, 2)).astype(np.uint32)

        self.divisor = min(2, self.g)
        self.h = self.g // self.divisor

        if(dist == "normal"):
            self.W = self.rng.standard_normal(size=(self.h, self.k, self.__KERNEL_LEN)).astype(self.dtype)
            self.W = self.W - np.mean(self.W)
            self.W = self.W / np.sum(np.abs(self.W))
        elif(dist == "binomial"):
            self.W = self.rng.choice([-1, 1], size=(self.h, self.k, self.__KERNEL_LEN), p=[0.5, 0.5]).astype(self.dtype)
        elif(dist == "tetranomial"):
            self.W = self.rng.choice([-2, -1, 1, 2], size=(self.h, self.k, self.__KERNEL_LEN), p=[0.1, 0.4, 0.4, 0.1]).astype(self.dtype)

        # Preliminary message with model dimensions
        self.evaluate_model_size()
        #print(f"nanoHydra model with Transform  size {self.size['transform']} kB and Complexity {self.complexity_mmacs['transform']} mflops")
        #print(f"nanoHydra model with Classifier size {self.size['classifier']} kB and Complexity {self.complexity_mmacs['classifier']} mflops")
        #print(f"Model Size  Partition: {100*self.size['transform']/self.size['total']:.2f} Transform % vs {100*self.size['classifier']/self.size['total']:.2f} % Classifier")
        #print(f"Model Compl Partition: {100*self.complexity_mmacs['transform']/self.complexity_mmacs['total']:.2f} Transform % vs {100*self.complexity_mmacs['classifier']/self.complexity_mmacs['total']:.2f} % Classifier")

    def __set_seed(self, seed):
        if(seed is None):
            seed = int(time.time())
        self.cfg.set_seed(seed)
        #print(f"Setted Seed: {self.cfg.get_seed()}")
        self.rng = np.random.default_rng(seed=self.cfg.get_seed())

    def evaluate_model_size(self):
        self.size             = {'transform': 0, 'classifier': 0, 'feature_vec': 0, 'total': 0}
        self.complexity_mmacs = {'transform': 0, 'classifier': 0}

        # Calculate number of weights in transform
        if(self.dist == "normal"):
            bits_per_w = 32
        elif(self.dist == "normal_q16"):
            bits_per_w = 16
        elif(self.dist == "tetranomial"):
            bits_per_w = 2
        elif(self.dist == "binomial"):
            bits_per_w = 1

        # Size of transform
        # Note that 1kB = 2**10 bytes
        self.size['transform']              = self.W.size * bits_per_w / 8 / (2**10)
        self.size['feature_vec']            = self.W.size / self.__KERNEL_LEN
        self.complexity_mmacs['transform']  = self.input_length * self.W.size / 1e6

        # Size of classifier. Assumer weights will be quantized to 8bits
        self.size['classifier']             = self.size['feature_vec'] / (2**10)
        self.complexity_mmacs['classifier'] = self.size['feature_vec'] / 1e6
        #self.size['classifier']             = self.cfg.get_classf().coef_.size / (2**10)
        #self.complexity_mmacs['classifier'] = self.cfg.get_classf().coef_.size

        # Total size (in Flash memory)
        self.size['total']             = self.size['transform'] + self.size['classifier']
        self.complexity_mmacs['total'] = self.complexity_mmacs['transform'] + self.complexity_mmacs['classifier']

    def fit_scaler(self, X, num_samples=None):
        if(num_samples is not None):
            Xs = X.take(np.random.choice(X.shape[0], num_samples), axis=0).astype(np.float32)
            Xs = self.forward(Xs)
        else:
            Xs = X
        self.cfg.get_scaler().fit(Xs)

    def forward_scaler(self, X):
        return self.cfg.get_scaler().transform(X)

    # transform in batches of *batch_size*
    def forward_batch(self, X, batch_size = 256, do_fit=True, do_scale=False):
        num_examples = X.shape[0]
        len_feat_vec = 2*self.k * self.g * self.num_dilations * self.num_channels
        #len_feat_vec = self.k * self.g * self.num_dilations * self.num_channels
        Z = np.empty((num_examples, len_feat_vec))

        for idx in tqdm(range(0, num_examples, batch_size)):
            Z[idx:min(idx+batch_size, num_examples), :] = self.forward(X[idx:min(idx+batch_size, num_examples), :, :])

        if(do_fit):
            self.fit_scaler(Z)
        
        if(do_scale):
            Z = self.forward_scaler(Z)

        return Z

    def forward(self, X):

        num_examples = X.shape[0]

        if self.divisor > 1:
            diff_X = np.diff(X, axis=2)
        
        # Making sure dimensions are coherent
        assert diff_X.shape[0]==X.shape[0],   "DiffX {diff_X.shape[0]} and X {X.shape[0]} must have the same number of examples"
        assert diff_X.shape[1]==X.shape[1],   "DiffX {diff_X.shape[1]} and X {X.shape[1]} must have the same number of channels"
        assert diff_X.shape[2]==X.shape[2]-1, "DiffX {diff_X.shape[2]} and X {X.shape[2]} must have the same length-1"

        Z = []

        for dilation_index in range(self.num_dilations):

            d = self.dilations[dilation_index]

            feats_diff = [None for i in range(self.divisor)]

            for diff_index in range(self.divisor):


                feats = [None for i in range(self.num_channels)]
                
                for channel in range(self.num_channels):
                    _X = X[:,channel,:] if diff_index == 0 else diff_X[:,channel,:]

                    # Perform convolution on all kernels of a given dilation
                    #print(f"Current Dilation: {d}")
                    #_Z = conv1d_opt_x_int16_w_b1(_X, self.W, dilation = d)
                    _Z = conv1d_opt_x_f32_w_f32(_X, self.W, dilation = d)

                    # For each example, calculate the (arg)max/min over the k kernels of a given group.
                    # Here we should "collapse" the second dimension of the tensor, where the kernel indices are.
                    # Both return vectors should have dimensions (num_examples, num_groups, input_len)
                    max_values, max_indices = np.max(_Z, axis=2).astype(np.float32), np.argmax(_Z, axis=2).astype(np.uint32)
                    min_values, min_indices = np.min(_Z, axis=2).astype(np.float32), np.argmin(_Z, axis=2).astype(np.uint32)

                    # Create a feature vector of size (num_groups, num_kernels) where each of the num_kernels position contains
                    # the count for the respective kernel with that index.
                    feats_hard_max = soft_counting_opt(max_indices, max_values, kernels_per_group=self.k)
                    feats_hard_min = hard_counting_opt(min_indices,             kernels_per_group=self.k)

                    feats_hard_max = feats_hard_max.reshape((num_examples, self.h*self.k))
                    feats_hard_min = feats_hard_min.reshape((num_examples, self.h*self.k))

                    feats[channel] = np.concatenate((feats_hard_max, feats_hard_min), axis=1)

                if(self.num_channels==1):
                    feats_diff[diff_index] = feats[0]
                else:
                    feats_diff[diff_index] = np.concatenate((feats[i] for i in range(self.num_channels)), axis=1)

            feats_dil = np.concatenate((feats_diff[0], feats_diff[1]), axis=1)

            if(dilation_index):
                Z = np.concatenate((Z,feats_dil), axis=1)
            else:
                Z = feats_dil

        num_feats = 2*self.k * self.g * self.num_dilations * self.num_channels
        assert len(Z[0]) == num_feats, f"Dimensions of feature vector ({len(Z[0])}) do not match expected features {num_feats}"

        # Immediately free up RAM by marking the large vectors for deletion and calling the GC.
        del X, diff_X, _X
        del feats, feats_diff, feats_dil
        gc.collect()

        return Z

    def __generate_filename_trf_cache(self, ds_name):
        ds_name = ds_name.lower()
        return f"{ds_name}_k_{self.k}_g_{self.g}_d_{self.num_dilations}"

    def save_transform(self, Z, ds_name, path, split):
        filepath = f"{path}/{self.__generate_filename_trf_cache(ds_name)}_{split}.h5"
        
        with h5py.File(filepath, "w") as f:
            print(f"Caching transform to file '{filepath}'...")
            ds = f.create_dataset(self.__generate_filename_trf_cache(ds_name), Z.shape, data=Z, chunks=True, compression="gzip")
            print(f"Seed: {self.cfg.get_seed()}")
            ds.attrs['Seed'] = self.cfg.get_seed()
            print("Done!")

    def load_transform(self, ds_name, path, split):
        filepath = f"{path}/{self.__generate_filename_trf_cache(ds_name)}_{split}.h5"
        try:
            with h5py.File(filepath, "r") as f:
                Z = np.array(f[self.__generate_filename_trf_cache(ds_name)][:])
                #print(f"Recorded Seed: {f[self.__generate_filename_trf_cache(ds_name)].attrs['Seed']}")
                self.__set_seed(f[self.__generate_filename_trf_cache(ds_name)].attrs['Seed'])
        except FileNotFoundError as e:
            print(f"Exception: {e}")
            return None

        return Z

    def fit_classifier(self, X, Y):
        self.cfg.get_classf().fit(X,Y)

    def fit_tf_classifier(self, X, Y, X_val, Y_val):
        early_stopping_cb = EarlyStopping(monitor='val_loss', mode='min', patience=50, restore_best_weights=True)
        self.history = self.cfg.get_classf().fit(X, Y, 
                                        epochs=100, 
                                        batch_size=64, 
                                        shuffle=1, 
                                        verbose=True, 
                                        validation_data=(X_val, Y_val),
                                        callbacks=[early_stopping_cb])

    def predict_tf(self, X):
        return self.cfg.get_classf().predict(X, verbose=1, use_multiprocessing=True)

    def predict_batch(self, X, batch_size = 256):
        num_examples = X.shape[0]
        Y = []
        for idx in range(0, num_examples, batch_size):
            partialY = self.cfg.get_classf().predict(X[idx:min(idx+batch_size, num_examples)])
            Y.append(partialY)
        return np.concatenate(Y)

    def score_manual(self, Ypred, Ytest, method="subset"):
        assert len(Ypred) == len(Ytest), f"The prediction array and the expected output arrays are not the same length. {len(Ypred)} vs {len(Ytest)}"

        # If Ypred are probs, first calculate Top-One
        if(method.lower() == "prob"):
            Ypred = np.argmax(Ypred, axis=1)
        
        return np.sum(Ypred == Ytest)/len(Ypred)

    def score(self, X, Y):
        return self.cfg.get_classf().score(X,Y)

class SparseScaler():

    def __init__(self, mask = True, exponent = 4):

        self.mask = mask
        self.exponent = exponent

        self.fitted = False

    def fit(self, X):

        X = np.clip(X, a_min=0, a_max=None)

        # Since X has dimensions (num_examples, num_features), we perform the operations 
        # on each example (feature vector). Therefore, from here on we perform operations on axis=1
        self.epsilon = np.mean((X == 0), axis=0) ** self.exponent + 1e-8


        self.mu = np.mean(X, axis=0)
        self.sigma = np.std(X, axis=0) + self.epsilon

        assert len(self.mu)    == X.shape[1], f"Scaling Vector *mean* is not the same length as the number of features {X.shape[1]}"
        assert len(self.sigma) == X.shape[1], f"Scaling Vector *sigm* is not the same length as the number of features {X.shape[1]}"
        
        self.fitted = True

    def transform(self, X):

        assert self.fitted, "Not fitted."

        X = np.clip(X, a_min=0, a_max=None)

        self.epsilon = np.mean((X == 0), axis=0) ** self.exponent + 1e-8
        #print(f"self.epsilon: {self.epsilon}")

        if self.mask:
            
            #print(f"Shape of X  = {X.shape}")
            #print(f"Shape of m  = {self.mu.shape}")
            #print(f"Shape of s  = {self.sigma.shape}")
            
            for col in range(X.shape[1]):
                X[:,col] = (X[:,col] - self.mu[col])*(X[:,col] != 0)
            for col in range(X.shape[1]):
                X[:,col] = X[:,col] / self.sigma[col]

            return X

    def fit_transform(self, X):

        self.fit(X)

        return self.transform(X)
