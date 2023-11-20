import time
import numpy as np
from sklearn.linear_model             import RidgeClassifierCV, SGDClassifier
from sklearn.tree                     import DecisionTreeClassifier, ExtraTreeClassifier
import h5py

from .optimized_fns.conv1d_opt        import conv1d_opt
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
            self.classf = SGDClassifier(loss='log_loss', alpha=0.01, shuffle=True, n_jobs=22, learning_rate='adaptive', eta0=1e-2, early_stopping=True, n_iter_no_change=5)
        elif(classifiertype.lower() == "ridge"):
            self.classf = RidgeClassifierCV(alphas=np.logspace(-3,3,10))
        elif(classifiertype.lower() == "perceptron"):
            self.classf = SGDClassifier(loss='perceptron', alpha=0.001, shuffle=True, n_jobs=22, learning_rate='adaptive', eta0=1e-2, early_stopping=True, n_iter_no_change=5)
        elif(classifiertype.lower() == "tree"):
            self.classf = ExtraTreeClassifier()

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

    def __init__(self, input_length, k = 8, g = 64, max_dilations=8, seed = None, dist = "normal", classifier="Logistic", scaler="Sparse"):

        self.cfg = NanoHydraCfg(seed= seed, scalertype=scaler, classifiertype=classifier)

        self.__set_seed(self.cfg.get_seed())

        self.k = k # num kernels per group
        self.g = g # num groups

        max_exponent = np.log2((input_length - 1) / (self.__KERNEL_LEN - 1))

        self.dilations = np.array(2 ** np.arange(int(max_exponent)), dtype=np.int32)[:max_dilations]
        self.dilations = np.insert(self.dilations, 0, 0)
        self.num_dilations = len(self.dilations)

        self.paddings = np.round(np.divide((9 - 1) * self.dilations, 2)).astype(np.uint32)

        self.divisor = min(2, self.g)
        self.h = self.g // self.divisor

        if(dist == "normal"):
            self.W = self.rng.standard_normal(size=(self.num_dilations, self.divisor, self.h, self.k, self.__KERNEL_LEN)).astype(np.float32)
            self.W = self.W - np.mean(self.W)
            self.W = self.W / np.sum(np.abs(self.W))
        elif(dist == "binomial"):
            self.W = self.rng.choice([-1, 1], size=(self.num_dilations, self.divisor, self.h, self.k, self.__KERNEL_LEN), p=[0.5, 0.5]).astype(np.float32)
            print(self.W)
        elif(dist == "tetranomial"):
            self.W = self.rng.choice([-2, -1, 1, 2], size=(self.num_dilations, self.divisor, self.h, self.k, self.__KERNEL_LEN), p=[0.1, 0.4, 0.4, 0.1]).astype(np.float32)

    def __set_seed(self, seed):
        if(seed is None):
            seed = int(time.time())
        self.cfg.set_seed(seed)
        print(f"Setted Seed: {self.cfg.get_seed()}")
        self.rng = np.random.default_rng(seed=self.cfg.get_seed())

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
    def forward_batch(self, X, batch_size = 256, do_fit=True, Y=None):
        num_examples = X.shape[0]

        Z = []
        for idx in range(0, num_examples, batch_size):
            print(f"Range: {idx}:{min(idx+batch_size, num_examples)}")
            Zt = self.forward(X[idx:min(idx+batch_size, num_examples)])
            self.fit_scaler(Zt)
            Zs = self.forward_scaler(Zt)

            if(do_fit):
                if(idx==0):
                    self.cfg.get_classf().partial_fit(Zs, Y[idx:min(idx+batch_size, num_examples)], np.unique(Y))
                else:
                    self.cfg.get_classf().partial_fit(Zs, Y[idx:min(idx+batch_size, num_examples)])
            else:
                Z.append(Zs)
        if(not do_fit):
            return np.vstack(Z)

    def forward(self, X):

        num_examples = X.shape[0]

        if self.divisor > 1:
            diff_X = np.diff(X, axis=1)

        Z = []

        for dilation_index in range(self.num_dilations):

            d = self.dilations[dilation_index]

            feats = [None for i in range(self.divisor)]

            for diff_index in range(self.divisor):

                #print(f"Transforming {num_examples} input samples for dilation {d} and diff_idx {diff_index}")

                _X = X if diff_index == 0 else diff_X
                # Perform convolution on all kernels of a given dilation
                #print(f"Current Dilation: {d}")
                _Z = conv1d_opt(_X, self.W[dilation_index, diff_index], dilation = d)

                # For each example, calculate the (arg)max/min over the k kernels of a given group.
                # Here we should "collapse" the second dimension of the tensor, where the kernel indices are.
                # Both return vectors should have dimensions (num_examples, num_groups, input_len)
                max_values, max_indices = np.max(_Z, axis=2).astype(np.float32), np.argmax(_Z, axis=2).astype(np.uint32)
                min_values, min_indices = np.min(_Z, axis=2).astype(np.float32), np.argmin(_Z, axis=2).astype(np.uint32)
                
                # Create a feature vector of size (num_groups, num_kernels) where each of the num_kernels position contains
                # the count for the respective kernel with that index.
                feats_hard_max = soft_counting_opt(max_indices, max_values, kernels_per_group=self.k)
                feats_hard_min = hard_counting_opt(min_indices, kernels_per_group=self.k)

                feats_hard_max = feats_hard_max.reshape((num_examples, self.h*self.k))
                feats_hard_min = feats_hard_min.reshape((num_examples, self.h*self.k))

                feats[diff_index] = np.concatenate((feats_hard_max, feats_hard_min), axis=1)

            feats = np.concatenate((feats[0], feats[1]), axis=1)
            
            if(dilation_index):
                Z = np.concatenate((Z,feats), axis=1)
            else:
                Z = feats

        return Z

    def __generate_filename_trf_cache(self, ds_name):
        ds_name = ds_name.lower()
        return f"{ds_name}_k_{self.k}_g_{self.g}_d_{self.num_dilations}"

    def save_transform(self, Z, ds_name, path, split):
        filepath = f"{path}/{self.__generate_filename_trf_cache(ds_name)}_{split}.h5"
        
        with h5py.File(filepath, "w") as f:
            print("Caching transform to file 'filepath'...")
            ds = f.create_dataset(self.__generate_filename_trf_cache(ds_name), Z.shape, data=Z, chunks=True, compression="gzip", compression_opts=9)
            print(f"Seed: {self.cfg.get_seed()}")
            ds.attrs['Seed'] = self.cfg.get_seed()
            print("Done!")

    def load_transform(self, ds_name, path, split):
        filepath = f"{path}/{self.__generate_filename_trf_cache(ds_name)}_{split}.h5"

        try:
            with h5py.File(filepath, "r") as f:
                Z = np.array(f[self.__generate_filename_trf_cache(ds_name)][:])
                print(f"Recorded Seed: {f[self.__generate_filename_trf_cache(ds_name)].attrs['Seed']}")
                self.__set_seed(f[self.__generate_filename_trf_cache(ds_name)].attrs['Seed'])
        except FileNotFoundError:
            return None

        return Z

    def fit_classifier(self, X, Y):
        self.cfg.get_classf().fit(X,Y)

    def predict_batch(self, X, batch_size = 256):
        num_examples = X.shape[0]
        Y = []
        for idx in range(0, num_examples, batch_size):
            partialY = self.cfg.get_classf().predict(X[idx:min(idx+batch_size, num_examples)])
            Y.append(partialY)
        return np.hstack(Y)

    def score_manual(self, Ypred, Ytest, method="subset"):
        assert len(Ypred) == len(Ytest), f"The prediction array and the expected output arrays are not the same length. {len(Ypred)} vs {len(Ytest)}"
        if(method.lower() == "subset"):
            return np.sum(Ypred == Ytest)/len(Ypred)

    def score(self, X, Y):
        return self.cfg.get_classf().score(X,Y)

class SparseScaler():

    def __init__(self, mask = True, exponent = 4):

        self.mask = mask
        self.exponent = exponent

        self.fitted = False

    def fit(self, X):

        X = np.sqrt(np.clip(X, a_min=0, a_max=None))

        # Since X has dimensions (num_examples, num_features), we perform the operations 
        # on each example (feature vector). Therefore, from here on we perform operations on axis=1
        #self.epsilon = (X == 0).float().mean(0) ** self.exponent + 1e-8

        self.mu = np.mean(X, axis=1).reshape(X.shape[0], 1)
        self.sigma = np.std(X, axis=1).reshape(X.shape[0], 1) #+ self.epsilon

        self.fitted = True

    def transform(self, X):

        assert self.fitted, "Not fitted."

        X = np.sqrt(np.clip(X, a_min=0, a_max=None))

        if self.mask:
            #return ((X - self.mu) * (X != 0)) / self.sigma
            return ((X - self.mu) ) / self.sigma
        else:
            return (X - self.mu) / self.sigma

    def fit_transform(self, X):

        self.fit(X)

        return self.transform(X)
