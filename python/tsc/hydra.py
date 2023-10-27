
import numpy as np


def conv1d(x, w, dilation):
    assert len(x.shape) == 2, f"X array must be dimension 2D, but has {len(x.shape)}"
    
    num_examples = x.shape[0]
    xlen = x.shape[1]
    H = w.shape[0]
    K = w.shape[1]

    _Y = np.empty((num_examples, H, K, int(xlen/(dilation+1))))

    for ex in range(num_examples):
        for h in range(H):
            for k in range(K):
                x_dil = np.take(x[ex,:], [(1+dilation)*i for i in range(int(xlen/(dilation+1)))], mode='wrap')
                _Y[ex, h, k] = np.convolve(x_dil, w[h,k], mode='same')
        #print(f"Processed {ex} of {num_examples} samples")

    return _Y

def hard_counting(max_idxs, kernels_per_group):

    num_examples, num_groups = max_idxs.shape[0], max_idxs.shape[1]

    feats = np.zeros((num_examples, num_groups, kernels_per_group))
    for ex in range(num_examples):
        for gr in range(num_groups):
            idxs,cnts = np.unique(max_idxs[ex,gr], return_counts=True)
            for idx,cnt in zip(idxs,cnts):
                feats[ex,gr,idx] = cnt

    return feats

class Hydra():

    __KERNEL_LEN = 9

    def __init__(self, input_length, k = 8, g = 64, seed = None):

        super().__init__()

        if seed is not None:
            # TODO: set numpy random num gen seed
            pass
        rng = np.random.default_rng()

        self.k = k # num kernels per group
        self.g = g # num groups

        max_exponent = np.log2((input_length - 1) / (9 - 1)) # kernel length = 9

        self.dilations = 2 ** np.arange(int(max_exponent))
        self.dilations = np.insert(self.dilations, 0, 0)
        print(self.dilations)
        self.num_dilations = len(self.dilations)

        self.paddings = np.round(np.divide((9 - 1) * self.dilations, 2)).astype(np.uint32)

        self.divisor = min(2, self.g)
        self.h = self.g // self.divisor

        self.W = rng.standard_normal(size=(self.num_dilations, self.divisor, self.h, self.k, self.__KERNEL_LEN))
        self.W = self.W - np.mean(self.W)
        self.W = self.W / np.sum(np.abs(self.W))

    # transform in batches of *batch_size*
    def batch(self, X, batch_size = 256):
        num_examples = X.shape[0]
        if num_examples <= batch_size:
            return self.forward(X)
        else:
            Z = []
            batches = np.arange(num_examples).split(batch_size)
            for batch in batches:
                Z.append(self.forward(X[batch]))
            return np.vstack(Z)

    def forward(self, X):

        num_examples = X.shape[0]

        if self.divisor > 1:
            diff_X = np.diff(X, axis=1)

        Z = []

        for dilation_index in range(self.num_dilations):

            d = self.dilations[dilation_index]
            p = self.paddings[dilation_index]

            feats = [None for i in range(self.divisor)]

            for diff_index in range(self.divisor):

                print(f"Transforming {num_examples} input samples for dilation {d} and diff_idx {diff_index}")

                #print(f"forward(): 'X' - {X.shape}")
                #print(f"forward(): 'diff_X'  = {diff_X.shape}")

                # Perform convolution on all kernels of a given
                _Z = conv1d(X if diff_index == 0 else diff_X, self.W[dilation_index, diff_index], dilation = d)

                #print(f"forward(): _Z - {_Z.shape}")

                # For each example, calculate the (arg)max/min over the k kernels of a given group.
                # Here we should "collapse" the second dimension of the tensor, where the kernel indices are.
                # Both return vectors should have dimensions (num_examples, num_groups, input_len)
                max_values, max_indices = np.max(_Z, axis=2), np.argmax(_Z, axis=2)
                min_values, min_indices = np.min(_Z, axis=2), np.argmin(_Z, axis=2)
                
                #print(f"forward(): max_indices - {max_indices.shape}")

                # Create a feature vector of size (num_groups, num_kernels) where each of the num_kernels position contains
                # the count for the respective kernel with that index.
                feats_hard_max = hard_counting(max_indices, kernels_per_group=self.k)
                feats_hard_min = hard_counting(min_indices, kernels_per_group=self.k)

                feats_hard_max = feats_hard_max.reshape((num_examples, self.h*self.k))
                feats_hard_min = feats_hard_min.reshape((num_examples, self.h*self.k))
                #print(f"forward(): feats_hard_max - {feats_hard_max[0]}")
                #print(f"forward(): feats_hard_min - {feats_hard_min[0]}")

                feats[diff_index] = np.concatenate((feats_hard_max, feats_hard_min), axis=1)
                #print(f"forward(): feats - {feats[diff_index].shape}")

            feats = np.concatenate((feats[0], feats[1]), axis=1)
            
            if(dilation_index):
                Z = np.concatenate((Z,feats), axis=1)
            else:
                Z = feats
        print(f"forward(): FINISH Z - {Z.shape}")

        return Z 