from nanohydra.optimized_fns.conv1d_opt_x_f32_w_f32         import conv1d_opt_x_f32_w_f32
from nanohydra.optimized_fns.conv1d_opt_x_f32_w_b1          import conv1d_opt_x_f32_w_b1
from nanohydra.optimized_fns.conv1d_opt_x_int16_w_b1        import conv1d_opt_x_int16_w_b1
from nanohydra.optimized_fns.conv1d_opt_x_int16_w_b1_notake import conv1d_opt_x_int16_w_b1_notake
import numpy as np
import time

# Input vector params
NUM_EXAMPLES  = 300
INPUT_VEC_LEN = 16000

# Weight matrix params
DIVISOR    = 2
G          = 64
K          = 8
KERNEL_LEN = 9


if __name__ == '__main__':

    # Constants
    FUNCS = ['orig', 'x_f32_w_b1', 'x_int16_w_b1', 'x_int16_w_b1_notake']

    # Test vars
    times  = {k:0 for k in FUNCS}
    errors = {k:0 for k in FUNCS}

    # Initialize RNG
    rng = np.random.default_rng(seed=42)

    # Test data
    X = rng.integers(low=-2**10, high=2**10, size=(NUM_EXAMPLES, INPUT_VEC_LEN)).astype(np.int16)
    W = rng.choice([-1, 1], size=(G // DIVISOR, K, KERNEL_LEN), p=[0.5, 0.5]).astype(np.int16)
    Y = {k:None for k in FUNCS}

    # Transform data
    start = time.perf_counter()
    Y['orig'] = conv1d_opt_x_f32_w_f32(X.astype(np.float32), W.astype(np.float32), 0)
    times['orig']  = time.perf_counter()-start

    start = time.perf_counter()
    Y['x_f32_w_b1']     = conv1d_opt_x_f32_w_b1(X.astype(np.float32), W, 0)
    times['x_f32_w_b1'] = time.perf_counter()-start
    errors['x_f32_w_b1'] = np.sum(np.abs(Y['x_f32_w_b1']-Y['orig']))

    start = time.perf_counter()
    Y['x_int16_w_b1']     = conv1d_opt_x_int16_w_b1(X, W, 0)
    times['x_int16_w_b1'] = time.perf_counter()-start
    errors['x_int16_w_b1'] = np.sum(np.abs(Y['x_int16_w_b1']-Y['orig']))

    print(np.abs(Y['x_int16_w_b1'][0,0,0]-Y['orig'][0,0,0]))
    start = time.perf_counter()
    Y['x_int16_w_b1_notake']      = conv1d_opt_x_int16_w_b1_notake(X, W, 0)
    times['x_int16_w_b1_notake']  = time.perf_counter()-start
    errors['x_int16_w_b1_notake'] = np.sum(np.abs(Y['x_int16_w_b1_notake']-Y['orig']))


    # Print Results
    for k,v in errors.items():
        print(f"'{k}': {v}, executed in {times[k]:.3f} seconds")
