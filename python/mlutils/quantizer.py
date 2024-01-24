import numpy as np
from fixedpoint import FixedPoint

class LayerQuantizer():

    def __init__(self, layer_sample, desired_bw):
        
        self.desired_bw = desired_bw

        self.Qm = None
        self.Qn = None

        self.analyze_samples(layer_sample.flatten())

    def analyze_samples(self, layer_sample):
        vmax = np.max(layer_sample)
        vmin = np.min(layer_sample)

        print(f"max: {vmax}, min: {vmin}")

        bmax = self.__get_max_N_bits(vmax)
        bmin = self.__get_max_N_bits(vmin)                    
    
        print(f"bmax: {bmax}, bmin: {bmin}")

        self.Qm = max(bmax, bmin)+1
        self.Qn = self.desired_bw - self.Qm

    def __get_max_N_bits(self, val):
        # From the absolute value (np.abs), we take the integer part (np.floor),
        # since the fractional will be represented by the M fractional bits. By taking the log2,
        # we evaluate the smallest power of two capable of representing the integer part, does not matter
        # if it is positive or negative. Since the number can be fractional, this means that it needs an aditional
        # integer bit to be represented, hence the use of np.ceil.
        return np.ceil(np.log2(np.floor(np.abs(val)))).astype(np.uint8) + (1 if val < 0 else 0)

    def __str__(self):
        return f"Layer Quantized for {self.desired_bw} bits, using Q{self.Qm}.{self.Qn}"

    def get_nm(self):
        return (self.Qn, self.Qm)

    def quantize(self, samples):
        return np.round(samples * (2**self.Qn)).astype(np.int16)

    def dequantize(self, samples):
        return samples / (2**self.Qn)

    def get_integer_bits(self):
        return self.Qm

    def get_fract_bits(self):
        return self.Qn

if __name__ == '__main__':
    layer = np.array([3.021, -7.234, 0.21, 0.231])
    lq = LayerQuantizer(layer, 16)
    print(lq)
    qlayer = lq.quantize(layer)
    print(qlayer)
    print([hex(n & 0xFFFF) for n in qlayer])
    dqlayer = lq.dequantize(qlayer)
    print(dqlayer)
    print(np.abs(dqlayer-layer))