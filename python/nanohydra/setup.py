from setuptools import setup
from setuptools.extension import Extension
import numpy as np
import multiprocessing 

ext_modules=[Extension("conv1d_opt_orig", ["./nanohydra/optimized_fns/conv1d_opt_orig.pyx"],
                extra_compile_args=['-fopenmp'],
                extra_link_args=['-fopenmp']),
             Extension("conv1d_opt_x_f32_w_f32", ["./nanohydra/optimized_fns/conv1d_opt_x_f32_w_f32.pyx"],
                extra_compile_args=['-fopenmp'],
                extra_link_args=['-fopenmp']),
             Extension("conv1d_opt_x_f32_w_b1", ["./nanohydra/optimized_fns/conv1d_opt_x_f32_w_b1.pyx"],
                extra_compile_args=['-fopenmp'],
                extra_link_args=['-fopenmp']),
             Extension("conv1d_opt_x_int16_w_b1", ["./nanohydra/optimized_fns/conv1d_opt_x_int16_w_b1.pyx"],
                extra_compile_args=['-fopenmp'],
                extra_link_args=['-fopenmp']),
             Extension("hard_counting_opt", ["./nanohydra/optimized_fns/hard_counting_opt_fn.pyx"],
                extra_compile_args=['-fopenmp'],
                extra_link_args=['-fopenmp']),
             Extension("soft_counting_opt", ["./nanohydra/optimized_fns/soft_counting_opt_fn.pyx"],
                extra_compile_args=['-fopenmp'],
                extra_link_args=['-fopenmp']),
             Extension("combined_counting_opt", ["./nanohydra/optimized_fns/combined_counting_opt_fn.pyx"],
                extra_compile_args=['-fopenmp'],
                extra_link_args=['-fopenmp'])
                ]


from Cython.Build import cythonize
setup(ext_modules=cythonize(ext_modules, 
                    compiler_directives={"language_level": "3"},
                    nthreads=24),
      include_dirs=[np.get_include()])