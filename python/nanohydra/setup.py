from setuptools import setup
from setuptools.extension import Extension
import numpy as np

ext_modules=[Extension("conv1d_opt", ["./nanohydra/optimized_fns/conv1d_opt_fn.pyx"],
                extra_compile_args=['-fopenmp'],
                extra_link_args=['-fopenmp']),
             Extension("hard_counting_opt", ["./nanohydra/optimized_fns/hard_counting_opt_fn.pyx"],
                extra_compile_args=['-fopenmp'],
                extra_link_args=['-fopenmp']),
             Extension("soft_counting_opt", ["./nanohydra/optimized_fns/soft_counting_opt_fn.pyx"],
                extra_compile_args=['-fopenmp'],
                extra_link_args=['-fopenmp'])]


from Cython.Build import cythonize
setup(ext_modules=cythonize(ext_modules, 
                    compiler_directives={"language_level": "3"}),
      include_dirs=[np.get_include()])