from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

setup(
  name='BMI203-FinalProject',
  ext_modules=cythonize("final_project/_alignment.pyx"),
  include_dirs=[np.get_include()],
)
