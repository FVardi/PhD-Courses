"""Run the TF-C checkout's main.py unmodified, under one in-memory compatibility shim.

Started by src/methods/tfc.py as a subprocess with the working directory set to the
checkout's code/TFC, which is where the README says to run main.py from and what its
relative paths (../../datasets, ../experiments_logs) assume. Arguments are passed through
to main.py untouched.

The shim: trainer.py calls np.float(0) when AUROC cannot be computed for a batch. np.float
was an alias for the builtin float and was removed in NumPy 1.24, so on this environment
that fallback would raise AttributeError. Restoring the alias here gives the behaviour the
code was written against without editing a file in the checkout. It changes no number
unless the fallback is reached, and then only by making it return 0.0 as intended.
"""

import os
import runpy
import sys

import numpy as np

np.float = float  # removed in NumPy 1.24; see module docstring

# Python put this file's folder first on sys.path. main.py expects its own folder there
# (it imports model, trainer, dataloader, ... as top-level modules).
sys.path[0] = os.getcwd()
runpy.run_path("main.py", run_name="__main__")
