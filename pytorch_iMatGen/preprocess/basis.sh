#!/bin/bash
#PBS -l select=1:ncpus=8

cd $PBS_O_WORKDIR

# activate venv
source ../../venv/bin/activate

# work around
export HDF5_USE_FILE_LOCKING=FALSE

# execute
PYTHONPATH=.. python get_basis_image.py

# deactivate venv
deactivate
