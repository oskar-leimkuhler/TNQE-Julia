Code to perform numerical simulations of the tensor network quantum eigensolver (TNQE) algorithm for chemical ground states, implemented using the ITensor Julia library (https://itensor.github.io/ITensors.jl/stable/index.html) and sparse matrices. Quantum chemical data is generated with the PySCF Python package (https://pyscf.org/).

Requires the following Julia packages:

ITensors
LinearAlgebra
SparseArrays
HDF5
Random
Plots
GraphRecipes
Optim
BlackBoxOptim
NLOpt
Combinatorics
Parameters
PyCall

And the following Python packages for the PySCF calculations:

pyscf
numpy
h5py
configparser

Example calculations are provided in the demo_tnqe notebook.
