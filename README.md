# Classical numerical simulations of the TNQE algorithm for quantum chemical ground state preparation

## Overview

Code to perform numerical simulations of the hybrid quantum-classical tensor network quantum eigensolver (TNQE) algorithm for chemical ground state preparation, implemented using the ITensor Julia library (https://itensor.github.io/ITensors.jl/stable/index.html) and sparse matrices. Quantum chemical data is generated with the PySCF Python package (https://pyscf.org/). This project is associated with the following paper:

**Citation**: [Oskar Leimkuhler and K. Birgitta Whaley], *A quantum eigenvalue solver based on tensor networks*, ArXiv Preprint, 2024, DOI: [https://doi.org/10.48550/arXiv.2404.10223].

## Installation

To get started, first clone this repository and install the necessary Julia and Python dependencies below. 

Then you can run the calculations in "notebooks/demo_tnqe.ipynb".

## Dependencies

Python: `pyscf, numpy, h5py, configparser`

Julia: `ITensors, LinearAlgebra, SparseArrays, HDF5, Random, Plots, GraphRecipes, Optim, BlackBoxOptim, NLOpt, Combinatorics, Parameters, PyCall`

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Citation
If you use this software in your research, please cite:

```
@misc{leimkuhler2024quantumeigenvaluesolverbased,
      title={A quantum eigenvalue solver based on tensor networks}, 
      author={Oskar Leimkuhler and K. Birgitta Whaley},
      year={2024},
      eprint={2404.10223},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      url={https://arxiv.org/abs/2404.10223}, 
}
```
