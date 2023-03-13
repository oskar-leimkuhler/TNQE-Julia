import numpy as np
import os
# The main libraries we will be using
# (with thanks to the authors: arXiv:1908.02721)
# Note: these libraries must be compiled and installed on your system, \\
# ... I used the default xerus (FastTT paper version) python bindings built in boost, \\
# ... and used pybind11 to generate python bindings and additional wrapper functions for the fast_tt script \\
import xerus
import fast_tt

def FastTT(dense_perm_array, pnum, maxdim, tol):
    
    reshape_dims = dense_perm_array.shape
    
    totdim = dense_perm_array.size
    
    dense_perm_array = dense_perm_array.reshape(totdim)
    
    nzi = np.nonzero(dense_perm_array)[0]
    
    nzv = dense_perm_array[np.logical_not(np.isclose(dense_perm_array, 0.0))]
    
    mpo_list = fast_tt.pysp2tt(nzv, nzi, reshape_dims, pnum, maxdim, tol)
    
    return mpo_list;

def BubbleSort2TT(N, swap_inds, phases, pnum, maxdim, tol):
    
    mpo_list = fast_tt.bubs2tt(N, swap_inds, phases, pnum, maxdim, tol)
    
    return mpo_list;
