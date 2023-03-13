# Import modules:
using ITensors
using HDF5
using Random
using PyCall
#using Graphs
#using SimpleWeightedGraphs
using Plots
using GraphRecipes
#using JuMP
#import Ipopt
#import ForwardDiff
using BlackBoxOptim
using Combinatorics
using LinearAlgebra
using Parameters

# Importing the other submodules:
include("./ChemData.jl")
include("./ChemRoutines.jl")
include("./GenEigRoutines.jl")
include("./GeometryHeuristics.jl")
include("./MutualInformation.jl")
include("./Permutations.jl")
include("./PlotRoutines.jl")
include("./Disentanglers.jl")
include("./Misc.jl")
include("./SubspaceRoutines.jl")
include("./SubspaceData.jl")
include("./Optimizer.jl")

# Import Python modules, including the RunPySCF subroutines in Python in order to use PySCF:
py"""
import sys
import os
import configparser
wd = os.getcwd()
sys.path.append(wd+'/../subroutines/')
import RunPySCF
import platform
print(platform.python_version())
"""

FastTT = pyimport("FastTT")
include("./FastTT.jl")