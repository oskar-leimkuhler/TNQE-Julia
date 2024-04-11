# Import modules:
using ITensors
using HDF5
using Random
using BlackBoxOptim
using Combinatorics
using LinearAlgebra
using Parameters
using ConfParser
using Dates

# Importing the other submodules:
include("../subroutines/ChemData.jl")
include("../subroutines/ChemRoutines.jl")
include("../subroutines/GenEigRoutines.jl")
include("../subroutines/MutualInformation.jl")
include("../subroutines/OrbitalRotations.jl")
include("../subroutines/Misc.jl")
include("../subroutines/SubspaceData.jl")
include("../subroutines/Optimizer.jl")
include("../subroutines/UCCSD.jl")

# Quick hard-coded script for a single result (not to be extended!)
md_fname = "h6_sto3g_031324%1756.hdf5"

chemical_data_list = ReadIn("../datasets/pyscf_data/"*md_fname)

conf = ConfParse("../configs/pyscf_configs/h6_octahedron_sto3g.ini")
parse_conf!(conf)
bond_lengths = parse.(Float64, retrieve(conf, "GEOMETRIES", "xyz_files"));

# A file to write the output to:
fid = h5open(pwd()*"/../datasets/tnqe_data/h6_octahedron_sto3g_uccsd_pe_curves_$(Dates.today()).h5", "w")

# Repeat over selected bond lengths:
n_x_list = [5,7,9,11,13,15,17,19,21,23,25,27,29,31]

for n_x in n_x_list
    
    chemical_data = chemical_data_list[n_x]

    e_opt, x_opt = uCCSDMinimize(chemical_data, maxiter=50)

    # Write the evec to file:
    create_group(fid, string(n_x))
    n_x_group = fid[string(n_x)]
    n_x_group["energies"] = [e_opt]
    n_x_group["bond_length"] = bond_lengths[n_x]
    
    println("Completed point n_x=$(n_x)!")
    
end

close(fid)