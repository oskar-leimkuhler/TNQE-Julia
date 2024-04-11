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
using SparseArrays
using Optim

# Importing the other submodules:
include("../subroutines/ChemData.jl")
include("../subroutines/ChemRoutines.jl")
include("../subroutines/GenEigRoutines.jl")
include("../subroutines/MutualInformation.jl")
include("../subroutines/OrbitalRotations.jl")
include("../subroutines/Misc.jl")
include("../subroutines/SubspaceData.jl")
include("../subroutines/Optimizer.jl")

# Quick hard-coded script for a single result
md_fname = "h2o_sto3g_032024%0918.hdf5"

chemical_data_list = ReadIn("../datasets/pyscf_data/"*md_fname)

conf = ConfParse("../configs/pyscf_configs/h6_octahedron_sto3g.ini")
parse_conf!(conf)
bond_lengths = parse.(Float64, retrieve(conf, "GEOMETRIES", "xyz_files"));

# A file to write the output to:
fid = h5open(pwd()*"/../datasets/tnqe_data/h2o_stretch_sto3g_tnqe_givens_pe_curves_$(Dates.today()).h5", "w")

# Repeat over selected bond lengths:
n_x_list = [2,4,6,8,10,12,14,16,18,20,22,24,26]

# Hard-coded parameter objects:
gp = GeomParameters(
    maxiter=300000,
    eta=-2,
    shrp=2.0,
    a_alpha=0.8
)

# The optimization parameters:
op1 = OptimParameters(
    maxiter=1, 
    numloop=1,  
    thresh="projection",
    eps=1e-12,
    sd_thresh="projection", 
    sd_eps=1.0e-12,
    delta=[1e-6,1e-7], # QPU noise
    noise=[0.0], # DMRG "noise" term
    sd_method="triple_geneig",
    sd_dtol=5e-3,
    sd_etol=1e-3
)


chi = 3
M_max = 4

delta_list = vcat([1e-8],[1e-10 for k=1:M_max])
noise_list = vcat([],[1e-10 for k=1:M_max])
etol_list = [2e-3,1e-3,5e-4,2e-4,1e-4,5e-5]

rotypes = ["fswap", "fswap", "givens"]

for n_x in n_x_list
    
    chemical_data = chemical_data_list[n_x]

    fci_array = FCIArray(chemical_data)
    fci_mps = MPS(fci_array, siteinds("Electron", chemical_data.N_spt), cutoff=1e-16, maxdim=2^16);

    S1, S2, Ipq = MutualInformation(fci_mps, collect(1:chemical_data.N_spt), chemical_data)

    opt_ord = InfDistAnnealing(
        Ipq, 
        1, 
        gp,
        verbose=false
    )[1]

    # Initialize these lists:
    tnqe3_evec = Float64[]
    
    tnqe3 = GenSubspace(
        chemical_data, 
        M=1, 
        mps_maxdim=chi, 
        ham_tol=1e-15,
        perm_tol=1e-15,
        thresh="projection", 
        init_ord = deepcopy(opt_ord),
        eps=1e-12, 
        sweep_num=20, 
        sweep_noise=(1e-2,1e-3,1e-4),
        dmrg_init=true,
        ovlp_opt=true,
        ovlp_weight=2.0,
        verbose=true
    );

    push!(tnqe3_evec, tnqe3.E[1])

    for k=1:M_max-1
    
        op1.delta = [delta_list[k]]
        op1.noise = [noise_list[k]]
        op1.sd_etol = etol_list[k]

        AddStates!(
            tnqe3;
            M_new=1,
            G_init="last",
            dmrg_init=(k==1),
            ovlp_opt=true,
            ovlp_weight=2.0,
            verbose=false
        )

        # Permute so new states are at the front:
        perm = circshift(collect(1:tnqe3.mparams.M), 1)

        ShuffleStates!(tnqe3, perm=perm, no_rev=true)

        println("\n$(tnqe3.mparams.M) states:\n")

        for l=1:4

            TwoSiteBlockSweep!(
                tnqe3,
                op1,
                verbose=true,
                nsite=vcat([2], [0 for i=2:tnqe3.mparams.M]),
                rotype=rotypes[1],
                jperm=vcat([1], shuffle(2:tnqe3.mparams.M))
            )

        end

        for l=1:10

            perm = randperm(tnqe3.mparams.M)
            ShuffleStates!(tnqe3, perm=perm, no_rev=true)

            TwoSiteBlockSweep!(
                tnqe3,
                op1,
                verbose=true,
                nsite=[2 for i=1:tnqe3.mparams.M],
                rotype=rotypes[2],
                jperm=collect(1:tnqe3.mparams.M),
            )

            ShuffleStates!(tnqe3, perm=invperm(perm), no_rev=true)

            perm = randperm(tnqe3.mparams.M)
            ShuffleStates!(tnqe3, perm=perm, no_rev=true)

            TwoSiteBlockSweep!(
                tnqe3,
                op1,
                verbose=true,
                nsite=[2 for i=1:tnqe3.mparams.M],
                rotype=rotypes[3],
                jperm=collect(1:tnqe3.mparams.M),
            )

            ShuffleStates!(tnqe3, perm=invperm(perm), no_rev=true)

        end

        push!(tnqe3_evec, tnqe3.E[1])

    end
    
    # Write the evec to file:
    create_group(fid, string(n_x))
    n_x_group = fid[string(n_x)]
    n_x_group["energies"] = tnqe3_evec
    n_x_group["bond_length"] = bond_lengths[n_x]
    
    println("Completed point n_x=$(n_x)!")
    
end

close(fid)
