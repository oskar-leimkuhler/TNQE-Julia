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
include("../subroutines/GeometryHeuristics.jl")
include("../subroutines/MutualInformation.jl")
include("../subroutines/Permutations.jl")
include("../subroutines/Misc.jl")
include("../subroutines/SubspaceRoutines.jl")
include("../subroutines/SubspaceData.jl")
include("../subroutines/Optimizer.jl")

# Read in parameters from config file:
conf_path = ARGS[1]

conf, op_list, gp_list, swp_list = FetchConfig("../$(conf_path)")

# Run computes and collect data:
subspace_vec = []
Ipq_vec = []

fid = h5open("../datasets/tnqe_data/$(conf.jobname)_$(Dates.today()).h5", "w")

fid["conf_path"] = conf_path

# Repeat for each geometry:
for m=1:conf.nmol
    
    create_group(fid, conf.mol_names[m])
    m_group = fid[conf.mol_names[m]]
    
    # Read in the chemical data:
    cdata = ReadIn("../"*conf.pyscf_paths[m])[1]
    
    # Determine bipartition entropies:
    if conf.Ipq_calc == "fci"
        fci_array = FCIArray(cdata)
        ipq_mps = MPS(fci_array, siteinds("Electron", cdata.N_spt), cutoff=1e-16, maxdim=2^16);
    elseif conf.Ipq_calc == "dmrg"
        ipq_dmrg = GenSubspace(cdata, 1, psi_maxdim=conf.Ipq_maxdim)
        GenStates!(ipq_dmrg, sweeps=conf.Ipq_sweeps)
        ipq_mps = ipq_dmrg.psi_list[1]
    end
    
    S1, S2, Ipq = MutualInformation(ipq_mps, collect(1:cdata.N_spt), cdata)
    push!(Ipq_vec, Ipq)
    
    if "Ipq" in conf.hdf5_out
        m_group["Ipq"] = Ipq
    end
    
    # Determine a quasi-optimal single ordering:
    opt_ord = InfDistAnnealing(
        Ipq, 
        1, 
        gp_list[conf.gp_optord]
    )[1]
    
    # Construct the ansatze:
    ansatze_m = []
    for a=1:conf.n_atz
        
        create_group(m_group, conf.atz_name[a])
        a_group = m_group[conf.atz_name[a]]
        
        M = conf.atz_M[a]
        psi_maxdim = conf.atz_m[a]
        
        ansatz = GenSubspace(cdata, M, psi_maxdim=psi_maxdim)
        
        # Fill in the ord list:
        if M==1 || conf.diff_ords[a]==false
            ansatz.ord_list = [opt_ord for j=1:M]
        else
            ansatz.ord_list = InfDistAnnealing(
                Ipq, 
                M, 
                gp_list[conf.gp_multord]
            )
        end
        
        # Initialize the MPO operators:
        GenPermOps!(ansatz)
        GenHams!(ansatz)
        
        # Initialize the states:
        if conf.init_sweeps[a]==0 # Randomize states
            GenStates!(ansatz, randomize=true)
        else # Use init_sweeps to initialize states
            GenStates!(ansatz, sweeps=swp_list[conf.init_sweeps[a]])
        end
        
        # Diagonalize:
        GenSubspaceMats!(ansatz)
        SolveGenEig!(ansatz)
        
        """
        # Optimize:
        if conf.do_opt[a]
            
            MultiGeomOptim!(
                ansatz,
                op_list,
                rep_struct=conf.rep_struct
            )
            
        end
        """
        
        # Output the data to HDF5 file:
        if "E" in conf.hdf5_out
            a_group["E"] = ansatz.E
        end
        
        if "C" in conf.hdf5_out
            a_group["C"] = ansatz.C
        end
        
        if "kappa" in conf.hdf5_out
            a_group["kappa"] = ansatz.kappa
        end
        
        if "H_mat" in conf.hdf5_out
            a_group["H_mat"] = ansatz.H_mat
        end
        
        if "S_mat" in conf.hdf5_out
            a_group["S_mat"] = ansatz.S_mat
        end
        
        if "ord_list" in conf.hdf5_out
            for (j,ord) in enumerate(ansatz.ord_list)
                a_group["ord_$(j)"] = ord
            end
        end
        
        if "psi_list" in conf.hdf5_out
            for (j,psi) in enumerate(ansatz.psi_list)
                a_group["psi_$(j)"] = psi
            end
        end
        
        if "ham_list" in conf.hdf5_out
            for (j,ham) in enumerate(ansatz.ham_list)
                a_group["ham_$(j)"] = ham
            end
        end
        
    end
    
end

close(fid)