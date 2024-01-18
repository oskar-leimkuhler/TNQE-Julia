# Functions for constructing (and modifying) the subspace data structure

# Packages:
using ITensors
using HDF5
import Base: copy


mutable struct MetaParameters
    # Subspace dimension:
    M::Int
    M_ex::Int
    # MPS/MPO constructor parameters:
    psi_maxdim::Int
    psi_tol::Float64
    ham_maxdim::Int
    ham_tol::Float64
    singleham::Bool
    # Permutation parameters:
    perm_maxdim::Int
    perm_tol::Float64
    # Diagonalization parameters:
    thresh::String
    eps::Float64
end


# The subspace properties data structure:
# Keeps track of a single experimental instance \\
# ...including molecular data, metaparameters, orderings, \\
# ...generated MPSs and MPOs, and subspace diag. results.
mutable struct SubspaceProperties
    chem_data::ChemProperties
    mparams::MetaParameters
    sites::Vector
    dflt_sweeps::Sweeps
    ord_list::Vector{Vector{Int}}
    psi_list::Vector{MPS}
    ham_list::Vector{MPO}
    perm_ops::Vector{Vector{MPO}}
    rev_flag::Vector{Vector{Bool}}
    H_mat::Matrix{Float64}
    S_mat::Matrix{Float64}
    E::Vector{Float64}
    C::Matrix{Float64}
    kappa::Float64
end


# Generates a subspace properties struct instance (with modifiable defaults):
function GenSubspace(
        chem_data,
        M;
        stype="Electron",
        # MPS/MPO constructor parameters:
        psi_maxdim=8,
        psi_tol=1e-12,
        ham_maxdim=2^16,
        ham_tol=1e-14,
        singleham=false,
        # Permutation parameters:
        perm_maxdim=512,
        perm_tol=1e-12,
        # Diagonalization parameters:
        thresh="inversion",
        eps=1e-8,
        # DMRG settings:
        sites=nothing,
        dflt_sweeps=nothing,
        sweep_num=4,
        sweep_noise=1e-2
    )
    
    # Default metaparameters:
    mparams = MetaParameters(
        # Subspace dimension:
        M,
        1,
        # MPS/MPO constructor parameters:
        psi_maxdim,
        psi_tol,
        ham_maxdim,
        ham_tol,
        singleham,
        # Permutation parameters:
        perm_maxdim,
        perm_tol,
        # Diagonalization parameters:
        thresh,
        eps,
    )
    
    # Default sites:
    if sites == nothing
        if stype=="Electron"
            sites = siteinds("Electron", chem_data.N_spt, conserve_qns=true)
        elseif stype=="Fermion"
            sites = siteinds("Fermion", chem_data.N, conserve_qns=true)
        elseif stype=="Qubit"
            sites = siteinds("Qubit", chem_data.N)
        else
            println("Invalid site type!")
        end
    end
    
    # Default sweeps:
    if dflt_sweeps == nothing
        dflt_sweeps = Sweeps(sweep_num)
        maxdim!(dflt_sweeps,psi_maxdim)
        mindim!(dflt_sweeps,psi_maxdim)
        cutoff!(dflt_sweeps,psi_tol)
        setnoise!(dflt_sweeps, sweep_noise...)
    end
    
    # Initialize with empty data:
    sd = SubspaceProperties(
        chem_data,
        mparams,
        sites,
        dflt_sweeps,
        [],
        [],
        [],
        [],
        [],
        zeros((mparams.M,mparams.M)),
        zeros((mparams.M,mparams.M)),
        zeros(mparams.M),
        zeros((mparams.M,mparams.M)),
        0.0
    )
    
    return sd
    
end


function copy(sd::SubspaceProperties)
    
    subspace_copy = SubspaceProperties(
        sd.chem_data,
        sd.mparams,
        sd.sites,
        sd.dflt_sweeps,
        deepcopy(sd.ord_list),
        deepcopy(sd.psi_list),
        deepcopy(sd.ham_list),
        deepcopy(sd.perm_ops),
        deepcopy(sd.rev_flag),
        deepcopy(sd.H_mat),
        deepcopy(sd.S_mat),
        deepcopy(sd.E),
        deepcopy(sd.C),
        sd.kappa
    )
    
    return subspace_copy
    
end


function copyto!(sd1::SubspaceProperties, sd2::SubspaceProperties)
    
    sd1.ord_list = deepcopy(sd2.ord_list)
    sd1.psi_list = deepcopy(sd2.psi_list)
    sd1.ham_list = deepcopy(sd2.ham_list)
    sd1.perm_ops = deepcopy(sd2.perm_ops)
    sd1.rev_flag = deepcopy(sd2.rev_flag)
    sd1.H_mat = deepcopy(sd2.H_mat)
    sd1.S_mat = deepcopy(sd2.S_mat)
    sd1.E = deepcopy(sd2.E)
    sd1.C = deepcopy(sd2.C)
    sd1.kappa = deepcopy(sd2.kappa)
    
end


function GenStates!(
        sd::SubspaceProperties;
        ovlp_opt=false,
        sweeps=nothing,
        weight=1.0,
        prior_states=[],
        prior_ords=[],
        denseify=false,
        randomize=false,
        verbose=false
    )
    
    if sweeps==nothing
        sweeps = sd.dflt_sweeps
    end
    
    if randomize
        
        if verbose
            println("Generating states:")
        end
        
        sd.psi_list = []
        
        # Generate random states:
        for j=1:sd.mparams.M
            hf_occ_j = [FillHF(sd.ord_list[j][p], sd.chem_data.N_el) for p=1:sd.chem_data.N_spt]
            psi_j = randomMPS(sd.sites, hf_occ_j, linkdims=sd.mparams.psi_maxdim)
            
            push!(sd.psi_list, psi_j)
            
            if verbose
                print("Progress: [",string(j),"/",string(sd.mparams.M),"] \r")
                flush(stdout)
            end
            
        end
        
        if verbose
            println("\nDone!")
        end
        
    else
        
        # Generate a set of states:
        sd.psi_list, sd.ham_list = GenStates(
            sd.chem_data, 
            sd.sites, 
            sd.ord_list, 
            sweeps, 
            ovlp_opt=ovlp_opt,
            weight=weight,
            prior_states=prior_states,
            prior_ords=prior_ords,
            perm_tol=sd.mparams.perm_tol, 
            perm_maxdim=sd.mparams.perm_maxdim, 
            ham_tol=sd.mparams.ham_tol, 
            ham_maxdim=sd.mparams.ham_maxdim, 
            singleham=sd.mparams.singleham,
            denseify=denseify,
            verbose=verbose
        )
        
    end
    
end


function GenPermOps!(
        sd::SubspaceProperties;
        no_rev=false,
        tol=1.0e-16,
        maxdim=2^16,
        verbose=false
    )
    
    M = sd.mparams.M
    
    if sd.perm_ops==[]
        # Construct identity MPOs:
        sd.perm_ops = [[MPO(sd.sites) for j=1:M-i] for i=1:M]
        sd.rev_flag = [[false for j=1:M-i] for i=1:M]
    end
    
    if verbose
        println("Generating permutation operators:")
    end
    
    c = 0
    c_tot = Int((M^2-M)/2)
    
    # Permute the identity MPOs to obtain \\
    # ...permutation MPOs:
    for i=1:M, j=1:M-i
        
        sd.perm_ops[i][j], sd.rev_flag[i][j] = FastPMPO(
            sd.sites,
            sd.ord_list[i],
            sd.ord_list[j+i],
            tol=tol,
            no_rev=no_rev,
            maxdim=maxdim
        )
        
        c += 1
        
        if verbose
            print("Progress: [$(c)/$(c_tot)] \r")
            flush(stdout)
        end
        
    end
    
    if verbose
        println("\nDone!\n")
    end
    
end


function GenSubspaceMats!(
        sd::SubspaceProperties;
        verbose=false
    )
    
    M = sd.mparams.M
    
    # Diagonal elements:
    for i=1:M
        sd.H_mat[i,i] = inner(sd.psi_list[i]', sd.ham_list[i], sd.psi_list[i])
        sd.S_mat[i,i] = 1.0
    end
    
    # Off-diagonal elements:
    for i=1:M, j=i+1:M
        
        psi_i = deepcopy(sd.psi_list[i])
        psi_j = deepcopy(sd.psi_list[j])
        ham_i = deepcopy(sd.ham_list[i])
        ham_j = deepcopy(sd.ham_list[j])
        
        if sd.rev_flag[i][j-i]
            psi_j = ReverseMPS(sd.psi_list[j])
            ham_j = ReverseMPO(sd.ham_list[j])
        end
        
        pmpo_ij = sd.perm_ops[i][j-i]
        
        sd.H_mat[i,j] = inner(pmpo_ij, psi_j, ham_i, psi_i)
        sd.S_mat[i,j] = inner(psi_i', pmpo_ij, psi_j)
        sd.H_mat[j,i] = sd.H_mat[i,j]
        sd.S_mat[j,i] = sd.S_mat[i,j]
        
    end
    
end


function SolveGenEig!(
        sd::SubspaceProperties;
        thresh=nothing,
        eps=nothing,
        verbose=false
    )
    
    if thresh==nothing
        thresh=sd.mparams.thresh
    end
    
    if eps==nothing
        eps=sd.mparams.eps
    end
    
    sd.E, sd.C, sd.kappa = SolveGenEig(
        sd.H_mat,
        sd.S_mat,
        thresh=thresh,
        eps=eps
    )
    
    if verbose
        DisplayEvalData(sd.chem_data, sd.H_mat, sd.E, sd.C, sd.kappa)
    end
    
end


# Generate the Hamiltonian MPOs
function GenHams!(
        sd::SubspaceProperties
    )
    
    sd.ham_list = []
    
    for j=1:sd.mparams.M
        
        opsum = GenOpSum(
            sd.chem_data, 
            sd.ord_list[j]
        )

        ham_j = MPO(
            opsum, 
            sd.sites, 
            cutoff=sd.mparams.ham_tol, 
            maxdim=sd.mparams.ham_maxdim
        )
        
        push!(sd.ham_list, ham_j)
        
    end
    
    
end


function ShuffleStates!(
        sd::SubspaceProperties;
        perm=nothing
    )
    
    if perm==nothing # Random shuffle
        perm = randperm(sd.mparams.M)
    end
    
    permute!(sd.ord_list, perm)
    permute!(sd.psi_list, perm)
    permute!(sd.ham_list, perm)
    
    GenPermOps!(sd)
    GenSubspaceMats!(sd)
    SolveGenEig!(sd)
    
end

function CycleStates!(
        sd::SubspaceProperties;
        c=1
    )
    
    sd.ord_list = circshift(sd.ord_list, c)
    sd.psi_list = circshift(sd.psi_list, c)
    sd.ham_list = circshift(sd.ham_list, c)
    
    GenPermOps!(sd)
    GenSubspaceMats!(sd)
    SolveGenEig!(sd)
    
end