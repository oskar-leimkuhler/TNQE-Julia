# Functions for constructing (and modifying) the subspace data structure

# Packages:
using ITensors
using HDF5


struct MetaParameters
    # Subspace dimension:
    M::Int
    # MPS/MPO constructor parameters:
    psi_maxdim::Int
    psi_tol::Float64
    ham_maxdim::Int
    ham_tol::Float64
    spatial::Bool
    spinpair::Bool
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
        # MPS/MPO constructor parameters:
        psi_maxdim=8,
        psi_tol=1e-12,
        ham_maxdim=512,
        ham_tol=1e-12,
        spatial=true,
        spinpair=false,
        singleham=false,
        # Permutation parameters:
        perm_maxdim=512,
        perm_tol=1e-12,
        # Diagonalization parameters:
        thresh="projection",
        eps=1e-12,
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
        # MPS/MPO constructor parameters:
        psi_maxdim,
        psi_tol,
        ham_maxdim,
        ham_tol,
        spatial,
        spinpair,
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
        if spatial == true
            sites = siteinds("Electron", chemical_data.N_spt, conserve_qns=true)
        else
            sites = siteinds("Fermion", chemical_data.N, conserve_qns=true)
        end
    end
    
    # Default sweeps:
    if dflt_sweeps == nothing
        dflt_sweeps = Sweeps(sweep_num)
        maxdim!(dflt_sweeps,psi_maxdim)
        cutoff!(dflt_sweeps,psi_tol)
        setnoise!(dflt_sweeps, sweep_noise...)
    end
    
    # Initialize with empty data:
    sdata = SubspaceProperties(
        chem_data,
        mparams,
        sites,
        dflt_sweeps,
        [],
        [],
        [],
        zeros((mparams.M,mparams.M)),
        zeros((mparams.M,mparams.M)),
        zeros(mparams.M),
        zeros((mparams.M,mparams.M)),
        0.0
    )
    
    return sdata
    
end


function copy(sdata::SubspaceProperties)
    
    subspace_copy = SubspaceProperties(
        sdata.chem_data,
        sdata.mparams,
        sdata.sites,
        sdata.dflt_sweeps,
        deepcopy(sdata.ord_list),
        deepcopy(sdata.psi_list),
        deepcopy(sdata.ham_list),
        deepcopy(sdata.H_mat),
        deepcopy(sdata.S_mat),
        deepcopy(sdata.E),
        deepcopy(sdata.C),
        sdata.kappa
    )
    
    return subspace_copy
    
end


function GenStates!(
        sdata::SubspaceProperties;
        ovlp_opt=false,
        weight=1.0,
        prior_states=[],
        prior_ords=[],
        verbose=false
    )
    
    # Generate a set of states:
    sdata.psi_list, sdata.ham_list = GenStates(
        sdata.chem_data, 
        sdata.sites, 
        sdata.ord_list, 
        sdata.dflt_sweeps, 
        ovlp_opt=ovlp_opt,
        weight=weight,
        prior_states=prior_states,
        prior_ords=prior_ords,
        perm_tol=sdata.mparams.perm_tol, 
        perm_maxdim=sdata.mparams.perm_maxdim, 
        ham_tol=sdata.mparams.ham_tol, 
        ham_maxdim=sdata.mparams.ham_maxdim, 
        spinpair=sdata.mparams.spinpair, 
        spatial=sdata.mparams.spatial, 
        singleham=sdata.mparams.singleham,
        verbose=verbose
    )
    
end


function GenSubspaceMats!(
        sdata::SubspaceProperties;
        verbose=false
    )
    
    # Generate a set of states:
    sdata.H_mat, sdata.S_mat = GenSubspaceMats(
        sdata.chem_data,
        sdata.sites,
        sdata.ord_list,
        sdata.psi_list,
        sdata.ham_list,
        perm_tol=sdata.mparams.perm_tol, 
        perm_maxdim=sdata.mparams.perm_maxdim,
        spinpair=sdata.mparams.spinpair, 
        spatial=sdata.mparams.spatial, 
        singleham=sdata.mparams.singleham,
        verbose=verbose
    )
    
end


function SolveGenEig!(
        sdata::SubspaceProperties;
        verbose=false
    )
    
    sdata.E, sdata.C, sdata.kappa = SolveGenEig(
        sdata.H_mat,
        sdata.S_mat,
        thresh=sdata.mparams.thresh,
        eps=sdata.mparams.eps
    )
    
    if verbose
        DisplayEvalData(sdata.chem_data, sdata.H_mat, sdata.E, sdata.C, sdata.kappa)
    end
    
end


function ScreenOrderings!(
        sdata::SubspaceProperties;
        maxiter=20,
        M_new=1,
        verbose=false
    )
    
    sdata.psi_list, sdata.ham_list, sdata.ord_list = ScreenOrderings(
        sdata.chem_data, 
        sdata.sites, 
        sdata.dflt_sweeps, 
        sdata.mparams.M; 
        maxiter=maxiter, 
        M_new=M_new, 
        verbose=verbose
    )
    
end


function BBOptimizeStates!(
        sdata::SubspaceProperties;
        loops=1,
        sweeps=1,
        verbose=false
    )
    
    sdata.psi_list = BBOptimizeStates(
        sdata.chem_data,
        sdata.psi_list,
        sdata.ham_list,
        sdata.ord_list;
        tol=sdata.mparams.psi_tol,
        maxdim=sdata.mparams.psi_maxdim,
        perm_tol=sdata.mparams.perm_tol,
        perm_maxdim=sdata.mparams.perm_maxdim,
        loops=loops,
        sweeps=sweeps,
        thresh=sdata.mparams.thresh,
        eps=sdata.mparams.eps,
        verbose=verbose
    )
    
end