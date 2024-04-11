# Functions for constructing (and modifying) the subspace data structure

# Packages:
using ITensors
using HDF5
import Base: copy


@with_kw mutable struct MetaParameters
    M::Int=1 # Number of MPS reference states
    mps_maxdim::Int=4 # Maximum bond dimension
    ham_maxdim::Int=2^30 # Maximum Hamiltonian bond dimension
    ham_tol::Float64=1e-14 # Cutoff for Hamiltonian svals
    perm_maxdim::Int=2^30 # Maximum PMPO bond dimension
    perm_tol::Float64=1e-14 # Cutoff for PMPO svals
    thresh::String="projection" # GenEig thresholding
    eps::Float64=1e-10 # GenEig threshold parameter
end


# The subspace data structure:
# Keeps track of a single experimental instance \\
# ...including molecular data, metaparameters, orderings, \\
# ...generated MPSs and MPOs, and subspace diag. results.
mutable struct SubspaceProperties
    chem_data::ChemProperties
    mparams::MetaParameters
    sites::Vector
    dflt_sweeps::Sweeps
    init_ord::Vector{Float64}
    phi_list::Vector{MPS}
    G_list::Vector{SparseMatrixCSC{Float64}}
    H_mpo::MPO
    H_sparse::SparseMatrixCSC{Float64}
    H_mat::Matrix{Float64}
    S_mat::Matrix{Float64}
    E::Vector{Float64}
    C::Matrix{Float64}
    kappa::Float64
end


# Generates a subspace properties struct instance (with modifiable defaults):
function GenSubspace(
        chem_data;
        M=1,
        stype="Electron",
        init_ord=nothing,
        # MPS/MPO constructor parameters:
        mps_maxdim=4,
        ham_maxdim=2^30,
        ham_tol=1e-14,
        # Permutation parameters:
        perm_maxdim=2^30,
        perm_tol=1e-14,
        # Diagonalization parameters:
        thresh="projection",
        eps=1e-10,
        # DMRG settings:
        sites=nothing,
        dflt_sweeps=nothing,
        sweep_num=4,
        sweep_noise=1e-2,
        dmrg_init=true,
        ovlp_opt=false,
        ovlp_weight=2.0,
        verbose=false
    )
    
    # Default metaparameters:
    mparams = MetaParameters(
        M,
        mps_maxdim,
        ham_maxdim,
        ham_tol,
        perm_maxdim,
        perm_tol,
        thresh,
        eps
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
        maxdim!(dflt_sweeps,mps_maxdim)
        mindim!(dflt_sweeps,mps_maxdim)
        #cutoff!(dflt_sweeps,mps_tol)
        setnoise!(dflt_sweeps, sweep_noise...)
    end
    
    # Generate the Hamiltonian:
    if init_ord==nothing
        init_ord = collect(1:chem_data.N_spt)
    end
    
    verbose && println("\nGenerating Hamiltonian MPO:")
    
    opsum = GenOpSum(chem_data, init_ord)
    H_mpo = MPO(opsum, sites, cutoff=ham_tol, maxdim=ham_maxdim)
    
    verbose && println("Done!\n")
    verbose && println("\nGenerating Hamiltonian sparse matrix:")
    
    H_tens = reduce(*, H_mpo);
    mpo_sites = vcat([dag(p_ind) for p_ind in sites],[p_ind' for p_ind in sites])
    
    H_sparse = sparse(reshape(Array(H_tens, mpo_sites), (4^chem_data.N_spt,4^chem_data.N_spt)))
    
    # Project onto the eta-subspace to save on computation:
    eta_vec = sparse(zeros(4^chem_data.N_spt))
    for b=1:4^chem_data.N_spt
        eta_vec[b] = Int(sum(digits(b-1, base=2))==chem_data.N_el)
    end
    eta_proj = sparse(diagm(eta_vec))
    
    H_sparse = eta_proj * H_sparse * eta_proj
    
    #H_sparse, eta_proj = HMatrix(chemical_data)
    
    #display(H_sparse)
    
    verbose && println("Done!\n")
    
    # Generate initial matrix product states:
    phi_list = MPS[]
    hf_occ = [FillHF(init_ord[p], chem_data.N_el) for p=1:chem_data.N_spt]
    
    verbose && println("\nGenerating states:")
    
    for i=1:M
        push!(phi_list, randomMPS(sites, hf_occ, linkdims=mps_maxdim))
    end
    
    if dmrg_init
        for i=1:M
            if ovlp_opt && i > 1
                _, phi_list[i] = dmrg(H_mpo, phi_list[1:i-1], phi_list[i], dflt_sweeps, outputlevel=0, weight=ovlp_weight)
            else
                _, phi_list[i] = dmrg(H_mpo, phi_list[i], dflt_sweeps, outputlevel=0)
            end
            
            if verbose
                print("Progress: [$(i)/$(M)] \r")
                flush(stdout)
            end
        end
    end
    
    verbose && println("\nDone!\n")
    
    # Initialize subspace data structure:
    sdata = SubspaceProperties(
        chem_data,
        mparams,
        sites,
        dflt_sweeps,
        init_ord,
        phi_list,
        [eta_proj for i=1:length(phi_list)],
        H_mpo,
        H_sparse,
        zeros((mparams.M,mparams.M)),
        zeros((mparams.M,mparams.M)),
        zeros(mparams.M),
        zeros((mparams.M,mparams.M)),
        0.0
    )
    
    # compute H, S, E, C, kappa:
    GenSubspaceMats!(sdata)
    SolveGenEig!(sdata)
    
    verbose && DisplayEvalData(sdata)
    
    return sdata
    
end


function copy(sd::SubspaceProperties)
    
    subspace_copy = SubspaceProperties(
        sd.chem_data,
        deepcopy(sd.mparams),
        sd.sites,
        sd.dflt_sweeps,
        sd.init_ord,
        deepcopy(sd.phi_list),
        deepcopy(sd.G_list),
        deepcopy(sd.H_mpo),
        deepcopy(sd.H_sparse),
        deepcopy(sd.H_mat),
        deepcopy(sd.S_mat),
        deepcopy(sd.E),
        deepcopy(sd.C),
        sd.kappa
    )
    
    return subspace_copy
    
end


function copyto!(sd1::SubspaceProperties, sd2::SubspaceProperties)
    
    sd1.phi_list = deepcopy(sd2.phi_list)
    sd1.G_list = deepcopy(sd2.G_list)
    sd1.H_mat = deepcopy(sd2.H_mat)
    sd1.S_mat = deepcopy(sd2.S_mat)
    sd1.E = deepcopy(sd2.E)
    sd1.C = deepcopy(sd2.C)
    sd1.kappa = deepcopy(sd2.kappa)
    
end


# Converts an MPS into a sparse vector:
function SparseVec(phi)
    phi_tens = reduce(*, phi);
    phi_vec = sparse(reshape(Array(phi_tens, siteinds(phi)), (4^length(phi))))
    return phi_vec
end


function GenSubspaceMats!(
        sd::SubspaceProperties;
        verbose=false
    )
    
    M = sd.mparams.M
    
    for i=1:M, j=i:M
        
        vec_i = sd.G_list[i] * SparseVec(sd.phi_list[i])
        vec_j = sd.G_list[j] * SparseVec(sd.phi_list[j])
        
        sd.H_mat[i,j] = transpose(vec_i) * sd.H_sparse * vec_j
        sd.S_mat[i,j] = transpose(vec_i) * vec_j
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


function ShuffleStates!(
        sd::SubspaceProperties;
        no_rev=false,
        perm=nothing
    )
    
    if perm==nothing # Random shuffle
        perm = randperm(sd.mparams.M)
    end
    
    permute!(sd.phi_list, perm)
    permute!(sd.G_list, perm)
    
    GenSubspaceMats!(sd)
    SolveGenEig!(sd)
    
end

"""
function ReverseAll!(sdata; no_rev=false)
    for i=1:sdata.mparams.M
        sdata.ord_list[i] = reverse(sdata.ord_list[i])
        sdata.phi_list[i] = ReverseMPS(sdata.phi_list[i])
        sdata.ham_list[i] = ReverseMPO(sdata.ham_list[i])
    end
end
"""

function AddStates!(
        sdata;
        M_new=1,
        G_init="last",
        dmrg_init=false,
        ovlp_opt=false,
        ovlp_weight=2.0,
        verbose=false
    )
    
    N = sdata.chem_data.N_spt
    
    for j=1:M_new
        
        sdata.mparams.M += 1
        M = sdata.mparams.M
        
        hf_occ = [FillHF(sdata.init_ord[p], sdata.chem_data.N_el) for p=1:sdata.chem_data.N_spt]
        new_state = randomMPS(sdata.sites, hf_occ, linkdims=sdata.mparams.mps_maxdim)
        normalize!(new_state)
        
        if dmrg_init
            if ovlp_opt
                _, opt_state = dmrg(sdata.H_mpo, sdata.phi_list, new_state, sdata.dflt_sweeps, outputlevel=0, weight=ovlp_weight)
            else
                _, opt_state = dmrg(sdata.H_mpo, new_state, sdata.dflt_sweeps, outputlevel=0)
            end
            new_state = opt_state
        end
        
        
        push!(sdata.phi_list, new_state)
        
        if G_init=="dflt"
            push!(sdata.G_list, sparse(I,4^N,4^N))
        elseif G_init=="last"
            push!(sdata.G_list, deepcopy(sdata.G_list[M-1]))
        end
        
        sdata.H_mat = zeros(M,M)
        sdata.S_mat = zeros(M,M)
        sdata.E = zeros(M)
        sdata.C = zeros(M,M)

        # compute H, S, E, C, kappa:
        GenSubspaceMats!(sdata)
        SolveGenEig!(sdata)
        
    end
    
    verbose && DisplayEvalData(sdata)
    
end