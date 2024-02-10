
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





# Functions for constructing (and modifying) the subspace data structure

# Packages:
using ITensors
using HDF5
import Base: copy


@with_kw mutable struct MetaParameters
    M::Int=1 # Number of MPS reference states
    psi_maxdim::Int=4 # Maximum bond dimension
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
    psi_list::Vector{MPS}
    ord_list::Vector{Vector{Int}}
    ham_mpo::MPO
    ham_ord::Vector{Int}
    eye_mpo::MPO
    perm_ops::Vector{MPO}
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
        ham_ord=nothing,
        # MPS/MPO constructor parameters:
        psi_maxdim=4,
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
        psi_maxdim,
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
        maxdim!(dflt_sweeps,psi_maxdim)
        mindim!(dflt_sweeps,psi_maxdim)
        #cutoff!(dflt_sweeps,psi_tol)
        setnoise!(dflt_sweeps, sweep_noise...)
    end
    
    # Generate the Hamiltonian:
    if ham_ord==nothing
        ham_ord = collect(1:chem_data.N_spt)
    end
    
    verbose && println("\nGenerating Hamiltonian MPO:")
    
    opsum = GenOpSum(chem_data, ham_ord)
    ham_mpo = MPO(opsum, sites, cutoff=ham_tol, maxdim=ham_maxdim)
    
    verbose && println("Done!\n")
    
    # Generate initial matrix product states:
    psi_list = MPS[]
    hf_occ = [FillHF(ham_ord[p], chem_data.N_el) for p=1:chem_data.N_spt]
    
    verbose && println("\nGenerating states:")
    
    for i=1:M
        push!(psi_list, randomMPS(sites, hf_occ, linkdims=psi_maxdim))
    end
    
    if dmrg_init
        for i=1:M
            if ovlp_opt && i > 1
                _, psi_list[i] = dmrg(ham_mpo, psi_list[1:i-1], psi_list[i], dflt_sweeps, outputlevel=0, weight=ovlp_weight)
            else
                _, psi_list[i] = dmrg(ham_mpo, psi_list[i], dflt_sweeps, outputlevel=0)
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
        psi_list,
        [deepcopy(ham_ord) for i=1:M],
        ham_mpo,
        ham_ord,
        MPO(sites, "I"),
        [MPO(sites, "I") for i=1:M],
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
        sd.mparams,
        sd.sites,
        sd.dflt_sweeps,
        deepcopy(sd.psi_list),
        deepcopy(sd.ord_list),
        deepcopy(sd.ham_mpo),
        deepcopy(sd.ham_ord),
        deepcopy(sd.eye_mpo),
        deepcopy(sd.perm_ops),
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
    sd1.ham_mpo = deepcopy(sd2.ham_mpo)
    sd1.perm_ops = deepcopy(sd2.perm_ops)
    sd1.H_mat = deepcopy(sd2.H_mat)
    sd1.S_mat = deepcopy(sd2.S_mat)
    sd1.E = deepcopy(sd2.E)
    sd1.C = deepcopy(sd2.C)
    sd1.kappa = deepcopy(sd2.kappa)
    
end


function GenPermOps!(
        sd::SubspaceProperties;
        no_rev=false,
        verbose=false
    )
    
    M = sd.mparams.M
    
    if verbose
        println("Generating permutation operators:")
    end
    
    for i=1:M
        
        sd.perm_ops[i], rev_flag = FastPMPO(
            sd.sites,
            sd.ord_list[i],
            sd.ham_ord,
            tol=sd.mparams.perm_tol,
            no_rev=no_rev,
            maxdim=sd.mparams.perm_maxdim
        )
        
        if rev_flag
            reverse!(sd.ord_list[i])
            sd.psi_list[i] = ReverseMPS(sd.psi_list[i])
        end
        
        if verbose
            print("Progress: [$(i)/$(M)] \r")
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
    
    # re-initialize:
    sd.H_mat = zeros((M,M))
    sd.S_mat = zeros((M,M))
    
    for i=1:M, j=i:M
        
        sd.H_mat[i,j] = FullContract(sd, sd.ham_mpo, i, j)
        sd.S_mat[i,j] = FullContract(sd, sd.eye_mpo, i, j)
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
        perm=nothing
    )
    
    if perm==nothing # Random shuffle
        perm = randperm(sd.mparams.M)
    end
    
    permute!(sd.ord_list, perm)
    permute!(sd.psi_list, perm)
    permute!(sd.perm_ops, perm)
    
    GenSubspaceMats!(sd)
    SolveGenEig!(sd)
    
end


function ReverseAll!(sdata)
    reverse!(sdata.ham_ord)
    sdata.ham_mpo = ReverseMPO(sdata.ham_mpo)
    for i=1:sdata.mparams.M
        reverse!(sdata.ord_list[i])
        sdata.psi_list[i] = ReverseMPS(sdata.psi_list[i])
    end
    GenPermOps!(sdata)
end


# Generate the Hamiltonian MPOs
function GenHams!(
        sd::SubspaceProperties;
        j_set=nothing
    )
    
    if j_set==nothing
        j_set=collect(1:sd.mparams.M)
        clear=true
        sd.ham_list = []
    else
        clear=false
    end
    
    for j in j_set
        
        opsum = GenOpSum(
            sd.chem_data, 
            sd.ord_list[j]
        )

        if clear
            push!(
                sd.ham_list, 
                MPO(
                    opsum, 
                    sd.sites, 
                    cutoff=sd.mparams.ham_tol, 
                    maxdim=sd.mparams.ham_maxdim
                )
            )
        else
            sd.ham_list[j] = MPO(
                opsum, 
                sd.sites, 
                cutoff=sd.mparams.ham_tol, 
                maxdim=sd.mparams.ham_maxdim
            )
        end
        
    end
    
    
end


function GenPermOps!(
        sd::SubspaceProperties;
        no_rev=false,
        tol=sd.maprams.perm_tol,
        maxdim=sd.maprams.perm_maxdim,
        verbose=false
    )
    
    M = sd.mparams.M
    
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
        ham_i = deepcopy(sd.ham_list[i])
        
        if sd.rev_flag[i][j-i]
            psi_j = ReverseMPS(sd.psi_list[j])
            ham_j = ReverseMPO(sd.ham_list[j])
        else
            psi_j = deepcopy(sd.psi_list[j])
            ham_j = deepcopy(sd.ham_list[j])
        end
        
        pmpo_ij = sd.perm_ops[i][j-i]
        
        sd.H_mat[i,j] = inner(pmpo_ij, psi_j, ham_i, psi_i)
        sd.S_mat[i,j] = inner(psi_i', pmpo_ij, psi_j)
        sd.H_mat[j,i] = sd.H_mat[i,j]
        sd.S_mat[j,i] = sd.S_mat[i,j]
        
    end
    
end