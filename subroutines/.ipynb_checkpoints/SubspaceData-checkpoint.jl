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
        chem_data;
        M=1,
        stype="Electron",
        init_ord=nothing,
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
    if init_ord==nothing
        init_ord = collect(1:chem_data.N_spt)
    end
    
    verbose && println("\nGenerating Hamiltonian MPO:")
    
    opsum = GenOpSum(chem_data, init_ord)
    ham_mpo = MPO(opsum, sites, cutoff=ham_tol, maxdim=ham_maxdim)
    
    ham_list = [deepcopy(ham_mpo) for i=1:M]
    
    verbose && println("Done!\n")
    
    # Generate initial matrix product states:
    psi_list = MPS[]
    hf_occ = [FillHF(init_ord[p], chem_data.N_el) for p=1:chem_data.N_spt]
    
    verbose && println("\nGenerating states:")
    
    for i=1:M
        push!(psi_list, randomMPS(sites, hf_occ, linkdims=psi_maxdim))
    end
    
    if dmrg_init
        for i=1:M
            if ovlp_opt && i > 1
                _, psi_list[i] = dmrg(ham_list[i], psi_list[1:i-1], psi_list[i], dflt_sweeps, outputlevel=0, weight=ovlp_weight)
            else
                _, psi_list[i] = dmrg(ham_list[i], psi_list[i], dflt_sweeps, outputlevel=0)
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
        [deepcopy(init_ord) for i=1:M],
        ham_list,
        [[MPO(sites, "I") for j=i+1:M] for i=1:M],
        [[false for j=i+1:M] for i=1:M],
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
        deepcopy(sd.psi_list),
        deepcopy(sd.ord_list),
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


function GenPermOps!(
        sd::SubspaceProperties;
        no_rev=false,
        tol=sd.mparams.perm_tol,
        maxdim=sd.mparams.perm_maxdim,
        verbose=false
    )
    
    M = sd.mparams.M
    
    verbose && println("Generating permutation operators:")
    
    c = 0
    c_tot = Int((M^2-M)/2)
    
    # Permute identity MPOs to obtain permutation MPOs:
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
        
        verbose && print("Progress: [$(c)/$(c_tot)] \r")
        verbose && flush(stdout)
        
    end
    
    # Double-check this:
    if no_rev
        sd.rev_flag = [[false for j=1:M-i] for i=1:M]
    end
    
    verbose && println("\nDone!\n")
    
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
    
    permute!(sd.ord_list, perm)
    permute!(sd.psi_list, perm)
    permute!(sd.ham_list, perm)
    
    GenPermOps!(sd, no_rev=no_rev)
    
    GenSubspaceMats!(sd)
    SolveGenEig!(sd)
    
end


function ReverseAll!(sdata; no_rev=false)
    for i=1:sdata.mparams.M
        sdata.ord_list[i] = reverse(sdata.ord_list[i])
        sdata.psi_list[i] = ReverseMPS(sdata.psi_list[i])
        sdata.ham_list[i] = ReverseMPO(sdata.ham_list[i])
    end
    GenPermOps!(sdata, no_rev=no_rev)
end


function AddStates!(
        sdata;
        M_new=1,
        init_ord=collect(1:sdata.chem_data.N_spt),
        dmrg_init=false,
        ovlp_opt=false,
        ovlp_weight=2.0,
        verbose=false
    )
    
    for j=1:M_new
        
        push!(sdata.ord_list, deepcopy(init_ord))
        
        sdata.mparams.M += 1
        M = sdata.mparams.M
        
        sdata.perm_ops = [[MPO(sdata.sites, "I") for j=i+1:M] for i=1:M]
        sdata.rev_flag = [[false for j=i+1:M] for i=1:M]
        
        GenHams!(sdata)
        
        GenPermOps!(sdata)
        
        verbose && println("\nGenerating states:")
        
        hf_occ = [FillHF(init_ord[p], sdata.chem_data.N_el) for p=1:sdata.chem_data.N_spt]
        new_state = randomMPS(sdata.sites, hf_occ, linkdims=sdata.mparams.psi_maxdim)
        normalize!(new_state)
        push!(sdata.psi_list, new_state)

        if dmrg_init
            if ovlp_opt
                _, sdata.psi_list[M] = dmrg(sdata.ham_list[M], deepcopy(sdata.psi_list[1:M-1]), deepcopy(sdata.psi_list[M]), sdata.dflt_sweeps, outputlevel=0, weight=ovlp_weight)
            else
                _, sdata.psi_list[M] = dmrg(sdata.ham_list[M], deepcopy(sdata.psi_list[M]), sdata.dflt_sweeps, outputlevel=0)
            end

            if verbose
                print("Progress: [$(j)/$(M_new)] \r")
                flush(stdout)
            end
        end
        
        verbose && println("\nDone!\n")
        
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


"""
function PopStates!(
        sdata;
        jset=[1]
    )
    
    
    
end
"""


function AddStates2(
        sdata;
        M_new=1,
        init_ord=collect(1:sdata.chem_data.N_spt),
        dmrg_init=false,
        ovlp_opt=false,
        ovlp_weight=2.0,
        verbose=false
    )
    
    new_ords = []
    new_states = []
    
    for j=1:M_new
        
        push!(new_ords, deepcopy(init_ord))
        
        #verbose && println("\nGenerating states:")
        
        hf_occ = [FillHF(init_ord[p], sdata.chem_data.N_el) for p=1:sdata.chem_data.N_spt]
        new_state = randomMPS(sdata.sites, hf_occ, linkdims=sdata.mparams.psi_maxdim)
        normalize!(new_state)
        push!(new_states, new_state)
        
    end
    
    mparams = MetaParameters(
        sdata.mparams.M + M_new,
        sdata.mparams.psi_maxdim,
        sdata.mparams.ham_maxdim,
        sdata.mparams.ham_tol,
        sdata.mparams.perm_maxdim,
        sdata.mparams.perm_tol,
        sdata.mparams.thresh,
        sdata.mparams.eps
    )
    
    new_sdata = SubspaceProperties(
        sdata.chem_data,
        mparams,
        sdata.sites,
        sdata.dflt_sweeps,
        vcat(sdata.psi_list, new_states),
        vcat(sdata.ord_list, new_ords),
        [MPO(sdata.sites, "I") for i=1:mparams.M],
        [[MPO(sdata.sites, "I") for j=i+1:mparams.M] for i=1:mparams.M],
        [[false for j=i+1:mparams.M] for i=1:mparams.M],
        zeros((mparams.M,mparams.M)),
        zeros((mparams.M,mparams.M)),
        zeros(mparams.M),
        zeros((mparams.M,mparams.M)),
        0.0
    )
    
    GenHams!(new_sdata)
    GenPermOps!(new_sdata)
    
    # compute H, S, E, C, kappa:
    GenSubspaceMats!(new_sdata)
    SolveGenEig!(new_sdata)
    
    verbose && DisplayEvalData(new_sdata)
    
    return new_sdata
    
end


function SplitStates(
        sdata;
        jset=collect(1:sdata.mparams.M),
        tdim=2
    )
    
    new_ords = []
    new_states = []
    
    for j=1:sdata.mparams.M
        
        push!(new_ords, deepcopy(sdata.ord_list[j]))
        deepcopy(sdata.ord_list[j])
        
        if j in jset
            
            push!(new_ords, deepcopy(sdata.ord_list[j]))
            
            psi_trunc = truncate(sdata.psi_list[j], maxdim=tdim)
            psi_rem = truncate(sdata.psi_list[j]-psi_trunc, maxdim=sdata.mparams.psi_maxdim)
            
            push!(new_states, normalize(psi_trunc))
            push!(new_states, normalize(psi_rem))
            
            """
            hf_occ = [FillHF(sdata.ord_list[j][p], sdata.chem_data.N_el) for p=1:sdata.chem_data.N_spt]
        
            delta_mps = randomMPS(sdata.sites, hf_occ, linkdims=sdata.mparams.psi_maxdim)

            normalize!(delta_mps)
            
            psi_plus = truncate(deepcopy(sdata.psi_list[j]) + 0.5*delta_mps, maxdim=sdata.mparams.psi_maxdim)
            psi_minus = truncate(deepcopy(sdata.psi_list[j]) - 0.5*delta_mps, maxdim=sdata.mparams.psi_maxdim)
            
            push!(new_states, normalize(psi_plus))
            push!(new_states, normalize(psi_minus))
            """
            
        else
            
            push!(new_states, deepcopy(sdata.psi_list[j]))
            
        end
        
    end
    
    mparams = MetaParameters(
        sdata.mparams.M + length(jset),
        sdata.mparams.psi_maxdim,
        sdata.mparams.ham_maxdim,
        sdata.mparams.ham_tol,
        sdata.mparams.perm_maxdim,
        sdata.mparams.perm_tol,
        sdata.mparams.thresh,
        sdata.mparams.eps
    )
    
    new_sdata = SubspaceProperties(
        sdata.chem_data,
        mparams,
        sdata.sites,
        sdata.dflt_sweeps,
        new_states,
        new_ords,
        [MPO(sdata.sites, "I") for i=1:mparams.M],
        [[MPO(sdata.sites, "I") for j=i+1:mparams.M] for i=1:mparams.M],
        [[false for j=i+1:mparams.M] for i=1:mparams.M],
        zeros((mparams.M,mparams.M)),
        zeros((mparams.M,mparams.M)),
        zeros(mparams.M),
        zeros((mparams.M,mparams.M)),
        0.0
    )
    
    GenHams!(new_sdata)
    GenPermOps!(new_sdata)
    
    # compute H, S, E, C, kappa:
    GenSubspaceMats!(new_sdata)
    SolveGenEig!(new_sdata)
    
    #verbose && DisplayEvalData(new_sdata)
    
    return new_sdata
    
end