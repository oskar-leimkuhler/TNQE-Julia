# Functions for constructing (and modifying) the subspace data structure

# Packages:
using ITensors
using HDF5
import Base: copy


struct MetaParameters
    # Subspace dimension:
    M::Int
    M_ex::Int
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
    rev_list::Vector{Bool}
    psi_list::Vector{MPS}
    ham_list::Vector{MPO}
    perm_ops::Vector{Vector{MPO}}
    perm_alt::Vector{Vector{Vector{MPO}}}
    #perm_hams::Vector{Vector{MPO}}
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
        1,
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
        mindim!(dflt_sweeps,psi_maxdim)
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
        [false for j=1:mparams.M],
        [],
        [],
        [],
        [],
        #[],
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
        deepcopy(sdata.rev_list),
        deepcopy(sdata.psi_list),
        deepcopy(sdata.ham_list),
        deepcopy(sdata.perm_ops),
        deepcopy(sdata.perm_alt),
        #deepcopy(sdata.perm_hams),
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
        sweeps=nothing,
        weight=1.0,
        prior_states=[],
        prior_ords=[],
        verbose=false
    )
    
    if sweeps==nothing
        sweeps = sdata.dflt_sweeps
    end
    
    # Generate a set of states:
    sdata.psi_list, sdata.ham_list = GenStates(
        sdata.chem_data, 
        sdata.sites, 
        sdata.ord_list, 
        sweeps, 
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


function GenExcited!(
        sdata::SubspaceProperties;
        sweeps=nothing,
        weight=1.0,
        levs=nothing,
        verbose=false
    )
    
    if levs==nothing
        levs = collect(1:sdata.mparams.M)
    end
    
    psi_list = []
    ham_list = []
    
    for j=1:sdata.mparams.M
        
        ords = [sdata.ord_list[j] for k=1:levs[j]]
        
        # Generate a set of states:
        psis, hams = GenStates(
            sdata.chem_data, 
            sdata.sites, 
            ords, 
            sweeps, 
            ovlp_opt=true,
            weight=weight,
            perm_tol=sdata.mparams.perm_tol, 
            perm_maxdim=sdata.mparams.perm_maxdim, 
            ham_tol=sdata.mparams.ham_tol, 
            ham_maxdim=sdata.mparams.ham_maxdim, 
            spinpair=sdata.mparams.spinpair, 
            spatial=sdata.mparams.spatial, 
            singleham=sdata.mparams.singleham,
            verbose=verbose
        )
        
        push!(psi_list, psis[end])
        push!(ham_list, hams[1])
        
    end
    
    sdata.psi_list = psi_list
    sdata.ham_list = ham_list
    
end


function SchmidtRankTruncate!(
        sdata::SubspaceProperties;
        verbose=false
    )
    
    M = sdata.mparams.M
    
    m_eff = 4*sdata.mparams.psi_maxdim
    
    # For each state, maximally entangle at each bond:
    for j=1:M
        
        psi = sdata.psi_list[j]
        
        for s=1:2
            
            orthogonalize!(psi,1)
        
            for p=1:sdata.chem_data.N_spt-1

                temp_tensor = (psi[p] * psi[p+1])
                temp_inds = uniqueinds(psi[p], psi[p+1])

                U,S,V = svd(temp_tensor, temp_inds, min_blockdim=m_eff, alg="recursive")

                m_j = length(diag(Array(S.tensor)))

                ids = inds(S)

                sig = minimum([m_j, j])

                for k=1:m_j

                    """
                    if k==1
                        S[ids[1]=>k,ids[2]=>k] = 1.0
                    else
                        S[ids[1]=>k,ids[2]=>k] = 1.0
                    end
                    """

                    S[ids[1]=>k,ids[2]=>k] = 1.0/sqrt(m_j)

                end

                temp_block = U*S*V

                # Replace the tensors of the MPS:
                spec = ITensors.replacebond!(
                    psi,
                    p,
                    temp_block;
                    #maxdim=sdata.mparams.psi_maxdim,
                    min_blockdim=sdata.mparams.psi_maxdim,
                    ortho="left",
                    normalize=true,
                    svd_alg="qr_iteration"
                )

            end

            truncate!(sdata.psi_list[j], maxdim=sdata.mparams.psi_maxdim)
            
        end
        
    end
    
end


function ApplyPhases!(mpo)
    
    #Apply phases:
    phase_mat = [1 0 0 0; 
                 0 1 0 0;
                 0 0 1 0;
                 0 0 0 -1]
            
    for p=1:length(mpo)

        mpo_inds = siteinds(mpo)[p]
        
        phase_gate = ITensor(phase_mat, dag(mpo_inds[1]), dag(mpo_inds[2]))
        setprime!(phase_gate, 2, plev=0)
        
        mpo[p] = mpo[p] * phase_gate
        setprime!(mpo[p], 1, plev=2)
        
    end
    
end


function GenPermOps!(
        sdata::SubspaceProperties;
        compute_alternates=true,
        verbose=false
    )
    
    M = sdata.mparams.M
    
    # Construct identity MPOs:
    #sdata.perm_ops = [[MPO(sdata.sites, "I") for j=1:M-i] for i=1:M]
    sdata.perm_alt = [[[MPO(sdata.sites, "I") for j=1:M-i] for i=1:M] for pa=1:4]
    
    #sdata.perm_hams = [[sdata.ham_list[i] for j=1:M-i] for i=1:M]
    
    if verbose
        println("Generating permutation operators:")
    end
    
    c = 0
    c_tot = Int((M^2-M)/2)
    
    # Permute the identity MPOs to obtain \\
    # ...permutation MPOs:
    for i=1:M, j=1:M-i
        
        sdata.perm_alt[1][i][j] = PermuteMPO(
            sdata.perm_alt[1][i][j],
            sdata.sites,
            sdata.ord_list[i],
            sdata.ord_list[j+i],
            tol=1e-16
        )
        
        # Alternate (flipped) configurations:
        if compute_alternates
            
            sdata.perm_alt[2][i][j] = PermuteMPO(
                sdata.perm_alt[2][i][j],
                sdata.sites,
                sdata.ord_list[i],
                reverse(sdata.ord_list[j+i]),
                tol=1e-16
            )

            sdata.perm_alt[3][i][j] = ReverseMPO(sdata.perm_alt[2][i][j])

            sdata.perm_alt[4][i][j] = ReverseMPO(sdata.perm_alt[1][i][j])
            
            # Apply phases:
            #phase_mpo = MPO(sdata.sites, "CZ")

            ApplyPhases!(sdata.perm_alt[3][i][j]) #sdata.perm_alt[3][i][j] = apply(sdata.perm_alt[3][i][j], phase_mpo)
            ApplyPhases!(sdata.perm_alt[4][i][j]) #sdata.perm_alt[4][i][j] = apply(sdata.perm_alt[4][i][j], phase_mpo)

            dag!(swapprime!(sdata.perm_alt[2][i][j],0,1))
            dag!(swapprime!(sdata.perm_alt[4][i][j],0,1))

            ApplyPhases!(sdata.perm_alt[2][i][j]) #sdata.perm_alt[2][i][j] = apply(sdata.perm_alt[2][i][j], phase_mpo)
            ApplyPhases!(sdata.perm_alt[4][i][j]) #sdata.perm_alt[4][i][j] = apply(sdata.perm_alt[4][i][j], phase_mpo)

            dag!(swapprime!(sdata.perm_alt[2][i][j],0,1))
            dag!(swapprime!(sdata.perm_alt[4][i][j],0,1))
        
        end
        
        #sdata.perm_hams[i][j] = apply(sdata.perm_hams[i][j], sdata.perm_ops[i][j], cutoff=1e-12)
        
        c += 1
        
        if verbose
            print("Progress: [$(c)/$(c_tot)] \r")
            flush(stdout)
        end
        
    end
    
    # Set everything to non-flipped config:
    sdata.perm_ops = [[deepcopy(sdata.perm_alt[1][i][j]) for j=1:M-i] for i=1:M]
    
    if verbose
        println("\nDone!\n")
    end
    
end


function GenHams!(sdata)
    
    # Update the Hamiltonian MPOs:
    for j=1:sdata.mparams.M
        opsum = GenOpSum(sdata.chem_data, sdata.ord_list[j])
        sdata.ham_list[j] = MPO(opsum, sdata.sites, cutoff=sdata.mparams.ham_tol, maxdim=sdata.mparams.ham_maxdim);
        if sdata.rev_list[j]
            sdata.ham_list[j] = ReverseMPO(sdata.ham_list[j])
        end
    end
    
end


# Apply FSWAP tensors to modify the permutation operators for state k:
# [May change later to include alternates]
function UpdatePermOps!(
        sdata::SubspaceProperties,
        new_ord_list
    )
    
    M = sdata.mparams.M
    
    # Update the permutation operators:
    for i=1:M, j=1:M-i
        
        # Un-apply the phases:
        ApplyPhases!(sdata.perm_alt[3][i][j])
        ApplyPhases!(sdata.perm_alt[4][i][j])

        dag!(swapprime!(sdata.perm_alt[2][i][j],0,1))
        dag!(swapprime!(sdata.perm_alt[4][i][j],0,1))

        ApplyPhases!(sdata.perm_alt[2][i][j])
        ApplyPhases!(sdata.perm_alt[4][i][j])

        dag!(swapprime!(sdata.perm_alt[2][i][j],0,1))
        dag!(swapprime!(sdata.perm_alt[4][i][j],0,1))
        
        for f=1:4
        
            ords_i = [deepcopy(sdata.ord_list[i]), deepcopy(new_ord_list[i])]
            ords_j = [deepcopy(sdata.ord_list[j+i]), deepcopy(new_ord_list[j+i])]

            if f==2 || f==4
                ords_j = [reverse(ords_j[k]) for k=1:2]
            end
            
            if f==3 || f==4
                ords_i = [reverse(ords_i[k]) for k=1:2]
            end
            
            sdata.perm_alt[f][i][j] = PermuteMPO(
                sdata.perm_alt[f][i][j],
                sdata.sites,
                ords_j[1],
                ords_j[2],
                tol=1e-16
            )
            
            dag!(swapprime!(sdata.perm_alt[f][i][j], 0,1))
            
            sdata.perm_alt[f][i][j] = PermuteMPO(
                sdata.perm_alt[f][i][j],
                sdata.sites,
                ords_i[1],
                ords_i[2],
                tol=1e-16
            )

            dag!(swapprime!(sdata.perm_alt[f][i][j], 0,1))
            
        end
        
        # Re-apply the phases:
        ApplyPhases!(sdata.perm_alt[3][i][j])
        ApplyPhases!(sdata.perm_alt[4][i][j])

        dag!(swapprime!(sdata.perm_alt[2][i][j],0,1))
        dag!(swapprime!(sdata.perm_alt[4][i][j],0,1))

        ApplyPhases!(sdata.perm_alt[2][i][j])
        ApplyPhases!(sdata.perm_alt[4][i][j])

        dag!(swapprime!(sdata.perm_alt[2][i][j],0,1))
        dag!(swapprime!(sdata.perm_alt[4][i][j],0,1))
        
        
    end
    
    SelectPermOps!(sdata)
    
end


function SelectPermOps!(
        sdata::SubspaceProperties
    )
    
    M = sdata.mparams.M
    
    phase_mpo = MPO(sdata.sites, "CZ")
    
    for i=1:M, j=1:M-i
        
        if !(sdata.rev_list[i]) && !(sdata.rev_list[j+i])
            # Non-flipped config:
            sdata.perm_ops[i][j] = deepcopy(sdata.perm_alt[1][i][j])
            
        elseif !(sdata.rev_list[i]) && sdata.rev_list[j+i]
            # Right-flipped config:
            sdata.perm_ops[i][j] = deepcopy(sdata.perm_alt[2][i][j])
            
            """
            dag!(swapprime!(sdata.perm_ops[i][j],0,1))
            sdata.perm_ops[i][j] = apply(sdata.perm_ops[i][j], phase_mpo)
            dag!(swapprime!(sdata.perm_ops[i][j],0,1))
            """
            
        elseif sdata.rev_list[i] && !(sdata.rev_list[j+i])
            # Left-flipped config:
            sdata.perm_ops[i][j] = deepcopy(sdata.perm_alt[3][i][j])
            
            """
            sdata.perm_ops[i][j] = apply(sdata.perm_ops[i][j], phase_mpo)
            """
            
        elseif sdata.rev_list[i] && sdata.rev_list[j+i]
            # Double-flipped config:
            sdata.perm_ops[i][j] = deepcopy(sdata.perm_alt[4][i][j])
            
            """
            sdata.perm_ops[i][j] = apply(sdata.perm_ops[i][j], phase_mpo)
            dag!(swapprime!(sdata.perm_ops[i][j],0,1))
            sdata.perm_ops[i][j] = apply(sdata.perm_ops[i][j], phase_mpo)
            dag!(swapprime!(sdata.perm_ops[i][j],0,1))
            """
            
        end
        
    end
    
end


function ReverseStates!(
        sdata::SubspaceProperties,
        j_list::Vector{Int};
        verbose=false
    )
    
    if verbose
        println("Reversing states...")
    end
    
    for j in j_list
        sdata.psi_list[j] = ReverseMPS(sdata.psi_list[j])
        sdata.ham_list[j] = ReverseMPO(sdata.ham_list[j])
        sdata.rev_list[j] = !(sdata.rev_list[j])
    end
    
    if verbose
        println("Done!")
    end

    SelectPermOps!(sdata)
    
end


function UnReverseStates!(
        sdata::SubspaceProperties;
        verbose=false
    )
    
    for j=1:sdata.mparams.M
        if sdata.rev_list[j]
            sdata.psi_list[j] = ReverseMPS(sdata.psi_list[j])
            sdata.rev_list[j]
        end
    end
    
    GenPermOps!(sdata, verbose=verbose)
    
end


function GenSubspaceMats!(
        sdata::SubspaceProperties;
        verbose=false
    )
    
    M = sdata.mparams.M
    
    # Diagonal elements:
    for i=1:M
        sdata.H_mat[i,i] = inner(sdata.psi_list[i]', sdata.ham_list[i], sdata.psi_list[i])
        sdata.S_mat[i,i] = 1.0
    end
    
    # Off-diagonal elements:
    for i=1:M, j=i+1:M
        
        sdata.H_mat[i,j] = inner(sdata.perm_ops[i][j-i], sdata.psi_list[j], sdata.ham_list[i], sdata.psi_list[i])
        #sdata.H_mat[i,j] = inner(sdata.psi_list[i]', sdata.perm_hams[i][j-i], sdata.psi_list[j])
        sdata.H_mat[j,i] = sdata.H_mat[i,j]
        
        sdata.S_mat[i,j] = inner(sdata.psi_list[i]', sdata.perm_ops[i][j-i], sdata.psi_list[j])
        sdata.S_mat[j,i] = sdata.S_mat[i,j]
        
    end
    
end


function SolveGenEig!(
        sdata::SubspaceProperties;
        thresh=nothing,
        eps=nothing,
        verbose=false
    )
    
    if thresh==nothing
        thresh=sdata.mparams.thresh
    end
    
    if eps==nothing
        eps=sdata.mparams.eps
    end
    
    sdata.E, sdata.C, sdata.kappa = SolveGenEig(
        sdata.H_mat,
        sdata.S_mat,
        thresh=thresh,
        eps=eps
    )
    
    if verbose
        DisplayEvalData(sdata.chem_data, sdata.H_mat, sdata.E, sdata.C, sdata.kappa)
    end
    
end
    
    
# The subspace "shadow" data structure:
# Preserves all matrix elements between subspace basis states
# without keeping MPS's etc.
mutable struct SubspaceShadow
    chem_data::ChemProperties
    M_list::Vector{Int}
    thresh::String
    eps::Float64
    vec_list::Vector{Vector{Float64}}
    X_list::Vector{Matrix{Float64}}
    H_full::Matrix{Float64}
    S_full::Matrix{Float64}
    H_mat::Matrix{Float64}
    S_mat::Matrix{Float64}
    E::Vector{Float64}
    C::Matrix{Float64}
    kappa::Float64
end


function copy(sh::SubspaceShadow)
    
    sh2 = SubspaceShadow(
        sh.chem_data,
        sh.M_list,
        sh.thresh,
        sh.eps,
        deepcopy(sh.vec_list),
        deepcopy(sh.X_list),
        deepcopy(sh.H_full),
        deepcopy(sh.S_full),
        deepcopy(sh.H_mat),
        deepcopy(sh.S_mat),
        deepcopy(sh.E),
        deepcopy(sh.C),
        sh.kappa
    )
    
    return sh2
    
end


function GenSubspaceMats!(shadow::SubspaceShadow)
    
    M_gm = length(shadow.M_list)
    
    for i=1:M_gm, j=i:M_gm
        
        i0 = sum(shadow.M_list[1:i-1])+1
        i1 = sum(shadow.M_list[1:i])
        
        j0 = sum(shadow.M_list[1:j-1])+1
        j1 = sum(shadow.M_list[1:j])
        
        # For readability:
        vec_list = shadow.vec_list
        X_list = shadow.X_list
        H_full = shadow.H_full
        S_full = shadow.S_full
        
        shadow.H_mat[i,j] = transpose(vec_list[i]) * H_full[i0:i1,j0:j1] * vec_list[j]
        shadow.H_mat[j,i] = shadow.H_mat[i,j]
        
        shadow.S_mat[i,j] = transpose(vec_list[i]) * S_full[i0:i1,j0:j1] * vec_list[j]
        shadow.S_mat[j,i] = shadow.S_mat[i,j]
        
    end
    
end


function SolveGenEig!(
        shadow::SubspaceShadow;
        thresh=nothing,
        eps=nothing,
        verbose=false
    )
    
    if thresh==nothing
        thresh=shadow.thresh
    end
    
    if eps==nothing
        eps=shadow.eps
    end
    
    shadow.E, shadow.C, shadow.kappa = SolveGenEig(
        shadow.H_mat,
        shadow.S_mat,
        thresh=thresh,
        eps=eps
    )
    
    if verbose
        DisplayEvalData(shadow.chem_data, shadow.H_mat, shadow.E, shadow.C, shadow.kappa)
    end
    
end