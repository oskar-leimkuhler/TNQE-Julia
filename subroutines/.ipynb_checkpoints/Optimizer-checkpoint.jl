# QPU-efficient site-by-site multi-geometry optimization functions:


# Optimizer function parameters data structure:
@with_kw mutable struct OptimParameters
    
    maxiter::Int=100 # Number of iterations
    numloop::Int=1 # Number of loops per iter
    numopt::Int=-1 # Number of states/pairs to optimize per iter
    
    # Site decomposition parameters:
    noise::Vector{Float64}=[1e-5] # Size of DMRG noise term at each iter
    delta::Vector{Float64}=[1e-3] # Size of Gaussian noise term
    theta::Float64=0.0 # Weight of the old state in the superposition 
    ttol::Float64=0.1 # Norm tolerance for truncated site tensors
    swap_mult::Float64=1.0 # "Swappiness"
    #gamma::Float64=1.0 # Selectivity of bond SWAP favorability
    
    # Generalized eigenvalue solver parameters:
    thresh::String="inversion" # "none", "projection", or "inversion"
    eps::Float64=1e-8 # Singular value cutoff
    
    # Site decomposition solver parameters:
    sd_method::String="geneig" # "geneig" or "triple_geneig"
    sd_thresh::String="inversion" # "none", "projection", or "inversion"
    sd_eps::Float64=1e-8 # Singular value cutoff
    sd_reps::Int=3 # Number of TripleGenEig repetitions
    sd_penalty::Float64=1.0 # Penalty factor for truncation error
    sd_swap_penalty::Float64=1.0 # Penalty factor for truncation error
    sd_dtol::Float64=1e-4 # OHT-state overlap discard tolerance
    sd_etol::Float64=1e-4 # Energy penalty tolerance
    
end


# This function takes a one-site or two-site tensor and returns a \\
# ...list of one-hot tensors:
function OneHotTensors(T; discards=[])
    
    T_inds = inds(T)
    k = 1
    
    oht_list = []
    
    # Generate a list of index values for the one-hot states \\
    # ...of appropriate N_el, S_z symmetry:
    for c in CartesianIndices(Array(T, T_inds))
        c_inds = Tuple(c)
        ivs = []
        for i=1:length(T_inds)
            push!(ivs, T_inds[i]=>c_inds[i])
        end
        
        oht = onehot(Tuple(ivs)...)
        
        if (flux(oht)==flux(T))
            if !(k in discards)
                push!(oht_list, oht)
            end
            k += 1
        end
    end
    
    return oht_list
    
end


# Check if states are overlapping too much and discard if they are:
function DiscardOverlapping(H_in, S_in, M_in, oht_in; tol=0.01, kappa_max=1e10, verbose=false)
    
    H_full = deepcopy(H_in)
    S_full = deepcopy(S_in)
    
    M_list = deepcopy(M_in)
    
    M = length(M_list)
    M_tot = sum(M_list)
    
    oht_list = deepcopy(oht_in)
    
    # Iterate over the comp index objects in oht_list:
    for j=M:(-1):1
        
        M_j = M_list[j]
        
        # Iterate over the states in the comp index object:
        for k=M_j:(-1):1
            
            # j0 and j1, keeping track of discards:
            j0, j1 = sum(M_list[1:j-1])+1, sum(M_list[1:j])
            
            # The current column of S_full, keeping track of discards:
            col = j0 + k - 1
            
            do_discard = false
            
            if (M_list[j] > 1)
                
                # Check for any Infs or NaNs in that column:
                if (Inf in S_full[:,col]) || (true in isnan.(S_full[:,col])) || (Inf in H_full[:,col]) || (true in isnan.(H_full[:,col]))
                    do_discard = true
                end
                
            end
                
            if (j != M) && (M_list[j] > 1)
                
                # First check the overlap with the previous subspace is not too large:
                S_red = deepcopy(S_full[j1+1:M_tot,j1+1:M_tot])
                vphi = deepcopy(S_full[j1+1:M_tot,col])
                
                F = svd(S_red, alg=LinearAlgebra.QRIteration())
                rtol = sqrt(eps(real(float(oneunit(eltype(S_red))))))
                S_inv = zeros(length(F.S))
                for l=1:length(F.S)
                    if F.S[l] >= maximum(F.S)*rtol
                        S_inv[l] = 1.0/F.S[l]
                    else
                        S_inv[l] = 0.0
                    end
                end
                S_red_inv = transpose(F.Vt) * Diagonal(S_inv) * transpose(F.U) 

                sqnm = transpose(vphi) * S_red_inv * vphi

                # Also double-check that the condition number does not blow up:
                kappa_new = cond(S_full[col:M_tot,col:M_tot])
                
                # Mark the state for discarding:
                if (sqnm > 1.0-tol) || (kappa_new > kappa_max) || isnan(kappa_new) || (kappa_new == Inf)
                    do_discard=true
                end
                
            end
            
            if do_discard
                
                H_full = H_full[1:end .!= col, 1:end .!= col]
                S_full = S_full[1:end .!= col, 1:end .!= col]
                
                oht_list[j] = oht_list[j][1:end .!= k]
                
                M_list[j] -= 1
                M_tot -= 1
                
            end
            
        end # loop over k
        
    end # loop over j
            
    return H_full, S_full, M_list, oht_list
    
end


function FSWAPModify(H_in, S_in, M_list, oht_list, sites, nsite, q_set, do_swaps)
    
    H_full = deepcopy(H_in)
    S_full = deepcopy(S_in)
    
    M = length(M_list)
    
    for i1=1:M
        
        if nsite[i1]==2

            i10 = sum(M_list[1:i1-1])+1
            i11 = sum(M_list[1:i1])

            if do_swaps[i1]

                # Construct FSWAP matrix:
                fswap = BuildFermionicSwap(sites, q_set[i1][1]; dim=4);
                fswap_mat = zeros(M_list[i1], M_list[i1])
                for k1=1:M_list[i1], k2=1:M_list[i1]
                    fswap_mat[k1,k2] = scalar(oht_list[i1][k1] * fswap * dag(setprime(oht_list[i1][k2],1, tags="Site")))
                end

                for i2=1:M

                    i20 = sum(M_list[1:i2-1])+1
                    i21 = sum(M_list[1:i2])

                    # Left-mult all subblocks in row i2:
                    H_full[i10:i11,i20:i21] = fswap_mat * H_full[i10:i11, i20:i21]
                    S_full[i10:i11,i20:i21] = fswap_mat * S_full[i10:i11, i20:i21]

                    # Right-mult all subblocks in col i2:
                    H_full[i20:i21,i10:i11] = H_full[i20:i21, i10:i11] * fswap_mat
                    S_full[i20:i21,i10:i11] = S_full[i20:i21, i10:i11] * fswap_mat

                end

            end
            
        end

    end
    
    return H_full, S_full
    
end


# Performs a sequence of alternating single-site decompositions \\
# ...to minimize truncation error from the two-site decomposition:
function TripleGenEigM(
        H_full,
        S_full,
        oht_list,
        T_init,
        M_list,
        op,
        nsite,
        ids,
        maxdim;
        nrep=3
    )
    
    M = length(M_list)
    
    #println(M_list)
    
    
    """
    # Do the initial diagonalization:
    E, C, kappa = SolveGenEig(
        H_full,
        S_full,
        thresh=op.sd_thresh,
        eps=op.sd_eps
    )
    """
    
    # Generate the initial T_i:
    T_list = T_init
    
    """
    for i=1:M
        
        i0 = sum(M_list[1:i-1])+1
        i1 = sum(M_list[1:i])

        t_vec = normalize(C[i0:i1,1])
        T_i = sum([t_vec[k] * oht_list[i][k] for k=1:M_list[i]])
            
        push!(T_list, T_i)
        
    end
    """
    
    E_new = 0.0 #E[1]
    
    T1_oht = Any[]
    V_list = Any[]
        
    T1_list = []
    
    # I can't believe I'm doing this...
    pos = zeros(Int64, M)
    c = 1
    for a=1:length(pos)
        if nsite[a]==2
            pos[a]=c
            c += 1
        end
    end
    
    for r=1:nrep, s=1:2
            
        T1_mats = []
        M1_list = Int[]

        T1_oht = Any[]
        V_list = Any[]

        for i=1:M
                
            if nsite[i]==2

                # Split by SVD and form single-site tensors
                linds = ids[pos[i]][s]
                U, S, V = svd(T_list[i], linds, maxdim=maxdim)
                push!(V_list, V)
                
                # Compute two-site tensors for the single-site one-hot states
                push!(T1_oht, OneHotTensors(U * S))
                push!(M1_list, length(T1_oht[end]))
                T1_twosite = [T1_oht[end][k] * V for k=1:M1_list[end]]

                # Form contraction matrix
                T1_mat = zeros(M1_list[i],M_list[i])
                for k1=1:M1_list[i], k2=1:M_list[i]
                    T1_mat[k1,k2] = scalar(T1_twosite[k1] * dag(oht_list[i][k2]))
                end

                push!(T1_mats, T1_mat)
                
            else
                
                push!(M1_list, M_list[i])
                push!(T1_mats, Matrix(I, M_list[i], M_list[i]))
                push!(T1_oht, oht_list[i])
                push!(V_list, 1.0)
                
            end

        end

        # Form reduced subspace H, S matrices
        H_red = zeros((sum(M1_list),sum(M1_list)))
        S_red = zeros((sum(M1_list),sum(M1_list)))

        for i=1:M, j=i:M

            i0 = sum(M_list[1:i-1])+1
            i1 = sum(M_list[1:i])
            j0 = sum(M_list[1:j-1])+1
            j1 = sum(M_list[1:j])

            i10 = sum(M1_list[1:i-1])+1
            i11 = sum(M1_list[1:i])
            j10 = sum(M1_list[1:j-1])+1
            j11 = sum(M1_list[1:j])

            H_red[i10:i11, j10:j11] = T1_mats[i] * H_full[i0:i1,j0:j1] * transpose(T1_mats[j])
            H_red[j10:j11, i10:i11] = transpose(H_red[i10:i11, j10:j11])

            S_red[i10:i11, j10:j11] = T1_mats[i] * S_full[i0:i1,j0:j1] * transpose(T1_mats[j])
            S_red[j10:j11, i10:i11] = transpose(S_red[i10:i11, j10:j11])

        end

        #println(M1_list)
        
        # Discard overlapping in the reduced space:
        H_red, S_red, M1_list, T1_oht = DiscardOverlapping(H_red, S_red, M1_list, T1_oht, tol=op.sd_dtol)

        #println(M1_list)
        
        """
        # "Pop" rows of the T1 matrices:
        for i=1:M
            T1_disc = zeros(M1_disc[i], M_list[i])
            for (k,l) in enumerate(setdiff(1:M1_list[i], rdiscards[i]))
                T1_disc[k,:] = T1_mats[i][l,:]
            end
            T1_mats[i] = T1_disc
        end
        """

        # Diagonalize in reduced one-site space
        E_, C_, kappa_ = SolveGenEig(
            H_red,
            S_red,
            thresh=op.sd_thresh,
            eps=op.sd_eps
        )

        E_new = E_[1]

        """
        # Convert coeffs back to two-site space
        for i=1:M
            i0 = sum(M_list[1:i-1])+1
            i1 = sum(M_list[1:i])

            i10 = sum(M1_disc[1:i-1])+1
            i11 = sum(M1_disc[1:i])

            t_vec = transpose(T1[i]) * normalize(real.(C_[i10:i11,1]))
            T_i = dag(ITensor(t_vec, cinds[i])) * oht_list[i]
            T_list[i] = T_i
        end
        """

        T1_list = []

        # Convert coeffs back to one-site space
        for i=1:M

            i10 = sum(M1_list[1:i-1])+1
            i11 = sum(M1_list[1:i])

            t_vec = normalize(real.(C_[i10:i11,1]))
            #println(M1_list[i])
            T_i = sum([t_vec[k] * T1_oht[i][k] for k=1:M1_list[i]])
            push!(T1_list, T_i)
        end
        
    end
    
    return T1_list, V_list, E_new
    
end


# Determine whether inserting an FSWAP reduces the \\ 
# ...truncation error for the two-site tensor T:
function TestFSWAP2(T, linds, maxdim; crit="fidelity")
    
    site_inds = inds(T, tags="Site")
    fswap = BuildFermionicSwap(site_inds, 1; dim=4);
    
    do_comp = true
    do_swap = false
    
    try 
        U_test,S_test,V_test = svd(T, linds, maxdim=maxdim)
    catch err
        do_comp = false
    end
    
    if do_comp
        T_ = T * fswap
        noprime!(T_)

        U,S,V = svd(T, linds)
        sigma = diag(Array(S, inds(S)))
        m = minimum([length(sigma), maxdim])
        fid = sum(reverse(sort(sigma))[1:m].^2)
        vne = -sum(sigma .* log.(sigma))

        U_,S_,V_ = svd(T_, linds, maxdim=maxdim)
        sigma_ = diag(Array(S_, inds(S_))) 
        m_ = minimum([length(sigma_), maxdim])
        fid_ = sum(reverse(sort(sigma_))[1:m_].^2)
        vne_ = -sum(sigma_ .* log.(sigma_))
        
        if crit=="fidelity"
            do_swap = (fid_ > fid)
        elseif crit=="entropy"
            do_swap = (vne_ < vne)
        end
    end
    
    return do_swap
    
end



function BlockSiteEntropy(T, prime_inds)

    totdim = prod(dim.(prime_inds))
    
    Tprime = deepcopy(T)

    for prime_ind in prime_inds
        setprime!(Tprime, 1, id=id(prime_ind))
    end

    Trdm = Tprime * dag(T)
    
    ind_set = vcat(
        [dag(prime_ind) for prime_ind in prime_inds], 
        [prime_ind' for prime_ind in prime_inds]
    )

    #println(inds(Trdm))
    #println(ind_set)
    
    rdm = reshape(Array(Trdm, ind_set), (totdim, totdim))

    return vnEntropy(rdm)
    
end


# Determine whether inserting an FSWAP is favorable by \\
# ...computing the mutual information with the left/right blocks:
function TestFSWAP3(T, sites, links)
    
    # left/right block entropies:
    s_b = []
    
    for lr = 1:2
        if links[lr]==nothing
            push!(s_b, 0.0)
        else
            push!(s_b, BlockSiteEntropy(T, [links[lr]]))
        end
    end
    
    # p/p+1 site entropies:
    s_p = [BlockSiteEntropy(T, [sites[q]]) for q=1:2]
    
    # Block/site mutual information:
    i_bp = []
    
    for q=1:2, lr=1:2
        if links[lr]==nothing
            push!(i_bp, 0.0)
        else
            s_bp = BlockSiteEntropy(T, [sites[q], links[lr]])
            push!(i_bp, s_p[q] + s_b[lr] - s_bp)
        end
    end
    
    # Compute FSWAP favorability (heuristic):
    fav = i_bp[2] + i_bp[3] - i_bp[1] - i_bp[4] 
    
    #println("\n  $(i_bp)  $(fav > 0.0)  \n")
    
    return (fav > 0.0)
    
end


# Add random noise to the MPS parameters for all states in the subspace:
function SeedNoise!(
        sdata::SubspaceProperties,
        delta::Float64,
        noise::Float64;
        jset=nothing,
        penalty=0.9999,
        verbose=false
    )
    
    sdata_copy = copy(sdata)
    
    N = sdata.chem_data.N_spt
    M = sdata.mparams.M
    
    if jset==nothing
        jset=collect(1:M)
    end
    
    for j in jset
        
        for p=1:N-1
            
            orthogonalize!(sdata.psi_list[j], p)
            
            T_old = sdata.psi_list[j][p] * sdata.psi_list[j][p+1]
            
            psi_decomp = OneHotTensors(T_old)
            
            t_vec = [scalar(psi_decomp[k] * dag(T_old)) for k=1:length(psi_decomp)]
            t_vec += delta*normalize(randn(length(t_vec)))
            normalize!(t_vec)
            
            T_new = sum([t_vec[k] * psi_decomp[k] for k=1:length(psi_decomp)])
            
            # Generate the "noise" term:
            pmpo = ITensors.ProjMPO(sdata.ham_list[j])
            ITensors.set_nsite!(pmpo,2)
            ITensors.position!(pmpo, sdata.psi_list[j], p)
            drho = noise*ITensors.noiseterm(pmpo,T_new,"left")
            
            # Replace the tensors of the MPS:
            spec = ITensors.replacebond!(
                sdata.psi_list[j],
                p,
                T_new;
                maxdim=sdata.mparams.psi_maxdim,
                eigen_perturbation=drho,
                ortho="left",
                normalize=true,
                svd_alg="qr_iteration"
                #min_blockdim=1
            )
            
            normalize!(sdata.psi_list[j])
            
        end
        
    end
    
    GenSubspaceMats!(sdata)
    SolveGenEig!(sdata)
    
    if sdata.E[1] > sdata_copy.E[1]*penalty
        copyto!(sdata, sdata_copy)
    end
    
end


function LocalFSWAP!(sdata, j, p)
    
    # For locally permuting the MPOs:
    lU, rV, qnvec = SWAPComponents(true)
    
    N = sdata.chem_data.N_spt
    
    # Permute the site ordering:
    sdata.ord_list[j][p:p+1] = reverse(sdata.ord_list[j][p:p+1])
    
    # Locally permute the Hamiltonian:
    PSWAP!(
        sdata.ham_list[j], 
        p, 
        lU, 
        rV, 
        qnvec, 
        do_trunc=true,
        tol=sdata.mparams.ham_tol,
        maxdim=sdata.mparams.ham_maxdim,
        prime_side=true
    )
    
    PSWAP!(
        sdata.ham_list[j], 
        p, 
        lU, 
        rV, 
        qnvec, 
        do_trunc=true,
        tol=sdata.mparams.ham_tol,
        maxdim=sdata.mparams.ham_maxdim,
        prime_side=false
    )
    
    # Locally permute the permutation MPOs:
    for i=1:M
        if i < j
            PSWAP!(
                sdata.perm_ops[i][j-i], 
                p, 
                lU, 
                rV, 
                qnvec, 
                do_trunc=true,
                tol=sdata.mparams.perm_tol,
                maxdim=sdata.mparams.perm_maxdim,
                prime_side=false
            )
        elseif i > j
            PSWAP!(
                sdata.perm_ops[j][i-j], 
                p, 
                lU, 
                rV, 
                qnvec, 
                do_trunc=true,
                tol=sdata.mparams.perm_tol,
                maxdim=sdata.mparams.perm_maxdim,
                prime_side=true
            )
        end
    end
    
end


function ListMPS(psi)
    return [psi[p] for p=1:length(psi)]
end


# Sets up H, S matrix element block contraction with left and right MPS tensor lists \\
# ...leaving gaps for the one-hot site decompositions, and assigns relevant MPOs:
function BlockSetup(sdata, i1, i2, p_i1, p_i2, ns_i1, ns_i2)
    
    N = sdata.chem_data.N_spt
    
    psi_i1 = ListMPS(sdata.psi_list[i1])
    
    for q=p_i1:(p_i1+ns_i1-1)
        psi_i1[q] = ITensor(1.0)
    end
    
    # Cases to determine the type of block for the CollectBlocks call:
    if i1==i2 # Diagonal block
        
        psi_i2 = ListMPS(sdata.psi_list[i1])
        
        for q=p_i2:(p_i2+ns_i2-1)
            psi_i2[q] = ITensor(1.0)
        end
        
        hmpo1, hmpo2 = sdata.ham_list[i1], nothing
        smpo1, smpo2 = nothing, nothing
        
    else # Off-diagonal block
        
        hmpo1 = sdata.perm_ops[i1][i2-i1]
        smpo1, smpo2 = sdata.perm_ops[i1][i2-i1], nothing
        
        if sdata.rev_flag[i1][i2-i1]
            
            psi_i2 = ListMPS(ReverseMPS(sdata.psi_list[i2]))
            
            for q=(N-p_i2+1):(-1):(N-p_i2-ns_i2+2)
                psi_i2[q] = ITensor(1.0)
            end
            
            hmpo2 = ReverseMPO(sdata.ham_list[i2])
            
        else
            
            psi_i2 = ListMPS(sdata.psi_list[i2])
            
            for q=p_i2:(p_i2+ns_i2-1)
                psi_i2[q] = ITensor(1.0)
            end
            
            hmpo2 = sdata.ham_list[i2]
            
        end
        
    end
    
    return psi_i1, psi_i2, hmpo1, hmpo2, smpo1, smpo2
    
end


# A "sweep" algorithm based on the two-site decomposition (efficient on classical hardware):
function TwoSiteBlockSweep!(
        sdata::SubspaceProperties,
        op::OptimParameters;
        nsite=nothing,
        jperm=nothing,
        no_swap=false,
        verbose=false,
        no_rev=false,
        debug_output=false
    )
    
    M = sdata.mparams.M
    N = sdata.chem_data.N_spt
    
    # Default is to just do twosite on everything
    if nsite==nothing
        nsite = [2 for i=1:M]
    end
    
    # Turn on no_rev if more than two states are decomposed
    if sum(nsite .> 0) > 2
        no_rev = true
    end
    
    GenPermOps!(sdata, no_rev=no_rev)
    
    # For locally permuting the MPOs:
    lU, rV, qnvec = SWAPComponents(true)
    
    # Default is to cycle through states one at a time:
    if jperm==nothing
        jperm = circshift(collect(1:M),1)
    end
    
    for l=1:op.maxiter
        
        ShuffleStates!(sdata, perm=jperm)
        
        # Choose random bonds to test fswaps:
        swap_bonds = [rand(1:N-1) for i=1:M]
        
        swap_counter = 0
        
        # Orthogonalize to site 1:
        for j=1:M
            orthogonalize!(sdata.psi_list[j], 1)
        end
        
        # Fill in the block_ref as we construct the "lock" tensors:
        block_ref = zeros(Int,(M,M))
        state_ref = []
        
        # Contract the "right" blocks and init the "left" blocks:
        rH_list, rS_list = Any[], Any[]
        lS, lH = Any[], Any[]
        
        for i1=1:M, i2=i1:M
            
            psi_i1, psi_i2, hmpo1, hmpo2, smpo1, smpo2 = BlockSetup(
                sdata, 
                i1, 
                i2, 
                1, 
                1, 
                nsite[i1], 
                nsite[i2]
            )
            
            rH = CollectBlocks(
                psi_i1,
                psi_i2,
                mpo1=hmpo1,
                mpo2=hmpo2,
                inv=true
            )
            
            rS = CollectBlocks(
                psi_i1,
                psi_i2,
                mpo1=smpo1,
                mpo2=smpo2,
                inv=true
            )

            push!(rH_list, rH)
            push!(rS_list, rS)
            push!(lH, ITensor(1.0))
            push!(lS, ITensor(1.0))

            block_ref[i2,i1] = length(lH)
            block_ref[i1,i2] = length(lH)
            push!(state_ref, [i1,i2])
            
        end
        
        # Iterate over all bonds:
        for p=1:N-1
            
            # Compile the one-hot tensor list:
            oht_list = [[ITensor(1.0)] for i=1:M]
            
            T_tensor_list = []
            oht_list = []
            
            for i=1:M
                if nsite[i]==2
                    push!(T_tensor_list, sdata.psi_list[i][p] * sdata.psi_list[i][p+1])
                    push!(oht_list, OneHotTensors(T_tensor_list[end]))
                elseif nsite[i]==1
                    push!(T_tensor_list, sdata.psi_list[i][p])
                    push!(oht_list, OneHotTensors(T_tensor_list[end]))
                elseif nsite[i]==0
                    push!(T_tensor_list, ITensor(1.0))
                    push!(oht_list, [ITensor(1.0)])
                end
            end
            
            M_list = [length(oht_list[i]) for i=1:M]
            M_tot = sum(M_list)
            
            # Construct the full H, S matrices:
            H_full = zeros(Float64, (M_tot, M_tot))
            S_full = zeros(Float64, (M_tot, M_tot))
            
            for i1=1:M, i2=i1:M
                
                i10, i11 = sum(M_list[1:i1-1])+1, sum(M_list[1:i1])
                i20, i21 = sum(M_list[1:i2-1])+1, sum(M_list[1:i2])
                
                bind = block_ref[i1, i2]
                
                psi_i1, psi_i2, hmpo1, hmpo2, smpo1, smpo2 = BlockSetup(
                    sdata, 
                    i1, 
                    i2, 
                    p, 
                    p, 
                    nsite[i1], 
                    nsite[i2]
                )
                
                H_tens = lH[bind]
                
                for q=p:p+1
                    H_tens = UpdateBlock(
                        H_tens, 
                        q,
                        psi_i1,
                        psi_i2,
                        hmpo1,
                        hmpo2
                    )
                end
                
                H_tens  *= rH_list[bind][p]
                
                H_array = zeros(M_list[i1],M_list[i2])
                for k1=1:M_list[i1], k2=1:M_list[i2]
                    #println(inds(H_tens))
                    #println(inds(dag(setprime(oht_list[i1][k1],1))))
                    #println(inds(oht_list[i2][k2]))
                    H_array[k1,k2] = scalar(dag(setprime(oht_list[i1][k1],1)) * H_tens * oht_list[i2][k2])
                end

                H_full[i10:i11, i20:i21] = H_array
                H_full[i20:i21, i10:i11] = conj.(transpose(H_full[i10:i11, i20:i21]))
                
                if i1==i2 # No contraction needed:
                    S_array = Matrix(I, (M_list[i1],M_list[i1]))
                else
                    S_tens = lS[bind]
                
                    for q=p:p+1
                        S_tens = UpdateBlock(
                            S_tens, 
                            q,
                            psi_i1,
                            psi_i2,
                            smpo1,
                            smpo2
                        )
                    end

                    S_tens  *= rS_list[bind][p]

                    S_array = zeros(M_list[i1],M_list[i2])
                    for k1=1:M_list[i1], k2=1:M_list[i2]
                        S_array[k1,k2] = scalar(dag(setprime(oht_list[i1][k1],1)) * S_tens * oht_list[i2][k2])
                    end
                end
                
                S_full[i10:i11, i20:i21] = S_array
                S_full[i20:i21, i10:i11] = conj.(transpose(S_full[i10:i11, i20:i21]))
                
            end
            
            # Make a copy to revert to at the end if the energy penalty is violated:
            sdata_copy = copy(sdata)
                
            H_all, S_all = deepcopy(H_full), deepcopy(S_full)
            oht_all = deepcopy(oht_list)
            M_list_all = deepcopy(M_list)
            
            #println(M_list)
            
            # Discard overlapping states:
            H_full, S_full, M_list, oht_list = DiscardOverlapping(
                H_full, 
                S_full, 
                M_list, 
                oht_list, 
                tol=op.sd_dtol,
                kappa_max=1e10
            )
            
            #println(M_list)
            
            E, C, kappa = SolveGenEig(
                H_full,
                S_full,
                thresh=op.sd_thresh,
                eps=op.sd_eps
            )
            
            # Test FSWAPS on each optimized two-site tensor:
            do_swaps = [false for i=1:M]
            if !(no_swap)
                for i=1:M

                    if nsite[i]==2

                        #if swap_bonds[i]==p
                        i0 = sum(M_list[1:i-1])+1
                        i1 = sum(M_list[1:i])

                        t_vec = normalize(C[i0:i1,1])
                        T = sum([t_vec[k]*oht_list[i][k] for k=1:M_list[i]])

                        linds = commoninds(T, sdata.psi_list[i][p])
                        llink = uniqueinds(linds, commoninds(T, sdata.psi_list[i][p], tags="Site"))
                        if length(llink)==0
                            llink = [nothing]
                        end

                        rinds = commoninds(T, sdata.psi_list[i][p+1])
                        rlink = uniqueinds(rinds, commoninds(T, sdata.psi_list[i][p+1], tags="Site"))
                        if length(rlink)==0
                            rlink = [nothing]
                        end

                        #do_swaps[i] = TestFSWAP2(T, linds, sdata.mparams.psi_maxdim, crit="fidelity")
                        do_swaps[i] = TestFSWAP3(T, sdata.sites[p:p+1], [llink[1], rlink[1]])
                        #end

                    end

                end
            end
            
            # Modify the H, S matrices to encode the FSWAPs:
            H_all, S_all = FSWAPModify(
                H_all, 
                S_all, 
                M_list_all, 
                oht_all, 
                sdata.sites, 
                nsite, 
                [[p,p+1] for i=1:M], 
                do_swaps
            )
            
            # Discard overlapping states:
            H_full, S_full, M_list, oht_list = DiscardOverlapping(
                H_all, 
                S_all, 
                M_list_all, 
                oht_all, 
                tol=op.sd_dtol,
                kappa_max=1e10
            )
            
            #println(M_list)
            
            E, C, kappa = SolveGenEig(
                H_full,
                S_full,
                thresh=op.sd_thresh,
                eps=op.sd_eps
            )
            
            # Generate the initial T_i:
            T_init = []

            for i=1:M

                i0 = sum(M_list[1:i-1])+1
                i1 = sum(M_list[1:i])

                t_vec = normalize(C[i0:i1,1])
                T_i = sum([t_vec[k] * oht_list[i][k] for k=1:M_list[i]])

                push!(T_init, T_i)

            end
            
            inds_list = []
            for j=1:M
                if nsite[j]==2
                    linds = commoninds(T_tensor_list[j], sdata.psi_list[j][p])
                    rinds = commoninds(T_tensor_list[j], sdata.psi_list[j][p+1])
                    push!(inds_list, [linds, rinds])
                end
            end

            # Do TripleGenEig on all states to lower energy:
            T_list, V_list, E_new = TripleGenEigM(
                H_all,
                S_all,
                oht_all,
                T_init,
                M_list_all,
                op,
                nsite,
                inds_list,
                sdata.mparams.psi_maxdim,
                nrep=20
            )
            
            #println("\n$(round(E[1], sigdigits=6))  $(round(E_new, sigdigits=6))\n")
            #println(E_new)

            do_replace = true
            for i=1:M
                if (NaN in T_list[i]) || (Inf in T_list[i])
                    do_replace = false
                end
            end
            
            #println("\n $(do_replace)")
            
            #println("$(real(E_new) < real(E[1]) + op.sd_etol)\n")
            
            if (real(E_new) < real(sdata.E[1]) + op.sd_etol) && do_replace
                
                # Update params, orderings
                for j=1:M
                    
                    if nsite[j]==2 && do_swaps[j]

                        swap_counter += 1
                        LocalFSWAP!(sdata, j, p)
                        
                    end

                end
                
            else # Revert to previous parameters:
                
                for j=1:M
                    
                    if nsite[j]==2
                        U,S,V = svd(
                            T_tensor_list[j], 
                            commoninds(sdata.psi_list[j][p], T_tensor_list[j]),
                            alg="qr_iteration",
                            maxdim=sdata.mparams.psi_maxdim
                        )

                        V_list[j] = U
                        T_list[j] = S*V
                    elseif nsite[j]==1
                        
                        T_list[j] = T_tensor_list[j]
                        
                    end
                    
                end
                
            end
                
            """
            #  One last check that the energy penalty limit has not been exceeded:
            GenSubspaceMats!(sdata)
            SolveGenEig!(sdata)

            if sdata.E[1] > sdata_copy.E[1] + op.sd_etol
                # Revert to the previous subspace:
                copyto!(sdata, sdata_copy)
            end
            """

            # Regardless of replacement, update state:
            for j=1:M
                
                if nsite[j]==2
                    
                    sdata.psi_list[j][p] = V_list[j]
                    sdata.psi_list[j][p+1] = T_list[j]
                    
                elseif nsite[j]==1
                    
                    T_j = T_list[j]*sdata.psi_list[j][p+1]

                    spec = ITensors.replacebond!(
                        sdata.psi_list[j],
                        p,
                        T_j;
                        maxdim=sdata.mparams.psi_maxdim,
                        #eigen_perturbation=drho,
                        ortho="left",
                        normalize=true,
                        svd_alg="qr_iteration"
                        #min_blockdim=1
                    )

                elseif nsite[j]==0
                    
                    T_j = sdata.psi_list[j][p]*sdata.psi_list[j][p+1]

                    spec = ITensors.replacebond!(
                        sdata.psi_list[j],
                        p,
                        T_j;
                        maxdim=sdata.mparams.psi_maxdim,
                        #eigen_perturbation=drho,
                        ortho="left",
                        normalize=true,
                        svd_alg="qr_iteration"
                        #min_blockdim=1
                    )
                    
                end
                
                # Make sure new state is normalized:
                #normalize!(sdata.psi_list[j])
                sdata.psi_list[j][p+1] *= 1.0/sqrt(norm(sdata.psi_list[j]))
                
            end
            
            #println(sdata.E[1])
            #println([norm(psi) for psi in sdata.psi_list])
            
            """
            # Add noise:
            for j in j_set

                T_j = sdata.psi_list[j][p]*sdata.psi_list[j][p+1]
                
                t_vec = [scalar(T_j*dag(oht_list[j][k])) for k=1:M_list[j]]
                
                t_vec += op.delta[1]*normalize(randn(M_list[j]))
                
                normalize!(t_vec)
                
                if !(true in isnan.(t_vec)) && !(Inf in t_vec)
                    T_j = sum([t_vec[k] * oht_list[j][k] for k=1:M_list[j]])
                end

                spec = ITensors.replacebond!(
                    sdata.psi_list[j],
                    p,
                    T_j;
                    maxdim=sdata.mparams.psi_maxdim,
                    #eigen_perturbation=drho,
                    ortho="left",
                    normalize=true,
                    svd_alg="qr_iteration"
                    #min_blockdim=1
                )

                # Make sure new state is normalized:
                #normalize!(sdata.psi_list[j])
                sdata.psi_list[j][p+1] *= 1.0/sqrt(norm(sdata.psi_list[j]))

            end
            """
            
            GenSubspaceMats!(sdata)
            SolveGenEig!(sdata)
            
            # Double-check that the energy is not too high!
            if sdata.E[1] > sdata_copy.E[1] + op.sd_etol
                
                # Revert to previous subspace:
                copyto!(sdata, sdata_copy)
                
                for j in j_set

                    T_j = sdata.psi_list[j][p]*sdata.psi_list[j][p+1]

                    spec = ITensors.replacebond!(
                        sdata.psi_list[j],
                        p,
                        T_j;
                        maxdim=sdata.mparams.psi_maxdim,
                        #eigen_perturbation=drho,
                        ortho="left",
                        normalize=true,
                        svd_alg="qr_iteration"
                        #min_blockdim=1
                    )

                    # Make sure new state is normalized:
                    #normalize!(sdata.psi_list[j])
                    sdata.psi_list[j][p+1] *= 1.0/sqrt(norm(sdata.psi_list[j]))

                end
                
            end

            for i1=1:M, i2=i1:M

                bind = block_ref[i1,i2]
                
                psi_i1, psi_i2, hmpo1, hmpo2, smpo1, smpo2 = BlockSetup(
                    sdata, 
                    i1, 
                    i2, 
                    p, 
                    p, 
                    0, 
                    0
                )

                lH[bind] = UpdateBlock(
                    lH[bind], 
                    p,
                    psi_i1,
                    psi_i2,
                    hmpo1,
                    hmpo2
                )
                
                lS[bind] = UpdateBlock(
                    lS[bind], 
                    p,
                    psi_i1,
                    psi_i2,
                    smpo1,
                    smpo2
                )

            end

            # Print some output
            if verbose
                print("Loop: ($(l)/$(op.maxiter)); ")
                print("Bond: $(p)/$(N-1); ")
                print("#swaps: $(swap_counter); ")
                print("E_min = $(round(sdata.E[1], digits=5)); ") 
                print("Delta = $(round(sdata.E[1]-sdata.chem_data.e_fci+sdata.chem_data.e_nuc, digits=5)); ")
                print("kappa_full = $(round(cond(S_full), sigdigits=3)); ")
                #print("kappa_disc = $(round(cond(S_disc), sigdigits=3)); ")
                print("kappa = $(round(sdata.kappa, sigdigits=3))     \r")
                flush(stdout)
            end

        end # loop over p
            
        for j=1:M # Make sure these states are normalized:
            normalize!(sdata.psi_list[j])
        end
        
        # Recompute H, S, E, C, kappa:
        GenPermOps!(sdata, no_rev=no_rev)
        GenSubspaceMats!(sdata)
        SolveGenEig!(sdata)
        
        l += 1
        
    end # loop over j-pairs
    
    # Exit loop
    if verbose
        println("\nDone!\n")
    end
    
end



# Optimizes all states in the subspace at random one- or two-site positions \\
# ...and inserts FSWAPS to reduce truncation error (permuting the orderings):
function AllStateFSWAP!(
        sdata::SubspaceProperties,
        op::OptimParameters;
        l=1,
        n_twos=sdata.mparams.M,
        no_swap=false,
        verbose=false
    )
    
    if verbose
        println("\nFSWAP DISENTANGLER:")
    end
    
    M = sdata.mparams.M
    N = sdata.chem_data.N_spt
    
    # Noise at this iteration:
    lnoise = op.noise[minimum([l,end])]
    ldelta = op.delta[minimum([l,end])]
    
    # Sweeps up to maxiter:
    for l=1:op.maxiter
        
        swap_counter = 0
        
        orb_ords = [randperm(N) for i=1:M]
        
        # Repeat over all bonds:
        for p=1:N

            # Find sites [q(p), q(p)+/-1] for each ordering
            #q_set = [[0,1] .+ rand(1:N-1) for j=1:M]
            q_set = []
            for (i,ord) in enumerate(sdata.ord_list)
                #q1 = findall(x -> x==p, ord)[1]
                q1 = orb_ords[i][p]
                #q2 = q1 + 1
                if q1 == 1
                    q2 = 2
                elseif q1 == N
                    q2 = N-1
                else
                    q2 = q1 + rand([1,-1])
                end
                push!(q_set, sort([q1,q2]))
            end

            #q_set = [[0,1] .+ bond_ords[i][p] for i=1:M]
            
            # Generate "one-hot" tensors:
            oht_list = []
            
            M_list = Int[]
            
            #j_set = sort(randperm(M)[1:n_twos])
            j_set = collect(1:n_twos)
            
            T_list = []
            
            for i=1:M
                
                orthogonalize!(sdata.psi_list[i], q_set[i][1])
                
                if i in j_set
                    T_i = sdata.psi_list[i][q_set[i][1]] * sdata.psi_list[i][q_set[i][2]]
                else
                    T_i = sdata.psi_list[i][q_set[i][1]]
                end
                
                push!(T_list, T_i)
                
            end
            
            for i=1:M
                push!(oht_list, OneHotTensors(T_list[i]))
                push!(M_list, length(oht_list[end]))
            end
        
            # Compute H, S matrix elements:
            H_full = zeros((sum(M_list), sum(M_list)))
            S_full = zeros((sum(M_list), sum(M_list)))
            
            for i=1:M
                
                i0 = sum(M_list[1:i-1])+1
                i1 = sum(M_list[1:i])
                
                psi_i = [sdata.psi_list[i][q] for q=1:N]
                
                if i in j_set
                    psi_i[q_set[i][1]] = ITensor(1.0)
                    psi_i[q_set[i][2]] = ITensor(1.0)
                else
                    psi_i[q_set[i][1]] = ITensor(1.0)
                end
                
                tenH_ii = FullContract(psi_i, psi_i, mpo1=sdata.ham_list[i])
                
                matH_ii = zeros(M_list[i],M_list[i]) #Array(tenH_ii, (ind_i, setprime(dag(ind_i),1)))
                
                for k1=1:M_list[i], k2=1:M_list[i]
                    matH_ii[k1,k2] = scalar(setprime(dag(oht_list[i][k1]),1) * tenH_ii * oht_list[i][k2])
                end
                
                H_full[i0:i1,i0:i1] = real.(matH_ii)
                
                S_full[i0:i1,i0:i1] = Matrix(I, M_list[i], M_list[i])
                
                # i-j blocks:
                for j=(i+1):M
                    
                    j0 = sum(M_list[1:j-1])+1
                    j1 = sum(M_list[1:j])

                    if sdata.rev_flag[i][j-i] # Construct a ReverseMPS list:
                        
                        revj = ReverseMPS(sdata.psi_list[j])
                        psi_j = [deepcopy(revj[q]) for q=1:N]
                        
                        q_revj = [N+1-q_set[j][2], N+1-q_set[j][1]]
                        if j in j_set
                            T_revj = psi_j[q_revj[1]] * psi_j[q_revj[2]]
                        else
                            T_revj = psi_j[q_revj[2]]
                        end
                        oht_revj = OneHotTensors(T_revj)
                        
                        if j in j_set
                            psi_j[q_revj[1]] = ITensor(1.0)
                            psi_j[q_revj[2]] = ITensor(1.0)
                        else
                            psi_j[q_revj[2]] = ITensor(1.0)
                        end
                    else
                        psi_j = [deepcopy(sdata.psi_list[j][q]) for q=1:N]
                        if j in j_set
                            psi_j[q_set[j][1]] = ITensor(1.0)
                            psi_j[q_set[j][2]] = ITensor(1.0)
                        else
                            psi_j[q_set[j][1]] = ITensor(1.0)
                        end
                    end
                    
                    tenH_ij = FullContract(psi_i, psi_j, mpo1=sdata.perm_ops[i][j-i], mpo2=sdata.ham_list[j])
                    tenS_ij = FullContract(psi_i, psi_j, mpo1=sdata.perm_ops[i][j-i])

                    matH_ij = zeros(M_list[i],M_list[j])
                    matS_ij = zeros(M_list[i],M_list[j])
                    
                    for k1=1:M_list[i], k2=1:M_list[j]
                        matH_ij[k1,k2] = scalar(setprime(dag(oht_list[i][k1]),1) * tenH_ij * oht_list[j][k2])
                        matS_ij[k1,k2] = scalar(setprime(dag(oht_list[i][k1]),1) * tenS_ij * oht_list[j][k2])
                    end
                    
                    H_full[i0:i1,j0:j1] = real.(matH_ij)
                    H_full[j0:j1,i0:i1] = transpose(matH_ij)
                    
                    S_full[i0:i1,j0:j1] = real.(matS_ij)
                    S_full[j0:j1,i0:i1] = transpose(matS_ij)
                    
                    
                end
                
            end
            
            """
            if (NaN in S_full) || (Inf in S_full)
                println("\n\nRuh-roh!\n\n")
            end
            """
            
            # Make a copy to revert to at the end if the energy penalty is violated:
            sdata_copy = copy(sdata)
                
            H_all, S_all = deepcopy(H_full), deepcopy(S_full)
            oht_all = deepcopy(oht_list)
            M_list_all = deepcopy(M_list)
            
            #println(M_list)
            
            # Discard overlapping states:
            H_full, S_full, M_list, oht_list = DiscardOverlapping(
                H_full, 
                S_full, 
                M_list, 
                oht_list, 
                tol=op.sd_dtol,
                kappa_max=1e10
            )
            
            #println(M_list)
            
            E, C, kappa = SolveGenEig(
                H_full,
                S_full,
                thresh=op.sd_thresh,
                eps=op.sd_eps
            )
            
            # Test FSWAPS on each optimized two-site tensor:
            do_swaps = [false for i=1:M]
            for i in j_set

                #if swap_bonds[i]==p
                i0 = sum(M_list[1:i-1])+1
                i1 = sum(M_list[1:i])

                t_vec = normalize(C[i0:i1,1])
                T = sum([t_vec[k]*oht_list[i][k] for k=1:M_list[i]])
                linds = commoninds(T, sdata.psi_list[i][p])
                do_swaps[i] = TestFSWAP2(T, linds, sdata.mparams.psi_maxdim, crit="fidelity")
                #end

            end

            # Enforce no swapping?
            if no_swap
                do_swaps = [false for i=1:M]
            end
            
            # Modify the H, S matrices to encode the FSWAPs:
            H_all, S_all = FSWAPModify(
                H_all, 
                S_all, 
                M_list_all, 
                oht_all, 
                sdata.sites, 
                j_set, 
                q_set, 
                do_swaps
            )

            """
            # Discard overlapping states:
            H_full, S_full, M_list, oht_list = DiscardOverlapping(
                H_all, 
                S_all, 
                M_list_all, 
                oht_all,
                tol=op.sd_dtol,
                kappa_max=1e10
            )
            
            #println(M_list)
            
            E, C, kappa = SolveGenEig(
                H_full,
                S_full,
                thresh=op.sd_thresh,
                eps=op.sd_eps
            )

            #println("\n$(E[1])\n")
            """
            
            inds_list = []
            for j in j_set
                linds = commoninds(T_list[j], sdata.psi_list[j][q_set[j][1]])
                rinds = commoninds(T_list[j], sdata.psi_list[j][q_set[j][2]])
                push!(inds_list, [linds, rinds])
            end
            
            # Do TripleGenEig on all states to lower energy:
            T_list, V_list, E_new = TripleGenEigM(
                H_all,
                S_all,
                oht_all,
                M_list_all,
                op,
                j_set,
                inds_list,
                sdata.mparams.psi_maxdim,
                nrep=20
            )
            
            #println("\n$(round(E[1], sigdigits=6))  $(round(E_new, sigdigits=6))\n")
            #println(E_new)

            do_replace = true
            for i=1:M
                if (NaN in T_list[i]) || (Inf in T_list[i])
                    do_replace = false
                end
            end

            if (real(E_new) < real(E[1]) + op.sd_etol) && do_replace

                # Update params, orderings
                for i=1:M

                    if i in j_set

                        """
                        spec = ITensors.replacebond!(
                            sdata.psi_list[i],
                            q_set[i][1],
                            T_list[i];
                            maxdim=sdata.mparams.psi_maxdim,
                            #eigen_perturbation=drho,
                            ortho="left",
                            normalize=true,
                            svd_alg="qr_iteration"
                            #min_blockdim=1
                        )
                        """
                        
                        sdata.psi_list[i][q_set[i][1]] = V_list[i]
                        sdata.psi_list[i][q_set[i][2]] = T_list[i]

                        if do_swaps[i]

                            swap_counter += 1
                            sdata.ord_list[i][q_set[i][1]:q_set[i][2]] = reverse(sdata.ord_list[i][q_set[i][1]:q_set[i][2]])

                            # Re-generate Hamiltonians and PMPOs:
                            GenHams!(sdata)
                            GenPermOps!(sdata)

                        end

                    else

                        sdata.psi_list[i][q_set[i][1]] = T_list[i]

                    end

                end

            end

            # Recompute H, S, E, C, kappa:
            GenSubspaceMats!(sdata)
            SolveGenEig!(sdata)

            # One last check that the energy penalty limit has not been exceeded:
            if sdata.E[1] > sdata_copy.E[1] + op.sd_etol
                # Revert to the previous subspace:
                copyto!(sdata, sdata_copy)
            end

            # Print some output
            if verbose
                print("Sweep: $(l)/$(op.maxiter); ")
                print("orbital: $(p)/$(N); ")
                print("#swaps: $(swap_counter); ")
                print("E_min = $(round(sdata.E[1], digits=5)); ") 
                print("Delta = $(round(sdata.E[1]-sdata.chem_data.e_fci+sdata.chem_data.e_nuc, digits=5)); ")
                print("kappa_full = $(round(cond(S_full), sigdigits=3)); ")
                print("kappa = $(round(sdata.kappa, sigdigits=3))     \r")
                flush(stdout)
            end
            
        end
        
    end
    
    # Exit loop
    if verbose
        println("\nDone!\n")
    end
    
end