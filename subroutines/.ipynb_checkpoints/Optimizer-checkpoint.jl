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
# ...decomposition tensor with an index running over the one-hot states:
function OHTCompIndexTensor(T; discards=[])
    
    T_inds = inds(T)
    ivs_list = []
    k = 1
    
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
                push!(ivs_list, ivs)
            end
            k += 1
        end
    end
    
    C_inds = vcat([Index(QN(("Nf",0,-1),("Sz",0)) => length(ivs_list), tags="c")], T_inds)
    C = ITensor(C_inds...)
    
    for (l, ivs) in enumerate(ivs_list)
        C_ivs = vcat([C_inds[1]=>l], ivs)
        C[C_ivs...] = 1.0
    end
    
    return C
    
end


# Check if states are overlapping too much and discard if they are:
function DiscardOverlapping(H_full_in, S_full_in, M_list_in; criterion="overlap", tol=0.01, kappa_max=1e10)
    
    H_full = deepcopy(H_full_in)
    S_full = deepcopy(S_full_in)
    
    M_list = deepcopy(M_list_in)
    
    M = length(M_list)
    M_tot = sum(M_list)
    
    discards = []
    
    # Iterate over the comp index objects in oht_list:
    for j=M:(-1):1
        
        discards_j = []
        
        M_j = M_list[j]
        
        # Iterate over the states in the comp index object:
        for k=M_j:(-1):1
            
            # j0 and j1, keeping track of discards:
            j0, j1 = sum(M_list[1:j-1])+1, sum(M_list[1:j])
            
            # The current column of S_full, keeping track of discards:
            col = j0 + k - 1
            
            do_discard = false
            
            if criterion=="InfNaN" && (M_list[j] > 1) # Check for any Infs or NaNs in that column:
                
                if (Inf in S_full[:,col]) || (NaN in S_full[:,col]) || (Inf in H_full[:,col]) || (NaN in H_full[:,col])
                    push!(discards_j, k)
                    do_discard = true
                end
                
            elseif criterion=="overlap" && (j != M) && (M_list[j] > 1)
                
                # First check the overlap with the previous subspace is not too large:
                S_red = deepcopy(S_full[j1+1:M_tot,j1+1:M_tot])
                vphi = S_full[j1+1:M_tot,col]
                
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
                
                #if kappa_new > kappa_max
                #    println("\nCond too high!\n")
                #end
                
                # Mark the state for discarding:
                if (sqnm > 1.0-tol) || (kappa_new > kappa_max) || isnan(kappa_new) || (kappa_new == Inf)
                    #println("\nDiscard triggered: kappa_new = $(kappa_new)\n")
                    push!(discards_j, k)
                    do_discard=true
                end
                
                
            end
            
            if do_discard
                
                H0 = H_full
                S0 = S_full
                
                H1 = zeros((M_tot-1,M_tot-1))
                H1[1:col-1,1:col-1] = H0[1:col-1,1:col-1]
                H1[1:col-1,col:end] = H0[1:col-1,col+1:end]
                H1[col:end,1:col-1] = H0[col+1:end,1:col-1]
                H1[col:end,col:end] = H0[col+1:end,col+1:end]

                S1 = zeros((M_tot-1,M_tot-1))
                S1[1:col-1,1:col-1] = S0[1:col-1,1:col-1]
                S1[1:col-1,col:end] = S0[1:col-1,col+1:end]
                S1[col:end,1:col-1] = S0[col+1:end,1:col-1]
                S1[col:end,col:end] = S0[col+1:end,col+1:end]

                
                H_full = H1
                S_full = S1
                
                M_list[j] -= 1
                M_tot -= 1
                
            end
            
        end # loop over k
        
        pushfirst!(discards, discards_j)
        
    end # loop over j
            
    return H_full, S_full, M_list, discards
    
end


# Update an inner-product block by contraction:
function UpdateBlock(
        block, 
        p,
        psi1, 
        psi2,
        mpo1,
        mpo2
    )
    
    if mpo1==nothing && mpo2==nothing
        block *= psi2[p] * setprime(dag(psi1[p]),1, tags="Link")
    elseif mpo2==nothing
        block *= psi2[p] * mpo1[p] * setprime(dag(psi1[p]),1)
    elseif mpo1==nothing
        block *= psi2[p] * mpo2[p] * setprime(dag(psi1[p]),1)
    else
        Ax = setprime(mpo1[p],2,plev=0) * setprime(dag(psi1[p]),1)
        yB = psi2[p] * dag(setprime(setprime(mpo2[p],2,plev=0,tags="Site"),0,plev=1),tags="Site")
        block *= Ax
        block *= yB
    end
    
    return block
    
end



function FullContract(
        psi1,
        psi2;
        mpo1=nothing,
        mpo2=nothing,
        combos=nothing,
        csites=nothing
    )
    
    block = ITensor(1.0)
    
    for p=1:length(psi1)
        
        block = UpdateBlock(block, p, psi1, psi2, mpo1, mpo2)
        
        if csites != nothing && p==csites[2]
            block *= combos[2]
        end
        if csites != nothing && p==csites[1]
            block *= setprime(dag(combos[1]),1)
        end
        
    end
    
    return block
    
end



# Collects partially contracted inner-product blocks \\
# (...) from two MPSs and up to two MPOs
function CollectBlocks(
        psi1,
        psi2;
        mpo1=nothing,
        mpo2=nothing,
        p0=length(psi1),
        p1=3,
        inv=false
    )
    
    p_block = ITensor(1.0)
    
    block_list = [p_block]
    
    for p = p0:sign(p1-p0):p1
        
        p_block = UpdateBlock(p_block, p, psi1, psi2, mpo1, mpo2)
        
        push!(block_list, deepcopy(p_block))
        
    end
    
    if inv
        return reverse(block_list)
    else
        return block_list
    end
    
end



# Performs a sequence of alternating single-site decompositions \\
# ...to minimize truncation error from the two-site decomposition:
function TripleGenEigM(
        H_disc,
        S_disc,
        oht_disc,
        M_disc,
        op,
        twos,
        ids,
        maxdim;
        nrep=3
    )
    
    M = length(M_disc)
    M_list = M_disc
    oht_list = oht_disc
    S_full = S_disc
    H_full = H_disc
    
    # Do the initial diagonalization:
    E, C, kappa = SolveGenEig(
        H_disc,
        S_disc,
        thresh=op.sd_thresh,
        eps=op.sd_eps
    )
    
    """
    println("Before discarding: M_list = $(M_list), discards = nothing")
    
    # Discard overlapping states:
    H_disc, S_disc, M_disc, discards = DiscardOverlapping(H_full, S_full, M_list, criterion="overlap", tol=op.sd_dtol)
    
    println("After discarding: M_list = $(M_list), discards = nothing")

    # Do the initial diagonalization:
    E, C, kappa = SolveGenEig(
        H_disc,
        S_disc,
        thresh=op.sd_thresh,
        eps=op.sd_eps
    )
    
    # Re-generate the one-hot tensor list:
    oht_disc = []

    for i=1:M
        if M_list[i] > 1
            T_i = ITensor(ones(M_list[i]), dag(inds(oht_list[i], tags="c")[1])) * oht_list[i]
            push!(oht_disc, OHTCompIndexTensor(T_i, discards=discards[i]))
        else
            push!(oht_disc, ITensor([1.0], inds(oht_list[i], tags="c")[1]))
        end
    end
    """
    
    # Generate the initial T_i:
    T_list = []
    
    cinds = []
    
    for i=1:M
        
        i0 = sum(M_disc[1:i-1])+1
        i1 = sum(M_disc[1:i])

        ind_i_disc = inds(oht_disc[i], tags="c")[1]
        ind_i = inds(oht_list[i], tags="c")[1]
        push!(cinds, ind_i)
        T_i = dag(ITensor(normalize(C[i0:i1,1]), ind_i_disc)) * oht_disc[i]
        push!(T_list, T_i)
        
    end
    
    E_new = E[1]
    
    for r=1:nrep
        for s=1:2
            
            T1 = []
            M1_list = Int[]
            
            for i=1:M
                if i in twos
                    
                    linds = ids[findall(x->x==i, twos)[1]][s]
                    
                    # Split by SVD and form single-site tensors
                    U, S, V = svd(T_list[i], linds, maxdim=maxdim, min_blockdim=maxdim)

                    T1_i = OHTCompIndexTensor(U * S)
                    push!(M1_list, dim(inds(T1_i, tags="c")[1]))

                    T1_i *= V

                    tenT1_i = T1_i * dag(oht_list[i])

                    # Form contraction matrix
                    matT1_i = Array(
                        tenT1_i, 
                        inds(T1_i,tags="c")[1], 
                        dag(inds(oht_list[i],tags="c")[1])
                    )

                    push!(T1, matT1_i)
                else
                    push!(M1_list, M_list[i])
                    push!(T1, Matrix(I, M_list[i], M_list[i]))
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
                
                H_red[i10:i11, j10:j11] = T1[i] * H_full[i0:i1,j0:j1] * transpose(T1[j])
                H_red[j10:j11, i10:i11] = transpose(H_red[i10:i11, j10:j11])
                
                S_red[i10:i11, j10:j11] = T1[i] * S_full[i0:i1,j0:j1] * transpose(T1[j])
                S_red[j10:j11, i10:i11] = transpose(S_red[i10:i11, j10:j11])
                
            end
            
            # Discard overlapping in the reduced space:
            H_rdisc, S_rdisc, M1_disc, rdiscards = DiscardOverlapping(H_red, S_red, M1_list, criterion="overlap", tol=op.sd_dtol)
            
            # "Pop" rows of the T1 matrices:
            for i=1:M
                T1_disc = zeros(M1_disc[i], M_list[i])
                for (k,l) in enumerate(setdiff(1:M1_list[i], rdiscards[i]))
                    T1_disc[k,:] = T1[i][l,:]
                end
                T1[i] = T1_disc
            end

            # Diagonalize in reduced one-site space
            E_, C_, kappa_ = SolveGenEig(
                H_rdisc,
                S_rdisc,
                thresh=op.sd_thresh,
                eps=op.sd_eps
            )
            
            E_new = E_[1]

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
            
        end
        
    end
    
    return T_list, E_new
    
end


# Determine whether inserting an FSWAP reduces the \\ 
# ...truncation error for the two-site tensor T:
function TestFSWAP2(T, linds, maxdim)
    
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

        U,S,V = svd(T, linds, maxdim=maxdim)
        sigma = diag(Array(S, inds(S)))
        m = minimum([length(sigma), maxdim])
        fid = sum(reverse(sort(sigma))[1:m].^2)

        U_,S_,V_ = svd(T_, linds, maxdim=maxdim)
        sigma_ = diag(Array(S_, inds(S_))) 
        m_ = minimum([length(sigma_), maxdim])
        fid_ = sum(reverse(sort(sigma_))[1:m_].^2)
        
        do_swap = (fid_ > fid)
    end
    
    return do_swap
    
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
            
            psi_decomp = OHTCompIndexTensor(T_old)
            
            t_tens = psi_decomp * dag(T_old)
            t_vec = Array(t_tens, inds(t_tens))
            t_vec += delta*normalize(randn(length(t_vec)))
            
            normalize!(t_vec)
            
            T_new = ITensor(t_vec, dag(inds(t_tens))) * psi_decomp
            
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


# A "sweep" algorithm for two states at a time (efficient on classical hardware):
function TwoSitePairSweep!(
        sdata::SubspaceProperties,
        op::OptimParameters;
        jperm=nothing,
        no_swap=false,
        verbose=false
    )
    
    M = sdata.mparams.M
    N = sdata.chem_data.N_spt
    
    # Default is to cycle through states one at a time:
    if jperm == nothing
        jperm = circshift(collect(1:M),1)
    end
    
    for l=1:op.maxiter
        
        ShuffleStates!(sdata, perm=jperm)
        
        j1, j2 = 1, 2
        
        swap_counter = 0
        
        # Check whether to reverse the j2 state:
        if sdata.rev_flag[j1][j2-j1]
            sdata.ord_list[j2] = reverse(sdata.ord_list[j2])
            sdata.psi_list[j2] = ReverseMPS(sdata.psi_list[j2])
            GenHams!(sdata)
            GenPermOps!(sdata)
        end
        
        orthogonalize!(sdata.psi_list[j1], 1)
        orthogonalize!(sdata.psi_list[j2], 1)
        
        # Fill in the block_ref as we construct the "lock" tensors:
        block_ref = zeros(Int,(M,M))
        state_ref = []
        
        # Contract the "right" blocks and init the "left" blocks:
        rH_list, rS_list = Any[], Any[]
        lS, lH = Any[], Any[]
        
        for i1=1:M, i2=i1:M
            
            if (i2 != i1)
                
                if sdata.rev_flag[i1][i2-i1]
                    psi_i2 = ReverseMPS(sdata.psi_list[i2])
                else
                    psi_i2 = sdata.psi_list[i2]
                end

                rH = CollectBlocks(
                    sdata.psi_list[i1],
                    psi_i2,
                    mpo1 = sdata.ham_list[i1],
                    mpo2 = sdata.perm_ops[i1][i2-i1],
                    p1=3,
                    inv=true
                )

                rS = CollectBlocks(
                    sdata.psi_list[i1],
                    psi_i2,
                    mpo1 = sdata.perm_ops[i1][i2-i1],
                    mpo2 = nothing,
                    p1=3,
                    inv=true
                )
                
            else
                
                rH = CollectBlocks(
                    sdata.psi_list[i1],
                    sdata.psi_list[i2],
                    mpo1 = sdata.ham_list[i1],
                    mpo2 = nothing,
                    p1=3,
                    inv=true
                )

                rS = nothing
                
            end

            push!(rH_list, rH)
            push!(rS_list, rS)
            push!(lH, ITensor(1.0))
            push!(lS, ITensor(1.0))

            block_ref[i2,i1] = length(lH)
            block_ref[i1,i2] = length(lH)
            push!(state_ref, [i1,i2])
            
        end
        
        # Orthogonalize to site p:
        orthogonalize!(sdata.psi_list[j1], 1)
        orthogonalize!(sdata.psi_list[j2], 1)
        
        # Iterate over all bonds:
        for p=1:N-1
            
            T_j1 = sdata.psi_list[j1][p] * sdata.psi_list[j1][p+1]
            T_j2 = sdata.psi_list[j2][p] * sdata.psi_list[j2][p+1]
            
            # Contract the OHT index tensors:
            oht_list = [ITensor([1.0], Index(QN(("Nf",0,-1),("Sz",0)) => 1, tags="c")) for i=1:M]
            oht_list[j1] = OHTCompIndexTensor(T_j1)
            oht_list[j2] = OHTCompIndexTensor(T_j2)
            
            M_list = [dim(inds(oht_list[i], tags="c")[1]) for i=1:M]
            M_tot = sum(M_list)
            
            # Construct the full H, S matrices:
            H_full = zeros((M_tot, M_tot))
            S_full = zeros((M_tot, M_tot))
            
            for i1=1:M, i2=i1:M
                
                i10, i11 = sum(M_list[1:i1-1])+1, sum(M_list[1:i1])
                i20, i21 = sum(M_list[1:i2-1])+1, sum(M_list[1:i2])
                
                ind_i1 = inds(oht_list[i1], tags="c")[1]
                ind_i2 = inds(oht_list[i2], tags="c")[1]
                
                if !(i1 in (j1, j2))
                    T_i1 = sdata.psi_list[i1][p] * sdata.psi_list[i1][p+1]
                    oht_i1 = oht_list[i1] * T_i1
                else
                    oht_i1 = oht_list[i1]
                end
                
                if !(i2 in (j1, j2)) && (i2 > i1)
                    if sdata.rev_flag[i1][i2-i1]
                        psi_i2 = ReverseMPS(sdata.psi_list[i2])
                    else
                        psi_i2 = sdata.psi_list[i2]
                    end
                    T_i2 = psi_i2[p] * psi_i2[p+1]
                    oht_i2 = oht_list[i2] * T_i2
                else
                    oht_i2 = oht_list[i2]
                end
                
                if i2==i1 # Diagonal block
                    
                    bind = block_ref[i1, i2]
                    
                    #tH = lH[bind]
                    #tH *= sdata.ham_list[i1][p] * sdata.ham_list[i1][p+1]
                    #tH *= rH_list[p][bind]
                    
                    mH = deepcopy(oht_i1)
                    mH *= sdata.ham_list[i1][p] * sdata.ham_list[i1][p+1]
                    mH *= dag(setprime(oht_i1,1))
                    
                    H_tens = lH[bind] * mH * rH_list[bind][p]
                    
                    H_full[i10:i11, i10:i11] = Array(H_tens, ind_i1, dag(setprime(ind_i1, 1)))
                    S_full[i10:i11, i10:i11] = Matrix(I, M_list[i1], M_list[i1])
                    
                else # Off-diagonal block
                    
                    bind = block_ref[i1, i2]
                    
                    mH = dag(setprime(oht_i1,1))
                    mH *= setprime(sdata.ham_list[i1][p],2,plev=0) * setprime(sdata.ham_list[i1][p+1],2,plev=0)
                    mH *= dag(setprime(setprime(sdata.perm_ops[i1][i2-i1][p],2,plev=0,tags="Site"),0,plev=1),tags="Site")
                    mH *= dag(setprime(setprime(sdata.perm_ops[i1][i2-i1][p+1],2,plev=0,tags="Site"),0,plev=1),tags="Site")
                    mH *= oht_i2
                    
                    H_tens = lH[bind] * mH * rH_list[bind][p]
                    
                    """
                    println("\n\n----------")
                    println(inds(lH[bind]))
                    println("----------")
                    println(inds(mH))
                    println("----------")
                    println(inds(rH_list[bind][p]))
                    println("----------\n\n")
                    """
                    
                    H_full[i10:i11, i20:i21] = Array(H_tens, dag(setprime(ind_i1, 1)), ind_i2)
                    H_full[i20:i21, i10:i11] = transpose(H_full[i10:i11, i20:i21])
                    
                    mS = dag(setprime(oht_i1,1))
                    mS *= sdata.perm_ops[i1][i2-i1][p] * sdata.perm_ops[i1][i2-i1][p+1]
                    mS *= oht_i2
                    
                    S_tens = lS[bind] * mS * rS_list[bind][p]
                    
                    S_full[i10:i11, i20:i21] = Array(S_tens, dag(setprime(ind_i1,1)), ind_i2)
                    S_full[i20:i21, i10:i11] = transpose(S_full[i10:i11, i20:i21])
                    
                end
                
            end
            
            # Make a copy to revert to at the end if the energy penalty is violated:
            sdata_copy = copy(sdata)
                
            #println("Before discarding: M_list = $(M_list), discards = nothing")
            
            # Discard any states with Infs and NaNs first:
            H_full, S_full, M_list, discards1 = DiscardOverlapping(H_full, S_full, M_list, criterion="InfNaN")
            oht_list[j1] = OHTCompIndexTensor(T_j1, discards=discards1[j1])
            oht_list[j2] = OHTCompIndexTensor(T_j2, discards=discards1[j2])

            #println("Discarded Infs/NaNs: M_list = $(M_list), discards = $(discards1)")
            
            # Now discard overlapping states:
            #H_disc, S_disc, M_disc, discards2 = DiscardOverlapping(H_full, S_full, M_list, criterion="overlap", tol=op.sd_dtol)
            H_full, S_full, M_list, discards2 = DiscardOverlapping(H_full, S_full, M_list, criterion="overlap", tol=op.sd_dtol)
            
            # Combine the two sets of discarded states:
            discards = []

            for i=1:M
                for i1 = 1:length(discards1[i]), i2 = 1:length(discards2[i])
                    if discards2[i][i2] >= discards1[i][i1]
                        discards2[i][i2] += 1
                    end
                end
                push!(discards, union(discards1[i], discards2[i]))
            end
            
            #println("Discarded overlapping: M_list = $(M_list), discards = $(discards)")

            """
            # Re-generate the one-hot tensor list:
            oht_disc = deepcopy(oht_list)
            oht_disc[j1] = OHTCompIndexTensor(T_j1, discards=discards[j1])
            oht_disc[j2] = OHTCompIndexTensor(T_j2, discards=discards[j2])
            """
            oht_list[j1] = OHTCompIndexTensor(T_j1, discards=discards[j1])
            oht_list[j2] = OHTCompIndexTensor(T_j2, discards=discards[j2])
            
            
            """
            # Do a full diagonalization:
            E, C, kappa = SolveGenEig(
                H_disc,
                S_disc,
                thresh=op.sd_thresh,
                eps=op.sd_eps
            )
            """
            
            E, C, kappa = SolveGenEig(
                H_full,
                S_full,
                thresh=op.sd_thresh,
                eps=op.sd_eps
            )

            """
            # Test FSWAPS on each optimized two-site tensor:
            do_swaps = [false for i=1:M]
            for i in (j1, j2)

                i0 = sum(M_disc[1:i-1])+1
                i1 = sum(M_disc[1:i])

                ind_i = inds(oht_disc[i], tags="c")[1]
                t_vec = ITensor(normalize(C[i0:i1,1]), ind_i)
                T = dag(t_vec) * oht_disc[i]
                linds = commoninds(T, sdata.psi_list[i][p])
                do_swaps[i] = TestFSWAP2(T, linds, sdata.mparams.psi_maxdim)

            end
            """
            
            # Test FSWAPS on each optimized two-site tensor:
            do_swaps = [false for i=1:M]
            for i in (j1, j2)

                i0 = sum(M_list[1:i-1])+1
                i1 = sum(M_list[1:i])

                ind_i = inds(oht_list[i], tags="c")[1]
                t_vec = ITensor(normalize(C[i0:i1,1]), ind_i)
                T = dag(t_vec) * oht_list[i]
                linds = commoninds(T, sdata.psi_list[i][p])
                do_swaps[i] = TestFSWAP2(T, linds, sdata.mparams.psi_maxdim)

            end
            

            # Enforce no swapping?
            if no_swap
                do_swaps = [false for i=1:M]
            end

            # Encode new FSWAPs into full H, S matrices:
            for i1 in (j1, j2)

                i10 = sum(M_list[1:i1-1])+1
                i11 = sum(M_list[1:i1])

                if do_swaps[i1]

                    # Construct FSWAP matrix:
                    fswap = BuildFermionicSwap(sdata.sites, p; dim=4);
                    fswap_tens = noprime(oht_list[i1] * fswap) * setprime(dag(oht_list[i1]), 1, tags="c") 
                    fswap_mat = Array(fswap_tens, inds(fswap_tens))

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

            inds_list = []
            for j in (j1, j2)
                T_j = sdata.psi_list[j][p] * sdata.psi_list[j][p+1]
                linds = commoninds(T_j, sdata.psi_list[j][p])
                rinds = commoninds(T_j, sdata.psi_list[j][p+1])
                push!(inds_list, [linds, rinds])
            end

            # Do TripleGenEig on all states to lower energy:
            T_list, E_new = TripleGenEigM(
                H_full,
                S_full,
                oht_list,
                M_list,
                op,
                (j1, j2),
                inds_list,
                sdata.mparams.psi_maxdim,
                nrep=8
            )
            
            #println("\n$(E[1])  $(E_new)\n")
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
                for j in (j1, j2)

                    if do_swaps[j]

                        swap_counter += 1
                        sdata.ord_list[j][p:p+1] = reverse(sdata.ord_list[j][p:p+1])

                        # Locally permute Hamiltonian and PMPOs:
                        fswap = BuildFermionicSwap(sdata.sites, p; dim=4);
                        fswap1 = setprime(fswap,2,plev=1)
                        fswap2 = setprime(fswap,2,plev=0)
                        
                        W_j = sdata.ham_list[j][p] * sdata.ham_list[j][p+1]
                        
                        W_j *= fswap1
                        setprime!(W_j,0,plev=2)
                        
                        W_j *= fswap2
                        setprime!(W_j,1,plev=2)
                        
                        linds = commoninds(W_j, sdata.ham_list[j][p])
                        
                        U,S,V = svd(W_j, linds)
                        sdata.ham_list[j][p] = U
                        sdata.ham_list[j][p+1] = S*V
                        
                        for i=1:M
                            if i<j
                                W_j = sdata.perm_ops[i][j-i][p] * sdata.perm_ops[i][j-i][p+1]
                                
                                W_j *= fswap1
                                setprime!(W_j,0,plev=2)
                                
                                linds = commoninds(W_j, sdata.perm_ops[i][j-i][p])
                                
                                U,S,V = svd(W_j, linds)
                                
                                sdata.perm_ops[i][j-i][p] = U
                                sdata.perm_ops[i][j-i][p+1] = S*V
                                
                            elseif i>j
                                W_j = sdata.perm_ops[j][i-j][p] * sdata.perm_ops[j][i-j][p+1]
                                
                                W_j *= fswap2
                                setprime!(W_j,1,plev=2)
                                
                                linds = commoninds(W_j, sdata.perm_ops[j][i-j][p])
                                
                                U,S,V = svd(W_j, linds)
                                
                                sdata.perm_ops[j][i-j][p] = U
                                sdata.perm_ops[j][i-j][p+1] = S*V
                                
                            end
                            
                        end
                        
                    end

                end
                
            else # Revert to previous parameters:
                
                for j in (j1, j2)
                    T_list[j] = sdata.psi_list[j][p] * sdata.psi_list[j][p+1]
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

            # Regardless of replacement, update "left" blocks:
            for j in (j1, j2)
                spec = ITensors.replacebond!(
                    sdata.psi_list[j],
                    p,
                    T_list[j];
                    maxdim=sdata.mparams.psi_maxdim,
                    #eigen_perturbation=drho,
                    ortho="left",
                    normalize=true,
                    svd_alg="qr_iteration"
                    #min_blockdim=1
                )
            end
            
            GenSubspaceMats!(sdata)
            SolveGenEig!(sdata)
            
            # Double-check that the energy is not too high!
            if sdata.E[1] > sdata_copy.E[1] + op.sd_etol
                
                # Revert to previous subspace:
                copyto!(sdata, sdata_copy)
                
                for j in (j1, j2)
                    
                    T_rev = sdata.psi_list[j][p]*sdata.psi_list[j][p+1]
                    
                    spec = ITensors.replacebond!(
                        sdata.psi_list[j],
                        p,
                        T_rev;
                        maxdim=sdata.mparams.psi_maxdim,
                        #eigen_perturbation=drho,
                        ortho="left",
                        normalize=true,
                        svd_alg="qr_iteration"
                        #min_blockdim=1
                    )
                    
                end
                
            end

            for i1=1:M, i2=i1:M

                bind = block_ref[i1,i2]

                if i1==i2 # Diagonal block

                    lH[bind] = UpdateBlock(
                        lH[bind], 
                        p, 
                        sdata.psi_list[i1], 
                        sdata.psi_list[i1], 
                        sdata.ham_list[i1], 
                        nothing
                    )

                else

                    if sdata.rev_flag[i1][i2-i1]
                        psi_i2 = ReverseMPS(sdata.psi_list[i2])
                    else
                        psi_i2 = sdata.psi_list[i2]
                    end

                    lH[bind] = UpdateBlock(
                        lH[bind], 
                        p, 
                        sdata.psi_list[i1], 
                        psi_i2, 
                        sdata.ham_list[i1], 
                        sdata.perm_ops[i1][i2-i1]
                    )

                    lS[bind] = UpdateBlock(
                        lS[bind], 
                        p, 
                        sdata.psi_list[i1], 
                        psi_i2, 
                        sdata.perm_ops[i1][i2-i1], 
                        nothing
                    )


                end

            end

            # Print some output
            if verbose
                print("Pair: [$(j1),$(j2)] ($(l)/$(op.maxiter)); ")
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
            
        # Recompute H, S, E, C, kappa:
        GenHams!(sdata)
        GenPermOps!(sdata)
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
# ...and insterts FSWAPS to reduce truncation error (permuting the orderings):`
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
        
        # Repeat over all orbitals:
        for p=1:N

            # Find sites [q(p), q(p)+/-1] for each ordering
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
            
            # Generate "one-hot" tensors:
            oht_list = []
            
            M_list = Int[]
            
            twos_states = sort(randperm(M)[1:n_twos])
            
            T_list = []
            
            for i=1:M
                
                orthogonalize!(sdata.psi_list[i], q_set[i][1])
                
                if i in twos_states
                    T_i = sdata.psi_list[i][q_set[i][1]] * sdata.psi_list[i][q_set[i][2]]
                else
                    T_i = sdata.psi_list[i][q_set[i][1]]
                end
                
                push!(T_list, T_i)
                
            end
            
            for i=1:M
                push!(oht_list, OHTCompIndexTensor(T_list[i]))
                push!(M_list, dim(inds(oht_list[end], tags="c")[1]))
            end
        
            # Compute H, S matrix elements:
            H_full = zeros((sum(M_list), sum(M_list)))
            S_full = zeros((sum(M_list), sum(M_list)))
            
            for i=1:M
                
                i0 = sum(M_list[1:i-1])+1
                i1 = sum(M_list[1:i])
                
                psi_i = [sdata.psi_list[i][q] for q=1:N]
                
                if i in twos_states
                    psi_i[q_set[i][1]] = oht_list[i]
                    psi_i[q_set[i][2]] = ITensor(1.0)
                else
                    psi_i[q_set[i][1]] = oht_list[i]
                end
                
                ind_i = inds(oht_list[i], tags="c")[1]
                
                tenH_ii = FullContract(psi_i, psi_i, mpo1=sdata.ham_list[i])
                
                matH_ii = Array(tenH_ii, (ind_i, setprime(dag(ind_i),1)))
                
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
                        if j in twos_states
                            T_revj = psi_j[q_revj[1]] * psi_j[q_revj[2]]
                        else
                            T_revj = psi_j[q_revj[2]]
                        end
                        oht_revj = OHTCompIndexTensor(T_revj)
                        
                        if j in twos_states
                            psi_j[q_revj[1]] = oht_revj
                            psi_j[q_revj[2]] = ITensor(1.0)
                        else
                            psi_j[q_revj[2]] = oht_revj
                        end
                        ind_j = inds(oht_revj, tags="c")[1]
                    else
                        psi_j = [deepcopy(sdata.psi_list[j][q]) for q=1:N]
                        if j in twos_states
                            psi_j[q_set[j][1]] = oht_list[j]
                            psi_j[q_set[j][2]] = ITensor(1.0)
                        else
                            psi_j[q_set[j][1]] = oht_list[j]
                        end
                        ind_j = inds(oht_list[j], tags="c")[1]
                    end
                    
                    tenH_ij = FullContract(psi_i, psi_j, mpo1=sdata.perm_ops[i][j-i], mpo2=sdata.ham_list[j])
                    tenS_ij = FullContract(psi_i, psi_j, mpo1=sdata.perm_ops[i][j-i])

                    #println(inds(tenH_ij))
                    matH_ij = Array(tenH_ij, (setprime(dag(ind_i),1), ind_j))
                    matS_ij = Array(tenS_ij, (setprime(dag(ind_i),1), ind_j))
                    
                    H_full[i0:i1,j0:j1] = real.(matH_ij)
                    H_full[j0:j1,i0:i1] = transpose(matH_ij)
                    
                    S_full[i0:i1,j0:j1] = real.(matS_ij)
                    S_full[j0:j1,i0:i1] = transpose(matS_ij)
                    
                    
                end
                
            end
            
            if (NaN in S_full) || (Inf in S_full)
                println("\n\nRuh-roh!\n\n")
            end
            
            # Make a copy to revert to at the end if the energy penalty is violated:
            sdata_copy = copy(sdata)
            
            # Make sure the svd is not going to bork...
            skip=false
            
            """
            try 
                F_test = svd(S_full, alg=LinearAlgebra.QRIteration())
            catch err
                skip=true
            end
            
            if cond(S_full) > 1e50
                skip=true
            end
            """
            
            if !(skip)
                
                # Discard any states with Infs and NaNs first:
                #println(NaN in S_full || Inf in S_full)
                H_full, S_full, M_list, discards1 = DiscardOverlapping(H_full, S_full, M_list, criterion="InfNaN")
                #println(NaN in S_full || Inf in S_full)
                oht_list = [OHTCompIndexTensor(T_list[i], discards=discards1[i]) for i=1:M]
                
                # Now discard overlapping states:
                H_disc, S_disc, M_disc, discards2 = DiscardOverlapping(H_full, S_full, M_list, criterion="overlap", tol=op.sd_dtol)
                
                # Combine the two sets of discarded states:
                discards = []
                
                for i=1:M
                    for i1 = 1:length(discards1[i]), i2 = 1:length(discards2[i])
                        if discards2[i][i2] >= discards1[i][i1]
                            discards2[i][i2] += 1
                        end
                    end
                    push!(discards, union(discards1[i], discards2[i]))
                end
                
                # Re-generate the one-hot tensor list:
                oht_disc = [OHTCompIndexTensor(T_list[i], discards=discards[i]) for i=1:M]

                #println(Inf in H_disc || NaN in H_disc)
                #println(Inf in S_disc || NaN in S_disc)
                
                # Do a full diagonalization:
                E, C, kappa = SolveGenEig(
                    H_disc,
                    S_disc,
                    thresh=op.sd_thresh,
                    eps=op.sd_eps
                )

                # Test FSWAPS on each optimized two-site tensor:
                do_swaps = [false for i=1:M]
                for i in twos_states

                    i0 = sum(M_disc[1:i-1])+1
                    i1 = sum(M_disc[1:i])

                    ind_i = inds(oht_disc[i], tags="c")[1]
                    t_vec = ITensor(normalize(C[i0:i1,1]), ind_i)
                    T = dag(t_vec) * oht_disc[i]
                    linds = commoninds(T, sdata.psi_list[i][q_set[i][1]])
                    do_swaps[i] = TestFSWAP2(T, linds, sdata.mparams.psi_maxdim)

                end

                # Enforce no swapping?
                if no_swap
                    do_swaps = [false for i=1:M]
                end

                # Encode new FSWAPs into full H, S matrices:
                for i in twos_states

                    i0 = sum(M_list[1:i-1])+1
                    i1 = sum(M_list[1:i])

                    if do_swaps[i]

                        # Construct FSWAP matrix:
                        fswap = BuildFermionicSwap(sdata.sites, q_set[i][1]; dim=4);
                        fswap_tens = noprime(oht_list[i] * fswap) * setprime(dag(oht_list[i]), 1, tags="c") 
                        fswap_mat = Array(fswap_tens, inds(fswap_tens))

                        for j=1:M

                            j0 = sum(M_list[1:j-1])+1
                            j1 = sum(M_list[1:j])

                            # Left-mult all subblocks in row i:
                            H_full[i0:i1,j0:j1] = fswap_mat * H_full[i0:i1, j0:j1]
                            S_full[i0:i1,j0:j1] = fswap_mat * S_full[i0:i1, j0:j1]

                            # Right-mult all subblocks in col i:
                            H_full[j0:j1,i0:i1] = H_full[j0:j1, i0:i1] * fswap_mat
                            S_full[j0:j1,i0:i1] = S_full[j0:j1, i0:i1] * fswap_mat

                        end

                    end

                end

                inds_list = []
                for i in twos_states
                    T_i = sdata.psi_list[i][q_set[i][1]] * sdata.psi_list[i][q_set[i][2]]
                    linds = commoninds(T_i, sdata.psi_list[i][q_set[i][1]])
                    rinds = commoninds(T_i, sdata.psi_list[i][q_set[i][2]])
                    push!(inds_list, [linds, rinds])
                end

                # Do TripleGenEig on all states to lower energy:
                T_list, E_new = TripleGenEigM(
                    H_full,
                    S_full,
                    oht_list,
                    M_list,
                    op,
                    twos_states,
                    inds_list,
                    sdata.mparams.psi_maxdim,
                    nrep=8
                )

                do_replace = true
                for i=1:M
                    if (NaN in T_list[i]) || (Inf in T_list[i])
                        do_replace = false
                    end
                end

                if (real(E_new) < real(E[1]) + op.sd_etol) && do_replace

                    # Update params, orderings
                    for i=1:M

                        if i in twos_states

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
                
            end

            # Print some output
            if verbose
                print("Sweep: $(l)/$(op.maxiter); ")
                print("orbital: $(p)/$(N); ")
                print("#swaps: $(swap_counter); ")
                print("E_min = $(round(sdata.E[1], digits=5)); ") 
                print("Delta = $(round(sdata.E[1]-sdata.chem_data.e_fci+sdata.chem_data.e_nuc, digits=5)); ")
                print("kappa_full = $(round(cond(S_full), sigdigits=3)); ")
                !(skip) && print("kappa_disc = $(round(cond(S_disc), sigdigits=3)); ")
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