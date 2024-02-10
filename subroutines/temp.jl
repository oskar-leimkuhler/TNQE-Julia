# A "sweep" algorithm for two states at a time (efficient on classical hardware):
function TwoSitePairSweep!(
        sdata::SubspaceProperties,
        op::OptimParameters;
        jperm=nothing,
        no_swap=false,
        verbose=false,
        debug_output=false
    )
    
    # For locally permuting the MPOs:
    lU, rV, qnvec = SWAPComponents(true)
    
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
        
        # Orthogonalize to site 1:
        orthogonalize!(sdata.psi_list[j1], 1)
        orthogonalize!(sdata.psi_list[j2], 1)
        
        # Do the same for any MPOs that may need to be locally modified:
        for j in (j1, j2)
            orthogonalize!(sdata.ham_list[j], 1)
            for i=j+1:M
                orthogonalize!(sdata.perm_ops[j][i-j], 1)
            end
        end
        
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
                    sdata.psi_list[i1],
                    mpo1 = sdata.ham_list[i1],
                    mpo2 = nothing,
                    p1=3,
                    inv=true
                )

                rS = CollectBlocks(
                    sdata.psi_list[i1],
                    sdata.psi_list[i1],
                    mpo1 = nothing,
                    mpo2 = nothing,
                    p1=3,
                    inv=true
                )
                
            end

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
                    
                    H_tens = lH[bind]
                    H_tens *= sdata.ham_list[i1][p]
                    H_tens *= sdata.ham_list[i1][p+1]
                    H_tens *= rH_list[bind][p]
                    
                    H_tens *= oht_i1
                    H_tens *= dag(setprime(oht_i1,1))
                    
                    S_tens = lS[bind]
                    S_tens *= oht_i1
                    S_tens *= setprime(setprime(dag(oht_i1),1,tags="Link"),1,tags="c")
                    S_tens *= rS_list[bind][p]
                    
                    H_full[i10:i11, i10:i11] = Hermitian(Array(H_tens, ind_i1, dag(setprime(ind_i1, 1))))
                    S_full[i10:i11, i10:i11] = Hermitian(Array(S_tens, ind_i1, dag(setprime(ind_i1, 1))))
                    
                else # Off-diagonal block
                    
                    bind = block_ref[i1, i2]
                    
                    H_tens = lH[bind]
                    H_tens *= setprime(sdata.ham_list[i1][p],2,plev=0)
                    H_tens *= setprime(sdata.perm_ops[i1][i2-i1][p],2,plev=1,tags="Site")
                    H_tens *= setprime(sdata.ham_list[i1][p+1],2,plev=0)
                    H_tens *= setprime(sdata.perm_ops[i1][i2-i1][p+1],2,plev=1,tags="Site")
                    H_tens *= rH_list[bind][p]
                    
                    H_tens *= dag(setprime(oht_i1,1))
                    H_tens *= oht_i2
                    
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
                    H_full[i20:i21, i10:i11] = conj.(transpose(H_full[i10:i11, i20:i21]))
                    
                    S_tens = lS[bind]
                    S_tens *= sdata.perm_ops[i1][i2-i1][p]
                    S_tens *= sdata.perm_ops[i1][i2-i1][p+1]
                    S_tens *= rS_list[bind][p]
                    
                    S_tens *= dag(setprime(oht_i1,1))
                    S_tens *= oht_i2
                    
                    S_full[i10:i11, i20:i21] = Array(S_tens, dag(setprime(ind_i1,1)), ind_i2)
                    S_full[i20:i21, i10:i11] = conj.(transpose(S_full[i10:i11, i20:i21]))
                    
                end
                
            end
            
            # Make a copy to revert to at the end if the energy penalty is violated:
            sdata_copy = copy(sdata)
                
            #println("Before discarding: M_list = $(M_list), discards = nothing")
            
            # Discard any states with Infs and NaNs first:
            H_full, S_full, M_list, discards1 = DiscardOverlapping(H_full, S_full, M_list, criterion="InfNaN")
            oht_list[j1] = OHTCompIndexTensor(T_j1, discards=discards1[j1])
            oht_list[j2] = OHTCompIndexTensor(T_j2, discards=discards1[j2])
            
            # Now discard overlapping states:
            H_full, S_full, M_list, discards2 = DiscardOverlapping(H_full, S_full, M_list, criterion="overlap", tol=op.sd_dtol, verbose=false)
            
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
            
            oht_list[j1] = OHTCompIndexTensor(T_j1, discards=discards[j1])
            oht_list[j2] = OHTCompIndexTensor(T_j2, discards=discards[j2])
            
            E, C, kappa = SolveGenEig(
                H_full,
                S_full,
                thresh=op.sd_thresh,
                eps=op.sd_eps
            )
            
            #println("\n$(E[1]) $(cond(S_full))\n")
            if (E[1] < -10.0) && (debug_output)
                #display(round.(H_full, sigdigits=3))
                #display(round.(S_full, sigdigits=3))
                #println(round.(diag(H_full), sigdigits=3))
                #println(round.(diag(S_full), sigdigits=3))
                
                # Find the erroneous block:
                
                println("Error detected! Diagnosing blocks...")
                for i1=1:M
                    i10, i11 = sum(M_list[1:i1-1])+1, sum(M_list[1:i1])
                    
                    H_i1i1 = H_full[i10:i11, i10:i11]
                    S_i1i1 = S_full[i10:i11, i10:i11]
                    
                    E_i1i1, C_i1i1, kappa_i1i1 = SolveGenEig(
                        H_i1i1,
                        S_i1i1,
                        thresh=op.sd_thresh,
                        eps=op.sd_eps
                    )
                    
                    println("$(i1), $(i1) block: E = $(E_i1i1[1])")
                    
                    for i2=i1+1:M
                        i20, i21 = sum(M_list[1:i2-1])+1, sum(M_list[1:i2])
                        
                        H_i1i2 = H_full[union(i10:i11, i20:i21), union(i10:i11, i20:i21)]
                        S_i1i2 = S_full[union(i10:i11, i20:i21), union(i10:i11, i20:i21)]

                        E_i1i2, C_i1i2, kappa_i1i2 = SolveGenEig(
                            H_i1i2,
                            S_i1i2,
                            thresh=op.sd_thresh,
                            eps=op.sd_eps
                        )
                        
                        println("$(i1), $(i2) block: E = $(E_i1i2[1])")
                        
                    end
                    
                end
                
            end
            
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
            
            E, C, kappa = SolveGenEig(
                H_full,
                S_full,
                thresh=op.sd_thresh,
                eps=op.sd_eps
            )

            #println("\n$(E[1])\n")

            inds_list = []
            for j in (j1, j2)
                T_j = sdata.psi_list[j][p] * sdata.psi_list[j][p+1]
                linds = commoninds(T_j, sdata.psi_list[j][p])
                rinds = commoninds(T_j, sdata.psi_list[j][p+1])
                push!(inds_list, [linds, rinds])
            end

            # Do TripleGenEig on all states to lower energy:
            T_list, V_list, E_new = TripleGenEigM(
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
                        
                        """
                        # Permute Hamiltonian and PMPO:
                        GenHams!(sdata, j_set=[j])
                        GenPermOps!(sdata)
                        GenSubspaceMats!(sdata)
                        SolveGenEig!(sdata)
                        """
                        
                        # Locally permute Hamiltonian and PMPOs:
                        PSWAP!(
                            sdata.ham_list[j], 
                            p, 
                            lU, 
                            rV, 
                            qnvec, 
                            prime_side=true, 
                            do_trunc=true, 
                            maxdim=sdata.mparams.ham_maxdim,
                            tol=sdata.mparams.ham_tol,
                        )
                        
                        PSWAP!(
                            sdata.ham_list[j], 
                            p, 
                            lU, 
                            rV, 
                            qnvec, 
                            prime_side=false, 
                            do_trunc=true, 
                            maxdim=sdata.mparams.ham_maxdim,
                            tol=sdata.mparams.ham_tol,
                        )
                        
                        #println(maxlinkdim(sdata.ham_list[j]))
                        
                        for i=1:M
                            if i<j # j is unprimed!
                                
                                """
                                if sdata.rev_flag[i][j-i]
                                    ApplyPhases!(sdata.perm_ops[i][j-i], sdata.sites)
                                    PSWAP!(sdata.perm_ops[i][j-i], p, lU, rV, qnvec)
                                    ApplyPhases!(sdata.perm_ops[i][j-i], sdata.sites)
                                else
                                    #dag!(swapprime!(sdata.perm_ops[i][j-i],1,0,tags="Site"),tags="Site")
                                    PSWAP!(sdata.perm_ops[i][j-i], p, lU, rV, qnvec)
                                    #dag!(swapprime!(sdata.perm_ops[i][j-i],1,0,tags="Site"),tags="Site")
                                end
                                """
                                
                                #PSWAP!(sdata.perm_ops[i][j-i], p, lU, rV, qnvec, prime_side=false)
                                
                                PSWAP!(
                                    sdata.perm_ops[i][j-i], 
                                    p, 
                                    lU, 
                                    rV, 
                                    qnvec, 
                                    prime_side=false, 
                                    do_trunc=true, 
                                    maxdim=sdata.mparams.perm_maxdim,
                                    tol=sdata.mparams.perm_tol,
                                )
                                #dag!(swapprime!(sdata.perm_ops[i][j-i],1,0,tags="Site"),tags="Site")
                                
                                #println("Permuted perm op [$(i),$(j)] on unprimed side")
                                
                            elseif i>j # j is primed!
                                
                                """
                                if sdata.rev_flag[j][i-j]
                                    ApplyPhases!(sdata.perm_ops[j][i-j], sdata.sites)
                                    PSWAP!(sdata.perm_ops[i][j-i], p, lU, rV, qnvec)
                                    ApplyPhases!(sdata.perm_ops[i][j-i], sdata.sites)
                                else
                                    #dag!(swapprime!(sdata.perm_ops[i][j-i],1,0,tags="Site"),tags="Site")
                                    PSWAP!(sdata.perm_ops[j][i-j], p, lU, rV, qnvec)
                                    #dag!(swapprime!(sdata.perm_ops[i][j-i],1,0,tags="Site"),tags="Site")
                                end
                                """
                                
                                #PSWAP!(sdata.perm_ops[j][i-j], p, lU, rV, qnvec, prime_side=true)
                                
                                PSWAP!(
                                    sdata.perm_ops[j][i-j], 
                                    p, 
                                    lU, 
                                    rV, 
                                    qnvec, 
                                    prime_side=true, 
                                    do_trunc=true, 
                                    maxdim=sdata.mparams.perm_maxdim,
                                    tol=sdata.mparams.perm_tol,
                                )
                                
                                #println("Permuted perm op [$(j),$(i)] on primed side")
                                
                            end
                            
                        end
                        
                    end

                end
                
                
                
            else # Revert to previous parameters:
                
                for j in (j1, j2)
                    V_list[j] = sdata.psi_list[j][p]
                    T_list[j] = sdata.psi_list[j][p+1]
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
                
                sdata.psi_list[j][p] = V_list[j]
                sdata.psi_list[j][p+1] = T_list[j]
                
                # Make sure new state is normalized:
                #normalize!(sdata.psi_list[j])
                sdata.psi_list[j][p+1] *= 1.0/sqrt(norm(sdata.psi_list[j]))
                
                # Shift ortho. center of locally modified MPOs:
                W_j = sdata.ham_list[j][p] * sdata.ham_list[j][p+1]
                U,S,V = svd(
                    W_j, 
                    commoninds(W_j, sdata.ham_list[j][p]), 
                    alg="qr_iteration", 
                    cutoff=sdata.mparams.ham_tol, 
                    maxdim=sdata.mparams.ham_maxdim
                )
                sdata.ham_list[j][p] = U
                sdata.ham_list[j][p+1] = S*V
                
                for i=j+1:M
                    W_j = sdata.perm_ops[j][i-j][p] * sdata.perm_ops[j][i-j][p+1]
                    U,S,V = svd(
                        W_j, 
                        commoninds(W_j, sdata.perm_ops[j][i-j][p]), 
                        alg="qr_iteration",
                        cutoff=sdata.mparams.perm_tol, 
                        maxdim=sdata.mparams.perm_maxdim
                    )
                    sdata.perm_ops[j][i-j][p] = U
                    sdata.perm_ops[j][i-j][p+1] = S*V
                end
                
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
                    
                    # Make sure new state is normalized:
                    #normalize!(sdata.psi_list[j])
                    sdata.psi_list[j][p+1] *= 1.0/sqrt(norm(sdata.psi_list[j]))
                    
                    # Shift ortho. center of locally modified MPOs:
                    W_j = sdata.ham_list[j][p] * sdata.ham_list[j][p+1]
                    U,S,V = svd(W_j, commoninds(W_j, sdata.ham_list[j][p]), alg="qr_iteration")
                    sdata.ham_list[j][p] = U
                    sdata.ham_list[j][p+1] = S*V

                    for i=j+1:M
                        W_j = sdata.perm_ops[j][i-j][p] * sdata.perm_ops[j][i-j][p+1]
                        U,S,V = svd(W_j, commoninds(W_j, sdata.perm_ops[j][i-j][p]), alg="qr_iteration")
                        sdata.perm_ops[j][i-j][p] = U
                        sdata.perm_ops[j][i-j][p+1] = S*V
                    end
                    
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
                    
                    lS[bind] = UpdateBlock(
                        lS[bind], 
                        p, 
                        sdata.psi_list[i1], 
                        sdata.psi_list[i1], 
                        nothing, 
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
            
        for j in (j1,j2) # Make sure these states are normalized:
            normalize!(sdata.psi_list[j])
        end
        
        # Recompute H, S, E, C, kappa:
        GenHams!(sdata, j_set=[j1,j2])
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