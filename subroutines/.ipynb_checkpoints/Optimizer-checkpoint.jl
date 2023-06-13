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
    alpha::Float64=1.0 # "Swappiness"
    gamma::Float64=1.0 # Selectivity of bond SWAP favorability
    
    # Generalized eigenvalue solver parameters:
    thresh::String="inversion" # "none", "projection", or "inversion"
    eps::Float64=1e-8 # Singular value cutoff
    
    # Site decomposition solver parameters:
    sd_method::String="geneig" # "geneig" or "triple_geneig"
    sd_thresh::String="inversion" # "none", "projection", or "inversion"
    sd_eps::Float64=1e-8 # Singular value cutoff
    sd_penalty::Float64=1.0 # Penalty factor for truncation error
    sd_swap_penalty::Float64=1.0 # Penalty factor for truncation error
    sd_dtol::Float64=1e-4 # OHT-state overlap discard tolerance
    
end



# returns a block of the full H or S matrix:
function MatrixBlock(psi1d, block_tensor; psi2d=[ITensor(1.0)], diag=false)
    
    dim1, dim2 = length(psi1d), length(psi2d)
    
    if diag
        
        A_block = zeros((dim1,dim1))
        
        for k=1:dim1, l=k:dim1
            A_block[k,l] = scalar( psi1d[k] * block_tensor * setprime(dag(psi1d[l]), 1) )
            A_block[l,k] = A_block[k,l]
        end
        
    else
        
        A_block = zeros((dim1,dim2))
        
        for k=1:dim1, l=1:dim2
            
            A_block[k,l] = scalar( psi2d[l] * block_tensor * setprime(dag(psi1d[k]), 1) )
            
        end
        
    end
    
    return A_block
    
end


# Expands the subspace matrices over the decomposition of the psi:
function ExpandSubspace(
        H_mat, 
        S_mat, 
        psid,
        H_blocks,
        S_blocks,
        b_ref
    )
    
    M = size(H_mat, 1)
    M_list = [length(psid[j]) for j=1:M]
    M_tot = sum(M_list)
    
    H_full = zeros((M_tot,M_tot))
    S_full = zeros((M_tot,M_tot))
    
    for i=1:M, j=i:M
        
        i0, i1 = sum(M_list[1:i-1])+1, sum(M_list[1:i])
        j0, j1 = sum(M_list[1:j-1])+1, sum(M_list[1:j])
        
        if M_list[i]==1 && M_list[j]==1
            H_full[i0,j0] = H_mat[i,j]
            H_full[j0,i0] = H_mat[i,j]
            S_full[i0,j0] = S_mat[i,j]
            S_full[j0,i0] = S_mat[i,j]
        else
            
            if i==j
                H_full[i0:i1,i0:i1] = MatrixBlock(psid[i], H_blocks[b_ref[i,i]], diag=true)
                S_full[i0:i1,i0:i1] = Matrix(1.0I, (M_list[i],M_list[i]))
            else
                H_full[i0:i1,j0:j1] = MatrixBlock(psid[i], H_blocks[b_ref[i,j]], psi2d=psid[j])
                H_full[j0:j1,i0:i1] = transpose(H_full[i0:i1,j0:j1])
                S_full[i0:i1,j0:j1] = MatrixBlock(psid[i], S_blocks[b_ref[i,j]], psi2d=psid[j])
                S_full[j0:j1,i0:i1] = transpose(S_full[i0:i1,j0:j1])
            end
            
        end
        
    end
    
    return H_full, S_full
    
end


# Return a list of all the one-hot tensors with the same inds and QN flux
function OneHotTensors(T; force_diag=false)
    
    T_inds = inds(T)
    
    # Scale factor in case T is not normalized
    # (OHTs must have the same norm as T)
    #sf = sqrt(scalar(dag(T)*T))
    
    oht_list = []
    
    for c in CartesianIndices(Array(T, T_inds))
        
        c_inds = Tuple(c)
        ivs = []
        for i=1:length(T_inds)
            push!(ivs, T_inds[i]=>c_inds[i])
        end
        
        oht = onehot(Tuple(ivs)...)#*sf

        if flux(oht)==flux(T)
            if (force_diag && all(y->y==c_inds[1],c_inds)) || !(force_diag)
                push!(oht_list, oht)
            end
        end
        
    end
    
    return oht_list
    
end


# Check if states are overlapping too much and discard if they are
function DiscardOverlapping(H_full, S_full, oht_list, j, tol)
    
    println("\nBefore: ", cond(S_full))
    E, C, kappa = SolveGenEig(H_full, S_full, thresh="inversion", eps=1e-8)
    println(E[1])
    
    M_list = [length(oht) for oht in oht_list]
    M = length(M_list)
    M_tot = sum(M_list)
            
    j0, j1 = sum(M_list[1:j-1])+1, sum(M_list[1:j])
    println(M_list)

    S_red = zeros((M_tot-M_list[j], M_tot-M_list[j]))
    S_red[1:j0-1,1:j0-1] = S_full[1:j0-1,1:j0-1]
    S_red[j0:end,1:j0-1] = S_full[j1+1:end,1:j0-1]
    S_red[1:j0-1,j0:end] = S_full[1:j0-1,j1+1:end]
    S_red[j0:end,j0:end] = S_full[j1+1:end,j1+1:end]
            
    discards = []
    
    for l=1:M_list[j]

        vphi = vcat(S_full[1:j0-1,j0+l-1], S_full[j1+1:end,j0+l-1])
        
        sqnm = transpose(vphi) * pinv(S_red, atol=1e-12) * vphi
        
        if 1.0-sqnm < tol # Mark the state for discarding
            push!(discards, l)
        end

    end
            
    # Discard the overlapping states
    oht_list[j] = [oht_list[j][l] for l in setdiff(1:M_list[j], discards)]
    
    H0 = H_full
    S0 = S_full
    
    for (i,l) in enumerate(discards)
        
        col = j0+l-i
        
        H1 = zeros((M_tot-i,M_tot-i))
        H1[1:col-1,1:col-1] = H0[1:col-1,1:col-1]
        H1[1:col-1,col:end] = H0[1:col-1,col+1:end]
        H1[col:end,1:col-1] = H0[col+1:end,1:col-1]
        H1[col:end,col:end] = H0[col+1:end,col+1:end]
        
        S1 = zeros((M_tot-i,M_tot-i))
        S1[1:col-1,1:col-1] = S0[1:col-1,1:col-1]
        S1[1:col-1,col:end] = S0[1:col-1,col+1:end]
        S1[col:end,1:col-1] = S0[col+1:end,1:col-1]
        S1[col:end,col:end] = S0[col+1:end,col+1:end]
        
        H0 = H1
        S0 = S1
        
    end
    
    H_full = H0
    S_full = S0
    
    println("After: ", cond(S_full))
    E, C, kappa = SolveGenEig(H_full, S_full, thresh="inversion", eps=1e-8)
    println(E[1])
    
    return H_full, S_full, oht_list
    
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
        yB = psi2[p] * setprime(mpo2[p],2,plev=1)
        block *= Ax
        block *= yB
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
        
        #println(inds(block))
        
        if csites != nothing && p==csites[2]
            block *= combos[2]
        end
        if csites != nothing && p==csites[1]
            block *= setprime(dag(combos[1]),1)
        end
        
    end
    
    return block
    
end


#Permutation operators computed relative to the indicated psi:
function RelativePermOps(sdata, j)
    
    M = length(sdata.ord_list)
    
    jperm_ops = []
    jrev_flag = []
    
    for i=1:M
        if i != j
            P_ij, fl_ij = FastPMPO(
                sdata.sites,
                sdata.ord_list[j],
                sdata.ord_list[i],
                tol=1e-16,
                maxdim=2^20
            )

            push!(jperm_ops, P_ij)
            push!(jrev_flag, fl_ij)
        else
            push!(jperm_ops, nothing)
            push!(jrev_flag, false)
        end
    end
    
    return jperm_ops, jrev_flag
    
end


function TruncError(c_j, j, p, psi_decomp, sd)
    
    T = sum([c_j[k]*psi_decomp[j][k] for k=1:length(c_j)])
    
    linds = commoninds(T, sd.psi_list[j][p])
    
    U,S,V = svd(T, linds, maxdim=sd.mparams.psi_maxdim)
    
    return norm(U*S*V - T)
    
end


function GeneralizedTwoSite!(
        sdata::SubspaceProperties,
        op::OptimParameters;
        j_offset=0,
        verbose=false
    )
    
    N = sdata.chem_data.N_spt
    M = sdata.mparams.M
    
    if op.numopt==-1
        op.numopt=M
    end
    
    if verbose
        println("\nGENERALIZED TWO-SITE SWEEP ALGORITHM:")
    end
    
    # The iteration loop:
    for l=1:op.maxiter
        
        # Noise at this iteration:
        lnoise = op.noise[minimum([l,end])]
        ldelta = op.delta[minimum([l,end])]
        
        for (jc, j) in enumerate(circshift(collect(1:M), j_offset)[1:op.numopt])
            
            # Re-compute permutation operators:
            jperm_ops, jrev_flag = RelativePermOps(sdata, j)
            
            for i=1:M
                orthogonalize!(sdata.psi_list[i], 1)
            end
            
            psi_list_c = deepcopy(sdata.psi_list)
            for i=1:M
                if jrev_flag[i]
                    psi_list_c[i] = ReverseMPS(psi_list_c[i])
                end
            end
            
            for s=1:op.numloop
                
                orthogonalize!(sdata.psi_list[j], 1)
                psi_list_c[j] = sdata.psi_list[j]
                
                # Contract the "right" blocks:
                rH_list = [CollectBlocks(
                        sdata.psi_list[j],
                        psi_list_c[i],
                        mpo1 = sdata.ham_list[j],
                        mpo2 = jperm_ops[i],
                        inv=true
                        ) for i=1:M]
                
                rS_list = [CollectBlocks(
                        sdata.psi_list[j],
                        psi_list_c[i],
                        mpo1 = jperm_ops[i],
                        inv=true
                        ) for i=1:M]
                
                # Initialize the "left" blocks:
                lH = [ITensor(1.0) for i=1:M]
                lS = [ITensor(1.0) for i=1:M]
                
                for p=1:N-1
                    
                    # Select the correct "right" blocks:
                    H_blocks = [rH_list[i][p] for i=1:M]
                    S_blocks = [rS_list[i][p] for i=1:M]
                    
                    # Generate "lock" tensors by fast contraction
                    for i in setdiff(1:M, j)
                        
                        yP1 = psi_list_c[i][p+1] * setprime(jperm_ops[i][p+1],2,plev=1)
                        yP2 = psi_list_c[i][p] * setprime(jperm_ops[i][p],2,plev=1)
                        H_blocks[i] *= yP1
                        H_blocks[i] *= setprime(sdata.ham_list[j][p+1],2,plev=0)
                        H_blocks[i] *= yP2
                        H_blocks[i] *= setprime(sdata.ham_list[j][p],2,plev=0)
                        H_blocks[i] *= lH[i]
                        
                        S_blocks[i] *= psi_list_c[i][p+1] * jperm_ops[i][p+1]
                        S_blocks[i] *= psi_list_c[i][p] * jperm_ops[i][p]
                        S_blocks[i] *= lS[i]
                        
                    end
                    
                    H_blocks[j] *= sdata.ham_list[j][p+1]
                    H_blocks[j] *= sdata.ham_list[j][p]
                    H_blocks[j] *= lH[j]
                    
                    block_ref = zeros(Int, (M,M))
                    block_ref[:,j] = collect(1:M)
                    block_ref[j,:] = collect(1:M)
                    
                    # Dagging and swap-priming:
                    for i in setdiff(1:M, j)
                        if i<j 
                            H_blocks[i] = swapprime(dag(H_blocks[i]),0,1)
                            S_blocks[i] = swapprime(dag(S_blocks[i]),0,1)
                        end
                    end
                    
                    # Generate the "key" tensors:
                    psi_decomp = [[ITensor(1.0)] for i=1:M]
                    T = sdata.psi_list[j][p] * sdata.psi_list[j][p+1]
                    psi_decomp[j] = OneHotTensors(T)
                    
                    # Generate the subspace matrices and one-hot tensor list:
                    H_full, S_full = ExpandSubspace(
                        sdata.H_mat,
                        sdata.S_mat,
                        psi_decomp,
                        H_blocks,
                        S_blocks,
                        block_ref
                    )
                    
                    #H_full, S_full, psi_decomp = DiscardOverlapping(H_full, S_full, psi_decomp, j, op.sd_dtol)
                    
                    M_list = [length(psi_decomp[i]) for i=1:M]
                    M_j = length(psi_decomp[j])
                    
                    do_replace = true
                    
                    if op.sd_method=="geneig"
                    
                        # Solve the generalized eigenvalue problem:
                        E, C, kappa = SolveGenEig(
                            H_full, 
                            S_full, 
                            thresh=op.sd_thresh,
                            eps=op.sd_eps
                        )

                        t_vec = deepcopy(real.(C[j:(j+M_j-1),1]))
                        normalize!(t_vec)
                        
                        if (NaN in t_vec) || (Inf in t_vec)
                            do_replace = false
                        end
                        
                        if E[1] >= op.sd_penalty*sdata.E[1]
                            do_replace = false
                        end
                        
                    elseif op.method=="triple_geneig"
                        
                        # Solve the generalized eigenvalue problem:
                        E, C, kappa = SolveGenEig(
                            H_full, 
                            S_full, 
                            thresh=op.sd_thresh,
                            eps=op.sd_eps
                        )
                        
                        j0, j1 = sum(M_list[1:j-1])+1, sum(M_list[1:j])
                        t_vec = real.(C[j0:j1,1])
                        if !(NaN in t_vec) && !(Inf in t_vec) && norm(t_vec) > 1e-16
                            t_vec = normalize(t_vec)
                        else
                            T_old = sdata.psi_list[j][p]*sdata.psi_list[j][p+1]
                            t_vec = [scalar(T_old*dag(psi_decomp[j][k])) for k=1:M_list[j]]
                        end
                        
                        ps = [p, p+1]
                        
                        # Construct the new tensor:
                        T = sum([t_vec[k]*psi_decomp[j][k] for k=1:M_list[j]])
                        
                        for s=1:3
                        
                            for p_ind=1:2

                                # Split by SVD:
                                linds = commoninds(T, sdata.psi_list[j][ps[p_ind]])

                                U, S, V = svd(T, linds, maxdim=sdata.mparams.psi_maxdim)

                                # single-site one-hot decomposition at site p:
                                psi_decomp_p = deepcopy(psi_decomp)

                                if p_ind==1
                                    psi_decomp_p[j] = OneHotTensors(U*S)
                                    psi_decomp_p[j] = [psid * V for psid in psi_decomp_p[j]]
                                else
                                    psi_decomp_p[j] = OneHotTensors(S*V)
                                    psi_decomp_p[j] = [psid * U for psid in psi_decomp_p[j]]
                                end

                                M_list_p = [length(psi_decomp_p[i]) for i=1:M]

                                # Re-compute H_full and S_full for site p:
                                H_full_p, S_full_p = ExpandSubspace(
                                    sdata.H_mat,
                                    sdata.S_mat,
                                    psi_decomp_p,
                                    H_blocks,
                                    S_blocks,
                                    block_ref
                                )

                                # Solve the generalized eigenvalue problem:
                                E_p, C_p, kappa_p = SolveGenEig(
                                    H_full_p, 
                                    S_full_p, 
                                    thresh=op.sd_thresh,
                                    eps=op.sd_eps
                                )

                                j0, j1 = sum(M_list_p[1:j-1])+1, sum(M_list_p[1:j])
                                t_vec_p = real.(C_p[j0:j1,1])
                                if !(NaN in t_vec_p) && !(Inf in t_vec_p) && norm(t_vec_p) > 1e-16
                                    t_vec_p = normalize(t_vec_p)
                                else
                                    t_vec_p = [scalar(T*dag(psi_decomp_p[j][k])) for k=1:M_list_p[j]]
                                end

                                T = sum([t_vec_p[k]*psi_decomp_p[j][k] for k=1:M_list_p[j]])

                                t_vec = [scalar(T*dag(psi_decomp[j][k])) for k=1:M_list[j]]

                                if s==3 && p_ind==2 && real(E_p[1]) > op.sd_penalty*sdata.E[1]
                                    do_replace = false
                                end

                            end
                            
                        end
                        
                    end
                    
                    # Check the truncation error is not too large:
                    if TruncError(t_vec, j, p, psi_decomp, sdata) > op.ttol    
                        do_replace = false
                    end
                    
                    # Check the coefficient vector is not numerically zero:
                    if abs(norm(t_vec)-0.0) < 1e-16
                        do_replace = false
                    end
                        
                    # Replace the tensors of the MPS:
                    if do_replace
                        
                        t_vec += ldelta*normalize(randn(M_j)) # Random noise term
                        normalize!(t_vec)
                        
                        # Construct the new tensor and plug it in:
                        T_new = sum([t_vec[k]*psi_decomp[j][k] for k=1:M_j])

                        # Mix the new tensor with the old tensor:
                        T_old = sdata.psi_list[j][p]*sdata.psi_list[j][p+1]
                        T_new = (1.0-op.theta)*T_new + op.theta*T_old
                        T_new *= 1.0/sqrt(scalar(T_new*dag(T_new)))
                        
                         # Generate the "noise" term:
                        pmpo = ITensors.ProjMPO(sdata.ham_list[j])
                        ITensors.set_nsite!(pmpo,2)
                        ITensors.position!(pmpo, sdata.psi_list[j], p)
                        drho = lnoise*ITensors.noiseterm(pmpo,T_new,"left")
                        
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
                        
                    end
                    
                    # Shift orthogonality center to site p+1:
                    orthogonalize!(sdata.psi_list[j], p+1)
                    normalize!(sdata.psi_list[j])
                    psi_list_c[j] = sdata.psi_list[j]
                    
                    # Update the "left" blocks:
                    for i=1:M
                        lH[i] = UpdateBlock(
                            lH[i], 
                            p, 
                            sdata.psi_list[j], 
                            psi_list_c[i], 
                            sdata.ham_list[j], 
                            jperm_ops[i]
                        )
                        
                        if i != j
                            lS[i] = UpdateBlock(
                                lS[i], 
                                p, 
                                sdata.psi_list[j], 
                                psi_list_c[i], 
                                jperm_ops[i],
                                nothing
                            )
                        end
                    end
                    
                    # Recompute H, S, E, C, kappa:
                    GenSubspaceMats!(sdata)
                    SolveGenEig!(sdata)
                    
                    # Print some output
                    if verbose
                        print("Iter: $(l)/$(op.maxiter); ")
                        print("state: $(jc)/$(op.numopt); ")
                        print("sweep: $(s)/$(op.numloop); ")
                        print("bond: $(p)/$(N-1); ")
                        print("E_min = $(round(sdata.E[1], digits=5)); ") 
                        print("kappa = $(round(sdata.kappa, sigdigits=3))     \r")
                        flush(stdout)
                    end
                    
                end
                
            end
            
        end
        
    end
    
    if verbose
        println("\nDone!\n")
    end
    
    
end



function OneSitePairSweep!(
        sdata::SubspaceProperties,
        op::OptimParameters;
        jpairs=nothing,
        verbose=false
    )
    
    N = sdata.chem_data.N_spt
    M = sdata.mparams.M
    
    if verbose
        println("\nTWO-STATE, ONE-SITE SWEEP ALGORITHM:")
    end
    
    if jpairs == nothing
        # Default is all pairs:
        jpairs = []
        for k=1:Int(floor(M/2))
            for j=1:M
                push!(jpairs, sort([j, mod1(j+k, M)]))
            end
        end
    end
    
    # The iteration loop:
    for l=1:op.maxiter
        
        # Noise at this iteration:
        lnoise = op.noise[minimum([l,end])]
        ldelta = op.delta[minimum([l,end])]
        
        for jc=1:length(jpairs)
            
            j1, j2 = jpairs[jc]
            
            # Do we need to reverse j2?
            if sdata.rev_flag[j1][j2-j1]
                sdata.ord_list[j2] = reverse(sdata.ord_list[j2])
                sdata.psi_list[j2] = ReverseMPS(sdata.psi_list[j2])
                ApplyPhases!(sdata.psi_list[j2])
                GenHams!(sdata)
                GenPermOps!(sdata)
            end
            
            # Re-compute permutation operators:
            j1perm_ops, j1rev_flag = RelativePermOps(sdata, j1)
            j2perm_ops, j2rev_flag = RelativePermOps(sdata, j2)
            jperm_ops = [j1perm_ops, j2perm_ops]
            jrev_flag = [j1rev_flag, j2rev_flag]
            
            for i=1:M
                orthogonalize!(sdata.psi_list[i], 1)
            end
            
            for s=1:op.numloop
                
                orthogonalize!(sdata.psi_list[j1], 1)
                orthogonalize!(sdata.psi_list[j2], 1)
                
                # Fill in the block_ref as we construct the "lock" tensors:
                block_ref = zeros(Int,(M,M))
                state_ref = []
                
                # Contract the "right" blocks and init the "left" blocks:
                rH_list, rS_list = Any[], Any[]
                lS, lH = Any[], Any[]
                
                # The j-i blocks:
                for i in setdiff(collect(1:M), [j1,j2])
                    for j_ind=1:2
                        
                        j = [j1,j2][j_ind]
                        
                        if jrev_flag[j_ind][i]
                            psi_i = ReverseMPS(sdata.psi_list[i])
                        else
                            psi_i = sdata.psi_list[i]
                        end
                    
                        rH_ji = CollectBlocks(
                            sdata.psi_list[j],
                            psi_i,
                            mpo1 = sdata.ham_list[j],
                            mpo2 = jperm_ops[j_ind][i],
                            p1=2,
                            inv=true
                        )

                        rS_ji = CollectBlocks(
                            sdata.psi_list[j],
                            psi_i,
                            mpo1 = jperm_ops[j_ind][i],
                            mpo2 = nothing,
                            p1=2,
                            inv=true
                        )
                    
                        push!(rH_list, rH_ji)
                        push!(rS_list, rS_ji)
                        push!(lH, ITensor(1.0))
                        push!(lS, ITensor(1.0))
                    
                        block_ref[i,j] = length(lH)
                        block_ref[j,i] = length(lH)
                        push!(state_ref, [j,i])
                        
                    end
                    
                end
                
                # The j-j blocks:
                for jj in [[j1,j1], [j2,j2], [j1,j2]]
                    
                    if jj[1]==jj[2]
                        jjperm_op = nothing
                    else
                        jjperm_op = jperm_ops[1][j2]
                    end
                    
                    rH_jj = CollectBlocks(
                        sdata.psi_list[jj[1]],
                        sdata.psi_list[jj[2]],
                        mpo1 = sdata.ham_list[jj[1]],
                        mpo2 = jjperm_op,
                        p1=2,
                        inv=true
                    )
                    
                    if jj[1]==jj[2]
                        rS_jj = [nothing for q=1:N]
                    else
                        rS_jj = CollectBlocks(
                            sdata.psi_list[jj[1]],
                            sdata.psi_list[jj[2]],
                            mpo1 = jjperm_op,
                            mpo2 = nothing,
                            p1=2,
                            inv=true
                        )
                    end
                    
                    push!(rH_list, rH_jj)
                    push!(rS_list, rS_jj)
                    
                    push!(lH, ITensor(1.0))
                    push!(lS, ITensor(1.0))
                    
                    block_ref[jj[1],jj[2]] = length(lH)
                    block_ref[jj[2],jj[1]] = length(lH)
                    push!(state_ref, jj)
                    
                end
                
                for p=1:N
                    
                    # Select the correct "right" blocks:
                    H_blocks = [rH_list[b][p] for b=1:length(rH_list)]
                    S_blocks = [rS_list[b][p] for b=1:length(rS_list)]
                    
                    # i-j blocks:
                    for bind=1:length(state_ref)-3
                        
                        j, i = state_ref[bind]
                        
                        j_ind = findall(x->x==j, [j1,j2])[1]
                        
                        if i==j # Diagonal block:
                
                            H_blocks[bind] *= sdata.ham_list[j][p]
                            H_blocks[bind] *= lH[bind]
                            
                        else # Off-diagonal block:
                            
                            if jrev_flag[j_ind][i]
                                psi_i = ReverseMPS(sdata.psi_list[i])
                            else
                                psi_i = sdata.psi_list[i]
                            end
                            
                            yP = psi_i[p] * setprime(jperm_ops[j_ind][i][p],2,plev=1)
                            H_blocks[bind] *= yP
                            H_blocks[bind] *= setprime(sdata.ham_list[j][p],2,plev=0)
                            H_blocks[bind] *= lH[bind]

                            S_blocks[bind] *= psi_i[p] * jperm_ops[j_ind][i][p]
                            S_blocks[bind] *= lS[bind]
                            
                        end
                        
                    end
                    
                    # j-j diagonal blocks:
                    for j in [j1, j2]
                        
                        bind = block_ref[j,j]
                        
                        H_blocks[bind] *= sdata.ham_list[j][p]
                        H_blocks[bind] *= lH[bind]
                        
                    end
                    
                    # j-j off-diagonal block:
                    bind = block_ref[j1,j2]
                    
                    H_blocks[bind] *= setprime(jperm_ops[1][j2][p],2,plev=1)
                    H_blocks[bind] *= setprime(sdata.ham_list[j1][p],2,plev=0)
                    H_blocks[bind] *= lH[bind]

                    S_blocks[bind] *= jperm_ops[1][j2][p]
                    S_blocks[bind] *= lS[bind]
                    
                    
                    psi_decomp = [[ITensor(1.0)] for i=1:M]
                    
                    # Dagging and swap-priming:
                    for bind=1:length(state_ref)
                        j,i = state_ref[bind]
                        if i < j
                            H_blocks[bind] = swapprime(dag(H_blocks[bind]), 1, 0)
                            S_blocks[bind] = swapprime(dag(S_blocks[bind]), 1, 0)
                        end
                    end
                    
                    # Generate the "key" tensors:
                    for j in [j1, j2]
                        T = sdata.psi_list[j][p]
                        psi_decomp[j] = OneHotTensors(T)
                    end
                    
                    # Generate the subspace matrices:
                    H_full, S_full = ExpandSubspace(
                        sdata.H_mat,
                        sdata.S_mat,
                        psi_decomp,
                        H_blocks,
                        S_blocks,
                        block_ref
                    )
                    
                    #H_full, S_full, psi_decomp = DiscardOverlapping(H_full, S_full, psi_decomp, j1, op.sd_dtol)
                    #H_full, S_full, psi_decomp = DiscardOverlapping(H_full, S_full, psi_decomp, j2, op.sd_dtol)
                    
                    M_list = [length(psi_decomp[i]) for i=1:M]

                    # Solve the generalized eigenvalue problem:
                    E, C, kappa = SolveGenEig(
                        H_full, 
                        S_full, 
                        thresh=op.sd_thresh,
                        eps=op.sd_eps
                    )

                    # Replace the tensors:
                    for i in [j1, j2]

                        i0, i1 = sum(M_list[1:i-1])+1, sum(M_list[1:i])

                        t_vec = real.(C[i0:i1,1])
                        
                        #display(t_vec)
                        
                        if !(NaN in t_vec) && !(Inf in t_vec) && (norm(t_vec) > 1e-16) && (real(E[1]) <= op.sd_penalty*sdata.E[1])
                            
                            normalize!(t_vec)
                            t_vec += ldelta*normalize(randn(M_list[i])) # Random noise term
                            normalize!(t_vec)

                            # Construct the new tensor and plug it in:
                            T_new = sum([t_vec[k]*psi_decomp[i][k] for k=1:M_list[i]])

                            # Mix the new tensor with the old tensor:
                            T_old = sdata.psi_list[i][p]
                            T_new = (1.0-op.theta)*T_new + op.theta*T_old
                            T_new *= 1.0/sqrt(scalar(T_new*dag(T_new)))

                            # Replace the tensor of the MPS:
                            sdata.psi_list[i][p] = T_new
                            
                        end
                        
                        # Shift orthogonality center to site p+1:
                        if p != N
                            orthogonalize!(sdata.psi_list[i], p+1)
                            normalize!(sdata.psi_list[i])
                        end

                    end

                    # Update the "left" blocks:
                    for bind=1:length(state_ref)
                        
                        j, i = state_ref[bind]
                        
                        j_ind = findall(x->x==j, [j1,j2])[1]

                        if i==j

                            lH[bind] = UpdateBlock(
                                lH[bind], 
                                p, 
                                sdata.psi_list[i], 
                                sdata.psi_list[i], 
                                sdata.ham_list[i], 
                                nothing
                            )

                        else
                            
                            if jrev_flag[j_ind][i]
                                psi_i = ReverseMPS(sdata.psi_list[i])
                            else
                                psi_i = sdata.psi_list[i]
                            end

                            lH[bind] = UpdateBlock(
                                lH[bind], 
                                p, 
                                sdata.psi_list[j], 
                                psi_i, 
                                sdata.ham_list[j], 
                                jperm_ops[j_ind][i]
                            )

                            lS[bind] = UpdateBlock(
                                lS[bind], 
                                p, 
                                sdata.psi_list[j], 
                                psi_i, 
                                jperm_ops[j_ind][i],
                                nothing
                            )

                        end

                    end
                    
                    # Recompute H, S, E, C, kappa:
                    GenSubspaceMats!(sdata)
                    SolveGenEig!(sdata)
                    
                    # Print some output
                    if verbose
                        print("Iter: $(l)/$(op.maxiter); ")
                        print("pair: $(jc)/$(length(jpairs)); ")
                        print("sweep: $(s)/$(op.numloop); ")
                        print("site: $(p)/$(N); ")
                        print("E_min = $(round(sdata.E[1], digits=5)); ") 
                        print("kappa = $(round(sdata.kappa, sigdigits=3))     \r")
                        flush(stdout)
                    end
                    
                end
                
            end
            
        end
        
    end
    
    if verbose
        println("\nDone!\n")
    end
    
end



function TwoSitePairSweep!(
        sdata::SubspaceProperties,
        op::OptimParameters;
        jpairs=nothing,
        verbose=false
    )
    
    N = sdata.chem_data.N_spt
    M = sdata.mparams.M
    
    if verbose
        println("\nTWO-STATE, TWO-SITE SWEEP ALGORITHM:")
    end
    
    if jpairs == nothing
        # Default is all pairs:
        jpairs = []
        for k=1:Int(floor(M/2))
            for j=1:M
                push!(jpairs, sort([j, mod1(j+k, M)]))
            end
        end
    end
    
    # The iteration loop:
    for l=1:op.maxiter
        
        # Noise at this iteration:
        lnoise = op.noise[minimum([l,end])]
        ldelta = op.delta[minimum([l,end])]
        
        for jc=1:length(jpairs)
            
            j1, j2 = jpairs[jc]
            
            # Do we need to reverse j2?
            if sdata.rev_flag[j1][j2-j1]
                sdata.ord_list[j2] = reverse(sdata.ord_list[j2])
                sdata.psi_list[j2] = ReverseMPS(sdata.psi_list[j2])
                ApplyPhases!(sdata.psi_list[j2])
                GenHams!(sdata)
                GenPermOps!(sdata)
            end
            
            # Re-compute permutation operators:
            j1perm_ops, j1rev_flag = RelativePermOps(sdata, j1)
            j2perm_ops, j2rev_flag = RelativePermOps(sdata, j2)
            jperm_ops = [j1perm_ops, j2perm_ops]
            jrev_flag = [j1rev_flag, j2rev_flag]
            
            for i=1:M
                orthogonalize!(sdata.psi_list[i], 1)
            end
            
            for s=1:op.numloop
                
                orthogonalize!(sdata.psi_list[j1], 1)
                orthogonalize!(sdata.psi_list[j2], 1)
                
                # Fill in the block_ref as we construct the "lock" tensors:
                block_ref = zeros(Int,(M,M))
                state_ref = []
                
                # Contract the "right" blocks and init the "left" blocks:
                rH_list, rS_list = Any[], Any[]
                lS, lH = Any[], Any[]
                
                # The j-i blocks:
                for i in setdiff(collect(1:M), [j1,j2])
                    for j_ind=1:2
                        
                        j = [j1,j2][j_ind]
                        
                        if jrev_flag[j_ind][i]
                            psi_i = ReverseMPS(sdata.psi_list[i])
                        else
                            psi_i = sdata.psi_list[i]
                        end
                    
                        rH_ji = CollectBlocks(
                            sdata.psi_list[j],
                            psi_i,
                            mpo1 = sdata.ham_list[j],
                            mpo2 = jperm_ops[j_ind][i],
                            inv=true
                        )

                        rS_ji = CollectBlocks(
                            sdata.psi_list[j],
                            psi_i,
                            mpo1 = jperm_ops[j_ind][i],
                            mpo2 = nothing,
                            inv=true
                        )
                    
                        push!(rH_list, rH_ji)
                        push!(rS_list, rS_ji)
                        push!(lH, ITensor(1.0))
                        push!(lS, ITensor(1.0))
                    
                        block_ref[i,j] = length(lH)
                        block_ref[j,i] = length(lH)
                        push!(state_ref, [j,i])
                        
                    end
                    
                end
                
                # The j-j blocks:
                for jj in [[j1,j1], [j2,j2], [j1,j2]]
                    
                    if jj[1]==jj[2]
                        jjperm_op = nothing
                    else
                        jjperm_op = jperm_ops[1][j2]
                    end
                    
                    rH_jj = CollectBlocks(
                        sdata.psi_list[jj[1]],
                        sdata.psi_list[jj[2]],
                        mpo1 = sdata.ham_list[jj[1]],
                        mpo2 = jjperm_op,
                        inv=true
                    )
                    
                    if jj[1]==jj[2]
                        rS_jj = [nothing for q=1:N-1]
                    else
                        rS_jj = CollectBlocks(
                            sdata.psi_list[jj[1]],
                            sdata.psi_list[jj[2]],
                            mpo1 = jjperm_op,
                            mpo2 = nothing,
                            inv=true
                        )
                    end
                    
                    push!(rH_list, rH_jj)
                    push!(rS_list, rS_jj)
                    
                    push!(lH, ITensor(1.0))
                    push!(lS, ITensor(1.0))
                    
                    block_ref[jj[1],jj[2]] = length(lH)
                    block_ref[jj[2],jj[1]] = length(lH)
                    push!(state_ref, jj)
                    
                end
                
                for p=1:N-1
                    
                    # Select the correct "right" blocks:
                    H_blocks = [rH_list[b][p] for b=1:length(rH_list)]
                    S_blocks = [rS_list[b][p] for b=1:length(rS_list)]
                    
                    # i-j blocks:
                    for bind=1:length(state_ref)-3
                        
                        j, i = state_ref[bind]
                        
                        j_ind = findall(x->x==j, [j1,j2])[1]
                        
                        if i==j # Diagonal block:
                            
                            H_blocks[bind] *= sdata.ham_list[j][p+1]
                            H_blocks[bind] *= sdata.ham_list[j][p]
                            H_blocks[bind] *= lH[bind]
                            
                        else # Off-diagonal block:
                            
                            if jrev_flag[j_ind][i]
                                psi_i = ReverseMPS(sdata.psi_list[i])
                            else
                                psi_i = sdata.psi_list[i]
                            end
                            
                            yP1 = psi_i[p+1] * setprime(jperm_ops[j_ind][i][p+1],2,plev=1)
                            yP2 = psi_i[p] * setprime(jperm_ops[j_ind][i][p],2,plev=1)
                            H_blocks[bind] *= yP1
                            H_blocks[bind] *= setprime(sdata.ham_list[j][p+1],2,plev=0)
                            H_blocks[bind] *= yP2
                            H_blocks[bind] *= setprime(sdata.ham_list[j][p],2,plev=0)
                            H_blocks[bind] *= lH[bind]

                            S_blocks[bind] *= psi_i[p+1] * jperm_ops[j_ind][i][p+1]
                            S_blocks[bind] *= psi_i[p] * jperm_ops[j_ind][i][p]
                            S_blocks[bind] *= lS[bind]
                            
                        end
                        
                    end
                    
                    # j-j diagonal blocks:
                    for j in [j1, j2]
                        
                        bind = block_ref[j,j]
                        
                        H_blocks[bind] *= sdata.ham_list[j][p+1]
                        H_blocks[bind] *= sdata.ham_list[j][p]
                        H_blocks[bind] *= lH[bind]
                        
                    end
                    
                    # j-j off-diagonal block:
                    bind = block_ref[j1,j2]
                    
                    H_blocks[bind] *= setprime(jperm_ops[1][j2][p+1],2,plev=1)
                    H_blocks[bind] *= setprime(sdata.ham_list[j1][p+1],2,plev=0)
                    H_blocks[bind] *= setprime(jperm_ops[1][j2][p],2,plev=1)
                    H_blocks[bind] *= setprime(sdata.ham_list[j1][p],2,plev=0)
                    H_blocks[bind] *= lH[bind]

                    S_blocks[bind] *= jperm_ops[1][j2][p+1]
                    S_blocks[bind] *= jperm_ops[1][j2][p]
                    S_blocks[bind] *= lS[bind]
                    
                    
                    psi_decomp = [[ITensor(1.0)] for i=1:M]
                    
                    # Dagging and swap-priming:
                    for bind=1:length(state_ref)
                        j,i = state_ref[bind]
                        if i < j
                            H_blocks[bind] = swapprime(dag(H_blocks[bind]), 1, 0)
                            S_blocks[bind] = swapprime(dag(S_blocks[bind]), 1, 0)
                        end
                    end
                    
                    # Generate the "key" tensors:
                    for j in [j1, j2]
                        T = sdata.psi_list[j][p] * sdata.psi_list[j][p+1]
                        psi_decomp[j] = OneHotTensors(T)
                    end
                    
                    # Generate the subspace matrices:
                    H_full, S_full = ExpandSubspace(
                        sdata.H_mat,
                        sdata.S_mat,
                        psi_decomp,
                        H_blocks,
                        S_blocks,
                        block_ref
                    )
                    
                    #H_full, S_full, psi_decomp = DiscardOverlapping(H_full, S_full, psi_decomp, j1, op.sd_dtol)
                    #H_full, S_full, psi_decomp = DiscardOverlapping(H_full, S_full, psi_decomp, j2, op.sd_dtol)
                    
                    M_list = [length(psi_decomp[i]) for i=1:M]
                    
                    do_replace = true
                    
                    if op.sd_method=="geneig"
                        
                        # Solve the generalized eigenvalue problem:
                        E, C, kappa = SolveGenEig(
                            H_full, 
                            S_full, 
                            thresh=op.sd_thresh,
                            eps=op.sd_eps
                        )
                        
                        t_vecs = []
                        for (idx, i) in enumerate([j1, j2])
                            i0, i1 = sum(M_list[1:i-1])+1, sum(M_list[1:i])
                            t_vec = real.(C[i0:i1,1])
                            push!(t_vecs, normalize(t_vec))
                        end
                        
                        if real(E[1]) >= op.sd_penalty*sdata.E[1]
                            do_replace = false
                        end
                    
                    elseif op.sd_method=="triple_geneig"
                        
                        # Solve the generalized eigenvalue problem:
                        E, C, kappa = SolveGenEig(
                            H_full, 
                            S_full, 
                            thresh=op.sd_thresh,
                            eps=op.sd_eps
                        )
                        
                        t_vecs = []
                        for (idx, i) in enumerate([j1, j2])
                            i0, i1 = sum(M_list[1:i-1])+1, sum(M_list[1:i])
                            t_vec = real.(C[i0:i1,1])
                            if !(NaN in t_vec) && !(Inf in t_vec) && norm(t_vec) > 1e-16
                                push!(t_vecs, normalize(t_vec))
                            else
                                T_old = sdata.psi_list[i][p]*sdata.psi_list[i][p+1]
                                push!(t_vecs, [scalar(T_old*dag(psi_decomp[i][k])) for k=1:M_list[i]])
                            end
                        end
                        
                        ps = [p, p+1]
                        
                        # Construct the new tensors:
                        Tj1 = sum([t_vecs[1][k]*psi_decomp[j1][k] for k=1:M_list[j1]])
                        Tj2 = sum([t_vecs[2][k]*psi_decomp[j2][k] for k=1:M_list[j2]])
                        
                        for s=1:3
                        
                            for p_ind=1:2

                                # Split by SVD:
                                linds_j1 = commoninds(Tj1, sdata.psi_list[j1][ps[p_ind]])
                                linds_j2 = commoninds(Tj2, sdata.psi_list[j2][ps[p_ind]])

                                Uj1, Sj1, Vj1 = svd(Tj1, linds_j1, maxdim=sdata.mparams.psi_maxdim)#, min_blockdim=sdata.mparams.psi_maxdim)
                                Uj2, Sj2, Vj2 = svd(Tj2, linds_j2, maxdim=sdata.mparams.psi_maxdim)#, min_blockdim=sdata.mparams.psi_maxdim)

                                # single-site one-hot decomposition at site p:
                                psi_decomp_p = deepcopy(psi_decomp)

                                if p_ind==1
                                    psi_decomp_p[j1] = OneHotTensors(Uj1*Sj1)
                                    psi_decomp_p[j1] = [psid * Vj1 for psid in psi_decomp_p[j1]]
                                    psi_decomp_p[j2] = OneHotTensors(Uj2*Sj2)
                                    psi_decomp_p[j2] = [psid * Vj2 for psid in psi_decomp_p[j2]]
                                else
                                    psi_decomp_p[j1] = OneHotTensors(Sj1*Vj1)
                                    psi_decomp_p[j1] = [psid * Uj1 for psid in psi_decomp_p[j1]]
                                    psi_decomp_p[j2] = OneHotTensors(Sj2*Vj2)
                                    psi_decomp_p[j2] = [psid * Uj2 for psid in psi_decomp_p[j2]]
                                end

                                M_list_p = [length(psi_decomp_p[i]) for i=1:M]

                                # Re-compute H_full and S_full for site p:
                                H_full_p, S_full_p = ExpandSubspace(
                                    sdata.H_mat,
                                    sdata.S_mat,
                                    psi_decomp_p,
                                    H_blocks,
                                    S_blocks,
                                    block_ref
                                )

                                # Solve the generalized eigenvalue problem:
                                E_p, C_p, kappa_p = SolveGenEig(
                                    H_full_p, 
                                    S_full_p, 
                                    thresh=op.sd_thresh,
                                    eps=op.sd_eps
                                )

                                t_vecs_p = []
                                for (idx, i) in enumerate([j1, j2])
                                    i0, i1 = sum(M_list_p[1:i-1])+1, sum(M_list_p[1:i])
                                    t_vec = real.(C_p[i0:i1,1])
                                    if !(NaN in t_vec) && !(Inf in t_vec) && norm(t_vec) > 1e-16
                                        push!(t_vecs_p, normalize(t_vec))
                                    else
                                        push!(t_vecs_p, [scalar([Tj1,Tj2][idx]*dag(psi_decomp_p[i][k])) for k=1:M_list_p[i]])
                                    end
                                end

                                Tj1 = sum([t_vecs_p[1][k]*psi_decomp_p[j1][k] for k=1:M_list_p[j1]])
                                Tj2 = sum([t_vecs_p[2][k]*psi_decomp_p[j2][k] for k=1:M_list_p[j2]])

                                t_vecs = [
                                    [scalar(Tj1*dag(psi_decomp[j1][k])) for k=1:M_list[j1]], 
                                    [scalar(Tj2*dag(psi_decomp[j2][k])) for k=1:M_list[j2]]
                                ]

                                #println("\n$(real(E_p[1]))  $(sdata.E[1])")

                                if s==3 && p_ind==2 && real(E_p[1]) > op.sd_penalty*sdata.E[1]
                                    do_replace = false
                                end

                            end
                            
                        end
                        
                    end
        
                    if (NaN in t_vecs[1]) || (Inf in t_vecs[1]) || (NaN in t_vecs[2]) || (Inf in t_vecs[2])
                        do_replace = false
                    end
                    
                    # Check the truncation error is not too large:
                    if TruncError(t_vecs[1],j1,p,psi_decomp,sdata) > op.ttol || TruncError(t_vecs[2],j2,p,psi_decomp,sdata) > op.ttol
                        do_replace = false
                    end
                    
                    # Do the replacement:
                    if do_replace
                        
                        for (idx, i) in enumerate([j1, j2])
                            
                            T_old = sdata.psi_list[i][p]*sdata.psi_list[i][p+1]
                            
                            t_vec = t_vecs[idx]
                                
                            if norm(t_vec) < 1e-16
                                t_vec = [scalar(T_old*dag(psi_decomp[i][k])) for k=1:M_list[i]]
                            end
                            
                            t_vec += ldelta*normalize(randn(M_list[i])) # Random noise term
                            normalize!(t_vec)

                            # Construct the new tensor and plug it in:
                            T_new = sum([t_vec[k]*psi_decomp[i][k] for k=1:M_list[i]])

                            # Mix the new tensor with the old tensor:
                            T_old = sdata.psi_list[i][p]*sdata.psi_list[i][p+1]
                            T_new = (1.0-op.theta)*T_new + op.theta*T_old
                            T_new *= 1.0/sqrt(scalar(T_new*dag(T_new)))
                            
                            # Generate the "noise" term:
                            pmpo = ITensors.ProjMPO(sdata.ham_list[i])
                            ITensors.set_nsite!(pmpo,2)
                            ITensors.position!(pmpo, sdata.psi_list[i], p)
                            drho = lnoise*ITensors.noiseterm(pmpo,T_new,"left")

                            # Replace the tensors of the MPS:
                            spec = ITensors.replacebond!(
                                sdata.psi_list[i],
                                p,
                                T_new;
                                maxdim=sdata.mparams.psi_maxdim,
                                eigen_perturbation=drho,
                                ortho="left",
                                normalize=true,
                                svd_alg="qr_iteration"
                                #min_blockdim=1
                            )
                            
                        end
                            
                    end
                    
                    # Shift orthogonality center to site p+1:
                    for i in [j1, j2]
                        orthogonalize!(sdata.psi_list[i], p+1)
                        normalize!(sdata.psi_list[i])
                    end

                    # Update the "left" blocks:
                    for bind=1:length(state_ref)
                        
                        j, i = state_ref[bind]
                        
                        j_ind = findall(x->x==j, [j1,j2])[1]

                        if i==j

                            lH[bind] = UpdateBlock(
                                lH[bind], 
                                p, 
                                sdata.psi_list[i], 
                                sdata.psi_list[i], 
                                sdata.ham_list[i], 
                                nothing
                            )

                        else
                            
                            if jrev_flag[j_ind][i]
                                psi_i = ReverseMPS(sdata.psi_list[i])
                            else
                                psi_i = sdata.psi_list[i]
                            end

                            lH[bind] = UpdateBlock(
                                lH[bind], 
                                p, 
                                sdata.psi_list[j], 
                                psi_i, 
                                sdata.ham_list[j], 
                                jperm_ops[j_ind][i]
                            )

                            lS[bind] = UpdateBlock(
                                lS[bind], 
                                p, 
                                sdata.psi_list[j], 
                                psi_i, 
                                jperm_ops[j_ind][i],
                                nothing
                            )

                        end

                    end
                    
                    # Recompute H, S, E, C, kappa:
                    GenSubspaceMats!(sdata)
                    SolveGenEig!(sdata)
                    
                    # Print some output
                    if verbose
                        print("Iter: $(l)/$(op.maxiter); ")
                        print("pair: $(jc)/$(length(jpairs)); ")
                        print("sweep: $(s)/$(op.numloop); ")
                        print("bond: $(p)/$(N-1); ")
                        print("E_min = $(round(sdata.E[1], digits=5)); ") 
                        print("kappa = $(round(sdata.kappa, sigdigits=3))     \r")
                        flush(stdout)
                    end
                    
                end
                
            end
            
        end
        
    end
    
    if verbose
        println("\nDone!\n")
    end
    
end


# Choose states (j1,j2)
function ChooseStates(
        sdata;
        j1=1,
        j2=nothing
    )

    M = sdata.mparams.M
    
    # Re-compute permutation operators:
    j1perm_ops, j1rev_flag = RelativePermOps(sdata, j1)
    
    # Apply reversals
    # Do we need to reverse j2?
    if j2 != nothing
        
        if sdata.rev_flag[j1][j2-j1]
            sdata.ord_list[j2] = reverse(sdata.ord_list[j2])
            sdata.psi_list[j2] = ReverseMPS(sdata.psi_list[j2])
            ApplyPhases!(sdata.psi_list[j2])
            GenHams!(sdata)
            GenPermOps!(sdata)
        end
        
        j1perm_ops, j1rev_flag = RelativePermOps(sdata, j1)
        j2perm_ops, j2rev_flag = RelativePermOps(sdata, j2)
        
        jperm_ops = [j1perm_ops, j2perm_ops]
        jrev_flag = [j1rev_flag, j2rev_flag]
        
    else
        
        # Re-compute permutation operators:
        jperm_ops, jrev_flag = RelativePermOps(sdata, j1)
        
    end

    for i=1:M
        orthogonalize!(sdata.psi_list[i], 1)
    end
    
    return jperm_ops, jrev_flag
    
end


# Pre-contract bottom blocks
function TwoStateCollectBlocks(
        sdata,
        jperm_ops,
        jrev_flag;
        j1=1,
        j2=2
    )
    
    M = sdata.mparams.M
    N = sdata.chem_data.N_spt
    
    orthogonalize!(sdata.psi_list[j1], 1)
    orthogonalize!(sdata.psi_list[j2], 1)

    # Fill in the block_ref as we construct the "lock" tensors:
    block_ref = zeros(Int,(M,M))
    state_ref = []

    # Contract the "right" blocks and init the "left" blocks:
    rH_list, rS_list = Any[], Any[]
    lS, lH = Any[], Any[]

    # The j-i blocks:
    for i in setdiff(collect(1:M), [j1,j2])
        for j_ind=1:2

            j = [j1,j2][j_ind]

            if jrev_flag[j_ind][i]
                psi_i = ReverseMPS(sdata.psi_list[i])
                ham_i = ReverseMPO(sdata.ham_list[i])
            else
                psi_i = sdata.psi_list[i]
                ham_i = sdata.ham_list[i]
            end

            rH_ji = CollectBlocks(
                sdata.psi_list[j],
                psi_i,
                mpo1 = jperm_ops[j_ind][i],
                mpo2 = ham_i,
                inv=true
            )

            rS_ji = CollectBlocks(
                sdata.psi_list[j],
                psi_i,
                mpo1 = jperm_ops[j_ind][i],
                mpo2 = nothing,
                inv=true
            )

            push!(rH_list, rH_ji)
            push!(rS_list, rS_ji)
            push!(lH, ITensor(1.0))
            push!(lS, ITensor(1.0))

            block_ref[i,j] = length(lH)
            block_ref[j,i] = length(lH)
            push!(state_ref, [j,i])

        end

    end

    # The j-j blocks:
    for jj in [[j1,j1], [j2,j2], [j1,j2]]

        if jj[1]==jj[2]
            jjperm_op = nothing
        else
            jjperm_op = jperm_ops[1][j2]
        end

        rH_jj = CollectBlocks(
            sdata.psi_list[jj[1]],
            sdata.psi_list[jj[2]],
            mpo1 = jjperm_op,
            mpo2 = sdata.ham_list[jj[2]],
            inv=true
        )

        if jj[1]==jj[2]
            rS_jj = [nothing for q=1:N-1]
        else
            rS_jj = CollectBlocks(
                sdata.psi_list[jj[1]],
                sdata.psi_list[jj[2]],
                mpo1 = jjperm_op,
                mpo2 = nothing,
                inv=true
            )
        end

        push!(rH_list, rH_jj)
        push!(rS_list, rS_jj)

        push!(lH, ITensor(1.0))
        push!(lS, ITensor(1.0))

        block_ref[jj[1],jj[2]] = length(lH)
        block_ref[jj[2],jj[1]] = length(lH)
        push!(state_ref, jj)

    end
    
    return lH, lS, rH_list, rS_list, block_ref, state_ref
    
end


function ContractLockTensors(
        sdata,
        jperm_ops,
        jrev_flag,
        p,
        rH_list,
        rS_list,
        lH,
        lS,
        block_ref,
        state_ref;
        j1=1,
        j2=2
    )
    
    # Select the correct "right" blocks:
    H_blocks = [rH_list[b][p] for b=1:length(rH_list)]
    S_blocks = [rS_list[b][p] for b=1:length(rS_list)]

    # Contract lock tensors

    # i-j blocks:
    for bind=1:length(state_ref)-3

        j, i = state_ref[bind]

        j_ind = findall(x->x==j, [j1,j2])[1]

        if i==j # Diagonal block:

            H_blocks[bind] *= sdata.ham_list[j][p+1]
            H_blocks[bind] *= sdata.ham_list[j][p]
            H_blocks[bind] *= lH[bind]

        else # Off-diagonal block:
            
            #println("######################")
            #println(inds(H_blocks[bind]))
            #println("----------------------")
            #println(inds(lH[bind]))
            #println("######################")

            if jrev_flag[j_ind][i]
                psi_i = ReverseMPS(sdata.psi_list[i])
                ham_i = ReverseMPO(sdata.ham_list[i])
            else
                psi_i = sdata.psi_list[i]
                ham_i = sdata.ham_list[i]
            end
            
            yP1 = psi_i[p+1] * setprime(ham_i[p+1],2,plev=1)
            yP2 = psi_i[p] * setprime(ham_i[p],2,plev=1)
            H_blocks[bind] *= yP1
            H_blocks[bind] *= setprime(jperm_ops[j_ind][i][p+1],2,plev=0)
            H_blocks[bind] *= yP2
            H_blocks[bind] *= setprime(jperm_ops[j_ind][i][p],2,plev=0)
            H_blocks[bind] *= lH[bind]

            S_blocks[bind] *= psi_i[p+1] * jperm_ops[j_ind][i][p+1]
            S_blocks[bind] *= psi_i[p] * jperm_ops[j_ind][i][p]
            S_blocks[bind] *= lS[bind]

        end

    end

    # j-j diagonal blocks:
    for j in [j1, j2]

        bind = block_ref[j,j]

        H_blocks[bind] *= sdata.ham_list[j][p+1]
        H_blocks[bind] *= sdata.ham_list[j][p]
        H_blocks[bind] *= lH[bind]

    end

    # j-j off-diagonal block:
    bind = block_ref[j1,j2]
    
    H_blocks[bind] *= setprime(sdata.ham_list[j2][p+1],2,plev=1)
    H_blocks[bind] *= setprime(jperm_ops[1][j2][p+1],2,plev=0)
    H_blocks[bind] *= setprime(sdata.ham_list[j2][p],2,plev=1)
    H_blocks[bind] *= setprime(jperm_ops[1][j2][p],2,plev=0)
    H_blocks[bind] *= lH[bind]

    S_blocks[bind] *= jperm_ops[1][j2][p+1]
    S_blocks[bind] *= jperm_ops[1][j2][p]
    S_blocks[bind] *= lS[bind]

    # Dagging and swap-priming:
    for bind=1:length(state_ref)
        j,i = state_ref[bind]
        if i < j
            H_blocks[bind] = swapprime(dag(H_blocks[bind]), 1, 0)
            S_blocks[bind] = swapprime(dag(S_blocks[bind]), 1, 0)
        end
    end
    
    return H_blocks, S_blocks
    
end


function TripleGenEig(
        H_full,
        S_full,
        psi_decomp,
        H_blocks,
        S_blocks,
        state_ref,
        block_ref,
        sdata,
        op,
        p;
        j1=1,
        j2=2
    )
    
    M = sdata.mparams.M
    
    M_list = [length(psi_decomp[i]) for i=1:M]
    
    # We will return this at the end:
    E_min = sdata.E[1]
    
    # Solve the generalized eigenvalue problem:
    E, C, kappa = SolveGenEig(
        H_full, 
        S_full, 
        thresh=op.sd_thresh,
        eps=op.sd_eps
    )

    t_vecs = []
    for (idx, i) in enumerate([j1, j2])
        i0, i1 = sum(M_list[1:i-1])+1, sum(M_list[1:i])
        t_vec = real.(C[i0:i1,1])
        if !(NaN in t_vec) && !(Inf in t_vec) && norm(t_vec) > 1e-16
            push!(t_vecs, normalize(t_vec))
        else
            T_old = sdata.psi_list[i][p]*sdata.psi_list[i][p+1]
            push!(t_vecs, [scalar(T_old*dag(psi_decomp[i][k])) for k=1:M_list[i]])
        end
    end

    ps = [p, p+1]

    # Construct the new tensors:
    Tj1 = sum([t_vecs[1][k]*psi_decomp[j1][k] for k=1:M_list[j1]])
    Tj2 = sum([t_vecs[2][k]*psi_decomp[j2][k] for k=1:M_list[j2]])

    for s=1:3

        for p_ind=1:2

            # Split by SVD:
            linds_j1 = commoninds(Tj1, sdata.psi_list[j1][ps[p_ind]])
            linds_j2 = commoninds(Tj2, sdata.psi_list[j2][ps[p_ind]])

            Uj1, Sj1, Vj1 = svd(Tj1, linds_j1, maxdim=sdata.mparams.psi_maxdim)#, min_blockdim=sdata.mparams.psi_maxdim)
            Uj2, Sj2, Vj2 = svd(Tj2, linds_j2, maxdim=sdata.mparams.psi_maxdim)#, min_blockdim=sdata.mparams.psi_maxdim)

            # single-site one-hot decomposition at site p:
            psi_decomp_p = deepcopy(psi_decomp)

            if p_ind==1
                psi_decomp_p[j1] = OneHotTensors(Uj1*Sj1)
                psi_decomp_p[j1] = [psid * Vj1 for psid in psi_decomp_p[j1]]
                psi_decomp_p[j2] = OneHotTensors(Uj2*Sj2)
                psi_decomp_p[j2] = [psid * Vj2 for psid in psi_decomp_p[j2]]
            else
                psi_decomp_p[j1] = OneHotTensors(Sj1*Vj1)
                psi_decomp_p[j1] = [psid * Uj1 for psid in psi_decomp_p[j1]]
                psi_decomp_p[j2] = OneHotTensors(Sj2*Vj2)
                psi_decomp_p[j2] = [psid * Uj2 for psid in psi_decomp_p[j2]]
            end

            M_list_p = [length(psi_decomp_p[i]) for i=1:M]

            # Re-compute H_full and S_full for site p:
            H_full_p, S_full_p = ExpandSubspace(
                sdata.H_mat,
                sdata.S_mat,
                psi_decomp_p,
                H_blocks,
                S_blocks,
                block_ref
            )

            # Solve the generalized eigenvalue problem:
            E_p, C_p, kappa_p = SolveGenEig(
                H_full_p, 
                S_full_p, 
                thresh=op.sd_thresh,
                eps=op.sd_eps
            )

            t_vecs_p = []
            for (idx, i) in enumerate([j1, j2])
                i0, i1 = sum(M_list_p[1:i-1])+1, sum(M_list_p[1:i])
                t_vec = real.(C_p[i0:i1,1])
                if !(NaN in t_vec) && !(Inf in t_vec) && norm(t_vec) > 1e-16
                    push!(t_vecs_p, normalize(t_vec))
                else
                    push!(t_vecs_p, [scalar([Tj1,Tj2][idx]*dag(psi_decomp_p[i][k])) for k=1:M_list_p[i]])
                end
            end

            Tj1 = sum([t_vecs_p[1][k]*psi_decomp_p[j1][k] for k=1:M_list_p[j1]])
            Tj2 = sum([t_vecs_p[2][k]*psi_decomp_p[j2][k] for k=1:M_list_p[j2]])

            t_vecs = [
                [scalar(Tj1*dag(psi_decomp[j1][k])) for k=1:M_list[j1]], 
                [scalar(Tj2*dag(psi_decomp[j2][k])) for k=1:M_list[j2]]
            ]
            
            E_min = E_p[1]

        end

    end
    
    return E_min, t_vecs
    
end


function TripleGenEig1(
        H_full,
        S_full,
        psi_decomp,
        H_blocks,
        S_blocks,
        state_ref,
        block_ref,
        sdata,
        op,
        p,
        j
    )
    
    M = sdata.mparams.M
    
    M_list = [length(psi_decomp[i]) for i=1:M]
    
    # We will return this at the end:
    E_min = sdata.E[1]
    
    # Solve the generalized eigenvalue problem:
    E, C, kappa = SolveGenEig(
        H_full, 
        S_full, 
        thresh=op.sd_thresh,
        eps=op.sd_eps
    )

    j0, j1 = sum(M_list[1:j-1])+1, sum(M_list[1:j])
    t_vec = real.(C[j0:j1,1])
    if (NaN in t_vec) || (Inf in t_vec) || norm(t_vec) < 1e-16
        T_old = sdata.psi_list[j][p]*sdata.psi_list[j][p+1]
        t_vec = [scalar(T_old*dag(psi_decomp[j][k])) for k=1:M_list[j]]
    end

    ps = [p, p+1]

    # Construct the new tensors:
    Tj = sum([t_vec[k]*psi_decomp[j][k] for k=1:M_list[j]])

    for s=1:3

        for p_ind=1:2

            # Split by SVD:
            linds = commoninds(Tj, sdata.psi_list[j][ps[p_ind]])

            U, S, V = svd(Tj, linds, maxdim=sdata.mparams.psi_maxdim)

            # single-site one-hot decomposition at site p:
            psi_decomp_p = deepcopy(psi_decomp)

            if p_ind==1
                psi_decomp_p[j] = OneHotTensors(U*S)
                psi_decomp_p[j] = [psid * V for psid in psi_decomp_p[j]]
            else
                psi_decomp_p[j] = OneHotTensors(S*V)
                psi_decomp_p[j] = [psid * U for psid in psi_decomp_p[j]]
            end

            M_list_p = [length(psi_decomp_p[i]) for i=1:M]

            # Re-compute H_full and S_full for site p:
            H_full_p, S_full_p = ExpandSubspace(
                sdata.H_mat,
                sdata.S_mat,
                psi_decomp_p,
                H_blocks,
                S_blocks,
                block_ref
            )

            # Solve the generalized eigenvalue problem:
            E_p, C_p, kappa_p = SolveGenEig(
                H_full_p, 
                S_full_p, 
                thresh=op.sd_thresh,
                eps=op.sd_eps
            )

            t_vec_p = []
            jp0, jp1 = sum(M_list_p[1:j-1])+1, sum(M_list_p[1:j])
            t_vec_p = real.(C_p[jp0:jp1,1])
            if (NaN in t_vec_p) || (Inf in t_vec_p) || norm(t_vec_p) < 1e-16
                t_vec_p = [scalar(Tj*dag(psi_decomp_p[j][k])) for k=1:M_list_p[j]]
            end

            Tj = sum([t_vec_p[k]*psi_decomp_p[j][k] for k=1:M_list_p[j]])

            t_vec = [scalar(Tj*dag(psi_decomp[j][k])) for k=1:M_list[j]]
            
            E_min = E_p[1]

        end

    end
    
    return E_min, t_vec
    
end


function ReplaceBond!(
        sdata,
        psi_decomp,
        j,
        p,
        t_vec,
        lnoise,
        ldelta,
        theta
    )
    
    M = sdata.mparams.M
    
    M_list = [length(psi_decomp[i]) for i=1:M]
    
    T_old = sdata.psi_list[j][p]*sdata.psi_list[j][p+1]
    
    if norm(t_vec) < 1e-16
        t_vec = [scalar(T_old*dag(psi_decomp[j][k])) for k=1:M_list[j]]
    end

    t_vec += ldelta*normalize(randn(M_list[j])) # Random noise term
    normalize!(t_vec)

    # Construct the new tensor and plug it in:
    T_new = sum([t_vec[k]*psi_decomp[j][k] for k=1:M_list[j]])

    # Mix the new tensor with the old tensor:
    T_old = sdata.psi_list[j][p]*sdata.psi_list[j][p+1]
    T_new = (1.0-theta)*T_new + theta*T_old
    T_new *= 1.0/sqrt(scalar(T_new*dag(T_new)))
    
    # Generate the "noise" term:
    pmpo = ITensors.ProjMPO(sdata.ham_list[j])
    ITensors.set_nsite!(pmpo,2)
    ITensors.position!(pmpo, sdata.psi_list[j], p)
    drho = lnoise*ITensors.noiseterm(pmpo,T_new,"left")

    #println(inds(T_new))
    #println(inds(drho))
    
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
    
end

function UpdateTopBlocks!(
        sdata,
        lH,
        lS,
        jperm_ops,
        jrev_flag,
        state_ref,
        p;
        j1=1,
        j2=2
    )
    
    for bind=1:length(state_ref)
                        
        j, i = state_ref[bind]

        j_ind = findall(x->x==j, [j1,j2])[1]

        if i==j

            lH[bind] = UpdateBlock(
                lH[bind], 
                p, 
                sdata.psi_list[i], 
                sdata.psi_list[i], 
                sdata.ham_list[i], 
                nothing
            )

        else

            if jrev_flag[j_ind][i]
                psi_i = ReverseMPS(sdata.psi_list[i])
                ham_i = ReverseMPO(sdata.ham_list[i])
            else
                psi_i = sdata.psi_list[i]
                ham_i = sdata.ham_list[i]
            end

            lH[bind] = UpdateBlock(
                lH[bind], 
                p, 
                sdata.psi_list[j], 
                psi_i, 
                jperm_ops[j_ind][i],
                ham_i
            )

            lS[bind] = UpdateBlock(
                lS[bind], 
                p, 
                sdata.psi_list[j], 
                psi_i, 
                jperm_ops[j_ind][i],
                nothing
            )

        end

    end
    
    return lH, lS
    
end


function BlockSwapMerge(
        H_blocks,
        S_blocks,
        block_ref,
        state_ref,
        sdata,
        j1,
        p
    )
    
    # Repeat the above with a SWAP inserted at sites p,p+1 of state j1:
    H_blocks_ = deepcopy(H_blocks)
    S_blocks_ = deepcopy(S_blocks)
    
    # Construct the SWAP tensor:
    swap_tensor = BuildFermionicSwap(sdata.sites, p, dim=4)
    setprime!(swap_tensor, 2, plev=0)
    
    # Apply the SWAP tensor to the site indices of j1:
    for (bnum, states) in enumerate(state_ref)
        
        if j1 in states 
            
            if length(setdiff(states, j1)) == 0 # j1-j1 diagonal block
                
                H_blocks_[bnum] *= dag(swap_tensor)
                setprime!(H_blocks_[bnum], 1, plev=2)
                
                H_blocks_[bnum] *= setprime(swap_tensor,0,plev=1)
                setprime!(H_blocks_[bnum], 0, plev=2)
                
            elseif setdiff(states, j1)[1] < j1 # i < j
                
                H_blocks_[bnum] *= setprime(swap_tensor,0,plev=1)
                setprime!(H_blocks_[bnum], 0, plev=2)
                
                S_blocks_[bnum] *= setprime(swap_tensor,0,plev=1)
                setprime!(S_blocks_[bnum], 0, plev=2)
                
            else # i > j
                
                H_blocks_[bnum] *= dag(swap_tensor)
                setprime!(H_blocks_[bnum], 1, plev=2)
                
                S_blocks_[bnum] *= dag(swap_tensor)
                setprime!(S_blocks_[bnum], 1, plev=2)
                
            end

        end

    end
    
    return H_blocks_, S_blocks_
    
end


function CoTwoSitePairSweep!(
        sdata::SubspaceProperties,
        op::OptimParameters;
        jpairs=nothing,
        verbose=false
    )
    
    N = sdata.chem_data.N_spt
    M = sdata.mparams.M
    
    if verbose
        println("\nTWO-STATE, TWO-SITE SWEEP ALGORITHM:")
    end
    
    if jpairs == nothing
        # Default is all pairs:
        jpairs = []
        for k=1:Int(floor(M/2))
            for j=1:M
                push!(jpairs, sort([j, mod1(j+k, M)]))
            end
        end
    end
    
    swap_counter = 0
    
    # The iteration loop:
    for l=1:op.maxiter
        
        # Noise at this iteration:
        lnoise = op.noise[minimum([l,end])]
        ldelta = op.delta[minimum([l,end])]
        
        for jc=1:length(jpairs)
            
            # Choose states (j1,j2)
            j1, j2 = jpairs[jc]
            
            jperm_ops, jrev_flag = ChooseStates(
                sdata,
                j1=j1,
                j2=j2
            )
            
            for s=1:op.numloop
                
                # Pre-contract bottom blocks
                lH, lS, rH_list, rS_list, block_ref, state_ref = TwoStateCollectBlocks(
                    sdata,
                    jperm_ops,
                    jrev_flag,
                    j1=j1,
                    j2=j2
                )
                
                for p=1:N-1
                    
                    # Loop start: select top/bottom blocks
                    
                    # Select the correct "right" blocks:
                    H_blocks = [rH_list[b][p] for b=1:length(rH_list)]
                    S_blocks = [rS_list[b][p] for b=1:length(rS_list)]
                    
                    #display(inds(H_blocks[block_ref[j1,j2]]))
                    
                    # Contract lock tensors
                    H_blocks, S_blocks = ContractLockTensors(
                        sdata,
                        jperm_ops,
                        jrev_flag,
                        p,
                        rH_list,
                        rS_list,
                        lH,
                        lS,
                        block_ref,
                        state_ref,
                        j1=j1,
                        j2=j2
                    )
                    
                    """
                    if [j1,j2] == [2,3]
                        for k=1:length(H_blocks)
                            display(inds(H_blocks[k]))
                        end
                    end
                    """
                    
                    # Generate OHT list + mat. els.
                    psi_decomp = [[ITensor(1.0)] for i=1:M]
                    
                    for j in [j1, j2]
                        T = sdata.psi_list[j][p] * sdata.psi_list[j][p+1]
                        psi_decomp[j] = OneHotTensors(T)
                    end
                    
                    # Generate the subspace matrices:
                    H_full, S_full = ExpandSubspace(
                        sdata.H_mat,
                        sdata.S_mat,
                        psi_decomp,
                        H_blocks,
                        S_blocks,
                        block_ref
                    )
                    
                    # Repeat the above with a SWAP inserted at sites p,p+1 of state j1:
                    
                    H_blocks_, S_blocks_ = BlockSwapMerge(
                        H_blocks,
                        S_blocks,
                        block_ref,
                        state_ref,
                        sdata,
                        j1,
                        p
                    )
                    
                    H_full_, S_full_ = ExpandSubspace(
                        sdata.H_mat,
                        sdata.S_mat,
                        psi_decomp,
                        H_blocks_,
                        S_blocks_,
                        block_ref
                    )
                    
                    #H_full, S_full, psi_decomp = DiscardOverlapping(H_full, S_full, psi_decomp, j1, op.sd_dtol)
                    #H_full, S_full, psi_decomp = DiscardOverlapping(H_full, S_full, psi_decomp, j2, op.sd_dtol)
                    
                    M_list = [length(psi_decomp[i]) for i=1:M]
                    
                    # Diagonalize, compare
                    
                    do_replace = true
                    do_swap = false
                    
                    if op.sd_method=="geneig"
                        
                        # Solve the generalized eigenvalue problem:
                        E, C, kappa = SolveGenEig(
                            H_full, 
                            S_full, 
                            thresh=op.sd_thresh,
                            eps=op.sd_eps
                        )
                        
                        # Solve the generalized eigenvalue problem:
                        E_, C_, kappa_ = SolveGenEig(
                            H_full_, 
                            S_full_, 
                            thresh=op.sd_thresh,
                            eps=op.sd_eps
                        )
                        
                        if E_[1] < E[1]
                            do_swap = true
                        end
                        
                        t_vecs = []
                        for (idx, i) in enumerate([j1, j2])
                            i0, i1 = sum(M_list[1:i-1])+1, sum(M_list[1:i])
                            if do_swap
                                t_vec = real.(C_[i0:i1,1])
                            else
                                t_vec = real.(C[i0:i1,1])
                            end
                            push!(t_vecs, normalize(t_vec))
                        end
                        
                        if minimum([real(E[1]), real(E_[1])]) >= op.sd_penalty*sdata.E[1]
                            do_replace = false
                        end
                    
                    elseif op.sd_method=="triple_geneig"
                        
                        # Solve the generalized eigenvalue problem:
                        E_min, t_vecs = TripleGenEig(
                            H_full,
                            S_full,
                            psi_decomp,
                            H_blocks,
                            S_blocks,
                            state_ref,
                            block_ref,
                            sdata,
                            op,
                            p,
                            j1=j1,
                            j2=j2
                        )
                        
                        # Solve the generalized eigenvalue problem:
                        E_min_, t_vecs_ = TripleGenEig(
                            H_full_,
                            S_full_,
                            psi_decomp,
                            H_blocks_,
                            S_blocks_,
                            state_ref,
                            block_ref,
                            sdata,
                            op,
                            p,
                            j1=j1,
                            j2=j2
                        )
                        
                        P = ExpProb(E_min, E_min_, op.alpha)
                        
                        if rand()[1] <= P
                            do_swap = true
                            E_min = E_min_
                            t_vecs = t_vecs_
                        end
                        
                        if real(E_min) > op.sd_penalty*sdata.E[1]
                            do_replace = false
                        end
                        
                    end
                    
                    # Update params, update top block
                    
                    if (NaN in t_vecs[1]) || (Inf in t_vecs[1]) || (NaN in t_vecs[2]) || (Inf in t_vecs[2])
                        do_replace = false
                    end
                    
                    # Check the truncation error is not too large:
                    if TruncError(t_vecs[1],j1,p,psi_decomp,sdata) > op.ttol || TruncError(t_vecs[2],j2,p,psi_decomp,sdata) > op.ttol
                        do_replace = false
                    end
                    
                    if do_replace
                        
                        # Do the replacement:
                        for (idx, i) in enumerate([j1, j2])
                            
                            ReplaceBond!(
                                sdata,
                                psi_decomp,
                                i,
                                p,
                                t_vecs[idx],
                                lnoise,
                                ldelta,
                                op.theta
                            )
                            
                        end
                        
                        if do_swap # Update ordering j1, re-generate block tensors
                            
                            swap_counter += 1
                            
                            sdata.ord_list[j1][p:p+1] = reverse(sdata.ord_list[j1][p:p+1])
                            
                            # Re-generate Hamiltonians and PMPOs:
                            GenHams!(sdata)
                            GenPermOps!(sdata)
                            
                            # Re-generate the top/bottom blocks:
                            jperm_ops, jrev_flag = ChooseStates(
                                sdata,
                                j1=j1,
                                j2=j2
                            )
                            
                            # Shift orthogonality center to site 1:
                            for i in [j1, j2]
                                orthogonalize!(sdata.psi_list[i], 1)
                                normalize!(sdata.psi_list[i])
                            end
                            
                            # Pre-contract bottom blocks
                            lH, lS, rH_list, rS_list, block_ref, state_ref = TwoStateCollectBlocks(
                                sdata,
                                jperm_ops,
                                jrev_flag,
                                j1=j1,
                                j2=j2
                            )
                            
                            # Re-contract the top blocks to position p-1:
                            for q=1:(p-1)
                                
                                # Shift orthogonality center to site q+1:
                                for i in [j1, j2]
                                    orthogonalize!(sdata.psi_list[i], q+1)
                                    normalize!(sdata.psi_list[i])
                                end
                                
                                lH, lS = UpdateTopBlocks!(
                                    sdata,
                                    lH,
                                    lS,
                                    jperm_ops,
                                    jrev_flag,
                                    state_ref,
                                    q,
                                    j1=j1,
                                    j2=j2
                                )
                            end

                        end
                            
                    end
                    
                    # Shift orthogonality center to site p+1:
                    for i in [j1, j2]
                        orthogonalize!(sdata.psi_list[i], p+1)
                        normalize!(sdata.psi_list[i])
                    end

                    # Update the "left" blocks:
                    lH, lS = UpdateTopBlocks!(
                        sdata,
                        lH,
                        lS,
                        jperm_ops,
                        jrev_flag,
                        state_ref,
                        p,
                        j1=j1,
                        j2=j2
                    )
                    
                    
                    # Recompute H, S, E, C, kappa:
                    GenSubspaceMats!(sdata)
                    SolveGenEig!(sdata)
                    
                    # Print some output
                    if verbose
                        print("Iter: $(l)/$(op.maxiter); ")
                        print("pair: $(jc)/$(length(jpairs)); ")
                        print("sweep: $(s)/$(op.numloop); ")
                        print("bond: $(p)/$(N-1); ")
                        print("#swaps: $(swap_counter); ")
                        print("E_min = $(round(sdata.E[1], digits=5)); ") 
                        print("kappa = $(round(sdata.kappa, sigdigits=3))     \r")
                        flush(stdout)
                    end
                    
                end
                
            end
            
        end
        
    end
    
    # Exit loop
    
    if verbose
        println("\nDone!\n")
    end
    
end


# Determine whether or not inserting an FSWAP will be beneficial:
function TestFSWAP(
        sdata,
        psi_decomp,
        H_full,
        S_full,
        op,
        p;
        j1=1,
        j2=2
    )
    
    M_list = [length(psi_decomp[i]) for i=1:length(psi_decomp)]
    
    do_swap = false
    
    # Do an initial geneig solve:
                    
    E, C, kappa = SolveGenEig(
        H_full, 
        S_full, 
        thresh=op.sd_thresh,
        eps=op.sd_eps
    )

    t_vecs = []
    for (idx, i) in enumerate([j1, j2])
        i0, i1 = sum(M_list[1:i-1])+1, sum(M_list[1:i])
        t_vec = real.(C[i0:i1,1])
        push!(t_vecs, normalize(t_vec))
    end

    Tj1 = sum([t_vecs[1][k]*psi_decomp[j1][k] for k=1:length(t_vecs[1])])
    linds = commoninds(Tj1, sdata.psi_list[j1][p])
    
    U,S,V = svd(Tj1, linds);
    L = size(S.tensor, 1)
    s_vec = [S[i,i] for i=1:L]
    s2_vec = s_vec .^ 2
    m_eff = minimum([length(s2_vec), sdata.mparams.psi_maxdim])
    fid = sum(reverse(sort(s2_vec))[1:m_eff])
    
    fswap = BuildFermionicSwap(sdata.sites, p; dim=4);
    Tj1_ = Tj1 * fswap
    noprime!(Tj1_)
    
    U_,S_,V_ = svd(Tj1_, linds);
    L_ = size(S_.tensor, 1)
    s_vec_ = [S_[i,i] for i=1:L_]
    s2_vec_ = s_vec_ .^ 2
    m_eff_ = minimum([length(s2_vec_), sdata.mparams.psi_maxdim])
    fid_ = sum(reverse(sort(s2_vec_))[1:m_eff_])
    
    if fid_ > fid
        do_swap = true
    end
    
    #println("\n $(fid)  $(fid_)  $(do_swap) \n")
    
    return do_swap
    
end


function CoTwoSitePairSweep2!(
        sdata::SubspaceProperties,
        op::OptimParameters;
        jpairs=nothing,
        verbose=false
    )
    
    N = sdata.chem_data.N_spt
    M = sdata.mparams.M
    
    if verbose
        println("\nTWO-STATE, TWO-SITE SWEEP ALGORITHM:")
    end
    
    if jpairs == nothing
        # Default is all pairs:
        jpairs = []
        for k=1:Int(floor(M/2))
            for j=1:M
                push!(jpairs, sort([j, mod1(j+k, M)]))
            end
        end
    end
    
    swap_counter = 0
    
    # The iteration loop:
    for l=1:op.maxiter
        
        # Noise at this iteration:
        lnoise = op.noise[minimum([l,end])]
        ldelta = op.delta[minimum([l,end])]
        
        for jc=1:length(jpairs)
            
            # Choose states (j1,j2)
            j1, j2 = jpairs[jc]
            
            jperm_ops, jrev_flag = ChooseStates(
                sdata,
                j1=j1,
                j2=j2
            )
            
            for s=1:op.numloop
                
                # Pre-contract bottom blocks
                lH, lS, rH_list, rS_list, block_ref, state_ref = TwoStateCollectBlocks(
                    sdata,
                    jperm_ops,
                    jrev_flag,
                    j1=j1,
                    j2=j2
                )
                
                for p=1:N-1
                    
                    # Loop start: select top/bottom blocks
                    
                    # Select the correct "right" blocks:
                    H_blocks = [rH_list[b][p] for b=1:length(rH_list)]
                    S_blocks = [rS_list[b][p] for b=1:length(rS_list)]
                    
                    #display(inds(H_blocks[block_ref[j1,j2]]))
                    
                    # Contract lock tensors
                    H_blocks, S_blocks = ContractLockTensors(
                        sdata,
                        jperm_ops,
                        jrev_flag,
                        p,
                        rH_list,
                        rS_list,
                        lH,
                        lS,
                        block_ref,
                        state_ref,
                        j1=j1,
                        j2=j2
                    )
                    
                    # Generate OHT list + mat. els.
                    psi_decomp = [[ITensor(1.0)] for i=1:M]
                    
                    for j in [j1, j2]
                        T = sdata.psi_list[j][p] * sdata.psi_list[j][p+1]
                        psi_decomp[j] = OneHotTensors(T)
                    end
                    
                    # Generate the subspace matrices:
                    H_full, S_full = ExpandSubspace(
                        sdata.H_mat,
                        sdata.S_mat,
                        psi_decomp,
                        H_blocks,
                        S_blocks,
                        block_ref
                    )
                    
                    # Test whether or not to insert a SWAP:
                    do_swap = TestFSWAP(
                        sdata,
                        psi_decomp,
                        H_full,
                        S_full,
                        op,
                        p;
                        j1=j1,
                        j2=j2
                    )
                    
                    if do_swap
                        
                        H_blocks, S_blocks = BlockSwapMerge(
                            H_blocks,
                            S_blocks,
                            block_ref,
                            state_ref,
                            sdata,
                            j1,
                            p
                        )
                        
                        H_full, S_full = ExpandSubspace(
                            sdata.H_mat,
                            sdata.S_mat,
                            psi_decomp,
                            H_blocks,
                            S_blocks,
                            block_ref
                        )
                        
                    end
                    
                    #H_full, S_full, psi_decomp = DiscardOverlapping(H_full, S_full, psi_decomp, j1, op.sd_dtol)
                    #H_full, S_full, psi_decomp = DiscardOverlapping(H_full, S_full, psi_decomp, j2, op.sd_dtol)
                    
                    M_list = [length(psi_decomp[i]) for i=1:M]
                    
                    # Diagonalize, compare
                    
                    do_replace = true
                    
                    if op.sd_method=="geneig"
                        
                        # Solve the generalized eigenvalue problem:
                        E, C, kappa = SolveGenEig(
                            H_full, 
                            S_full, 
                            thresh=op.sd_thresh,
                            eps=op.sd_eps
                        )
                        
                        t_vecs = []
                        for (idx, i) in enumerate([j1, j2])
                            i0, i1 = sum(M_list[1:i-1])+1, sum(M_list[1:i])
                            t_vec = real.(C[i0:i1,1])
                            push!(t_vecs, normalize(t_vec))
                        end
                        
                        if do_swap && real(E[1]) >= op.sd_swap_penalty*sdata.E[1]
                            do_replace = false
                        elseif do_swap==false && real(E[1]) >= op.sd_penalty*sdata.E[1]
                            do_replace = false
                        end
                    
                    elseif op.sd_method=="triple_geneig"
                        
                        # Solve the generalized eigenvalue problem:
                        E_min, t_vecs = TripleGenEig(
                            H_full,
                            S_full,
                            psi_decomp,
                            H_blocks,
                            S_blocks,
                            state_ref,
                            block_ref,
                            sdata,
                            op,
                            p,
                            j1=j1,
                            j2=j2
                        )
                        
                        if do_swap && real(E_min) >= op.sd_swap_penalty*sdata.E[1]
                            do_replace = false
                        elseif do_swap==false && real(E_min) >= op.sd_penalty*sdata.E[1]
                            do_replace = false
                        end
                        
                    end
                    
                    # Update params, update top block
                    
                    if (NaN in t_vecs[1]) || (Inf in t_vecs[1]) || (NaN in t_vecs[2]) || (Inf in t_vecs[2])
                        do_replace = false
                    end
                    
                    # Check the truncation error is not too large:
                    if TruncError(t_vecs[1],j1,p,psi_decomp,sdata) > op.ttol || TruncError(t_vecs[2],j2,p,psi_decomp,sdata) > op.ttol
                        do_replace = false
                    end
                    
                    if do_replace
                        
                        # Do the replacement:
                        for (idx, i) in enumerate([j1, j2])
                            
                            ReplaceBond!(
                                sdata,
                                psi_decomp,
                                i,
                                p,
                                t_vecs[idx],
                                lnoise,
                                ldelta,
                                op.theta
                            )
                            
                        end
                        
                        if do_swap # Update ordering j1, re-generate block tensors
                            
                            swap_counter += 1
                            
                            sdata.ord_list[j1][p:p+1] = reverse(sdata.ord_list[j1][p:p+1])
                            
                            # Re-generate Hamiltonians and PMPOs:
                            GenHams!(sdata)
                            GenPermOps!(sdata)
                            
                            # Re-generate the top/bottom blocks:
                            jperm_ops, jrev_flag = ChooseStates(
                                sdata,
                                j1=j1,
                                j2=j2
                            )
                            
                            # Shift orthogonality center to site 1:
                            for i in [j1, j2]
                                orthogonalize!(sdata.psi_list[i], 1)
                                normalize!(sdata.psi_list[i])
                            end
                            
                            # Pre-contract bottom blocks
                            lH, lS, rH_list, rS_list, block_ref, state_ref = TwoStateCollectBlocks(
                                sdata,
                                jperm_ops,
                                jrev_flag,
                                j1=j1,
                                j2=j2
                            )
                            
                            # Re-contract the top blocks to position p-1:
                            for q=1:(p-1)
                                
                                # Shift orthogonality center to site q+1:
                                for i in [j1, j2]
                                    orthogonalize!(sdata.psi_list[i], q+1)
                                    normalize!(sdata.psi_list[i])
                                end
                                
                                lH, lS = UpdateTopBlocks!(
                                    sdata,
                                    lH,
                                    lS,
                                    jperm_ops,
                                    jrev_flag,
                                    state_ref,
                                    q,
                                    j1=j1,
                                    j2=j2
                                )
                            end

                        end
                            
                    end
                    
                    # Shift orthogonality center to site p+1:
                    for i in [j1, j2]
                        orthogonalize!(sdata.psi_list[i], p+1)
                        normalize!(sdata.psi_list[i])
                    end

                    # Update the "left" blocks:
                    lH, lS = UpdateTopBlocks!(
                        sdata,
                        lH,
                        lS,
                        jperm_ops,
                        jrev_flag,
                        state_ref,
                        p,
                        j1=j1,
                        j2=j2
                    )
                    
                    
                    # Recompute H, S, E, C, kappa:
                    GenSubspaceMats!(sdata)
                    SolveGenEig!(sdata)
                    
                    # Print some output
                    if verbose
                        print("Iter: $(l)/$(op.maxiter); ")
                        print("pair: $(jc)/$(length(jpairs)); ")
                        print("sweep: $(s)/$(op.numloop); ")
                        print("bond: $(p)/$(N-1); ")
                        print("#swaps: $(swap_counter); ")
                        print("E_min = $(round(sdata.E[1], digits=5)); ") 
                        print("kappa = $(round(sdata.kappa, sigdigits=3))     \r")
                        flush(stdout)
                    end
                    
                end
                
            end
            
        end
        
    end
    
    # Exit loop
    
    if verbose
        println("\nDone!\n")
    end
    
end


function SeedNoise!(
        sdata::SubspaceProperties,
        delta::Float64,
        noise::Float64;
        jset=nothing,
        verbose=false
    )
    
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
            
            t_vec = [scalar(T_old*dag(psi_decomp[k])) for k=1:length(psi_decomp)]
            
            t_vec += delta*normalize(randn(length(t_vec)))
            
            normalize!(t_vec)
            
            T_new = sum([t_vec[k]*psi_decomp[k] for k=1:length(t_vec)])
            
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
    
end


# Bond SWAP selection based on Ipq
function SelectBond(ord, gamma, Ipq)
    
    N = length(ord)
    
    dI_vec = []
    
    for p=1:N-1
        
        if p==1
            dI = Ipq[p,p+2] - Ipq[p+1,p+2]
        elseif p==N-1
            dI = Ipq[p-1,p+1] - Ipq[p-1,p]
        else
            dI = Ipq[p-1,p+1] + Ipq[p,p+2]  - Ipq[p-1,p] - Ipq[p+1,p+2]
        end
        
        push!(dI_vec, dI)
        
    end
    
    exp_vec = exp.(dI_vec.*gamma)
    
    prob_vec = exp_vec./sum(exp_vec)
    
    pfloat = rand()[1]
    
    q = 0
    
    for p=1:length(prob_vec)
        if pfloat >= sum(prob_vec[1:p-1]) && pfloat < sum(prob_vec[1:p])
            q = p
        end
    end
    
    return q
    
end


function TwoSiteLock(
        psi1, 
        psi2,
        mpo1,
        mpo2,
        p;
        diag=false
    )
    
    N = length(psi1)
    
    # Contract the top block:
    top_block = ITensor(1.0)
    for q=1:p-1
        top_block = UpdateBlock(
            top_block,
            q,
            psi1,
            psi2,
            mpo1,
            mpo2
        )
    end
    
    # Contract the bottom block:
    bottom_block = ITensor(1.0)
    for q=N:(-1):p+2
        bottom_block = UpdateBlock(
            bottom_block,
            q,
            psi1,
            psi2,
            mpo1,
            mpo2
        )
    end
    
    # Fill in the middle block:
    block = top_block
    for q in [p, p+1]
        if diag
            if mpo2 == nothing && mpo1 != nothing
                block *= mpo1[q]
            elseif mpo1 == nothing && mpo2 != nothing
                block *= mpo2[q]
            elseif mpo2 != nothing && mpo1 != nothing
                block *= setprime(mpo1[q],2,plev=0)
                block *= setprime(mpo2[q],2,plev=1)
            end
        else
            if mpo1 == nothing && mpo2 == nothing
                block *= psi2[q]
            elseif mpo2 == nothing
                block *= psi2[q] * mpo1[q]
            elseif mpo1 == nothing
                block *= psi2[q] * mpo2[q]
            else
                Ax = setprime(mpo1[q],2,plev=0)
                yB = psi2[q] * setprime(mpo2[q],2,plev=1)
                #println(p)
                #println(inds(block))
                #println(inds(yB))
                block *= yB
                block *= Ax
            end
        end
    end
    
    block *= bottom_block
    
    return block
    
end


function SWAPtimize!(
        sdata::SubspaceProperties,
        op::OptimParameters;
        verbose=false
    )
    
    M = sdata.mparams.M
    N = sdata.chem_data.N_spt
    
    if verbose
        println("\nSWAPTIMIZER:")
    end
    
    swap_counter = 0
    
    for l=1:op.maxiter
        
        # Noise at this iteration:
        lnoise = op.noise[minimum([l,end])]
        ldelta = op.delta[minimum([l,end])]
        
        # Iterate through each state:
        for j=1:M
            
            # Re-compute permutation operators:
            jperm_ops, jrev_flag = RelativePermOps(sdata, j)
            
            psi_list_c = deepcopy(sdata.psi_list)
            for i=1:M
                if jrev_flag[i]
                    psi_list_c[i] = ReverseMPS(psi_list_c[i])
                end
            end
            
            # Choose a bond at which to apply an FSWAP:
            p = SelectBond(sdata.ord_list[j], op.gamma, Ipq)
            
            orthogonalize!(sdata.psi_list[j], p)
            psi_list_c[j] = sdata.psi_list[j]
            
            # Compute subspace expansions:
            
            ## Generate lock tensors
            block_ref = zeros(Int, (M,M))
            block_ref[:,j] = collect(1:M)
            block_ref[j,:] = collect(1:M)
            state_ref = []
            
            H_blocks = []
            S_blocks = Any[]
            
            for i=1:M
                
                #block_ref[i,j] = i
                push!(state_ref, sort([i,j]))
                
                if i==j
                    
                    H_block = TwoSiteLock(
                        sdata.psi_list[j], 
                        sdata.psi_list[j],
                        sdata.ham_list[j],
                        nothing,
                        p;
                        diag=true
                    )
                    
                    S_block = ITensor(1.0)
                    
                else
                    
                    H_block = TwoSiteLock(
                        sdata.psi_list[j], 
                        psi_list_c[i],
                        sdata.ham_list[j],
                        jperm_ops[i],
                        p;
                        diag=false
                    )
                    
                    S_block = TwoSiteLock(
                        sdata.psi_list[j], 
                        psi_list_c[i],
                        jperm_ops[i],
                        nothing,
                        p;
                        diag=false
                    )
                    
                end
                
                push!(H_blocks, H_block)
                push!(S_blocks, S_block)
                
            end
            
            # Dagging and swap-priming:
            for i in setdiff(1:M, j)
                if i<j 
                    H_blocks[i] = swapprime(dag(H_blocks[i]),0,1)
                    S_blocks[i] = swapprime(dag(S_blocks[i]),0,1)
                end
            end
            
            # Generate OHT list + mat. els.
            psi_decomp = [[ITensor(1.0)] for i=1:M]

            T = sdata.psi_list[j][p] * sdata.psi_list[j][p+1]
            psi_decomp[j] = OneHotTensors(T)

            # Generate the subspace matrices:
            H_full, S_full = ExpandSubspace(
                sdata.H_mat,
                sdata.S_mat,
                psi_decomp,
                H_blocks,
                S_blocks,
                block_ref
            )

            # Repeat the above with a SWAP inserted at sites p,p+1 of state j:

            H_blocks_, S_blocks_ = BlockSwapMerge(
                H_blocks,
                S_blocks,
                block_ref,
                state_ref,
                sdata,
                j,
                p
            )

            H_full_, S_full_ = ExpandSubspace(
                sdata.H_mat,
                sdata.S_mat,
                psi_decomp,
                H_blocks_,
                S_blocks_,
                block_ref
            )
            
            M_list = [length(psi_decomp[i]) for i=1:M]
                    
            # Diagonalize, compare

            do_replace = true
            do_swap = false

            if op.sd_method=="geneig"

                # Solve the generalized eigenvalue problem:
                E, C, kappa = SolveGenEig(
                    H_full, 
                    S_full, 
                    thresh=op.sd_thresh,
                    eps=op.sd_eps
                )

                # Solve the generalized eigenvalue problem:
                E_, C_, kappa_ = SolveGenEig(
                    H_full_, 
                    S_full_, 
                    thresh=op.sd_thresh,
                    eps=op.sd_eps
                )

                if E_[1] < E[1]
                    do_swap = true
                end

                t_vecs = []
                for (idx, i) in enumerate([j1, j2])
                    i0, i1 = sum(M_list[1:i-1])+1, sum(M_list[1:i])
                    if do_swap
                        t_vec = real.(C_[i0:i1,1])
                    else
                        t_vec = real.(C[i0:i1,1])
                    end
                    push!(t_vecs, normalize(t_vec))
                end

                if minimum([real(E[1]), real(E_[1])]) >= op.sd_penalty*sdata.E[1]
                    do_replace = false
                end

            elseif op.sd_method=="triple_geneig"

                # Solve the generalized eigenvalue problem:
                E_min, t_vec = TripleGenEig1(
                    H_full,
                    S_full,
                    psi_decomp,
                    H_blocks,
                    S_blocks,
                    state_ref,
                    block_ref,
                    sdata,
                    op,
                    p,
                    j
                )

                # Solve the generalized eigenvalue problem:
                E_min_, t_vec_ = TripleGenEig1(
                    H_full_,
                    S_full_,
                    psi_decomp,
                    H_blocks_,
                    S_blocks_,
                    state_ref,
                    block_ref,
                    sdata,
                    op,
                    p,
                    j
                )

                P = ExpProb(E_min, E_min_, op.alpha)
                #println("\n$(E_min)  $(E_min_)  $(P)\n")

                if rand()[1] <= P
                    do_swap = true
                    E_min = E_min_
                    t_vec = t_vec_
                end

                if real(E_min) > op.sd_penalty*sdata.E[1]
                    #println("\n do_replace=false: E penalty \n")
                    do_replace = false
                end

            end

            # Update params, update top block

            if (NaN in t_vec) || (Inf in t_vec)
                #println("\n do_replace=false: NaN or Inf \n")
                do_replace = false
            end

            # Check the truncation error is not too large:
            if TruncError(t_vec,j,p,psi_decomp,sdata) > op.ttol
                #println("\n do_replace=false: truncation error \n")
                do_replace = false
            end

            if do_replace

                # Do the replacement:
                ReplaceBond!(
                    sdata,
                    psi_decomp,
                    j,
                    p,
                    t_vec,
                    lnoise,
                    ldelta,
                    op.theta
                )
                
                #println("\n Did replace! \n")

                if do_swap # Update ordering j, re-generate block tensors

                    swap_counter += 1

                    sdata.ord_list[j][p:p+1] = reverse(sdata.ord_list[j][p:p+1])

                    # Re-generate Hamiltonians and PMPOs:
                    GenHams!(sdata)
                    GenPermOps!(sdata)

                end

            end
            
            # Recompute H, S, E, C, kappa:
            GenSubspaceMats!(sdata)
            SolveGenEig!(sdata)

            # Print some output
            if verbose
                print("Iter: $(l)/$(op.maxiter); ")
                print("State: $(j)/$(M); ")
                print("#swaps: $(swap_counter); ")
                print("E_min = $(round(sdata.E[1], digits=5)); ") 
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


# A general wrapper function for multi-geometry optimizations:
function MultiGeomOptim!(
        sdata::SubspaceProperties,
        op_list::Vector{OptimParameters};
        reps=1,
        rep_struct=[["twosite",1]],
        verbose=false
    )
    
    N = sdata.chem_data.N_spt
    M = sdata.mparams.M
    
    if verbose
        println("\n----| Starting MultiGeomOptim!\n")
    end
    
    # The repetition loop:
    for r=1:reps
        
        if verbose
            println("\n----| Repetition $(r)/$(reps):\n")
        end
        
        for action in rep_struct
            
            # Select the OptimParameters object:
            op = op_list[action[2]]
                
            if action[1]=="seednoise" # Seed with noise:
                
                SeedNoise!(
                    sdata, 
                    op.delta[r], 
                    op.noise[r], 
                    verbose=verbose
                )
                
            elseif action[1]=="twosite" # Complete a generalized two-site sweep:
                
                GeneralizedTwoSite!(
                    sdata, 
                    op, 
                    verbose=verbose
                )
                
            elseif action[1]=="onesitepair" # Do a paired one-site sweep
                
                OneSitePairSweep!(
                    sdata, 
                    op, 
                    verbose=verbose
                )
                
            elseif action[1]=="twositepair" # Do a paired two-site sweep
                
                TwoSitePairSweep!(
                    sdata, 
                    op,  
                    verbose=verbose
                )
                
            else
                println("----| Invalid action: \"$(action[1])\"")
            end
            
        end
        
        
        if verbose
            println("\n----| Repetition $(r)/$(reps) complete!")
            println("----| E = $(sdata.E[1]+sdata.chem_data.e_nuc)")
        end
        
    end
    
    if verbose
        println("\n----| MultiGeomOptim complete!\n")
    end
    
end
