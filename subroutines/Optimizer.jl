# QPU-efficient site-by-site multi-geometry optimization functions:


# Optimizer function parameters data structure:
@with_kw mutable struct OptimParameters
    
    maxiter::Int=100 # Number of iterations
    numloop::Int=1 # Number of loops per iter
    numopt::Int=-1 # Number of states/pairs to optimize per iter
    
    # Geometry update move parameters:
    move_type::String="none" # "none", "random" or "shuffle"
    swap_num::Int=-1 # How may states to permute (-1 => all)
    swap_mult::Float64=0.4 # Swappiness multiplier
    permute_states::Bool=false # Permute the states after geom. update?
    
    # State "recycling" parameters:
    rnum::Int=1 # How may states to recycle? (0 for 'off')
    rswp::Int=1 # Number of sweeps
    rnoise::Float64=1e-5 # Noise level
    
    # Move acceptance hyperparameters:
    afunc::String="exp" # "step", "poly", "exp", or "stun"
    tpow::Float64=3.0 # Power for "poly" function
    alpha::Float64=1e1 # Sharpness for "exp" and "stun" functions
    gamma::Float64=1e2 # Additional parameter for "stun" function
    
    # Site decomposition parameters:
    noise::Vector{Float64}=[1e-5] # Size of DMRG noise term at each iter
    delta::Vector{Float64}=[1e-3] # Size of Gaussian noise term
    theta::Float64=0.0 # Weight of the old state in the superposition 
    ttol::Float64=0.1 # Norm tolerance for truncated site tensors
    
    # Generalized eigenvalue solver parameters:
    thresh::String="inversion" # "none", "projection", or "inversion"
    eps::Float64=1e-8 # Singular value cutoff
    
    # Site decomposition solver parameters:
    sd_method::String="geneig" # "annealing", "bboptim", or "geneig"
    sd_maxiter::Int=2000 # Number of iterations
    sd_stun::Bool=true # Use stochastic tunnelling?
    sd_alpha::Float64=1e-4 # Sharpness for "exp" and "stun" functions
    sd_delta::Float64=1e-3 # Step size
    sd_gamma::Float64=1e3 # Additional parameter for "stun" function
    sd_thresh::String="inversion" # "none", "projection", or "inversion"
    sd_eps::Float64=1e-8 # Singular value cutoff
    sd_penalty::Float64=1.0 # Penalty factor for truncation error
    
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


function ApplyRandomSwaps!(
        sd::SubspaceProperties,
        op::OptimParameters,
        l
    )
    
    M = sd.mparams.M
    
    # Randomly select geometries:
    j_set = circshift(collect(1:M), l-1)[1:op.swap_num]

    # Randomly apply swaps:
    for j in j_set

        # Number of applied swaps to generate a new ordering (sampled from an exponential distribution):
        num_swaps = Int(floor(op.swap_mult*randexp()[1]))

        # Apply these swaps randomly:
        for swap=1:num_swaps
            p = rand(1:sd.chem_data.N_spt-1)
            sd.ord_list[j][p:p+1]=reverse(sd.ord_list[j][p:p+1])
        end

    end
    
end


function GeometryMove!(sd, op, l)
    
    old_ords = deepcopy(sd.ord_list)
    
    # Permute the orderings:
    if op.move_type=="random"
        ApplyRandomSwaps!(sd, op, l)
    elseif op.move_type=="shuffle"
        sd.ord_list=shuffle(sd.ord_list)
    end

    # Re-compute permutation tensors
    GenPermOps!(sd)
    GenHams!(sd)

    # Permute the states:
    if op.permute_states
        for j=1:sd.mparams.M
            sd.psi_list[j] = Permute(
                sd.psi_list[j], 
                siteinds(sd.psi_list[j]), 
                old_ords[j], 
                sd.ord_list[j],
                # Heavy truncation!
                maxdim=sd.mparams.psi_maxdim
            )
        end
    end
    
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
        block *= psi2[p] * setprime(dag(psi1[p]),1)
    elseif mpo2==nothing
        block *= psi2[p] * mpo1[p] * setprime(dag(psi1[p]),1)
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


function BBCostFunc(
        c::Vector{Float64}, 
        j_list,
        p,
        psi_decomp,
        shadow_in::SubspaceShadow, 
        sd::SubspaceProperties,
        op::OptimParameters;
        verbose=false
    )
    
    shadow = copy(shadow_in)
    
    j0 = 1
    j1 = 0
    
    t_err = 0.0
    
    for j in j_list
        
        j1 += shadow.M_list[j]
        
        c_j = normalize(c[j0:j1])
        
        T = sum([c_j[k]*psi_decomp[j][k] for k=1:shadow.M_list[j]])
        
        linds = commoninds(T, sd.psi_list[j][p])
        
        U,S,V = svd(T, linds, maxdim=sd.mparams.psi_maxdim)
        
        T_trunc = U*S*V
        
        t_j = normalize([scalar(dag(psi_decomp[j][k])*T_trunc) for k=1:shadow.M_list[j]])
        
        shadow.vec_list[j] = t_j
        
        j0 = j1+1
        
    end
    
    GenSubspaceMats!(shadow)
    
    SolveGenEig!(shadow, thresh=op.sd_thresh, eps=op.sd_eps)
    
    return shadow.E[1]
    
end


function MultiGeomBB(
        H_full,
        S_full,
        j_list,
        p,
        psi_decomp,
        sd::SubspaceProperties,
        op::OptimParameters
    )
    
    M_list = [length(psid) for psid in psi_decomp]
    M = length(M_list)
    M_tot = sum(M_list)
            
    shadow = SubspaceShadow(
        sd.chem_data,
        M_list,
        op.sd_thresh,
        op.sd_eps,
        [zeros(M_list[j]) for j=1:M],
        H_full,
        S_full,
        zeros((M,M)),
        zeros((M,M)),
        zeros(M),
        zeros((M,M)),
        0.0
    )
    
    E, C, kappa = SolveGenEig(
        shadow.H_full, 
        shadow.S_full, 
        thresh=op.sd_thresh, 
        eps=op.sd_eps
    )
    
    c0 = Float64[]
    
    # Fill in the initial vec list:
    for j=1:length(shadow.M_list)
        
        j0 = sum(shadow.M_list[1:j-1])+1
        j1 = sum(shadow.M_list[1:j])
        
        shadow.vec_list[j] = normalize(C[j0:j1,1])
        
        if j in j_list
            #T_0 = sd.psi_list[j][p] * sd.psi_list[j][p+1]
            #shadow.vec_list[j] = normalize([scalar(dag(psi_decomp[j][k])*T_0) for k=1:M_list[j]])
            c0 = vcat(c0, shadow.vec_list[j])
        #else
            #shadow.vec_list[j] = [1.0]
        end
        
    end
    
    GenSubspaceMats!(shadow)
    SolveGenEig!(shadow)
            
    f(c) = BBCostFunc(
        c, 
        j_list,
        p,
        psi_decomp,
        shadow,
        sd,
        op
    )

    res = bboptimize(
        f, 
        c0; 
        NumDimensions=length(c0), 
        SearchRange = (-1.0, 1.0), 
        MaxFuncEvals=op.sd_maxiter, 
        TraceMode=:silent
    )

    c_opt = best_candidate(res)
    e_opt = best_fitness(res)
    
    """
    BBCostFunc(
        c_opt, 
        j_list,
        p,
        psi_decomp,
        shadow,
        sd,
        op,
        verbose=true
    )
    """
    
    j0 = 1
    j1 = 0
    
    # Replace the vectors:
    for j in j_list

        j1 += shadow.M_list[j]

        c_j = normalize(c_opt[j0:j1])
        
        T = sum([c_j[k]*psi_decomp[j][k] for k=1:shadow.M_list[j]])
        
        linds = commoninds(T, sd.psi_list[j][p])
        
        U,S,V = svd(T, linds, maxdim=sd.mparams.psi_maxdim)
        
        T_trunc = U*S*V
        
        t_j = normalize([scalar(dag(psi_decomp[j][k])*T_trunc) for k=1:shadow.M_list[j]])
        
        shadow.vec_list[j] = t_j

        j0 = j1+1

    end

    GenSubspaceMats!(shadow)
    SolveGenEig!(shadow)
    
    return shadow
    
end


function MultiGeomAnneal!(
        shadow::SubspaceShadow,  
        op::OptimParameters
    )
    
    M_tot = sum(shadow.M_list)

    c0 = zeros(M_tot)

    for j=1:length(shadow.M_list)
        j0 = sum(shadow.M_list[1:j-1])+1
        j1 = sum(shadow.M_list[1:j])
        c0[j0:j1] = shadow.vec_list[j]
    end
    
    E_best = shadow.E[1]
            
    for l=1:maxiter

        sh2 = copy(shadow)

        # Update all states
        for j=1:length(sh2.M_list)
            j0 = sum(sh2.M_list[1:j-1])+1
            j1 = sum(sh2.M_list[1:j])

            c_j = sh2.vec_list[j] + delta*randn(sh2.M_list[j])#/sqrt(l)
            sh2.vec_list[j] = c_j/sqrt(transpose(c_j)*sh2.S_full[j0:j1,j0:j1]*c_j)
        end

        # re-compute matrix elements and diagonalize
        GenSubspaceMats!(sh2)
        SolveGenEig!(sh2)
        
        E_old = shadow.E[1]
        E_new = sh2.E[1]

        # Accept move with some probability
        beta = alpha*l#^4

        if stun
            F_0 = Fstun(E_best, E_old, gamma)
            F_1 = Fstun(E_best, E_new, gamma)
            P = ExpProb(F_1, F_0, beta)
        else
            P = ExpProb(E_old, E_new, beta)
        end

        if E_new < E_best
            E_best = E_new
        end

        if rand(Float64) < P
            
            # Replace the vectors:
            for j=1:length(shadow.M_list)
                shadow.vec_list[j] = sh2.vec_list[j]
            end

            GenSubspaceMats!(shadow)

            SolveGenEig!(shadow)

        end
        
    end
    
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
                        
                    elseif op.sd_method=="bboptim"
                        
                        shadow = MultiGeomBB(
                            H_full,
                            S_full,
                            [j],
                            p,
                            psi_decomp,
                            sdata,
                            op
                        )
                        
                        t_vec = shadow.vec_list[j]
                        
                        if shadow.E[1] >= op.sd_penalty*sdata.E[1]
                            do_replace = false
                        end
                        
                    end
                    
                    # Check the truncation error is not too large:
                    if TruncError(t_vec, j, p, psi_decomp, sdata) > op.ttol    
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
        println("\nGENERALIZED TWO-SITE SWEEP ALGORITHM:")
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
                        
                        if !(NaN in t_vec) && !(Inf in t_vec)
                        
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

                            # Shift orthogonality center to site p+1:
                            if p != N
                                orthogonalize!(sdata.psi_list[i], p+1)
                            end
                            
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
        println("\nGENERALIZED TWO-SITE SWEEP ALGORITHM:")
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
                        
                    elseif op.sd_method=="bboptim"
                        
                        shadow = MultiGeomBB(
                            H_full,
                            S_full,
                            [j1,j2],
                            p,
                            psi_decomp,
                            sdata,
                            op
                        )
                        
                        t_vecs = [shadow.vec_list[j1], shadow.vec_list[j2]]
                        
                        #println("\n  $(shadow.E[1])  $(sdata.E[1])")
                        
                        if shadow.E[1] > op.sd_penalty*sdata.E[1]
                            do_replace = false
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
    
    # Initialize the cost function:
    f = sdata.E[1] 
    f_best = f
    f_new = f
    
    # The "master" OptimParameters object:
    op1 = op_list[1]
    
    n_accept = 0
    
    if verbose
        println("\n----| Starting MultiGeomOptim!\n")
    end
    
    # The repetition loop:
    for r=1:reps
        
        if verbose
            println("\n----| Repetition $(r)/$(reps):\n")
        end
        
        # Copy the sdata before making changes:
        sdata_c = copy(sdata)
        
        j_offset = 1 - r
        
        for action in rep_struct
            
            # Select the OptimParameters object:
            op = op_list[action[2]]
            
            if action[1]=="ordmove" # Perform a geometry (ordering) update:
                
                if op.swap_num==-1
                    op.swap_num = M
                end
                
                # Update the geometries:
                if op.move_type != "none"
                    if verbose
                        println("----| Updating geometries...\n")
                    end
                    GeometryMove!(sdata_c, op, r)
                else
                    println("Move type is \"none\".\n")
                end
                
            elseif action[1]=="recycle" # "Recycle" some of the MPSs:
                
                if op.rnum==-1
                    op.rnum = M
                end
                
                # "Recycle" the basis states:
                if op.rnum != 0
                    if verbose
                        println("----| Recycling states...\n")
                    end

                    j_set = circshift(collect(1:M), j_offset)[1:op.rnum]
                        
                    for j in j_set
                        
                        hf_occ_j = [FillHF(
                                sdata_c.ord_list[j][p], 
                                sdata_c.chem_data.N_el) for p=1:N]
                        
                        sdata_c.psi_list[j] = randomMPS(
                            sdata_c.sites, 
                            hf_occ_j, 
                            linkdims=sdata_c.mparams.psi_maxdim
                        )
                    end
                    
                else
                    println("----| Recycle number is 0.\n")
                end
                
            elseif action[1]=="twosite" # Complete a generalized two-site sweep:
                
                GeneralizedTwoSite!(sdata_c, op, j_offset=j_offset, verbose=verbose)
                
            elseif action[1]=="onesitepair" # Do a paired one-site sweep
                
                jpairs = []
                for k=1:Int(floor(M/2))
                    for j=1:M
                        push!(jpairs, sort([j, mod1(j+k, M)]))
                    end
                end
                
                OneSitePairSweep!(
                    sdata_c, 
                    op, 
                    #jpairs=jpairs[1:op.numopt], 
                    verbose=verbose
                )
                
            elseif action[1]=="twositepair" # Do a paired two-site sweep
                
                jpairs = []
                for k=1:Int(floor(M/2))
                    for j=1:M
                        push!(jpairs, sort([j, mod1(j+k, M)]))
                    end
                end
                
                TwoSitePairSweep!(
                    sdata_c, 
                    op, 
                    jpairs=jpairs[1:op.numopt], 
                    verbose=verbose
                )
                
            else
                println("----| Invalid action: \"$(action[1])\"")
            end
            
        end
        
        # Accept move with some probability:
        f_new = sdata_c.E[1]

        if op1.afunc=="step"
            P = StepProb(f, f_new)
        elseif op1.afunc=="poly"
            P = PolyProb(f, f_new, op1.alpha, tpow=op1.tpow)
        elseif op1.afunc=="exp"
            P = ExpProb(f, f_new, op1.alpha)
        elseif op1.afunc=="stun"
            F_0 = Fstun(f_best, f, op1.gamma)
            F_1 = Fstun(f_best, f_new, op1.gamma)
            P = ExpProb(F_1, F_0, op1.alpha)
        elseif op1.afunc=="flat"
            P = 1
        else
            println("----| Invalid probability function!")
            return nothing
        end

        if f_new < f_best
            f_best = f_new
        end

        if rand()[1] < P
            copyto!(sdata, sdata_c)
            n_accept += 1
            f=f_new
            outcome="accept!"
        else
            outcome="reject!"
        end
        
        if verbose
            println("\n----| Repetition $(r)/$(reps) complete!")
            println("----| E_new = $(f_new); E_best = $(f)")
            ratio = Int(round(100.0*n_accept/r, digits=0))
            println("----| Outcome = \"$(outcome)\"; acceptance ratio = $(ratio)%\n")
        end
        
    end
    
    if verbose
        println("\n----| MultiGeomOptim complete!\n")
    end
    
end
