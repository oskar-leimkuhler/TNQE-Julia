# A QPU-efficient site-by-site multi-geometry optimizer:


function SiteSVD(psi, p; restrict_svals=false)
    temp_tensor = (psi[p] * psi[p+1])
    temp_inds = uniqueinds(psi[p], psi[p+1]) 
    if restrict_svals
        return svd(temp_tensor,temp_inds,maxdim=maxlinkdim(psi))
    else
        return svd(temp_tensor,temp_inds)
    end
end


function SiteQR(psi, p)
    temp_tensor = psi[p]
    temp_inds = uniqueinds(psi[p], psi[p+1])
    
    dense_tensors = [dense(psi[p]),dense(psi[p+1])]
    dense_inds = uniqueinds(dense_tensors[1],dense_tensors[2])
    
    dQ,dR = qr(dense_tensors[1],dense_inds)
    
    Q = Itensor(Array(dQ.tensor), )
    
    return qr(temp_tensor,temp_inds, positive=true)
end


function BBCostFunc2(c::Vector{Float64}, shadow_in::SubspaceShadow)
    
    shadow = copy(shadow_in)
    
    for j=1:length(shadow.M_list)
        
        j0 = sum(shadow.M_list[1:j-1])+1
        j1 = sum(shadow.M_list[1:j])
        
        c_j = c[j0:j1]

        shadow.vec_list[j] = normalize(c_j)
    end
    
    GenSubspaceMats!(shadow)
    
    SolveGenEig!(shadow, thresh="projection", eps=1e-8)
    
    return shadow.E[1]
    
end


# Return a subspace shadow generated
# by Schmidt-decomposing each state at site p:
function SiteDecomposition(sdata, U_list, sigma_list, V_list, H_diag, H_offdiag, S_offdiag, p; anchor=false)
    
    M = sdata.mparams.M
    tol = sdata.mparams.psi_tol
    m = sdata.mparams.psi_maxdim
    
    psi_decomp = []
    vec_list = []
    M_list = Int[]
    
    for j=1:M
        
        if anchor && j==1
            
            sigma = diag(Array(sigma_list[j].tensor))

            m_eff = length(sigma)

            ids = inds(H_diag[j], plev=0)

            S_jk = ITensor(zeros((dim(ids[1]),dim(ids[2]))),ids[1],ids[2])
                
            for k=1:m_eff

                S_jk[ids[1]=>k,ids[2]=>k] = sigma[k]

            end
            
            push!(psi_decomp, S_jk)
            push!(vec_list, [1.0])
            push!(M_list, 1)
            
        else
        
            sigma = diag(Array(sigma_list[j].tensor))

            m_eff = length(sigma)

            ids = inds(H_diag[j], plev=0)

            for k=1:m_eff

                S_jk = ITensor(zeros((dim(ids[1]),dim(ids[2]))),ids[1],ids[2])

                S_jk[ids[1]=>k,ids[2]=>k] = 1.0

                push!(psi_decomp, S_jk)

            end

            push!(vec_list, sigma)
            push!(M_list, m_eff)
        end
        
    end
    
    M_gm = length(M_list)
    M_tot = sum(M_list)
    
    H_full = zeros((M_tot,M_tot))
    S_full = zeros((M_tot,M_tot))
    
    # Diagonal blocks:
    for i=1:M_gm
        
        i0 = sum(M_list[1:i-1])+1
        i1 = sum(M_list[1:i])
        
        psi_decomp_i = psi_decomp[i0:i1]
        
        H_block = zeros((M_list[i],M_list[i]))
        S_block = zeros((M_list[i],M_list[i]))
        
        for k=1:M_list[i], l=k:M_list[i]
            H_block[k,l] = scalar( psi_decomp_i[k] * dag(H_diag[i]) * setprime(dag(psi_decomp_i[l]),1) )
            H_block[l,k] = H_block[k,l]
            
            temp_block_k = U_list[i]*dag(psi_decomp_i[k])*V_list[i]
            temp_block_l = U_list[i]*dag(psi_decomp_i[l])*V_list[i]
            
            S_block[k,l] = scalar( temp_block_k*dag(temp_block_l) )
            S_block[l,k] = S_block[k,l]
        end
        
        H_full[i0:i1,i0:i1] = H_block
        S_full[i0:i1,i0:i1] = S_block
        
    end
    
    # Off-diagonal blocks:
    for i=1:M_gm, j=i+1:M_gm
        
        i0 = sum(M_list[1:i-1])+1
        i1 = sum(M_list[1:i])
        j0 = sum(M_list[1:j-1])+1
        j1 = sum(M_list[1:j])
        
        psi_decomp_i = psi_decomp[i0:i1]
        psi_decomp_j = psi_decomp[j0:j1]
        
        H_block = zeros((M_list[i],M_list[j]))
        S_block = zeros((M_list[i],M_list[j]))
        
        for k=1:M_list[i], l=1:M_list[j]
            H_block[k,l] = scalar( psi_decomp_j[l] * dag(H_offdiag[i][j-i]) * setprime(dag(psi_decomp_i[k]),1) )
            S_block[k,l] = scalar( psi_decomp_j[l] * dag(S_offdiag[i][j-i]) * setprime(dag(psi_decomp_i[k]),1) )
        end
            
        H_full[i0:i1,j0:j1] = H_block
        H_full[j0:j1,i0:i1] = transpose(H_block)
        
        S_full[i0:i1,j0:j1] = S_block
        S_full[j0:j1,i0:i1] = transpose(S_block)
        
    end
    
    # Construct the "subspace shadow" object:
    shadow = SubspaceShadow(
        sdata.chem_data,
        M_list,
        sdata.mparams.thresh,
        sdata.mparams.eps,
        vec_list,
        [],
        H_full,
        S_full,
        zeros((M_gm,M_gm)),
        zeros((M_gm,M_gm)),
        zeros(M_gm),
        zeros((M_gm,M_gm)),
        0.0
    )
    
    GenSubspaceMats!(shadow)
    
    SolveGenEig!(shadow)
    
    return shadow
    
end


# Replace the site tensors of the matrix product states at p:
function ReplaceStates!(sdata, U_list, S_list, V_list, p, vec_list, eps)
    
    M = sdata.mparams.M
    tol = sdata.mparams.psi_tol
    
    for j=1:sdata.mparams.M
        
        m_eff = length(vec_list[j])
        
        # Replace the tensors:
        for k=1:m_eff
            S_list[j][k,k] = vec_list[j][k]
        end
        
        temp_inds = uniqueinds(sdata.psi_list[j][p], sdata.psi_list[j][p+1])
        
        temp_block = U_list[j]*S_list[j]*V_list[j]
        
        # Generate the "noise" term:
        pmpo = ITensors.ProjMPO(sdata.ham_list[j])
        ITensors.set_nsite!(pmpo,2)
        ITensors.position!(pmpo, sdata.psi_list[j], p)
        drho = eps*ITensors.noiseterm(pmpo,temp_block,"left")
        
        # Replace the tensors of the MPS:
        spec = ITensors.replacebond!(
            sdata.psi_list[j],
            p,
            temp_block;
            maxdim=sdata.mparams.psi_maxdim,
            eigen_perturbation=drho,
            ortho="left",
            normalize=true,
            svd_alg="qr_iteration"
        )
        
    end
    
end


# Pre-contract the "top" blocks prior to sweep:
function ContractTopBlocks(sdata::SubspaceProperties)
    
    n = sdata.chem_data.N_spt
    M = sdata.mparams.M
    
    # Diagonal blocks:
    H_top_diag = Any[1.0 for i=1:M]
    
    # Off-diagonal blocks:
    H_top_offdiag = Any[Any[1.0 for j=i+1:M] for i=1:M]
    S_top_offdiag = Any[Any[1.0 for j=i+1:M] for i=1:M]
    
    # Initialize the lists:
    H_top_diag_list = [deepcopy(H_top_diag)]
    H_top_offdiag_list = [deepcopy(H_top_offdiag)]
    S_top_offdiag_list = [deepcopy(S_top_offdiag)]
    
    # Update the top blocks and push to list:
    for p=n:(-1):3
        
        for i=1:M
            
            H_top_diag[i] *= sdata.psi_list[i][p] * sdata.ham_list[i][p] * setprime(dag(sdata.psi_list[i][p]),1)
            
            for j=i+1:M
                
                yP = sdata.psi_list[j][p] * setprime(sdata.perm_ops[i][j-i][p],2,plev=1)
                Hx = setprime(sdata.ham_list[i][p],2,plev=0) * setprime(dag(sdata.psi_list[i][p]),1)
                
                H_top_offdiag[i][j-i] *= yP
                H_top_offdiag[i][j-i] *= Hx
                
                S_top_offdiag[i][j-i] *= sdata.psi_list[j][p] * sdata.perm_ops[i][j-i][p] * setprime(dag(sdata.psi_list[i][p]),1)
                
            end
            
        end
        
        push!(H_top_diag_list, deepcopy(H_top_diag))
        push!(H_top_offdiag_list, deepcopy(H_top_offdiag))
        push!(S_top_offdiag_list, deepcopy(S_top_offdiag))

    end
    
    return reverse(H_top_diag_list), reverse(H_top_offdiag_list), reverse(S_top_offdiag_list)
    
end


function UpdateBlocks!(sdata, p, Q_list, H_diag, H_offdiag, S_offdiag)
    
    M = sdata.mparams.M
    
    for i=1:M

        H_diag[i] *= Q_list[i] * sdata.ham_list[i][p] * setprime(dag(Q_list[i]),1)

        for j=i+1:M

            yP = Q_list[j] * setprime(sdata.perm_ops[i][j-i][p],2,plev=1)
            Hx = setprime(sdata.ham_list[i][p],2,plev=0) * setprime(dag(Q_list[i]),1)

            H_offdiag[i][j-i] *= yP
            H_offdiag[i][j-i] *= Hx

            S_offdiag[i][j-i] *= Q_list[j] * sdata.perm_ops[i][j-i][p] * setprime(dag(Q_list[i]),1)

        end

    end
    
end


function StatePrepCostFunc(
        shadow::SubspaceShadow,
        theta::Vector{Float64}
    )
    
    normalize!(theta)
    
    M = length(shadow.M_list)
    
    # Maximize von Neumann entropy:
    f_entropy = 0.0
    
    # Minimize the condition number:
    f_overlap = shadow.kappa
    
    for j=1:M
        
        j0 = sum(shadow.M_list[1:j-1])+1
        j1 = sum(shadow.M_list[1:j])
        
        sig2 = (shadow.vec_list[j]).^(2)
        
        f_entropy += sum([sig2[k]*log(sig2[k]) for k=1:length(sig2)])
        
        #f_overlap += sum([abs(shadow.S_mat[j,k]) for k=(j+1):M])
        
    end
    
    f_tot = theta[1]*shadow.E[1] + theta[2]*f_overlap - theta[3]*f_entropy
    
    return f_tot
    
end


function MultiGeomBB!(
        shadow::SubspaceShadow, 
        c0::Vector{Float64}, 
        maxiter::Int
    )
    
    c0 *= 0.99/maximum(c0)
            
    f(c) = BBCostFunc2(c, shadow)

    res = bboptimize(
        f, 
        c0; 
        NumDimensions=length(c0), 
        SearchRange = (-1.0, 1.0), 
        MaxFuncEvals=maxiter, 
        TraceMode=:silent
    )

    c_opt = best_candidate(res)
    e_opt = best_fitness(res)

    if e_opt <= shadow.E[1]

        # Replace the vectors:
        for j=1:length(shadow.M_list)
            j0 = sum(shadow.M_list[1:j-1])+1
            j1 = sum(shadow.M_list[1:j])
            shadow.vec_list[j] = normalize(c_opt[j0:j1])
        end

        GenSubspaceMats!(shadow)

        SolveGenEig!(shadow)

    end
    
end


function MultiGeomAnneal!(
        shadow::SubspaceShadow, 
        c0::Vector{Float64}, 
        maxiter::Int; 
        alpha=1e2,
        delta=1e-2,
        gamma=1.0,
        costfunc="energy",
        theta=[1.0,1.0,1.0],
        stun=true
    )
    
    if costfunc=="energy"
        E_best = shadow.E[1]
    elseif costfunc=="stateprep"
        E_best = StatePrepCostFunc(shadow, theta)
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
            #println(norm(sh2.vec_list[j]))
        end

        # re-compute matrix elements and diagonalize
        GenSubspaceMats!(sh2)
        SolveGenEig!(sh2)
        
        if costfunc=="energy"
            E_old = shadow.E[1]
            E_new = sh2.E[1]
        elseif costfunc=="stateprep"
            E_old = StatePrepCostFunc(shadow, theta)
            E_new = StatePrepCostFunc(sh2, theta)
        end

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


function MultiGeomGenEig!(
        shadow::SubspaceShadow, 
        thresh::String,
        eps::Float64;
        costfunc="energy",
        theta=[1.0,1.0,1.0,1.0]
    )
    
    M_list = shadow.M_list
    M_tot = sum(M_list)
    M_gm = length(M_list)
    
    # Construct 
    
    if costfunc=="energy"
        
        E, C, kappa = SolveGenEig(shadow.H_full, shadow.S_full; thresh=thresh, eps=eps)
        
    elseif costfunc=="stateprep"
        
        # Subspace overlap minimization matrix:
        A = zeros((M_tot, M_tot))
        
        for i=1:M_gm, j=1:M_gm
            
            if i!=j
                i0 = sum(M_list[1:i-1])+1
                i1 = sum(M_list[1:i])
                j0 = sum(M_list[1:j-1])+1
                j1 = sum(M_list[1:j])

                A[i0:i1,i0:i1] += Symmetric(shadow.S_full[i0:i1,j0:j1]*shadow.S_full[j0:j1,i0:i1])
            end
            
        end
        
        # Offdiagonal H-element maximization matrix:
        D = zeros((M_tot, M_tot))
        
        for i=1:M_gm, j=1:M_gm
            
            if i!=j
                i0 = sum(M_list[1:i-1])+1
                i1 = sum(M_list[1:i])
                j0 = sum(M_list[1:j-1])+1
                j1 = sum(M_list[1:j])

                D[i0:i1,i0:i1] += Symmetric(shadow.H_full[i0:i1,j0:j1]*shadow.H_full[j0:j1,i0:i1])
            end
            
        end
        
        # Entanglement growth matrix:
        B = zeros((M_tot, M_tot))
        
        for i=1:M_gm
            i0 = sum(M_list[1:i-1])+1
            i1 = sum(M_list[1:i])
            
            B[i0:i1,i0:i1] = Matrix(I, (M_list[i],M_list[i])) - ones((M_list[i],M_list[i]))
        end
        
        E, C, kappa = ModifiedGenEig(shadow.H_full, shadow.S_full, [A, B, D], theta; eps=eps)
        
    end
    
    #println(E[1])
    
    for j=1:length(shadow.M_list)
        j0 = sum(shadow.M_list[1:j-1])+1
        j1 = sum(shadow.M_list[1:j])
        c_j = C[j0:j1,1]
        nrm = norm(c_j)
        if nrm != 0.0
            shadow.vec_list[j] = c_j/nrm
        else
            shadow.vec_list[j] = c_j
            shadow.vec_list[j][1] = 1.0
        end
        #println(shadow.vec_list[j])
    end
    
    GenSubspaceMats!(shadow)
    SolveGenEig!(shadow)
    
    #println(shadow.E[1])
    
end


function MultiGeomOptim!(
        sdata::SubspaceProperties; 
        sweeps=1,
        maxiter=1000,
        method="bboptim",
        delta=1e-2,
        alpha=1e2,
        stun=true,
        gamma=1.0,
        noise=[0.0],
        thresh="projection",
        eps=1e-12,
        restrict_svals=false,
        costfunc="energy",
        theta=[1.0,1.0,1.0,1.0],
        anchor=false,
        verbose=false
    )
    
    M = sdata.mparams.M
    
    n = sdata.chem_data.N_spt
    
    E_min = sdata.E[1]
    kappa = sdata.kappa
    
    # The sweep loop:
    for s=1:sweeps
        
        # Sweep noise parameter:
        if s > length(noise)
            ns = noise[end]
        else
            ns = noise[s]
        end
        
        # Right-orthogonalize all vectors:
        for j=1:M
            #truncate!(sdata.psi_list[j], cutoff=1e-12, min_blockdim=3, maxdim=3)
            orthogonalize!(sdata.psi_list[j],1)
        end
        
        # Pre-contract the "top" blocks prior to sweep:
        H_top_diag_list, H_top_offdiag_list, S_top_offdiag_list = ContractTopBlocks(sdata)
        
        # Initialize the "bottom" blocks:
        H_bot_diag = Any[1.0 for i=1:M]
        H_bot_offdiag = Any[Any[1.0 for j=i+1:M] for i=1:M]
        S_bot_offdiag = Any[Any[1.0 for j=i+1:M] for i=1:M]
        
        # The site loop:
        for p=1:n-1
            
            # Select the correct "top" blocks:
            H_top_diag = H_top_diag_list[p]
            H_top_offdiag = H_top_offdiag_list[p]
            S_top_offdiag = S_top_offdiag_list[p]
            
            # Decompose at p:
            U_list = []
            sigma_list = []
            V_list = []
            
            # Obtain decompositions at site p:
            for i=1:M
                U,S,V = SiteSVD(sdata.psi_list[i],p, restrict_svals=restrict_svals)
                push!(U_list, U)
                push!(sigma_list, S)
                push!(V_list, V)
            end
            
            # Make a copy for later:
            H_bot_diag_copy = deepcopy(H_bot_diag)
            H_bot_offdiag_copy = deepcopy(H_bot_offdiag)
            S_bot_offdiag_copy = deepcopy(S_bot_offdiag)
            
            # contract the final layer of each block:
            UpdateBlocks!(sdata, p, U_list, H_bot_diag, H_bot_offdiag, S_bot_offdiag)
            UpdateBlocks!(sdata, p+1, V_list, H_top_diag, H_top_offdiag, S_top_offdiag)
            
            # Pre-contract the block tensors for obtaining the H and S matrix elements:
            H_diag = [H_top_diag[i]*H_bot_diag[i] for i=1:M]
            H_offdiag = [[H_top_offdiag[i][j]*H_bot_offdiag[i][j] for j=1:M-i] for i=1:M]
            S_offdiag = [[S_top_offdiag[i][j]*S_bot_offdiag[i][j] for j=1:M-i] for i=1:M]
            
            # Generate the state decompositions \\
            # ...and store as a subspace shadow object:
            shadow = SiteDecomposition(sdata, U_list, sigma_list, V_list, H_diag, H_offdiag, S_offdiag, p, anchor=anchor)
            
            M_tot = sum(shadow.M_list)
            
            c0 = zeros(M_tot)
            
            for j=1:length(shadow.M_list)
                j0 = sum(shadow.M_list[1:j-1])+1
                j1 = sum(shadow.M_list[1:j])
                c0[j0:j1] = shadow.vec_list[j]
            end
            
            shadow_copy = copy(shadow)
            
            if method=="annealing"
                
                # Simulated annealing:
                MultiGeomAnneal!(shadow, c0, maxiter, alpha=alpha, delta=delta, gamma=gamma, stun=stun, costfunc=costfunc, theta=theta)
                
            elseif method=="bboptim"
                
                # Black-box optimizer:
                MultiGeomBB!(shadow, c0, maxiter)
                
            elseif method=="geneig"
                
                # Full diagonalization:
                MultiGeomGenEig!(shadow, thresh, eps, costfunc=costfunc, theta=theta)
                
            end
            
            # Replace the state tensors:
            ReplaceStates!(sdata, U_list, sigma_list, V_list, p, shadow.vec_list, ns)
            
            GenSubspaceMats!(shadow)
            SolveGenEig!(shadow)
            E_min = shadow.E[1]
            kappa = shadow.kappa 
            
            # Update the bottom blocks:
            new_U_list = [sdata.psi_list[j][p] for j=1:M]
            UpdateBlocks!(sdata, p, new_U_list, H_bot_diag_copy, H_bot_offdiag_copy, S_bot_offdiag_copy)
            
            H_bot_diag = H_bot_diag_copy
            H_bot_offdiag = H_bot_offdiag_copy
            S_bot_offdiag = S_bot_offdiag_copy
            
            if verbose
                print("Sweep: [$(s)/$(sweeps)]; site: [$(p)/$(sdata.chem_data.N_spt-1)]; E_min = $(E_min); kappa = $(kappa)             \r")
                flush(stdout)
            end
            
            # Free up some memory:
            H_top_diag_list[p] = [nothing]
            H_top_offdiag_list[p] = [[nothing]]
            S_top_offdiag_list[p] = [[nothing]]
            
        end
        
        # After each sweep, truncate back to the max bond dim:
        for j=1:M
            truncate!(sdata.psi_list[j], maxdim=sdata.mparams.psi_maxdim)
            normalize!(sdata.psi_list[j])
        end
        
    end
    
end


# Co-optimize the parameters and the geometries:
function CoGeomOptim!(
        sdata::SubspaceProperties; 
        sweeps=1,
        maxiter=20,
        mg_maxiter=8000,
        method="geneig",
        swap_mult = 0.5,
        alpha=1e-3,
        stun=true,
        gamma=1e2,
        noise=[0.0],
        thresh="inversion",
        eps=1e-8,
        restrict_svals=false,
        costfunc="energy",
        theta=[1.0,1.0,1.0,1.0],
        anchor=false,
        n_flip=0,
        verbose=false
    )
    
    N = sdata.chem_data.N_spt
    M = sdata.mparams.M
    
    # Initialize the cost function:
    f = sdata.E[1]
    f_best = f
    
    for l=1:maxiter
        
        if verbose
            println("\nLoop $(l) of $(maxiter):")
        end
        
        if l > length(noise)
            ns = noise[end]
        else
            ns = noise[l]
        end
        
        # Make a copy of the sdata:
        new_sdata = copy(sdata)
        
        new_ord_list = deepcopy(new_sdata.ord_list)
    
        # Try permuting each ordering in the list:
        for (j,ord) in enumerate(new_ord_list)
            
            # Number of applied swaps to generate a new ordering (sampled from an exponential distribution):
            num_swaps = Int(floor(swap_mult*randexp()[1]))

            # Apply these swaps randomly:
            for swap=1:num_swaps
                p = rand(1:N-1)
                ord[p:p+1]=reverse(ord[p:p+1])
            end
            
        end
        
        #PrintPrimedInds(new_sdata)
        
        # Re-generate the permutation operators:
        UpdatePermOps!(new_sdata, new_ord_list)
        new_sdata.ord_list = new_ord_list
        #GenPermOps!(new_sdata, verbose=verbose)
        GenHams!(new_sdata)
        
        #PrintPrimedInds(new_sdata)
        
        # Flip the states:
        if n_flip != 0
            j_list = shuffle(collect(1:M))[1:n_flip]
            ReverseStates!(new_sdata, j_list, verbose=false)
        end
        
        # Optimize the states:
        MultiGeomOptim!(
            new_sdata, 
            sweeps=sweeps, 
            method=method,
            maxiter=mg_maxiter,
            noise=[ns],
            thresh=thresh,
            eps=eps,
            restrict_svals=restrict_svals,
            costfunc=costfunc,
            theta=theta,
            verbose=verbose
        )
        
        GenSubspaceMats!(new_sdata)
        SolveGenEig!(new_sdata)
        
        # New cost function:
        f_new = new_sdata.E[1]
        
        # Accept move with some probability
        beta = alpha*l/maxiter

        if stun
            F_0 = Fstun(f_best, f, gamma)
            F_1 = Fstun(f_best, f_new, gamma)
            P = ExpProb(F_1, F_0, beta)
        else
            P = ExpProb(f, f_new, beta)
        end
        
        #println(f)
        #println(f_new)
        #println(P)

        if f_new < f_best
            f_best = f_new
        end

        if rand()[1] < P
            # Accept move:
            sdata.ord_list = new_sdata.ord_list
            sdata.rev_list = new_sdata.rev_list
            sdata.psi_list = new_sdata.psi_list
            sdata.ham_list = new_sdata.ham_list
            sdata.perm_ops = new_sdata.perm_ops
            sdata.perm_alt = new_sdata.perm_alt
            sdata.H_mat = new_sdata.H_mat
            sdata.S_mat = new_sdata.S_mat
            sdata.E = new_sdata.E
            sdata.C = new_sdata.C
            sdata.kappa = new_sdata.kappa
            
            f = f_new
            
            if verbose
                println("\nAccept!")
            end
        else
            if verbose
                println("\nReject!")
            end
        end
        
        
    end
    
    
end
