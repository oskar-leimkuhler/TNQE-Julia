# A QPU-efficient site-by-site multi-geometry optimizer:


function SiteSVD(psi, p; restrict_svals=false)
    temp_tensor = (psi[p] * psi[p+1])
    temp_inds = uniqueinds(psi[p], psi[p+1]) 
    if restrict_svals
        return svd(temp_tensor,temp_inds,maxdim=maxlinkdim(psi))
    else
        ids = inds(temp_tensor,tags="Link")
        min_svals = 4*minimum(dim(ids))
        return svd(temp_tensor,temp_inds,min_blockdim=min_svals)
    end
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
function SiteDecomposition(
        sd, 
        U_list, 
        sigma_list, 
        V_list, 
        H_diag, 
        H_offdiag, 
        S_offdiag;
        anchor=false
    )
    
    M = sd.mparams.M
    tol = sd.mparams.psi_tol
    m = sd.mparams.psi_maxdim
    
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
        sd.chem_data,
        M_list,
        sd.mparams.thresh,
        sd.mparams.eps,
        vec_list,
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


function DeltaPerturb(tensor, delta)
    
    array = Array(tensor, inds(tensor))
    
    nz_inds = findall(!iszero, array)
    
    for nz in nz_inds
        array[nz] += delta*randn()
    end
    
    tensor2 = ITensor(array, inds(tensor))
    
    tensor2 *= 1.0/sqrt(scalar(tensor2*dag(tensor2)))
    
    return tensor2
    
end


# Replace the site tensors of the matrix product states at p:
function ReplaceStates!(sdata, U_list, S_list, V_list, p_list, vec_list, eps, op)
    
    M = sdata.mparams.M
    tol = sdata.mparams.psi_tol
    
    for j=1:sdata.mparams.M
        
        p = p_list[j]
        
        m_eff = length(vec_list[j])
        
        # Replace the tensors:
        for k=1:m_eff
            S_list[j][k,k] = vec_list[j][k]
        end
        
        temp_inds = uniqueinds(sdata.psi_list[j][p], sdata.psi_list[j][p+1])
        
        # Apply a unitary perturbation to U and V:
        temp_block = U_list[j]*S_list[j]*V_list[j]
        
        # Apply a random perturbation to the temp block:
        temp_block = DeltaPerturb(temp_block, op.delta)
        
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


function MultiGeomBB!(
        shadow::SubspaceShadow, 
        maxiter::Int
    )
    
    M_tot = sum(shadow.M_list)
            
    c0 = zeros(M_tot)

    for j=1:length(shadow.M_list)
        j0 = sum(shadow.M_list[1:j-1])+1
        j1 = sum(shadow.M_list[1:j])
        c0[j0:j1] = shadow.vec_list[j]
    end
    
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
        maxiter::Int; 
        alpha=1e2,
        delta=1e-2,
        gamma=1.0,
        stun=true
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


function MultiGeomGenEig!(
        shadow::SubspaceShadow, 
        thresh::String,
        eps::Float64;
        lev=1
    )
    
    M_list = shadow.M_list
    M_tot = sum(M_list)
    M_gm = length(M_list)
        
    E, C, kappa = SolveGenEig(shadow.H_full, shadow.S_full; thresh=thresh, eps=eps)
    
    for j=1:length(shadow.M_list)
        j0 = sum(shadow.M_list[1:j-1])+1
        j1 = sum(shadow.M_list[1:j])
        c_j = C[j0:j1,lev]
        nrm = norm(c_j)
        if nrm != 0.0
            shadow.vec_list[j] = c_j/nrm
        else
            shadow.vec_list[j] = c_j
            shadow.vec_list[j][1] = 1.0
        end

    end
    
    GenSubspaceMats!(shadow)
    SolveGenEig!(shadow)
    
end


# Optimizer function parameters data structure:

@with_kw mutable struct OptimParameters
    
    maxiter::Int=100 # Number of iterations
    numloop::Int=1 # Number of loops per iter
    
    # State "recycling" parameters:
    rnum::Int=1 # How may states to recycle? (0 for 'off')
    rswp::Int=1 # Number of sweeps
    rnoise::Float64=1e-5 # Noise level
    tethering::Bool=true # "Tether" to the excited states?
    tweight::Float64=2.0 # Strength of tethering
    
    # Move acceptance hyperparameters:
    afunc::String="exp" # "step", "poly", "exp", or "stun"
    tpow::Float64=3.0 # Power for "poly" function
    alpha::Float64=1e1 # Sharpness for "exp" and "stun" functions
    gamma::Float64=1e2 # Additional parameter for "stun" function
    
    # Site decomposition parameters:
    sweep::Bool=false # Do a site sweep instead of random sites?
    restrict_svals::Bool=false # Restrict svals?
    noise::Vector{Float64}=[1e-5] # Size of noise term at each iter
    delta::Float64=1e-3 # Size of random perturbation term
    
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
    
end


# Generate "lock" tensors by fast contraction
function BlockContract(sd, U_list, V_list, p_list)
    
    M = sd.mparams.M
    N = sd.chem_data.N_spt
    
    H_diag = Any[1.0 for i=1:M]
    H_offdiag = Any[Any[1.0 for j=1:M-i] for i=1:M]
    S_offdiag = Any[Any[1.0 for j=1:M-i] for i=1:M]

    for i=1:M 

        psi_i = [deepcopy(sd.psi_list[i][p]) for p=1:N]
        psi_i[p_list[i]] = deepcopy(U_list[i])
        psi_i[p_list[i]+1] = deepcopy(V_list[i])

        for p=1:N
            H_diag[i] *= psi_i[p] * sd.ham_list[i][p] * setprime(dag(psi_i[p]),1)
        end

        for j=1:M-i
            
            mps_j = deepcopy(sd.psi_list[j+i])
            ham_j = deepcopy(sd.ham_list[j+i])
            ppj = [p_list[j+i], p_list[j+i]+1]
            if sd.rev_flag[i][j]
                mps_j = deepcopy(ReverseMPS(mps_j))
                ham_j = deepcopy(ReverseMPO(ham_j))
                ppj = [N-p_list[j+i]+1, N-p_list[j+i]]
            end
            
            psi_j = [mps_j[p] for p=1:N]
            psi_j[ppj[1]] = deepcopy(U_list[j+i])
            psi_j[ppj[2]] = deepcopy(V_list[j+i])
            
            if sd.rev_flag[i][j]
                sites=siteinds(sd.psi_list[j+i])
                replaceind!(psi_j[ppj[1]], sites[p_list[j+i]], sites[ppj[1]])
                replaceind!(psi_j[ppj[2]], sites[p_list[j+i]+1], sites[ppj[2]])
            end

            for p=1:N
                
                yP = psi_j[p] * setprime(sd.perm_ops[i][j][p],2,plev=1)
                Hx = setprime(sd.ham_list[i][p],2,plev=0) * setprime(dag(psi_i[p]),1)

                H_offdiag[i][j] *= yP
                H_offdiag[i][j] *= Hx

                S_offdiag[i][j] *= psi_j[p] * sd.perm_ops[i][j][p] * setprime(dag(psi_i[p]),1)

            end

        end

    end
    
    return H_diag, H_offdiag, S_offdiag
    
end


function RecycleStates!(sd, op, rsweeps, l)
    
    M = sd.mparams.M
    
    j_set = collect(mod1(l,M):mod1(l,M)+op.rnum-1)
            
    for j in j_set

        if op.tethering
            _, sd.psi_list[j] = dmrg(
                sd.ham_list[j],
                sd.tethers[j],
                sd.psi_list[j], 
                rsweeps, 
                weight=op.tweight,
                outputlevel=0
            )
        else
            _, sd.psi_list[j] = dmrg(
                sd.ham_list[j],
                sd.psi_list[j], 
                rsweeps, 
                outputlevel=0
            )
        end

    end
    
end


function RandomSiteDecomp!(
        sdata::SubspaceProperties,
        op::OptimParameters;
        verbose=false
    )
    
    N = sdata.chem_data.N_spt
    M = sdata.mparams.M
    
    # The recycling sweep object:
    rsweeps = Sweeps(op.rswp)
    maxdim!(rsweeps, sdata.mparams.psi_maxdim)
    mindim!(rsweeps, sdata.mparams.psi_maxdim)
    setnoise!(rsweeps, op.rnoise)
    
    # Initialize the cost function:
    f = sdata.E[1] 
    f_best = f
    f_new = f
    
    n_accept = 0
    
    if verbose
        println("\nRANDOM SITE DECOMPOSITION:")
    end
    
    # The iteration loop:
    for l=1:op.maxiter
        
        # Noise at this iteration:
        lnoise = op.noise[minimum([n_accept+1,end])]
        
        # Make a copy of the subspace:
        sdata_c = copy(sdata)
        
        # "Recycle" the basis states:
        if op.rnum != 0
            RecycleStates!(
                sdata_c, 
                op, 
                rsweeps, 
                l
            )
        end
        
        p_lists = [collect(1:N-1) for j=1:M]
        
        # Decomposition loop:
        for s=1:op.numloop
            
            # Choose sites randomly and orthogonalize states
            p_list = [rand(1:N-1) for j=1:M]
            
            if op.sweep
                # Sweep through the sites instead:
                p_list = [p_lists[j][mod1(s,N-1)] for j=1:M]
            end
            
            for j=1:M
                orthogonalize!(sdata_c.psi_list[j], p_list[j])
            end
            
            # Perform site decompositions at chosen sites
            U_list, sigma_list, V_list = [], [], []
            
            for j=1:M
                U,S,V = SiteSVD(
                    sdata_c.psi_list[j],
                    p_list[j], 
                    restrict_svals=op.restrict_svals
                )
                push!(U_list, U)
                push!(sigma_list, S)
                push!(V_list, V)
            end
            
            # Generate "lock" tensors by fast contraction
            H_diag, H_offdiag, S_offdiag = BlockContract(
                sdata_c, 
                U_list, 
                V_list, 
                p_list
            )
            
            # Compute matrix elements:
            shadow = SiteDecomposition(
                sdata_c, 
                U_list, 
                sigma_list, 
                V_list, 
                H_diag, 
                H_offdiag, 
                S_offdiag
            )
            
            # Minimize energy by chosen method:
            if op.sd_method=="annealing" # Simulated annealing:
                MultiGeomAnneal!(
                    shadow, 
                    op.sd_maxiter, 
                    alpha=op.sd_alpha, 
                    delta=op.sd_delta, 
                    gamma=op.sd_gamma, 
                    stun=op.sd_stun
                )
            elseif op.sd_method=="bboptim" # Black-box optimizer:
                MultiGeomBB!(
                    shadow, 
                    op.sd_maxiter
                )
            elseif op.sd_method=="geneig" # Full diagonalization:
                MultiGeomGenEig!(
                    shadow, 
                    op.sd_thresh, 
                    op.sd_eps
                )
            else
                println("Invalid optimization method!")
                return nothing
            end
            
            # Contract output => update MPSs
            ReplaceStates!(
                sdata_c, 
                U_list, 
                sigma_list, 
                V_list, 
                p_list, 
                shadow.vec_list, 
                lnoise,
                op
            )
            
            """
            # After each decomp, truncate back to the max bond dim:
            for j=1:M
                #orthogonalize!(sdata_c.psi_list[j], p_list[j])
                #truncate!(sdata_c.psi_list[j], maxdim=sdata_c.mparams.psi_maxdim)
                normalize!(sdata_c.psi_list[j])
            end
            """
            
            # Copy the subspace mats:
            sdata_c.H_mat = shadow.H_mat
            sdata_c.S_mat = shadow.S_mat
            sdata_c.E = shadow.E
            sdata_c.C = shadow.C
            sdata_c.kappa = shadow.kappa

            f_new = sdata_c.E[1]
            
            # Print some output
            if verbose
                print("Iter: $(l)/$(op.maxiter); loop: $(s)/$(op.numloop); E_min = $(round(sdata_c.E[1], digits=5)); kappa = $(round(sdata_c.kappa, sigdigits=3)); E_best = $(round(f_best, digits=5)); accept = $(n_accept)/$(l-1) ($(Int(round(100.0*n_accept/maximum([l-1, 1]), digits=0)))%)     \r")
                flush(stdout)
            end
            
        end
        
        # Accept move with some probability:
        beta = op.alpha*l/op.maxiter
        
        if op.afunc=="step"
            P = StepProb(f, f_new)
        elseif op.afunc=="poly"
            P = PolyProb(f, f_new, beta, tpow=op.tpow)
        elseif op.afunc=="exp"
            P = ExpProb(f, f_new, beta)
        elseif op.afunc=="stun"
            F_0 = Fstun(f_best, f, op.gamma)
            F_1 = Fstun(f_best, f_new, op.gamma)
            P = ExpProb(F_1, F_0, beta)
        elseif op.afunc=="flat"
            P = 1
        else
            println("Invalid probability function!")
            return nothing
        end

        if f_new < f_best
            f_best = f_new
        end
        
        if rand()[1] < P
            copyto!(sdata, sdata_c)
            n_accept += 1
            f=f_new
        end
        
    end
    
    if verbose
        println("\nDone!\n")
    end
    
end