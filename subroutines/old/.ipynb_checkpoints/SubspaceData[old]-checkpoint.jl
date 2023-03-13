function BBOptimizeStates!(
        sdata::SubspaceProperties;
        loops=1,
        sweeps=1,
        verbose=false
    )
    
    sdata.psi_list = BBOptimizeStates(
        sdata.chem_data,
        sdata.psi_list,
        sdata.ham_list,
        sdata.ord_list;
        tol=sdata.mparams.psi_tol,
        maxdim=sdata.mparams.psi_maxdim,
        perm_tol=sdata.mparams.perm_tol,
        perm_maxdim=sdata.mparams.perm_maxdim,
        loops=loops,
        sweeps=sweeps,
        thresh=sdata.mparams.thresh,
        eps=sdata.mparams.eps,
        verbose=verbose
    )
    
end


function ScreenOrderings!(
        sdata::SubspaceProperties;
        maxiter=20,
        M_new=1,
        annealing=false,
        alpha=1.0,
        verbose=false
    )
    
    sdata.psi_list, sdata.ham_list, sdata.ord_list = ScreenOrderings(
        sdata.chem_data, 
        sdata.sites, 
        sdata.dflt_sweeps, 
        sdata.mparams.M; 
        maxiter=maxiter, 
        M_new=M_new, 
        annealing=annealing,
        alpha=alpha,
        verbose=verbose
    )
    
end

# Expand the subspace to include excited configurations for each ordering:
function ExpandSubspaces!(
        sdata::SubspaceProperties,
        M_ex;
        weight=1.0,
        sweeps=nothing,
        randomize=false,
        verbose=false
    )
    
    if sweeps==nothing
        sweeps=sdata.dlft_sweeps
    end
    
    mpm1 = sdata.mparams
    
    # Expand the metaparameters:
    mpm2 = MetaParameters(
        # Subspace dimension:
        mpm1.M*M_ex,
        M_ex,
        # MPS/MPO constructor parameters:
        mpm1.psi_maxdim,
        mpm1.psi_tol,
        mpm1.ham_maxdim,
        mpm1.ham_tol,
        mpm1.spatial,
        mpm1.spinpair,
        mpm1.singleham,
        # Permutation parameters:
        mpm1.perm_maxdim,
        mpm1.perm_tol,
        # Diagonalization parameters:
        mpm1.thresh,
        mpm1.eps
    )
    
    sdata.mparams = mpm2
    
    psi_list_copy = deepcopy(sdata.psi_list)
    ham_list_copy = deepcopy(sdata.ham_list)
    ord_list_copy = deepcopy(sdata.ord_list)
    
    sdata.psi_list = []
    sdata.ham_list = []
    sdata.ord_list = []
    
    if verbose
        println("Expanding subspaces:")
    end
    
    for j=1:mpm1.M
        
        if verbose
            print("Progress: [$(j)/$(mpm1.M)]\r")
            flush(stdout)
        end
        
        new_ord_list_j = [ord_list_copy[j] for l=2:M_ex]
        #new_ord_list_j = [randperm(sdata.chem_data.N_spt) for l=2:M_ex]
        
        if randomize
            
            if mpm2.spinpair
                spnord = Spatial2SpinOrd(ord_list_copy[j])
            else
                spnord = ord_list_copy[j]
            end
            
            if mpm2.spatial
                hf_occ = [FillHF(spnord[i], sdata.chem_data.N_el) for i=1:sdata.chem_data.N_spt]
            else
                hf_occ = [FillHF(spnord[i], sdata.chem_data.N_el) for i=1:sdata.chem_data.N]
            end
            
            new_psi_list_j = [randomMPS(sdata.sites, hf_occ, linkdims=mpm2.psi_maxdim) for l=2:M_ex]
            new_ham_list_j = [ham_list_copy[j] for l=2:M_ex]
            
        else
            # Generate a set of states:
            new_psi_list_j, new_ham_list_j = GenStates(
                sdata.chem_data, 
                sdata.sites, 
                new_ord_list_j, 
                sweeps, 
                ovlp_opt=true,
                weight=weight,
                prior_states=[psi_list_copy[j]],
                prior_ords=[ord_list_copy[j]],
                perm_tol=sdata.mparams.perm_tol, 
                perm_maxdim=sdata.mparams.perm_maxdim, 
                ham_tol=sdata.mparams.ham_tol, 
                ham_maxdim=sdata.mparams.ham_maxdim, 
                spinpair=sdata.mparams.spinpair, 
                spatial=sdata.mparams.spatial, 
                singleham=sdata.mparams.singleham,
                verbose=false
            )
        end
        
        sdata.ord_list = vcat(sdata.ord_list, [ord_list_copy[j]])
        sdata.ord_list = vcat(sdata.ord_list, new_ord_list_j)
        
        sdata.psi_list = vcat(sdata.psi_list, [psi_list_copy[j]])
        sdata.psi_list = vcat(sdata.psi_list, new_psi_list_j)
        
        sdata.ham_list = vcat(sdata.ham_list, [ham_list_copy[j]])
        sdata.ham_list = vcat(sdata.ham_list, new_ham_list_j)
        
    end
        
    if verbose
        println("\nDone!\n")
    end
    
end



# Expands the space of states per geometry and returns an expanded subspace shadow:
function GenSubspaceShadow(
        sdata_in::SubspaceProperties,
        M_ex;
        weight=1.0,
        sweeps=nothing,
        randomize=false,
        verbose=false
    )
    
    sdata = copy(sdata_in)
    
    if sweeps==nothing
        sweeps=sdata.dflt_sweeps
    end
    
    ExpandSubspaces!(
        sdata, 
        M_ex, 
        weight=weight, 
        sweeps=sweeps, 
        randomize=randomize,
        verbose=verbose
    )
    
    GenSubspaceMats!(sdata, verbose=verbose)
    
    mpm = sdata.mparams
    
    M_gm = Int(mpm.M/mpm.M_ex)
    
    # Generate subspace transformation matrices
    # and initial vectors:
    X_list = []
    vec_list = [zeros(M_ex) for j=1:M_gm]
    for j=1:M_gm
        vec_list[j][1] = 1.0
    end
    
    for j=1:M_gm
        
        j0 = (j-1)*mpm.M_ex+1
        j1 = j*mpm.M_ex
        
        S_jj = Symmetric(sdata.S_mat[j0:j1,j0:j1])
        sqrt_S_jj = Symmetric(sqrt(S_jj))
        #X_j = Symmetric(inv(sqrt_S_jj))
        vec_j = sqrt_S_jj[1:M_ex,1]
        
        #push!(X_list, X_j)
        
    end
    
    M_list = [M_ex for j=1:M_gm]
    
    # Construct the "subspace shadow" object:
    shadow = SubspaceShadow(
        sdata.chem_data,
        M_list,
        mpm.thresh,
        mpm.eps,
        vec_list,
        X_list,
        sdata.H_mat,
        sdata.S_mat,
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


function BBCostFunc(c_j::Vector{Float64}, shadow_in::SubspaceShadow, j::Int)
    
    shadow = copy(shadow_in)
    
    j0 = sum(shadow.M_list[1:j-1])+1
    j1 = sum(shadow.M_list[1:j])
    
    shadow.vec_list[j] = c_j/sqrt(transpose(c_j)*shadow.S_full[j0:j1,j0:j1]*c_j)
    
    #println(shadow.X_list[j]*shadow.vec_list[j])
    
    GenSubspaceMats!(shadow)
    
    SolveGenEig!(shadow, thresh="projection", eps=1e-8)
    
    return shadow.E[1]
    
end


function BBOptimizeStates!(
        shadow::SubspaceShadow;
        loops=1,
        verbose=false
    )
    
    for l=1:loops
        for j=1:length(shadow.M_list)
            
            j0 = sum(shadow.M_list[1:j-1])+1
            j1 = sum(shadow.M_list[1:j])
            
            c0 = shadow.vec_list[j]
            c0 *= 0.99/maximum(abs.(c0))
            
            f(c) = BBCostFunc(c, shadow, j)
            
            res = bboptimize(f, c0; NumDimensions=length(c0), SearchRange = (-1.0, 1.0), MaxFuncEvals=1000, TraceMode=:silent)
            
            c_opt = best_candidate(res)
            e_opt = best_fitness(res)
            
            if e_opt <= shadow.E[1]
                
                shadow.vec_list[j] = c_opt/sqrt(transpose(c_opt)*shadow.S_full[j0:j1,j0:j1]*c_opt)
                
                GenSubspaceMats!(shadow)
                
                SolveGenEig!(shadow)
                
            end
            
            if verbose
                print("Loop: [$(l)/$(loops)]; geometry: [] $(shadow.E[1])  $(shadow.kappa)\r")
                flush(stdout)
            end
            
        end
    end
    
    
end


function AnnealStates!(
        sh_in::SubspaceShadow;
        maxiter=100,
        delta=1e-2,
        alpha=1e2,
        stun=false,
        gamma=1.0,
        verbose=false
    )
    
    sh1 = copy(sh_in)
    
    E_best = sh1.E[1]
    
    for l=1:maxiter
        
        sh2 = copy(sh1)
        
        # Update all states
        for j=1:sh2.M_gm
            j0 = sum(sh2.M_list[1:j-1])+1
            j1 = sum(sh2.M_list[1:j])
            
            c_j = sh2.vec_list[j] + delta*randn(sh2.M_list[j])#/sqrt(l)
            sh2.vec_list[j] = c_j/sqrt(transpose(c_j)*sh2.S_full[j0:j1,j0:j1]*c_j)
        end
        
        # re-compute matrix elements and diagonalize
        GenSubspaceMats!(sh2)
        SolveGenEig!(sh2)
        
        # Accept move with some probability
        beta = alpha*l#^4
        
        if stun
            F_0 = Fstun(E_best, sh1.E[1], gamma)
            F_1 = Fstun(E_best, sh2.E[1], gamma)
            P = ExpProb(F_1, F_0, beta)
        else
            P = ExpProb(sh1.E[1], sh2.E[1], beta)
        end
            
        if sh2.E[1] < E_best
            E_best = sh2.E[1]
        end
        
        if rand(Float64) < P
            sh1 = sh2
        end
        
        if verbose && (div(l,1000)>div((l-1),1000))
            print("$(sh1.E[1])   $(sh1.kappa)      \r")
            flush(stdout)
        end
        
    end
    
    sh_in.vec_list = sh1.vec_list
    GenSubspaceMats!(sh_in)
    SolveGenEig!(sh_in)
    
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

function GenPermOps2!(
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


function GenPermOps!(sdata; tol=1.0e-12, maxdim=2^16, compute_alternates=true, verbose=false)
    
    M = sdata.mparams.M
    N = sdata.chem_data.N_spt
    
    complete_swap = FastPMPO(
        N,
        collect(1:N),
        reverse(collect(1:N)),
        [false,false],
        sdata.sites,
        tol=tol,
        maxdim=maxdim
    )
    
    ApplyPhases!(complete_swap)
    
    if verbose
        println("Generating permutation operators:")
    end
    
    # Declare empty MPOs:
    sdata.perm_alt = [[[MPO(sdata.sites) for j=1:M-i] for i=1:M] for pa=1:4]
    
    c = 0
    c_tot = Int((M^2-M)/2)
    
    # Permute the identity MPOs to obtain \\
    # ...permutation MPOs:
    for i=1:M, j=1:M-i
        
        sdata.perm_alt[1][i][j] = FastPMPO(
            N,
            sdata.ord_list[j+i],
            sdata.ord_list[i],
            [false,false],
            sdata.sites,
            tol=tol,
            maxdim=maxdim
        )
        
        # Alternate (flipped) configurations:
        if compute_alternates
            sdata.perm_alt[2][i][j] = apply(sdata.perm_alt[1][i][j], complete_swap)
            sdata.perm_alt[3][i][j] = apply(complete_swap, sdata.perm_alt[1][i][j])
            sdata.perm_alt[4][i][j] = apply(complete_swap, sdata.perm_alt[2][i][j])
        end
        
        """
        # Alternate (flipped) configurations:
        if compute_alternates
            
            sdata.perm_alt[2][i][j] = FastPMPO(
                N,
                reverse(sdata.ord_list[j+i]),
                sdata.ord_list[i],
                [false,false],
                sdata.sites,
                tol=tol,
                maxdim=maxdim
            )
            
            #sdata.perm_alt[3][i][j] = ReverseMPO(sdata.perm_alt[2][i][j])

            #sdata.perm_alt[4][i][j] = ReverseMPO(sdata.perm_alt[1][i][j])
            
            sdata.perm_alt[2][i][j] = apply()
            
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
        """
        
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


function ReplacePermOps!(sdata, k; compute_alternates=true, verbose=false)
    
    M = sdata.mparams.M
    
    c = 0
    
    if verbose
        println("Updating permutation operators:")
    end
    
    for i=1:M, j=1:M-i
        
        if i==k || j+i==k
        
            sdata.perm_alt[1][i][j] = PermuteMPO(
                MPO(sdata.sites, "I"),
                sdata.sites,
                sdata.ord_list[i],
                sdata.ord_list[j+i],
                tol=1e-16
            )

                # Alternate (flipped) configurations:
            if compute_alternates

                sdata.perm_alt[2][i][j] = PermuteMPO(
                    MPO(sdata.sites, "I"),
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
                print("Progress: [$(c)/$(M-1)] \r")
                flush(stdout)
            end
            
        end
        
    end
    
    if verbose
        println("\nDone!\n")
    end
    
end


function GenHams!(sdata; denseify=false)
    
    # Update the Hamiltonian MPOs:
    for j=1:sdata.mparams.M
        opsum = GenOpSum(sdata.chem_data, sdata.ord_list[j])
        sdata.ham_list[j] = MPO(opsum, sdata.sites, cutoff=sdata.mparams.ham_tol, maxdim=sdata.mparams.ham_maxdim);
        if sdata.rev_list[j]
            sdata.ham_list[j] = ReverseMPO(sdata.ham_list[j])
        end
        if denseify
            sdata.ham_list[j] = dense(sdata.ham_list[j])
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
