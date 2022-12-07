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