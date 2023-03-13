# Functions to perform black-box optimization on the MPS tensors \\
# ...with black-box cost function = GenEig E[1].

# Packages:
using LinearAlgebra
using GraphRecipes
using JuMP
import Ipopt
import ForwardDiff
using BlackBoxOptim


function ReplaceTensors(psi, p, tensor; tol=1e-12, maxdim=1e4)

    temp_inds = uniqueinds(psi[p],psi[p+1])
    
    U,S,V = svd(tensor,temp_inds,cutoff=tol,maxdim=maxdim,alg="qr_iteration")
    
    psi[p] = U
    psi[p+1] = S*V
    
    psi[p+1] *= 1.0/norm(psi)
    
    return psi
end


function ReplaceMatEls(H_mat, S_mat, x_tensor, j, H_blocks, S_blocks)
    M = size(H_mat,1)
    
    for i in setdiff(collect(1:M), [j])
        H_mat[i,j] = scalar(x_tensor * H_blocks[i])
        H_mat[j,i] = H_mat[i,j]
        S_mat[i,j] = scalar(x_tensor * S_blocks[i])
        S_mat[j,i] = S_mat[i,j]
    end
    
    H_mat[j,j] = scalar(x_tensor * H_blocks[j] * setprime(dag(x_tensor),1))
    
    return H_mat, S_mat
end


function nzInds(tensor)
    
    t_inds = inds(tensor)
    array = Array(tensor, t_inds)
    nz_inds = findall(!iszero, array)
    return nz_inds, t_inds
    
end


function ConstructTensor(x, t_inds, nz_inds)
    
    dims = [dim(t_ind) for t_ind in t_inds]
    x_array = zeros(Tuple(dims))
    
    for (i,nz_ind) in enumerate(nz_inds)
        x_array[nz_ind] = x[i]
    end
    
    x_tensor = ITensor(x_array, t_inds)
    
    return x_tensor 
end


function ReMPSCostFunc(
        x,
        ids,
        nz_els,
        H_in,
        S_in,
        H_blocks,
        S_blocks,
        j,
        p;
        tol=1e-12,
        maxdim=1e4,
        thresh="none",
        eps=1e-8,
        verbose=false,
        debug=false
    )
    
    H_mat = deepcopy(H_in)
    S_mat = deepcopy(S_in)
    M = size(H_mat,1)
    
    # Construct the tensor:
    x_tensor = ConstructTensor(x, ids,nz_els)
    
    # Normalize:
    tnorm = sqrt(scalar(x_tensor*S_blocks[j]*setprime(dag(x_tensor),1,tags="Link")))
    x_tensor *= 1.0/tnorm
    
    H_mat, S_mat = ReplaceMatEls(H_mat, S_mat, x_tensor, j, H_blocks, S_blocks)
    
    E, C, kappa = SolveGenEig(
        H_mat, 
        S_mat, 
        thresh=thresh, 
        eps=eps
    )
    
    if debug
        display(H_mat-H_in)
        display(S_mat-S_in)
    end
    
    return E[1]
    
end


# Pre-contract the "top" blocks prior to sweep:
function ContractTopBlocks(psi_j, dag_psi, ham, j)
    
    n = length(ham)
    M = length(dag_psi)
    
    # Initialize the top blocks:
    H_top_blocks_list = Any[[1.0 for i=1:M]]
    S_top_blocks_list = Any[[1.0 for i=1:M]]
    H_top_blocks = [psi_j[n] * ham[n] * setprime(dag_psi[i][n],1) for i=1:M]
    S_top_blocks = [psi_j[n] * setprime(dag_psi[i][n],1,tags="Link") for i=1:M]
    push!(H_top_blocks_list, deepcopy(H_top_blocks))
    push!(S_top_blocks_list, deepcopy(S_top_blocks))
    
    # Update the top blocks and push to list:
    for p=n-1:(-1):3
        
        for i=1:M
            H_top_blocks[i] *= psi_j[p] * ham[p] * setprime(dag_psi[i][p],1)
            S_top_blocks[i] *= psi_j[p] * setprime(dag_psi[i][p],1,tags="Link")
        end
        push!(H_top_blocks_list, deepcopy(H_top_blocks))
        push!(S_top_blocks_list, deepcopy(S_top_blocks))

    end
    
    return reverse(H_top_blocks_list), reverse(S_top_blocks_list)
    
end


function BBOptimizeMPS(
        psi_in,
        ham_in,
        ord_list,
        j,
        H_mat,
        S_mat;
        sweeps=1,
        tol=1e-12,
        maxdim=1e4,
        perm_tol=1e-12,
        perm_maxdim=1e4,
        thresh="none",
        eps=1e-10,
        verbose=false
    )
    
    sites = siteinds(psi_in[1])
    psi_list = deepcopy(psi_in)
    ham = deepcopy(ham_in)
    M = length(psi_list)
    N_sites = length(psi_list[j])
    
    # Permute the states into the j-ordering:
    for i=1:M
        psi_list[i] = Permute(
            psi_list[i],
            sites, 
            ord_list[i], 
            ord_list[j], 
            tol=perm_tol, 
            maxdim=perm_maxdim, 
            spinpair=false, 
            locdim=4
        )   
    end
    
    # "Dagger" the states:
    dag_psi = [dag(psi) for psi in psi_list]
    
    E, C, kappa = SolveGenEig(
        H_mat, 
        S_mat, 
        thresh=thresh, 
        eps=eps
    )
    
    # The sweep loop:
    for s=1:sweeps

        # Pre-contract the "top" blocks prior to sweep:
        H_top_blocks_list, S_top_blocks_list = ContractTopBlocks(psi_list[j], dag_psi, ham, j)
        
        # Initialize the bottom blocks to 1:
        H_bottom_blocks = [1.0 for i=1:M]
        S_bottom_blocks = [1.0 for i=1:M]
        
        # Loop over the local indices of the mps:
        for p=1:N_sites-1
            
            # Select the correct "top" blocks:
            H_top_blocks = H_top_blocks_list[p]
            S_top_blocks = S_top_blocks_list[p]
            
            # Contract the "middle" blocks:
            H_middle_blocks = []
            S_middle_blocks = []
            for i=1:M
                if i!=j
                    H_middle_block_i = setprime(dag_psi[i][p],1) * setprime(dag_psi[i][p+1],1)
                    H_middle_block_i *= ham[p] * ham[p+1]
                    S_middle_block_i = setprime(dag_psi[i][p],1,tags="Link") * setprime(dag_psi[i][p+1],1,tags="Link")
                else
                    H_middle_block_i = ham[p] * ham[p+1]
                    S_middle_block_i = 1.0
                end
                push!(H_middle_blocks, H_middle_block_i)
                push!(S_middle_blocks, S_middle_block_i)
            end
            
            if p==2 # Initialize the "bottom" blocks:
                H_bottom_blocks = [psi_list[j][1] * ham[1] * setprime(dag_psi[i][1],1) for i=1:M]
                S_bottom_blocks = [psi_list[j][1] * setprime(dag_psi[i][1],1,tags="Link")  for i=1:M]
            elseif p>2 #  Update the "bottom" blocks:
                for i=1:M
                    H_bottom_blocks[i] *= psi_list[j][p-1] * ham[p-1] * setprime(dag_psi[i][p-1],1)
                    S_bottom_blocks[i] *= psi_list[j][p-1] * setprime(dag_psi[i][p-1],1,tags="Link")
                end
            end
            
            # Pre-contract the block tensors for obtaining the H and S matrix elements:
            H_blocks = []
            S_blocks = []
            for i=1:M
                H_block_i = H_top_blocks[i] * H_middle_blocks[i] * H_bottom_blocks[i]
                S_block_i = S_top_blocks[i] * S_middle_blocks[i] * S_bottom_blocks[i]
                push!(H_blocks, H_block_i)
                push!(S_blocks, S_block_i)
            end
            
            # Obtain the initial values:
            init_tensor = psi_list[j][p] * psi_list[j][p+1]
            nz_els,ids = nzInds(init_tensor)
            n_nz = length(nz_els)
            x0 = [init_tensor[nz_el] for nz_el in nz_els]
            x0 .*= 0.999/maximum(abs.(x0))
            
            # Now construct the cost function:
            f(x) = ReMPSCostFunc(
                x,
                ids,
                nz_els,
                H_mat,
                S_mat,
                H_blocks,
                S_blocks,
                j,
                p,
                tol=tol,
                maxdim=maxdim,
                thresh=thresh,
                eps=eps,
                verbose=false,
                debug=false
            )
            
            # Optimize!
            res = bboptimize(f, x0; NumDimensions=n_nz, SearchRange = (-1.0, 1.0), MaxFuncEvals=1000, TraceMode=:silent)
            x_opt = best_candidate(res)
            e_opt = best_fitness(res)
            
            if e_opt <= f(x0)
                
                # Construct the tensor:
                x_tensor = ConstructTensor(x_opt,ids,nz_els)
                
                #Normalize:
                tnorm = sqrt(scalar(x_tensor*S_blocks[j]*setprime(dag(x_tensor),1,tags="Link")))
                x_tensor *= 1.0/tnorm
                
                # Replace matrix elements:
                H_mat, S_mat = ReplaceMatEls(H_mat, S_mat, x_tensor, j, H_blocks, S_blocks)
                
                # Replace the state MPS:
                psi_list[j] = ReplaceTensors(psi_list[j], p, x_tensor; tol=1e-12,maxdim=maxdim)

                dag_psi[j] = dag(psi_list[j])
                
            end
            
            if verbose
                print("State: [$(j)/$(M)]; sweep: [$(s)/$(sweeps)]; site: [$(p)/$(N_sites-1)];  min. eval = $(E[1]) \r")
                flush(stdout)
            end
            
        end
        
        E, C, kappa = SolveGenEig(
            H_mat, 
            S_mat, 
            thresh=thresh, 
            eps=eps
        )
        
    end
    
    return psi_list[j], H_mat, S_mat
    
end


function BBOptimizeStates(
        chemical_data,
        psi_in,
        ham_list,
        ord_list;
        tol=1e-12,
        maxdim=31,
        perm_tol=1e-12,
        perm_maxdim=5000,
        loops=1,
        sweeps=1,
        thresh="none",
        eps=1e-10,
        verbose=false
    )
    
    sites = siteinds(psi_in[1])
    
    psi_list = deepcopy(psi_in)
    
    M = length(psi_list)
    
    H_mat, S_mat = GenSubspaceMats(
        chemical_data, 
        sites, 
        ord_list, 
        psi_list,
        ham_list,
        perm_tol=perm_tol, 
        perm_maxdim=perm_maxdim, 
        spinpair=false, 
        spatial=true, 
        singleham=false,
        verbose=false
    )

    E, C, kappa = SolveGenEig(
        H_mat, 
        S_mat, 
        thresh=thresh, 
        eps=eps
    )
    
    if verbose
        println("Starting energy: $(E[1])")
    end
    
    for l=1:loops
        
        if verbose
            println("\nStarting run $(l) of $(loops):")
        end
        
        for j=1:M
            
            psi_list[j], H_mat, S_mat = BBOptimizeMPS(
                psi_list,
                ham_list[j],
                ord_list,
                j,
                H_mat,
                S_mat;
                sweeps=sweeps,
                tol=1e-12,
                maxdim=maxdim,
                perm_tol=perm_tol,
                perm_maxdim=perm_maxdim,
                thresh=thresh,
                eps=eps,
                verbose=verbose
            )
            
        end
        
        E, C, kappa = SolveGenEig(
            H_mat, 
            S_mat, 
            thresh=thresh, 
            eps=eps
        )
        
        if verbose
            println("\nRun $(l) complete!")
            println("Ground state energy: $(E[1])")
            println("Condition number: $(kappa)")
        end
        
    end
    
    return psi_list
    
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
    
    GenSubspaceMats!(sdata)
    SolveGenEig!(sdata, thresh="inversion", eps=1e-8)
    
end


# Co-optimize the parameters and the geometries:
function CoGeomOptim!(
        sdata::SubspaceProperties; 
        sweeps=1,
        maxiter=20,
        mg_maxiter=8000,
        method="geneig",
        opt_all=false,
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
    
    # Need to globally init this variable:
    j_index = 1
    
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
        
        if opt_all
            
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
            
        else
            
            # Select a random state to permute:
            j_index = rand(1:M)

            # Number of applied swaps to generate a new ordering (sampled from an exponential distribution):
            num_swaps = Int(floor(swap_mult*randexp()[1]))

            # Apply these swaps randomly:
            for swap=1:num_swaps
                p = rand(1:N-1)
                new_ord_list[j_index][p:p+1]=reverse(new_ord_list[j_index][p:p+1])
            end
            
        end
        
        #PrintPrimedInds(new_sdata)
        
        # Re-generate the permutation operators:
        #UpdatePermOps!(new_sdata, new_ord_list)
        
        new_sdata.ord_list = new_ord_list
        
        if opt_all
            GenPermOps!(new_sdata, verbose=verbose)
        else
            ReplacePermOps!(new_sdata, j_index; verbose=verbose)
        end
        
        SelectPermOps!(new_sdata)
        
        GenHams!(new_sdata, denseify=true)
        
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
        
        #P = StepProb(f, f_new)
        
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



"""
# Return a subspace shadow generated
# by Schmidt-decomposing state k at site p:
function OneStateSiteDecomposition(sdata, U_k, sigma_k, V_k, H_list, S_list, p, k)
    
    M = sdata.mparams.M
    tol = sdata.mparams.psi_tol
    m = sdata.mparams.psi_maxdim
    
    psi_decomp = []
    vec = []
        
    sigma = diag(Array(sigma_k.tensor))
        
    m_eff = length(sigma)
        
    ids = inds(H_diag[j], plev=0)
        
    for j=1:m_eff
            
        S_jk = ITensor(zeros((dim(ids[1]),dim(ids[2]))),ids[1],ids[2])
            
        S_jk[ids[1]=>j,ids[2]=>j] = 1.0

        push!(psi_decomp, S_jk)
            
    end
        
    vec = sigma
    
    M_gm = sdata.mparams.M
    M_tot = M_gm+m_eff-1
    
    H_full = zeros((M_tot,M_tot))
    S_full = zeros((M_tot,M_tot))
    
    k0 = k
    k1 = k+m_eff-1
    
    # Existing blocks:
    H_full[1:k0-1,1:k0-1] = sdata.H_mat[1:k-1,1:k-1]
    H_full[k1+1:M_tot,k1+1:M_tot] = sdata.H_mat[k+1:M,k+1:M]
    H_full[1:k0-1,k1+1:M_tot] = sdata.H_mat[1:k-1,k+1:M]
    H_full[k1+1:M_tot,1:k0-1] = sdata.H_mat[k+1:M,1:k-1]
    
    S_full[1:k0-1,1:k0-1] = sdata.S_mat[1:k-1,1:k-1]
    S_full[k1+1:M_tot,(k+m_eff-1):M_tot] = sdata.S_mat[k+1:M,k+1:M]
    S_full[1:k0-1,k1+1:M_tot] = sdata.S_mat[1:k-1,k+1:M]
    S_full[k1+1:M_tot,1:k0-1] = sdata.S_mat[k+1:M,1:k-1]
    
    # The middle block:
    H_block = zeros((m_eff,m_eff))
        
    for i=k0:k1, j=k0:k1
        H_block[i,j] = scalar( psi_decomp[i-k0+1] * dag(H_list[k]) * setprime(dag(psi_decomp[j-k0+1]),1) )
        H_block[j,i] = H_block[i,j]
    end
        
    H_full[k0:k1,k0:k1] = deepcopy(H_block)
    S_full[k0:k1,k0:k1] = Matrix(I, m_eff, m_eff)
    
    # Off-diagonal blocks:
    H_block = zeros((k-1,m_eff))
    S_block = zeros((k-1,m_eff))
    for i=1:k0-1, j=k0:k1
        
        H_block[i,j] = scalar( psi_decomp_j[l] * dag(H_offdiag[i][j-i]) * setprime(dag(psi_decomp_i[k]),1) )
        S_block[k,l] = scalar( psi_decomp_j[l] * dag(S_offdiag[i][j-i]) * setprime(dag(psi_decomp_i[k]),1) )
            
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
"""

"""
function OneStateTopBlocks(sdata::SubspaceProperties, k)
    
    n = sdata.chem_data.N_spt
    M = sdata.mparams.M

    H_top = Any[1.0 for j=1:M]
    S_top = Any[1.0 for j=1:M]
    
    # Initialize the lists:
    H_top_list = [deepcopy(H_top)]
    S_top_list = [deepcopy(S_top)]
    
    # Update the top blocks and push to list:
    for p=n:(-1):3
        
        for j=1:M
            
            if j==k
                H_top[j] *= sdata.psi_list[j][p] * sdata.ham_list[j][p] * setprime(dag(sdata.psi_list[j][p]),1)
            else
                yP = sdata.psi_list[j][p] * setprime(sdata.perm_ops[k][j-k][p],2,plev=1)
                Hx = setprime(sdata.ham_list[k][p],2,plev=0) * setprime(dag(sdata.psi_list[k][p]),1)

                H_top[j] *= yP
                H_top[j] *= Hx

                S_top[j] *= sdata.psi_list[j][p] * sdata.perm_ops[k][j-k][p] * setprime(dag(sdata.psi_list[k][p]),1)
            end

        end
        
        push!(H_top_list, deepcopy(H_top))
        push!(S_top_list, deepcopy(S_top))

    end
    
    return reverse(H_top_list), reverse(S_top_list)
    
end
"""

"""
function OneStateUpdateBlocks!(sdata, p, Q, H_list, S_list, k)
    
    M = sdata.mparams.M
    
    for j=1:M
        
        if j==k
            H_list[j] *= Q * sdata.ham_list[j][p] * setprime(dag(Q),1)
        else

            yP = sdata.psi_list[j][p] * setprime(sdata.perm_ops[k][j-k][p],2,plev=1)
            Hx = setprime(sdata.ham_list[k][p],2,plev=0) * setprime(dag(Q),1)

            H_list[j] *= yP
            H_list[j] *= Hx

            S_list[j] *= sdata.psi_list[j][p] * sdata.perm_ops[k][j-k][p] * setprime(dag(Q),1)

        end

    end
    
end
"""


"""
function SingleGeomOptim!(
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
        verbose=false
    )
    
    M = sdata.mparams.M
    
    n = sdata.chem_data.N_spt
    
    E_min = sdata.E[1]
    kappa = sdata.kappa
    
    # The sweep loop:
    for l=1:loops
        
        # Sweep noise parameter:
        if l > length(noise)
            eps = noise[end]
        else
            eps = noise[l]
        end
        
        # Optimize state k:
        for k=1:sdata.mparams.M
            
            for s=1:sweeps
                
                # Right-orthogonalize all vectors:
                for j=1:M
                    #truncate!(sdata.psi_list[j], cutoff=1e-12, min_blockdim=3, maxdim=3)
                    orthogonalize!(sdata.psi_list[j],1)
                end
                
                # Pre-contract the "top" blocks prior to sweep:
                H_top_list, S_top_list = OneStateTopBlocks(sdata, k)

                # Initialize the "bottom" blocks:
                H_bot = Any[1.0 for i=1:M]
                S_bot = Any[1.0 for i=1:M]

                # The site loop:
                for p=1:n-1

                    # Select the correct "top" blocks:
                    H_top = H_top_list[p]
                    S_top = S_top_list[p]

                    # Decompose state k at p:
                    U,S,V = SiteSVD(sdata.psi_list[k],p, restrict_svals=restrict_svals)

                    # Make a copy for later:
                    H_bot_copy = deepcopy(H_bot)
                    S_bot_copy = deepcopy(S_bot)

                    # contract the final layer of each block:
                    OneStateUpdateBlocks!(sdata, p, U, H_bot, S_bot, k)
                    OneStateUpdateBlocks!(sdata, p+1, V, H_top, S_top, k)
        
                    # Pre-contract the block tensors for obtaining the H and S matrix elements:
                    H_list = [H_top[j]*H_bot[j] for j=1:M]
                    S_list = [S_top[j]*S_bot[j] for j=1:M]

                    # Generate the state decompositions \\
                    # ...and store as a subspace shadow object:
                    shadow = SiteDecomposition(sdata, U, sigma, V, H_list, S_list, p)

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
                        MultiGeomAnneal!(shadow, c0, maxiter, alpha=alpha, delta=delta, gamma=gamma, stun=stun)

                    elseif method=="bboptim"

                        # Black-box optimizer:
                        MultiGeomBB!(shadow, c0, maxiter)

                    elseif method=="geneig"

                        # Full diagonalization:
                        MultiGeomGenEig!(shadow, thresh, eps)

                    end

                    # Replace the state tensors:
                    ReplaceStates!(sdata, U_list, sigma_list, V_list, p, shadow.vec_list, eps)

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
                    #H_top_diag_list[p] = [nothing]
                    #H_top_offdiag_list[p] = [[nothing]]
                    #S_top_offdiag_list[p] = [[nothing]]

                end

                # After each sweep, truncate back to the max bond dim:
                for j=1:M
                    truncate!(sdata.psi_list[j], maxdim=sdata.mparams.psi_maxdim)
                    normalize!(sdata.psi_list[j])
                end

            end
            
        end
        
    end
    
end
"""


"""
# Applies a small random perturbation to U
function QPerturbation(U, rind, delta)
    
    linds = uniqueinds(inds(U), rind)
    
    U1 = deepcopy(U)
    
    U1 += delta*randomITensor(inds(U1))
    
    U2 = dense(U1)
    
    Q,R = qr(U2, removeqns(linds); positive=true)
    
    #display(U2.tensor)
    
    U3 = ITensor(Q.tensor, inds(U))
    
    c1 = combiner(inds(U3, tags="u")[1], tags="c1")
    c2 = combiner(dag(rind), tags="c2")
    
    cinds = [inds(c1, tags="c1")[1], inds(c2, tags="c2")[1]]
    
    cd = ITensors.delta(dag(cinds[1]), dag(cinds[2]))
    
    U3 *= c1
    U3 *= cd
    U3 *= c2
    
    #display(U2.tensor)
    
    return U3
    
end
"""


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


"""

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


"""


# Optimizer function parameters data structure:

@with_kw mutable struct OptimParameters
    
    maxiter::Int=100 # Number of iterations
    numloop::Int=1 # Number of loops per iter
    
    # Geometry update move parameters:
    theta::Vector{Float64}=[1.0,0,0,0,0] # Weights for E, log(kappa), fit
    permute_states::Bool=false # Permute the states after geom. update?
    move_type::String="random" # "none", "random" or "shuffle"
    
    # Move acceptance hyperparameters:
    afunc::String="exp" # "step", "poly", "exp", or "stun"
    tpow::Float64=3.0 # Power for "poly" function
    alpha::Float64=1e1 # Sharpness for "exp" and "stun" functions
    gamma::Float64=1e2 # Additional parameter for "stun" function
    
    # Geometry permutation parameters:
    swap_num::Int=-1 # How may states to permute (-1 => all)
    swap_mult::Float64=1.0 # Swappiness multiplier
    
    # Site decomposition parameters:
    sweep::Bool=false # Do a site sweep instead of random sites?
    restrict_svals::Bool=false # Restrict svals?
    noise::Vector{Float64}=[1e-5] # Size of noise term at each iter
    delta::Float64=1e-3 # Size of unitary Q-perturbation
    
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


function ApplyRandomSwaps!(
        sd::SubspaceProperties,
        op::OptimParameters
    )
    
    # Randomly select geometries:
    swap_states = randperm(sd.mparams.M)[1:op.swap_num]

    # Randomly apply swaps:
    for j in swap_states

        # Number of applied swaps to generate a new ordering (sampled from an exponential distribution):
        num_swaps = Int(floor(op.swap_mult*randexp()[1]))

        # Apply these swaps randomly:
        for swap=1:num_swaps
            p = rand(1:sd.chem_data.N_spt-1)
            sd.ord_list[j][p:p+1]=reverse(sd.ord_list[j][p:p+1])
        end

    end
    
end


function GeometryMove!(sd, op, gp, biparts, entrops)
    
    old_ords = deepcopy(sd.ord_list)
    
    # Permute the orderings:
    if op.move_type=="random"
        ApplyRandomSwaps!(sd, op)
    elseif op.move_type=="shuffle"
        sd.ord_list=shuffle(sd.ord_list)
    elseif op.move_type=="anneal"
        sd.ord_list = BipartiteAnnealing(
            sd.ord_list, 
            biparts[1], 
            entrops[1], 
            sd.mparams.psi_maxdim,
            gp,
            statevec=sd.C[:,1],
            verbose=true
        )
    end

    # Re-compute permutation tensors
    GenPermOps!(sd)

    # Update the Hamiltonian MPOs and states:
    for j=1:sd.mparams.M

        opsum = GenOpSum(sd.chem_data, sd.ord_list[j])

        sd.ham_list[j] = MPO(
            opsum, 
            sd.sites, 
            cutoff=sd.mparams.ham_tol, 
            maxdim=sd.mparams.ham_maxdim
        )

        if op.permute_states
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



function RandomSiteDecomp!(
        sdata::SubspaceProperties,
        op::OptimParameters;
        biparts=nothing,
        entrops=nothing,
        gp=nothing,
        tether_list=nothing,
        noisy_sweeps=nothing,
        verbose=false
    )
    
    N = sdata.chem_data.N_spt
    M = sdata.mparams.M
    
    if op.swap_num==-1
        op.swap_num = M
    end
    
    # Initialize the cost function:
    fit = 0.0
    """
    if gp != nothing
        if gp.costfunc=="simple"
            fit = BipartiteFitness(
                sdata.ord_list, 
                biparts, 
                entrops, 
                sdata.mparams.psi_maxdim, 
                statevec=sdata.C[:,1], 
                zeta=gp.zeta_list[1], 
                xi=gp.xi
            )
        else
            fit = CompositeCostFunc(
                sdata.ord_list, 
                biparts, 
                entrops, 
                sdata.mparams.psi_maxdim, 
                sdata.C[:,1], 
                gp
            )
        end
    end
    """
    
    f = op.theta[1]*sdata.E[1] + op.theta[2]*log(sdata.kappa) + op.theta[3]*log(maximum([fit+2.0, 1.0]))
        
    f_best = f
    
    fit_new = fit
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
        
        # Update the geometries:
        if op.move_type != "none"
            GeometryMove!(sdata_c, op, gp, biparts, entrops)
        end
        
        # Refresh one of the basis states:
        j = mod1(l, sdata.mparams.M)
        _, sdata_c.psi_list[j] = dmrg(
            sdata_c.ham_list[j],
            tether_list[j],
            sdata_c.psi_list[j], 
            noisy_sweeps, 
            weight=2.0,
            outputlevel=0
        )
        
        #p_lists = [randperm(N-1) for j=1:M]
        p_lists = [collect(1:N-1) for j=1:M]
        
        levs = [1]
        
        # Decomposition loop:
        for s=1:op.numloop
            
            # Loop initializations
            
            slev = levs[minimum([s,end])]
            
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
            
            M_tot = sum(shadow.M_list)
            
            c0 = zeros(M_tot)
            
            for j=1:length(shadow.M_list)
                j0 = sum(shadow.M_list[1:j-1])+1
                j1 = sum(shadow.M_list[1:j])
                c0[j0:j1] = shadow.vec_list[j]
            end
            
            # Minimize energy by chosen method:
            if op.sd_method=="annealing" # Simulated annealing:
                MultiGeomAnneal!(
                    shadow, 
                    c0, 
                    op.sd_maxiter, 
                    alpha=op.sd_alpha, 
                    delta=op.sd_delta, 
                    gamma=op.sd_gamma, 
                    stun=op.sd_stun
                )
            elseif op.sd_method=="bboptim" # Black-box optimizer:
                MultiGeomBB!(
                    shadow, 
                    c0, 
                    op.sd_maxiter
                )
            elseif op.sd_method=="geneig" # Full diagonalization:
                MultiGeomGenEig!(
                    shadow, 
                    op.sd_thresh, 
                    op.sd_eps,
                    lev=slev
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
            
            # After each decomp, truncate back to the max bond dim:
            for j=1:M
                orthogonalize!(sdata_c.psi_list[j], p_list[j])
                #truncate!(sdata_c.psi_list[j], maxdim=sdata_c.mparams.psi_maxdim)
                normalize!(sdata_c.psi_list[j])
            end
            
            # Copy the subspace mats:
            sdata_c.H_mat = shadow.H_mat
            sdata_c.S_mat = shadow.S_mat
            sdata_c.E = shadow.E
            sdata_c.C = shadow.C
            sdata_c.kappa = shadow.kappa
            
            fit_new = 0.0
            """
            if gp != nothing
                if gp.costfunc=="simple"
                    fit_new = BipartiteFitness(
                        sdata_c.ord_list, 
                        biparts, 
                        entrops, 
                        sdata_c.mparams.psi_maxdim, 
                        statevec=sdata_c.C[:,1], 
                        zeta=gp.zeta_list[1], 
                        xi=gp.xi
                    )
                else
                    fit_new = CompositeCostFunc(
                        sdata_c.ord_list, 
                        biparts, 
                        entrops, 
                        sdata_c.mparams.psi_maxdim, 
                        sdata_c.C[:,1], 
                        gp
                    )
                end
            end
            """

            f_new = op.theta[1]*sdata_c.E[1] + op.theta[2]*log(sdata_c.kappa) + op.theta[3]*log(maximum([fit_new+2.0, 1.0]))
            
            # Print some output
            if verbose
                print("Iter: $(l)/$(op.maxiter); loop: $(s)/$(op.numloop); E=$(round(sdata_c.E[1], digits=5)); kap=$(round(sdata_c.kappa, sigdigits=3)); fit=$(round(fit_new, digits=5)); cf=$(round(f_new, digits=5)); acc=$(n_accept)/$(l-1) ($(Int(round(100.0*n_accept/maximum([l-1, 1]), digits=0)))%)     \r")
                println("")
                flush(stdout)
            end
            
        end
        
        if verbose
            println("----------------------------------------------------")
        end
        
        # Accept move with some probability:
        """
        fit_new = 0.0
        if gp != nothing
            if gp.costfunc=="simple"
                fit_new = BipartiteFitness(
                    sdata_c.ord_list, 
                    biparts, 
                    entrops, 
                    sdata_c.mparams.psi_maxdim, 
                    statevec=sdata_c.C[:,1], 
                    zeta=gp.zeta_list[1], 
                    xi=gp.xi
                )
            else
                fit_new = CompositeCostFunc(
                    sdata_c.ord_list, 
                    biparts, 
                    entrops, 
                    sdata_c.mparams.psi_maxdim, 
                    sdata_c.C[:,1], 
                    gp
                )
            end
        end
        
        f_new = op.theta[1]*sdata_c.E[1] + op.theta[2]*log(sdata_c.kappa) + op.theta[3]*fit_new
        """
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
            fit=fit_new
        end
        
    end
    
    if verbose
        println("\nDone!\n")
    end
    
end