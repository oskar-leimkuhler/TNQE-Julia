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


function TwoSiteFullSweep!(
        sdata::SubspaceProperties,
        op::OptimParameters;
        verbose=false
    )
    
    N = sdata.chem_data.N_spt
    M = sdata.mparams.M
    
    if verbose
        println("\nGENERALIZED TWO-SITE SWEEP ALGORITHM:")
    end
    
    # Generate the permutation operators:
    GenPermOps!(sdata, no_rev=true, verbose=verbose)
    
    # The iteration loop:
    for l=1:op.maxiter
        
        # Noise at this iteration:
        lnoise = op.noise[minimum([l,end])]
        ldelta = op.delta[minimum([l,end])]
        
        for j=1:M
            orthogonalize!(sdata.psi_list[j], 1)
        end
        
        # Fill in the block_ref as we construct the "lock" tensors:
        block_ref = zeros(Int,(M,M))
        
        # Contract the "right" blocks:
        rH_list = Any[]
        rS_list = Any[]
        
        # Initialize the "left" blocks:
        lH = Any[]
        lS = Any[]
        
        for i=1:M, j=i:M
            
            if i==j
                
                push!(rH_list, CollectBlocks(
                        sdata.psi_list[i],
                        sdata.psi_list[i],
                        mpo1 = sdata.ham_list[i],
                        mpo2 = nothing,
                        inv=true
                    )
                )
                
                push!(rS_list, [nothing for p=1:N-1])
                
            else
                
                push!(rH_list, CollectBlocks(
                        sdata.psi_list[i],
                        sdata.psi_list[j],
                        mpo1 = sdata.ham_list[i],
                        mpo2 = sdata.perm_ops[i][j-i],
                        inv=true
                    )
                )
                
                push!(rS_list, CollectBlocks(
                        sdata.psi_list[i],
                        sdata.psi_list[j],
                        mpo1 = sdata.perm_ops[i][j-i],
                        mpo2 = nothing,
                        inv=true
                    )
                )
                
            end
            
            push!(lH, ITensor(1.0))
            push!(lS, ITensor(1.0))
            
            block_ref[i,j] = length(rH_list)
            block_ref[j,i] = length(rH_list)
            
        end

        for p=1:N-1

            psi_decomp = [[ITensor(1.0)] for i=1:M]

            H_blocks = []
            S_blocks = []
            
            # Generate the "key" tensors:
            for j=1:M
                T = sdata.psi_list[j][p] * sdata.psi_list[j][p+1]
                psi_decomp[j] = OneHotTensors(T)
            end

            # Generate the "lock" tensors:
            for i=1:M, j=i:M

                bind = block_ref[i,j]
                
                # Choose the correct "right" blocks:
                H_block = rH_list[bind][p]
                S_block = rS_list[bind][p]
                
                if i==j
                    
                    H_block *= sdata.ham_list[i][p+1]
                    H_block *= sdata.ham_list[i][p]
                    H_block *= lH[bind]

                else
                    
                    H_block *= setprime(sdata.perm_ops[i][j-i][p+1],2,plev=1)
                    H_block *= setprime(sdata.ham_list[i][p+1],2,plev=0)
                    H_block *= setprime(sdata.perm_ops[i][j-i][p],2,plev=1)
                    H_block *= setprime(sdata.ham_list[i][p],2,plev=0)
                    H_block *= lH[bind]
                    
                    S_block *= sdata.perm_ops[i][j-i][p+1]
                    S_block *= sdata.perm_ops[i][j-i][p]
                    S_block *= lS[bind]

                end

                push!(H_blocks, H_block)
                push!(S_blocks, S_block)

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
                thresh="inversion",
                eps=op.sd_eps
            )
            
            # Replace the tensors:
            for i=1:M

                i0, i1 = sum(M_list[1:i-1])+1, sum(M_list[1:i])

                t_vec = real.(C[i0:i1,1])
                normalize!(t_vec)
                t_vec += ldelta*normalize(randn(M_list[i])) # Random noise term
                normalize!(t_vec)

                # Construct the new tensor and plug it in:
                T_new = sum([t_vec[k]*psi_decomp[i][k] for k=1:M_list[i]])

                # Mix the new tensor with the old tensor:
                T_old = sdata.psi_list[i][p]*sdata.psi_list[i][p+1]
                T_new = (1.0-op.theta)*T_new + op.theta*T_old
                T_new *= 1.0/sqrt(scalar(T_new*dag(T_new)))

                # Generate the "noise" term:
                """
                pmpo = ITensors.ProjMPO(sdata.ham_list[i])
                ITensors.set_nsite!(pmpo,2)
                ITensors.position!(pmpo, sdata.psi_list[i], q)
                drho = 0.0*ITensors.noiseterm(pmpo,T_new,ortho_str)
                """

                # Replace the tensors of the MPS:
                spec = ITensors.replacebond!(
                    sdata.psi_list[i],
                    p,
                    T_new;
                    maxdim=sdata.mparams.psi_maxdim,
                    #eigen_perturbation=drho,
                    ortho="left",
                    normalize=true,
                    svd_alg="qr_iteration"
                )

            end
            
            # Update the "left" blocks:
            for i=1:M, j=i:M
                
                bind = block_ref[i,j]
                
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
                    
                    lH[bind] = UpdateBlock(
                        lH[bind], 
                        p, 
                        sdata.psi_list[i], 
                        sdata.psi_list[j], 
                        sdata.ham_list[i], 
                        sdata.perm_ops[i][j-i]
                    )
                    
                    lS[bind] = UpdateBlock(
                        lS[bind], 
                        p, 
                        sdata.psi_list[i], 
                        sdata.psi_list[j], 
                        sdata.perm_ops[i][j-i],
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
                print("bond: $(p)/$(N-1); ")
                print("E_min = $(round(sdata.E[1], digits=5)); ") 
                print("kappa = $(round(sdata.kappa, sigdigits=3))     \r")
                flush(stdout)
            end

        end
                
    end
    
    # Replace the perm ops:
    GenPermOps!(sdata, no_rev=false, verbose=false)
    
    if verbose
        println("\nDone!\n")
    end
    
end


function GeneralizedTwoSite_v2!(
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
        
        for j=1:M
            orthogonalize!(sdata.psi_list[j], 1)
        end

        for p=1:N-1

            psi_decomp = [[ITensor(1.0)] for i=1:M]

            # Fill in the block_ref as we construct the "lock" tensors:
            block_ref = zeros(Int,(M,M))

            H_blocks = []
            S_blocks = []
            
            combo = []
            
            # Generate the key tensors:
            for j=1:M
                T = sdata.psi_list[j][p] * sdata.psi_list[j][p+1]
                ct1, ci1 = TensorCombiner(commoninds(T, sdata.psi_list[j][p]))
                ct2, ci2 = TensorCombiner(commoninds(T, sdata.psi_list[j][p+1]))
                ct3, ci3 = TensorCombiner((ci1, ci2))
                push!(combo, [ct1, ct2, ct3])
                psi_decomp[j] = OneHotTensors(T*ct1*ct2*ct3)
            end

            # Generate the lock tensors:
            for i=1:M, j=i:M

                psi1 = [deepcopy(sdata.psi_list[i][q]) for q=1:N]
                psi2 = [deepcopy(sdata.psi_list[j][q]) for q=1:N]
                psi1[p], psi1[p+1] = dag(combo[i][1]), dag(combo[i][2])
                psi2[p], psi2[p+1] = dag(combo[j][1]), dag(combo[j][2])

                if i==j

                    push!(H_blocks, FullContract(
                            psi1, 
                            psi2, 
                            mpo1=sdata.ham_list[i],
                            combos=[dag(combo[i][3]), dag(combo[i][3])],
                            csites=[p+1,p+1]
                            )
                    )

                    push!(S_blocks, ITensor(1.0))

                else
                    pr = p+1
                    if sdata.rev_flag[i][j-i]
                        psi2 = reverse(psi2)
                        for q=1:N
                            replaceind!(psi2[q], sdata.sites[N-q+1], sdata.sites[q])
                        end
                        pr = N-p+1
                    end

                    push!(H_blocks, FullContract(
                            psi1, 
                            psi2, 
                            mpo1=sdata.ham_list[i], 
                            mpo2=sdata.perm_ops[i][j-i],
                            combos=[dag(combo[i][3]), dag(combo[j][3])],
                            csites=[p+1,pr]
                            )
                    )

                    push!(S_blocks, FullContract(
                            psi1, 
                            psi2, 
                            mpo1=sdata.perm_ops[i][j-i],
                            combos=[dag(combo[i][3]), dag(combo[j][3])],
                            csites=[p+1,pr]
                            )
                    )

                end

                block_ref[i,j] = length(H_blocks)
                block_ref[j,i] = length(H_blocks)

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
                thresh="inversion",
                eps=op.sd_eps
            )

            for i=1:M

                i0, i1 = sum(M_list[1:i-1])+1, sum(M_list[1:i])

                t_vec = real.(C[i0:i1,1])
                normalize!(t_vec)
                t_vec += ldelta*normalize(randn(M_list[i])) # Random noise term
                normalize!(t_vec)

                # Construct the new tensor and plug it in:
                T_new = sum([t_vec[k]*psi_decomp[i][k] for k=1:M_list[i]])

                # Mix the new tensor with the old tensor:
                T_old = sdata.psi_list[i][p]*sdata.psi_list[i][p+1]
                T_old *= combo[i][1]*combo[i][2]
                T_old *= combo[i][3]
                T_new = (1.0-op.theta)*T_new + op.theta*T_old
                T_new *= 1.0/sqrt(scalar(T_new*dag(T_new)))

                # Generate the "noise" term:
                """
                pmpo = ITensors.ProjMPO(sdata.ham_list[i])
                ITensors.set_nsite!(pmpo,2)
                ITensors.position!(pmpo, sdata.psi_list[i], q)
                drho = 0.0*ITensors.noiseterm(pmpo,T_new,ortho_str)
                """
                
                T_new *= dag(combo[i][3])
                T_new *= dag(combo[i][1]) * dag(combo[i][2])

                # Replace the tensors of the MPS:
                spec = ITensors.replacebond!(
                    sdata.psi_list[i],
                    p,
                    T_new;
                    maxdim=sdata.mparams.psi_maxdim,
                    #eigen_perturbation=drho,
                    ortho="left",
                    normalize=true,
                    svd_alg="qr_iteration"
                )

            end

            # Recompute H, S, E, C, kappa:
            GenSubspaceMats!(sdata)
            SolveGenEig!(sdata)

            # Print some output
            if verbose
                print("Iter: $(l)/$(op.maxiter); ")
                print("bond: $(p)/$(N-1); ")
                print("E_min = $(round(sdata.E[1], digits=5)); ") 
                print("kappa = $(round(sdata.kappa, sigdigits=3))     \r")
                flush(stdout)
            end

        end
                
    end
    
    if verbose
        println("\nDone!\n")
    end
    
    
    
end

function TensorCombiner(ids)
    
    c = combiner(ids...)
    
    ci = combinedind(c)
    
    ct = ITensor(Matrix(1.0I, (dim(ci), dim(ci))), ci', dag(ci)) * c
    
    noprime!(ct)
    
    return ct, ci
    
end

function TwoSiteBondPivoting!(
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
            jperm_ops, jrev_flag = Any[], Bool[]
            for i=1:M
                if i != j
                    P_ij, fl_ij = SlowPMPO(
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
            
            for s=1:op.numloop
                
                for i=1:M    
                    if jrev_flag[i]
                        orthogonalize!(sdata.psi_list[i], N)
                    else
                        orthogonalize!(sdata.psi_list[i], 1)
                    end
                end
                
                psi_list_c = deepcopy(sdata.psi_list)
                
                for i=1:M
                    if jrev_flag[i]
                        psi_list_c[i] = ReverseMPS(psi_list_c[i])
                    end
                end
                
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
                    
                    psi_decomp = [[ITensor(1.0)] for i=1:M]
                    
                    """
                    for i=1:M
                        if jrev_flag[i]
                            orthogonalize!(sdata.psi_list[i], N-p+1)
                        else
                            orthogonalize!(sdata.psi_list[i], p)
                        end    
                    end
                    
                    psi_list_c = deepcopy(sdata.psi_list)
                    
                    for i=1:M
                        if jrev_flag[i]
                            psi_list_c[i] = ReverseMPS(psi_list_c[i])
                        end
                    end
                    """
                    
                    # Do the bond pivot decompositions:
                    U_list = [ITensor(1.0) for i=1:M]
                    V_list = [ITensor(1.0) for i=1:M]
                    Sigma_list = [ITensor(1.0) for i=1:M]
                    
                    for i in setdiff(collect(1:M), j)
                        
                        U_list[i], Sigma_list[i], V_list[i] = SiteSVD(
                            psi_list_c[i],
                            p,
                            restrict_svals=op.restrict_svals
                        )
                        
                        psi_decomp[i] = OneHotTensors(Sigma_list[i], force_diag=true)
                        
                    end
                    
                    # Fill in the block_ref as we construct the "lock" tensors:
                    block_ref = zeros(Int,(M,M))
                    
                    H_blocks = []
                    S_blocks = []
                    
                    # Generate the bond-bond lock tensors:
                    for i1 in setdiff(1:M, j), i2 in setdiff(i1:M, j)
                        
                        psi1 = [deepcopy(sdata.psi_list[i1][p]) for p=1:N]
                        psi2 = [deepcopy(sdata.psi_list[i2][p]) for p=1:N]
                        ham1 = deepcopy(sdata.ham_list[i1])
                        ham2 = deepcopy(sdata.ham_list[i2])
                        
                        if jrev_flag[i1]
                            psi1[N-p+1] = replaceind(U_list[i1], sdata.sites[p], sdata.sites[N-p+1])
                            psi1[N-p] = replaceind(V_list[i1], sdata.sites[p+1], sdata.sites[N-p])
                        else
                            psi1[p] = deepcopy(U_list[i1])
                            psi1[p+1] = deepcopy(V_list[i1])
                        end
                        
                        if jrev_flag[i2]
                            psi2[N-p+1] = replaceind(U_list[i2], sdata.sites[p], sdata.sites[N-p+1])
                            psi2[N-p] = replaceind(V_list[i2], sdata.sites[p+1], sdata.sites[N-p])
                        else
                            psi2[p] = deepcopy(U_list[i2])
                            psi2[p+1] = deepcopy(V_list[i2])
                        end
                        
                        if i1==i2
                            
                            push!(H_blocks, FullContract(
                                    psi1, 
                                    psi2, 
                                    mpo1=ham1
                                    )
                            )
                            
                            push!(S_blocks, ITensor(1.0))
                            
                        else
                            
                            if sdata.rev_flag[i1][i2-i1]
                                
                                sites = siteinds(sdata.psi_list[i2])
                                psi2 = reverse(psi2)
                                
                                for q=1:N
                                    replaceind!(psi2[q], sites[N-q+1], sites[q])
                                end
                                #println(inds(psi2[1]))
                                
                                ham2 = ReverseMPO(ham2)
                            end
                            
                            #println(inds(psi1[1]))
                            #println(inds(psi2[1]))
                            #println(inds(ham2[1]))
                            #println(inds(sdata.perm_ops[i1][i2-i1][1]))
                            
                            push!(H_blocks, FullContract(
                                    psi1, 
                                    psi2, 
                                    mpo1=ham1, 
                                    mpo2=sdata.perm_ops[i1][i2-i1]
                                    )
                            )
                            
                            push!(S_blocks, FullContract(
                                    psi1, 
                                    psi2, 
                                    mpo1=sdata.perm_ops[i1][i2-i1]
                                    )
                            )
                            
                        end
                        
                        block_ref[i1,i2] = length(H_blocks)
                        block_ref[i2,i1] = length(H_blocks)
                        
                    end
                    
                    # The two-site-bond lock tensors:
                    # Select the correct "right" blocks:
                    bb_len = length(H_blocks)
                    H_blocks = vcat(H_blocks, [rH_list[i][p] for i=1:M])
                    S_blocks = vcat(S_blocks, [rS_list[i][p] for i=1:M])
                    
                    # Generate "lock" tensors by fast contraction
                    for i in setdiff(collect(1:M), j)
                        
                        yP1 = V_list[i] * setprime(jperm_ops[i][p+1],2,plev=1)
                        yP2 = U_list[i] * setprime(jperm_ops[i][p],2,plev=1)
                        H_blocks[i+bb_len] *= yP1
                        H_blocks[i+bb_len] *= setprime(sdata.ham_list[j][p+1],2,plev=0)
                        H_blocks[i+bb_len] *= yP2
                        H_blocks[i+bb_len] *= setprime(sdata.ham_list[j][p],2,plev=0)
                        H_blocks[i+bb_len] *= lH[i]
                        
                        S_blocks[i+bb_len] *= V_list[i] * jperm_ops[i][p+1]
                        S_blocks[i+bb_len] *= U_list[i] * jperm_ops[i][p]
                        S_blocks[i+bb_len] *= lS[i]
                        
                    end
                    
                    H_blocks[j+bb_len] *= sdata.ham_list[j][p+1]
                    H_blocks[j+bb_len] *= sdata.ham_list[j][p]
                    H_blocks[j+bb_len] *= lH[j]
                    
                    H_blocks = vcat(H_blocks, [swapprime(dag(H_block),0,1) for H_block in H_blocks[1+bb_len:end]])
                    S_blocks = vcat(S_blocks, [swapprime(dag(S_block),0,1) for S_block in S_blocks[1+bb_len:end]])
                    
                    T = sdata.psi_list[j][p] * sdata.psi_list[j][p+1]
                    psi_decomp[j] = OneHotTensors(T)
                    
                    block_ref[:,j] = collect((bb_len+M+1):(bb_len+2*M))
                    block_ref[j,:] = collect((bb_len+1):(bb_len+M))
                    
                    # Generate the subspace matrices:
                    H_full, S_full = ExpandSubspace(
                        sdata.H_mat,
                        sdata.S_mat,
                        psi_decomp,
                        H_blocks,
                        S_blocks,
                        block_ref
                    )
                    
                    M_j = length(psi_decomp[j])
                    
                    # Solve the generalized eigenvalue problem:
                    E, C, kappa = SolveGenEig(
                        H_full, 
                        S_full, 
                        thresh="inversion",
                        eps=op.sd_eps
                    )
                    
                    M_list = [length(psi_decomp[i]) for i=1:M]
                    
                    for i=1:M
                        
                        i0, i1 = sum(M_list[1:i-1])+1, sum(M_list[1:i])
                        
                        if jrev_flag[i]
                            q = N-p
                            ortho_str="right"
                        else
                            q = p
                            ortho_str="left"
                        end
                        
                        t_vec = real.(C[i0:i1,1])
                        normalize!(t_vec)
                        t_vec += ldelta*normalize(randn(M_list[i])) # Random noise term
                        normalize!(t_vec)
                        
                        # Construct the new tensor and plug it in:
                        if i==j
                            T_new = sum([t_vec[k]*psi_decomp[i][k] for k=1:M_list[i]])
                        elseif jrev_flag[i]
                            T_new = replaceind(U_list[i], sdata.sites[p], sdata.sites[N-p+1]) * sum([t_vec[k]*psi_decomp[i][k] for k=1:M_list[i]]) * replaceind(V_list[i], sdata.sites[p+1], sdata.sites[N-p])
                        else
                            T_new = U_list[i] * sum([t_vec[k]*psi_decomp[i][k] for k=1:M_list[i]]) * V_list[i]
                        end
                        
                        # Mix the new tensor with the old tensor:
                        T_new = (1.0-op.theta)*T_new + op.theta*sdata.psi_list[i][q]*sdata.psi_list[i][q+1]
                        T_new *= 1.0/sqrt(scalar(T_new*dag(T_new)))
                        
                        #display(T_new.tensor)

                        # Generate the "noise" term:
                        """
                        pmpo = ITensors.ProjMPO(sdata.ham_list[i])
                        ITensors.set_nsite!(pmpo,2)
                        ITensors.position!(pmpo, sdata.psi_list[i], q)
                        drho = 0.0*ITensors.noiseterm(pmpo,T_new,ortho_str)
                        """

                        # Replace the tensors of the MPS:
                        spec = ITensors.replacebond!(
                            sdata.psi_list[i],
                            q,
                            T_new;
                            maxdim=sdata.mparams.psi_maxdim,
                            #eigen_perturbation=drho,
                            ortho=ortho_str,
                            normalize=true,
                            svd_alg="qr_iteration"
                        )
                        
                        #println(siteinds(sdata.psi_list[i]))
                        
                        # Shift orthogonality center to site p:
                        #orthogonalize!(sdata.psi_list[j], p)
                        if jrev_flag[i]
                            psi_list_c[i] = ReverseMPS(sdata.psi_list[i])
                        else
                            psi_list_c[i] = sdata.psi_list[i]
                        end
                        
                    end
                    
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


# Return subspace matrices corresponding to the two-site decomposition
function TwoSiteDecomp(sd, j, p, H_blocks, S_blocks)
    
    # Generate the "one-hot" tensors:
    T = sd.psi_list[j][p] * sd.psi_list[j][p+1]
    T_inds = inds(T)
    T_dim = [dim(T_ind) for T_ind in T_inds]
    
    oht_list = ITensor[]
    
    if length(T_dim) == 3
        # Tensor indices a,b,c:
        for a=1:T_dim[1], b=1:T_dim[2], c=1:T_dim[3]

            oht = onehot(
                T_inds[1]=>a,
                T_inds[2]=>b,
                T_inds[3]=>c
            )

            if flux(oht)==flux(T)
                push!(oht_list, oht)
            end

        end
    else
        # Tensor indices a,b,c,d:
        for a=1:T_dim[1], b=1:T_dim[2], c=1:T_dim[3], d=1:T_dim[4]

            oht = onehot(
                T_inds[1]=>a,
                T_inds[2]=>b,
                T_inds[3]=>c,
                T_inds[4]=>d
            )

            if flux(oht)==flux(T)
                push!(oht_list, oht)
            end

        end
    end
    
    M_j = length(oht_list)
    
    M_tot = M_j + sd.mparams.M - 1
    
    H_full, S_full = zeros((M_tot,M_tot)), zeros((M_tot,M_tot))
    
    # Pre-existing blocks:
    H_full[1:(j-1),1:(j-1)] = sd.H_mat[1:(j-1), 1:(j-1)]
    S_full[1:(j-1),1:(j-1)] = sd.S_mat[1:(j-1), 1:(j-1)]
    
    H_full[(j+M_j):end,(j+M_j):end] = sd.H_mat[(j+1):end, (j+1):end]
    S_full[(j+M_j):end,(j+M_j):end] = sd.S_mat[(j+1):end, (j+1):end]
    
    H_full[1:(j-1),(j+M_j):end] = sd.H_mat[1:(j-1), (j+1):end]
    S_full[1:(j-1),(j+M_j):end] = sd.S_mat[1:(j-1), (j+1):end]
    
    H_full[(j+M_j):end,1:(j-1)] = sd.H_mat[(j+1):end, 1:(j-1)]
    S_full[(j+M_j):end,1:(j-1)] = sd.S_mat[(j+1):end, 1:(j-1)]
    
    # j-state diagonal block:
    for k=1:M_j, l=k:M_j
        H_full[(k+j-1),(l+j-1)] = scalar( oht_list[k] * H_blocks[j] * setprime(dag(oht_list[l]),1) )
        H_full[(l+j-1),(k+j-1)] = H_full[(k+j-1),(l+j-1)]
    end
    S_full[j:(j+M_j-1),j:(j+M_j-1)] = Matrix(1.0I, M_j, M_j)
    
    # Off-diagonal blocks:
    for i in setdiff(collect(1:sd.mparams.M), j)
        if i < j
            offset = 0
        else
            offset = M_j-1
        end
        
        for k=1:M_j
            H_full[(i+offset),(k+j-1)] = scalar( setprime(dag(oht_list[k]),1) * H_blocks[i] )
            H_full[(k+j-1),(i+offset)] = H_full[(i+offset),(k+j-1)]

            S_full[(i+offset),(k+j-1)] = scalar( setprime(dag(oht_list[k]),1) * S_blocks[i] )
            S_full[(k+j-1),(i+offset)] = S_full[(i+offset),(k+j-1)]
        end
        
    end
    
    return H_full, S_full, oht_list
    
end


# Generate "lock" tensors by fast contraction
function TwoSiteBlocks(sd, j, p0, p_ops, r_flg)
    
    M = sd.mparams.M
    N = sd.chem_data.N_spt
    
    H_diag = 1.0
    H_offdiag = Any[1.0 for i=1:M]
    S_offdiag = Any[1.0 for i=1:M]
    
    for i in setdiff(collect(1:M), j) 
        if r_flg[i]
            psi_i = ReverseMPS(sd.psi_list[i])
            ham_i = ReverseMPO(sd.ham_list[i])
        else
            psi_i = sd.psi_list[i]
            ham_i = sd.ham_list[i]
        end

        psi_j = sd.psi_list[j]
        ham_j = sd.ham_list[j]

        for p=1:N
            yP = psi_i[p] * setprime(p_ops[i][p],2,plev=1)
            Hx = setprime(ham_j[p],2,plev=0)
            if (p != p0) && (p != p0+1)
                Hx *= setprime(dag(psi_j[p]),1)
            end
            H_offdiag[i] *= yP
            H_offdiag[i] *= Hx

            if (p != p0) && (p != p0+1)
                S_offdiag[i] *= psi_i[p] * p_ops[i][p] * setprime(dag(psi_j[p]),1)
            else
                S_offdiag[i] *= psi_i[p] * p_ops[i][p]
            end

        end
    end
    
    for p=1:N
        if (p==p0) || (p==p0+1)
            H_diag *= sd.ham_list[j][p]
        else
            H_diag *= sd.psi_list[j][p] * sd.ham_list[j][p] * setprime(dag(sd.psi_list[j][p]),1)
        end
    end
    
    return H_diag, H_offdiag, S_offdiag
    
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
    
    randvec = [randn() for n=1:length(nz_inds)]
    
    normalize!(randvec)
    
    for (n,nz) in enumerate(nz_inds)
        array[nz] += delta*randvec[n]
    end
    
    tensor2 = ITensor(array, inds(tensor))
    
    tensor2 *= 1.0/sqrt(scalar(tensor2*dag(tensor2)))
    
    return tensor2
    
end


# Replace the site tensors of the matrix product states at p:
function ReplaceStates!(sdata, U_list, S_list, V_list, p_list, vec_list, lnoise, ldelta, op)
    
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
        temp_block = DeltaPerturb(temp_block, ldelta)
        
        # Generate the "noise" term:
        pmpo = ITensors.ProjMPO(sdata.ham_list[j])
        ITensors.set_nsite!(pmpo,2)
        ITensors.position!(pmpo, sdata.psi_list[j], p)
        drho = lnoise*ITensors.noiseterm(pmpo,temp_block,"left")
        
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


function RandomSiteDecomp!(
        sdata::SubspaceProperties,
        op::OptimParameters;
        verbose=false
    )
    
    N = sdata.chem_data.N_spt
    M = sdata.mparams.M
    
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
        ldelta = op.delta[minimum([n_accept+1,end])]
        
        #p_lists = [collect(1:N-1) for j=1:M]        
        p_lists = [randperm(N-1) for j=1:M]
        
        # Decomposition loop:
        for s=1:op.numloop
            
            # Choose sites and orthogonalize states
            
            if op.sweep
                if mod1(s,N-1)==1
                    # re-randomize the sweep lists:
                    p_lists = [randperm(N-1) for j=1:M]
                end
                # Sweep through the sites instead:
                p_list = [p_lists[j][mod1(s,N-1)] for j=1:M]
            else
                p_list = [rand(1:N-1) for j=1:M]
            end
            
            for j=1:M
                orthogonalize!(sdata.psi_list[j], p_list[j])
            end
            
            # Perform site decompositions at chosen sites
            U_list, sigma_list, V_list = [], [], []
            
            for j=1:M
                U,S,V = SiteSVD(
                    sdata.psi_list[j],
                    p_list[j], 
                    restrict_svals=op.restrict_svals
                )
                push!(U_list, U)
                push!(sigma_list, S)
                push!(V_list, V)
            end
            
            # Generate "lock" tensors by fast contraction
            H_diag, H_offdiag, S_offdiag = BlockContract(
                sdata, 
                U_list, 
                V_list, 
                p_list
            )
            
            # Compute matrix elements:
            shadow = SiteDecomposition(
                sdata, 
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
                sdata, 
                U_list, 
                sigma_list, 
                V_list, 
                p_list, 
                shadow.vec_list, 
                lnoise,
                ldelta,
                op
            )
            
            # Copy the subspace mats:
            sdata.H_mat = shadow.H_mat
            sdata.S_mat = shadow.S_mat
            sdata.E = shadow.E
            sdata.C = shadow.C
            sdata.kappa = shadow.kappa
            
            # Print some output
            if verbose
                print("Iter: $(l)/$(op.maxiter); ")
                print("loop: $(s)/$(op.numloop); ")
                print("E_min = $(round(sdata.E[1], digits=5)); ")
                print("kappa = $(round(sdata.kappa, sigdigits=3));       \r")
                flush(stdout)
            end
            
        end
        
    end
    
    if verbose
        println("\nDone!\n")
    end
    
end


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


function RecycleStates!(sd, op, rsweeps, l)
    
    M = sd.mparams.M
    
    j_set = circshift(collect(1:M), l-1)[1:op.rnum]
            
    for j in j_set
            _, sd.psi_list[j] = dmrg(
                sd.slh_list[j],
                sd.psi_list[j], 
                rsweeps, 
                outputlevel=0
            )

    end
    
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



"""
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
"""

"""

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

"""


"""
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
"""


"""

function SWAPHam!(
        sdata,
        j,
        p,
        swap_tensor
    )
    
    swap_tensor2 = deepcopy(swap_tensor)
    setprime!(swap_tensor2, 0, plev=1)
    
    temp_tensor = sdata.ham_list[j][p] * sdata.ham_list[j][p+1]
    
    temp_tensor *= dag(swap_tensor)
    
    setprime!(temp_tensor, 1, plev=2)
    
    temp_tensor *= swap_tensor2
    
    setprime!(temp_tensor, 0, plev=2)
    
    temp_inds = uniqueinds(sdata.ham_list[j][p],sdata.ham_list[j][p+1])
    
    U,S,V = svd(
        temp_tensor,
        temp_inds,
        cutoff=sdata.mparams.ham_tol,
        maxdim=sdata.mparams.ham_maxdim,
        mindim=1,
        alg="qr_iteration"
    )
    
    sdata.ham_list[j][p] = U
    sdata.ham_list[j][p+1] = S*V
    
    swaptags!(sdata.ham_list[j][p], "u", "l=$(p)")
    swaptags!(sdata.ham_list[j][p+1], "u", "l=$(p)")
    
end

function MergeSwap!(
        sdata,
        jperm_ops,
        swap_tensor,
        p;
        j1=1,
        j2=2
    )
    
    M = sdata.mparams.M
    
    p_pos = findall(x->x==p, sdata.ord_list[j1])[1]
    pp1_pos = findall(x->x==p+1, sdata.ord_list[j1])[1]

    sdata.ord_list[j1][p_pos] = p+1
    sdata.ord_list[j1][pp1_pos] = p
    
    # Re-generate Hamiltonians and PMPOs:
    GenHams!(sdata)
    GenPermOps!(sdata)
    
    for i in setdiff(collect(1:M), [j1])

        jperm_ops[1][i] = ApplySwapMPO(
            jperm_ops[1][i], 
            dag(swap_tensor), 
            p, 
            sdata.mparams.perm_tol, 
            sdata.mparams.perm_maxdim, 
            1
        )

    end

    SWAPHam!(
        sdata,
        j1,
        p,
        swap_tensor
    )
    
    GenPermOps!(sdata)
    
end

"""


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


"""
# Check if states are overlapping too much and discard if they are
function DiscardOverlapping(H_full, S_full, oht_list, j, tol)
    
    #println("\nBefore: ", cond(S_full))
    E, C, kappa = SolveGenEig(H_full, S_full, thresh="inversion", eps=1e-8)
    #println(E[1])
    
    M_list = [length(oht) for oht in oht_list]
    M = length(M_list)
    M_tot = sum(M_list)
            
    j0, j1 = sum(M_list[1:j-1])+1, sum(M_list[1:j])
    #println(M_list)

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
    
    #println("After: ", cond(S_full))
    E, C, kappa = SolveGenEig(H_full, S_full, thresh="inversion", eps=1e-8)
    #println(E[1])
    
    return H_full, S_full, oht_list
    
end
"""


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


function LockContract(
        psi1,
        psi2;
        mpo1=nothing,
        mpo2=nothing,
        combos=nothing,
        p=nothing,
        nsite=0
    )
    
    N = length(psi1)
    
    top_block = ITensor(1.0)
    
    bottom_block = ITensor(1.0)
    
    if nsite==0
        psites=[N+1,N]
    elseif nsite==1
        psites=[p,p]
    elseif nsite==2
        psites=[p,p+1]
    end
    
    for q=1:(psites[1]-1)
        
        top_block = UpdateBlock(top_block, q, psi1, psi2, mpo1, mpo2)
        
    end
    
    for q=N:(-1):(psites[2]+1)
        
        bottom_block = UpdateBlock(bottom_block, q, psi1, psi2, mpo1, mpo2)
        
    end
    
    if nsite==1
        
        top_block = UpdateBlock(top_block, p, psi1, psi2, mpo1, mpo2)
        
    elseif nsite==2
        
        top_block = UpdateBlock(top_block, p, psi1, psi2, mpo1, mpo2)
        top_block = UpdateBlock(top_block, p+1, psi1, psi2, mpo1, mpo2)
        
    end
    
    return top_block * bottom_block
    
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
        nsites=2,
        j1=1,
        j2=2
    )
    
    if nsites==2
        p1 = 3
    elseif nsites==1
        p1 = 2
    end
    
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
                p1=p1,
                inv=true
            )

            rS_ji = CollectBlocks(
                sdata.psi_list[j],
                psi_i,
                mpo1 = jperm_ops[j_ind][i],
                mpo2 = nothing,
                p1=p1,
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
            p1=p1,
            inv=true
        )

        if jj[1]==jj[2]
            if nsites==2
                rS_jj = [nothing for q=1:N-1]
            elseif nsites==1
                rS_jj = [nothing for q=1:N]
            end
            
        else
            rS_jj = CollectBlocks(
                sdata.psi_list[jj[1]],
                sdata.psi_list[jj[2]],
                mpo1 = jjperm_op,
                mpo2 = nothing,
                p1=p1,
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
        nsites=2,
        j1=1,
        j2=2
    )
    
    if nsites==2
        do_pp1 = true
    elseif nsites==1
        do_pp1 = false
    end
    
    # Select the correct "right" blocks:
    H_blocks = [rH_list[b][p] for b=1:length(rH_list)]
    S_blocks = [rS_list[b][p] for b=1:length(rS_list)]

    # Contract lock tensors

    # i-j blocks:
    for bind=1:length(state_ref)-3

        j, i = state_ref[bind]

        j_ind = findall(x->x==j, [j1,j2])[1]

        if i==j # Diagonal block:

            if do_pp1
                H_blocks[bind] *= sdata.ham_list[j][p+1]
            end
            H_blocks[bind] *= sdata.ham_list[j][p]
            H_blocks[bind] *= lH[bind]

        else # Off-diagonal block:

            if jrev_flag[j_ind][i]
                psi_i = ReverseMPS(sdata.psi_list[i])
                ham_i = ReverseMPO(sdata.ham_list[i])
            else
                psi_i = sdata.psi_list[i]
                ham_i = sdata.ham_list[i]
            end
            
            if do_pp1
                yP1 = psi_i[p+1] * setprime(ham_i[p+1],2,plev=1)
                H_blocks[bind] *= yP1
                H_blocks[bind] *= setprime(jperm_ops[j_ind][i][p+1],2,plev=0)
            end
            yP2 = psi_i[p] * setprime(ham_i[p],2,plev=1)
            H_blocks[bind] *= yP2
            H_blocks[bind] *= setprime(jperm_ops[j_ind][i][p],2,plev=0)
            H_blocks[bind] *= lH[bind]

            if do_pp1
                S_blocks[bind] *= psi_i[p+1] * jperm_ops[j_ind][i][p+1]
            end
            S_blocks[bind] *= psi_i[p] * jperm_ops[j_ind][i][p]
            S_blocks[bind] *= lS[bind]

        end

    end

    # j-j diagonal blocks:
    for j in [j1, j2]

        bind = block_ref[j,j]

        if do_pp1
            H_blocks[bind] *= sdata.ham_list[j][p+1]
        end
        H_blocks[bind] *= sdata.ham_list[j][p]
        H_blocks[bind] *= lH[bind]

    end

    # j-j off-diagonal block:
    bind = block_ref[j1,j2]
    
    if do_pp1
        H_blocks[bind] *= setprime(sdata.ham_list[j2][p+1],2,plev=1)
        H_blocks[bind] *= setprime(jperm_ops[1][j2][p+1],2,plev=0)
    end
    H_blocks[bind] *= setprime(sdata.ham_list[j2][p],2,plev=1)
    H_blocks[bind] *= setprime(jperm_ops[1][j2][p],2,plev=0)
    H_blocks[bind] *= lH[bind]

    if do_pp1
        S_blocks[bind] *= jperm_ops[1][j2][p+1]
    end
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


function TripleGenEig2(
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
    
    t_vecs = []
    
    for i=1:M
        i0, i1 = sum(M_list[1:i-1])+1, sum(M_list[1:i])
        t_vec = real.(C[i0:i1,1])
        if (NaN in t_vec) || (Inf in t_vec) || norm(t_vec) < 1e-16
            if i==j
                T_old = sdata.psi_list[j][p] * sdata.psi_list[j][p+1]
            else
                ipos = findall(x -> x==sdata.ord_list[j][p], sdata.ord_list[i])[1]
                T_old = sdata.psi_list[i][ipos]
            end
            t_vec = [scalar(T_old*dag(psi_decomp[i][k])) for k=1:M_list[i]]
        end
        push!(t_vecs, normalize(t_vec))
    end

    ps = [p, p+1]

    # Construct the new tensors:
    Tj = sum([t_vecs[j][k]*psi_decomp[j][k] for k=1:M_list[j]])

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

            t_vecs_p = []
    
            for i=1:M
                i0, i1 = sum(M_list_p[1:i-1])+1, sum(M_list_p[1:i])
                t_vec_p = real.(C_p[i0:i1,1])
                if (NaN in t_vec_p) || (Inf in t_vec_p) || norm(t_vec_p) < 1e-16
                    t_vec_p = deepcopy(t_vecs[i])
                end
                push!(t_vecs_p, normalize(t_vec_p))
            end

            Tj = sum([t_vecs_p[j][k]*psi_decomp_p[j][k] for k=1:M_list_p[j]])

            t_vecs[j] = [scalar(Tj*dag(psi_decomp[j][k])) for k=1:M_list[j]]
            
            for i in setdiff(1:M, j)
                t_vecs[i] = deepcopy(t_vecs_p[i])
            end
            
            E_min = E_p[1]

        end

    end
    
    return E_min, t_vecs
    
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
        normalize!(t_vec)
        if (NaN in t_vec) || (Inf in t_vec)
            T_old = sdata.psi_list[i][p] * sdata.psi_list[i][p+1]
            t_vec = [scalar(T_old*dag(psid)) for psid in psi_decomp[i]]
        end
        push!(t_vecs, t_vec)
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


# Determine whether or not inserting an FSWAP will be beneficial:
function TestFSWAP1(
        sdata,
        psi_decomp,
        H_full,
        S_full,
        op,
        p,
        j
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
    
    j0, j1 = sum(M_list[1:j-1])+1, sum(M_list[1:j])
    t_vec = real.(C[j0:j1,1])
    normalize!(t_vec)
    if (NaN in t_vec) || (Inf in t_vec)
        T_old = sdata.psi_list[j][p] * sdata.psi_list[j][p+1]
        t_vec = [scalar(T_old*dag(psid)) for psid in psi_decomp[j]]
    end

    Tj1 = sum([t_vec[k]*psi_decomp[j][k] for k=1:length(t_vec)])
    linds = commoninds(Tj1, sdata.psi_list[j][p])
    
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
    
    """
    if rand()[1] < ExpProb(fid_, fid, 1.0/op.swap_mult)
        do_swap = true
    end
    """
    
    if fid_ > fid
        do_swap = true
    end
    
    #println("\n $(fid)  $(fid_)  $(do_swap) \n")
    
    return do_swap
    
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
                        
                        if real(E[1]) >= op.sd_penalty*sdata.E[1]
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
                        
                        if real(E_min) >= op.sd_penalty*sdata.E[1]
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
                        print("E_min = $(round(sdata.E[1], digits=5)); ") 
                        print("Delta = $(round(sdata.E[1]-sdata.chem_data.e_fci+sdata.chem_data.e_nuc, digits=5)); ")
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
                    nsites=1,
                    j1=j1,
                    j2=j2
                )
                
                for p=1:N
                    
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
                        nsites=1,
                        j1=j1,
                        j2=j2
                    )
                    
                    # Generate OHT list + mat. els.
                    psi_decomp = [[ITensor(1.0)] for i=1:M]
                    
                    for j in [j1, j2]
                        psi_decomp[j] = OneHotTensors(sdata.psi_list[j][p])
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
                    
                    # Diagonalize, compare
                    
                    do_replace = true
                        
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
                    
                    # Update params, update top block
                    
                    if (NaN in t_vecs[1]) || (Inf in t_vecs[1]) || (NaN in t_vecs[2]) || (Inf in t_vecs[2])
                        do_replace = false
                    end
                    
                    # Check the truncation error is not too large:
                    if TruncError(t_vecs[1],j1,p,psi_decomp,sdata) > op.ttol || TruncError(t_vecs[2],j2,p,psi_decomp,sdata) > op.ttol
                        do_replace = false
                    end
                    
                    # Save the old state
                    sdata2 = copy(sdata)
                    
                    if do_replace
                        
                        # Do the replacement:
                        for (idx, i) in enumerate([j1, j2])

                            t_vec = t_vecs[idx]

                            T_old = sdata.psi_list[i][p]
    
                            if norm(t_vec) < 1e-16
                                t_vec = [scalar(T_old*dag(psi_decomp[i][k])) for k=1:M_list[i]]
                            end

                            t_vec += ldelta*normalize(randn(M_list[i])) # Random noise term
                            normalize!(t_vec)

                            # Construct the new tensor and plug it in:
                            T_new = sum([t_vec[k]*psi_decomp[i][k] for k=1:M_list[i]])

                            # Mix the new tensor with the old tensor:
                            T_new = (1.0-op.theta)*T_new + op.theta*T_old
                            T_new *= 1.0/sqrt(scalar(T_new*dag(T_new)))

                            # Replace the tensor of the MPS:
                            if !(NaN in T_new) && !(Inf in T_new) # Quick double-check
                                sdata.psi_list[i][p] = T_new
                            end
                            
                        end
                            
                    end
                    
                    # Recompute H, S, E, C, kappa:
                    GenSubspaceMats!(sdata)
                    
                    # Make sure the GenEig is not going to bork...
                    try 
                        E_test, C_test, kappa_test = SolveGenEig(
                            sdata.H_mat, 
                            sdata.S_mat, 
                            thresh=sdata.mparams.thresh, 
                            eps=sdata.mparams.eps
                        )
                    catch err
                        if isa(err, LoadError) || isa(err, ArgumentError)
                            #println("\n#############################\nInf or NaN detected!\n")
                            #display(sdata.H_mat)
                            #display(sdata.S_mat)
                            copyto!(sdata, sdata2) # Revert to the unaltered subspace
                            #println("\nReverting to previous subspace:\n")
                            #display(sdata.H_mat)
                            #display(sdata.S_mat)
                            #println("\n#############################\n\n\n")
                        end
                    end
                    
                    SolveGenEig!(sdata)
                    
                    if p != N
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
                    end
                    
                    # Print some output
                    if verbose
                        print("Iter: $(l)/$(op.maxiter); ")
                        print("pair: $(jc)/$(length(jpairs)); ")
                        print("sweep: $(s)/$(op.numloop); ")
                        print("bond: $(p)/$(N); ")
                        print("E_min = $(round(sdata.E[1], digits=5)); ") 
                        print("Delta = $(round(sdata.E[1]-sdata.chem_data.e_fci+sdata.chem_data.e_nuc, digits=5)); ")
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



function CoTwoSitePairSweep!(
        sdata::SubspaceProperties,
        op::OptimParameters;
        jpairs=nothing,
        verbose=false
    )
    
    N = sdata.chem_data.N_spt
    M = sdata.mparams.M
    
    if verbose
        println("\nCO-OPT. TWO-STATE, TWO-SITE SWEEP ALGORITHM:")
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
                        print("Delta = $(round(sdata.E[1]-sdata.chem_data.e_fci+sdata.chem_data.e_nuc, digits=5)); ")
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


function FSwapDisentangle!(
        sdata::SubspaceProperties,
        op::OptimParameters;
        jset=nothing,
        l=1,
        no_swap=false,
        verbose=false
    )
    
    if verbose
        println("\nFSWAP DISENTANGLER:")
    end
    
    M = sdata.mparams.M
    N = sdata.chem_data.N_spt
    
    if jset==nothing
        jset = collect(1:M)
    end
    
    # Noise at this iteration:
    lnoise = op.noise[minimum([l,end])]
    ldelta = op.delta[minimum([l,end])]
    
    # Repeat for each state in the jset:
    for j in jset
        
        swap_counter = 0
        
        # Rank bonds in terms of entanglement
        #rbonds = RankBonds(sdata.psi_list[j])
        rbonds = randperm(N-1)
        
        jperm_ops, jrev_flag = RelativePermOps(sdata, j)
        
        # Try to "disentangle" each bond in order from largest --> smallest entanglement
        for (b,p) in enumerate(rbonds)
            
            # Decompose state j at sites p, p+1
            # Decompose each state i != j at the same orbital as site p
            
            # Find the site position of the orbital given by sdata.ord_list[j][p] on each state
            orbs = [sdata.ord_list[j][p], sdata.ord_list[j][p+1]]
            
            pos_list = []
            for i=1:M
                if i==j
                    orb = orbs[1]
                else
                    orb = orbs[rand((1,2))]
                end
                push!(pos_list, findall(x -> x==orb, sdata.ord_list[j])[1])
            end
            
            for i=1:M
                orthogonalize!(sdata.psi_list[i], pos_list[i])
            end
        
            # Convert the MPS objects to lists, \\
            # ... replace with ITensor(1.0) at relevant positions
            psi_tlists = []
            
            for i=1:M
                push!(psi_tlists, [deepcopy(sdata.psi_list[i][q]) for q=1:N])
                psi_tlists[end][pos_list[i]] = ITensor(1.0)
            end
            
            # Include an additional ITensor(1.0) object for state j:
            psi_tlists[j][p+1] = ITensor(1.0)
            
            # Fill in the block_ref as we construct the "lock" tensors:
            block_ref = zeros(Int,(M,M))
            state_ref = []
            
            H_blocks = []
            S_blocks = []
            
            ## Compute the lock tensors:
            for i1 in setdiff(1:M, j), i2 in setdiff(i1:M, j)
                
                push!(state_ref, [i1, i2])
                block_ref[i1, i2] = length(state_ref)
                
                if i1==i2 # Diagonal block
                    push!(H_blocks, LockContract(psi_tlists[i1], psi_tlists[i1], mpo1=sdata.ham_list[i1], p=pos_list[i1], nsite=1))
                    push!(S_blocks, ITensor(1.0))
                else # Off-diagonal block
                    push!(H_blocks, LockContract(psi_tlists[i1], psi_tlists[i2], mpo1=sdata.perm_ops[i1][i2-i1], mpo2=sdata.ham_list[i2], p=pos_list[i1], nsite=1))
                    push!(S_blocks, LockContract(psi_tlists[i1], psi_tlists[i2], mpo1=sdata.perm_ops[i1][i2-i1], p=pos_list[i1], nsite=1))
                end
                
            end
            
            
            for i=1:M
                
                push!(state_ref, sort([i,j]))
                block_ref[i,j] = length(state_ref)
                block_ref[j,i] = length(state_ref)
                
                if i==j # Diagonal block
                    push!(H_blocks, LockContract(psi_tlists[j], psi_tlists[j], mpo1=sdata.ham_list[j], p=p, nsite=2))
                    push!(S_blocks, ITensor(1.0))
                else # Off-diagonal block
                    if jrev_flag[i]
                        push!(H_blocks, LockContract(psi_tlists[j], reverse(psi_tlists[i]), mpo1=jperm_ops[i], mpo2=sdata.ham_list[i], p=p, nsite=2))
                    push!(S_blocks, LockContract(psi_tlists[j], reverse(psi_tlists[i]), mpo1=jperm_ops[i], p=p, nsite=2))
                    else
                        push!(H_blocks, LockContract(psi_tlists[j], psi_tlists[i], mpo1=jperm_ops[i], mpo2=sdata.ham_list[i], p=p, nsite=2))
                    push!(S_blocks, LockContract(psi_tlists[j], psi_tlists[i], mpo1=jperm_ops[i], p=p, nsite=2))
                    end
                    
                    if i < j
                        H_blocks[end] = dag(swapprime(H_blocks[end], 0, 1))
                        S_blocks[end] = dag(swapprime(S_blocks[end], 0, 1))
                    end
                    
                end
                
            end
            
            ## Generate OHT decomp list:
            psi_decomp = [[ITensor(1.0)] for i=1:M]
            for i in setdiff(1:M, j)
                psi_decomp[i] = OneHotTensors(sdata.psi_list[i][pos_list[i]])
            end
            psi_decomp[j] = OneHotTensors(sdata.psi_list[j][p] * sdata.psi_list[j][p+1])
            
            M_list = [length(psi_decomp[i]) for i=1:M]
            
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
            if no_swap
                do_swap = false
            else
                do_swap = TestFSWAP1(
                    sdata,
                    psi_decomp,
                    H_full,
                    S_full,
                    op,
                    p,
                    j
                )
            end

            if do_swap

                H_blocks, S_blocks = BlockSwapMerge(
                    H_blocks,
                    S_blocks,
                    block_ref,
                    state_ref,
                    sdata,
                    j,
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
            
            # Update ordering and parameters
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
                for i=1:M
                    i0, i1 = sum(M_list[1:i-1])+1, sum(M_list[1:i])
                    t_vec = real.(C[i0:i1,1])
                    push!(t_vecs, normalize(t_vec))
                end

                if real(E[1]) >= op.sd_swap_penalty*sdata.E[1]
                    do_replace = false
                end

            elseif op.sd_method=="triple_geneig"

                # Solve the generalized eigenvalue problem:
                E_min, t_vecs = TripleGenEig2(
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

                if real(E_min) >= op.sd_swap_penalty*sdata.E[1]
                    do_replace = false
                end

            end

            # Update params, update top block

            if (NaN in t_vecs) || (Inf in t_vecs)
                do_replace = false
            end

            # Check the truncation error is not too large:
            if TruncError(t_vecs[j],j,p,psi_decomp,sdata) > op.ttol
                do_replace = false
            end

            if do_replace

                ReplaceBond!(
                    sdata,
                    psi_decomp,
                    j,
                    p,
                    t_vecs[j],
                    lnoise,
                    ldelta,
                    op.theta
                )
                
                for i in setdiff(1:M, j)
                    sdata.psi_list[i][pos_list[i]] = sum([t_vecs[i][k]*psi_decomp[i][k] for k=1:M_list[i]])
                end

                if do_swap # Update ordering j, re-generate perm ops

                    swap_counter += 1
                    sdata.ord_list[j][p:p+1] = reverse(sdata.ord_list[j][p:p+1])

                    # Re-generate Hamiltonians and PMPOs:
                    GenHams!(sdata)
                    GenPermOps!(sdata)
                    
                    jperm_ops, jrev_flag = RelativePermOps(sdata, j)

                end
                
                GenSubspaceMats!(sdata)
                SolveGenEig!(sdata)

            end
            
            # Recompute H, S, E, C, kappa:
            GenSubspaceMats!(sdata)
            SolveGenEig!(sdata)

            # Print some output
            if verbose
                print("State: $(j)/$(length(jset)); ")
                print("bond: $(b)/$(N-1); ")
                print("#swaps: $(swap_counter); ")
                print("E_min = $(round(sdata.E[1], digits=5)); ") 
                print("Delta = $(round(sdata.E[1]-sdata.chem_data.e_fci+sdata.chem_data.e_nuc, digits=5)); ")
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


function SeedSwaps!(
        sdata::SubspaceProperties,
        op::OptimParameters;
        plimit=true,
        penalty=nothing,
        jset=nothing,
        verbose=false
    )
    
    N = sdata.chem_data.N_spt
    M = sdata.mparams.M
    
    if jset==nothing
        jset=collect(1:M)
    end
    
    if penalty==nothing
        penalty = op.sd_swap_penalty
    end
    
    swap_prob = op.swap_mult
    
    for j in jset
        
        #num_swaps = Int(floor(swap_prob*randexp()[1]))
        num_swaps = rand(0:Int(floor(swap_prob)))
        
        # Apply these swaps randomly:
        for s=1:num_swaps
            
            if plimit
                sdata_copy = copy(sdata)
            end
            
            p = rand(1:N-1)
            sdata.ord_list[j][p:p+1]=reverse(sdata.ord_list[j][p:p+1])
            
            orthogonalize!(sdata.psi_list[j], p)
            
            fswap = BuildFermionicSwap(sdata.sites, p; dim=4);
            T = sdata.psi_list[j][p] * sdata.psi_list[j][p+1]
            T *= fswap
            
            noprime!(T)
            
            # Replace the tensors of the MPS:
            spec = ITensors.replacebond!(
                sdata.psi_list[j],
                p,
                T;
                maxdim=sdata.mparams.psi_maxdim,
                #eigen_perturbation=drho,
                ortho="left",
                normalize=true,
                svd_alg="qr_iteration"
                #min_blockdim=1
            )
            
            if plimit # Check to see if energy penalty limit has been exceeded
                
                GenHams!(sdata)
                GenPermOps!(sdata)
                GenSubspaceMats!(sdata)
                SolveGenEig!(sdata)

                if sdata.E[1] > sdata_copy.E[1]*penalty
                    copyto!(sdata, sdata_copy)
                end
                
            end
            
        end
        
    end
    
    GenHams!(sdata)
    GenPermOps!(sdata)
    GenSubspaceMats!(sdata)
    SolveGenEig!(sdata)
    
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
                
            elseif action[1]=="cotwositepair" # Do a two-site co-optimization sweep:
                
                CoTwoSitePairSweep!(
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
                
            elseif action[1]=="custom1" 
            
                ShuffleStates!(sdata)
    
                jpair_list = [[p,p+1] for p=1:sdata.mparams.M-1]

                for jpair in jpair_list
                    
                    SeedNoise!(
                        sdata,
                        0.02,
                        0.0,
                    )
                    
                    SeedSwaps!(
                        sdata,
                        op.swap_mult,
                        jset=[jpair[1]]
                    )

                    println(op.sd_penalty)
                    println(op.sd_swap_penalty)
                    
                    CoTwoSitePairSweep!(
                        sdata,
                        op,
                        jpairs=[jpair],
                        verbose=verbose
                    )
                end
                
                OneSitePairSweep!(
                    sdata,
                    op,
                    verbose=verbose
                )
                
            elseif action[1]=="custom2" 
                
                ShuffleStates!(sdata)
    
                jpair_list = [[p,p+1] for p=1:sdata.mparams.M-1]

                for jpair in jpair_list

                    SeedNoise!(
                        sdata,
                        0.02,
                        0.0,
                    )

                    TwoSitePairSweep!(
                        sdata,
                        op,
                        jpairs=[jpair],
                        verbose=verbose
                    )
                end
                
                OneSitePairSweep!(
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


function TruncError(c_j, j, p, psi_decomp, sd)
    
    T = sum([c_j[k]*psi_decomp[j][k] for k=1:length(c_j)])
    
    linds = commoninds(T, sd.psi_list[j][p])
    
    U,S,V = svd(T, linds, maxdim=sd.mparams.psi_maxdim)
    
    return norm(U*S*V - T)
    
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
    
    # Double-check there are no Infs or NaNs:
    if !(Inf in T_new) && !(NaN in T_new)
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
    
end


function RankBonds(
        psi_in::MPS
    )
    
    psi = deepcopy(psi_in)
    
    N = length(psi)
    
    vn_entropies = []
    
    for p=1:N-1
        
        orthogonalize!(psi, p)
        
        T = psi[p] * psi[p+1]
        
        linds = uniqueinds(psi[p],psi[p+1])
        
        U,S,V = svd(T, linds)
        
        sigma = diag(Array(S, inds(S)))
        
        vne = sum([-sigma[k]^2*log(sigma[k]^2) for k=1:length(sigma)])
        
        push!(vn_entropies, vne)
        
    end
    
    perm1 = sortperm(vn_entropies)
    
    rbonds = reverse(collect(1:N-1)[perm1])
    
    return rbonds
    
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



# A random bond optimization algorithm:
function TwoSiteRandomBond!(
        sdata::SubspaceProperties,
        op::OptimParameters;
        n_twos=2,
        jperm=nothing,
        no_swap=false,
        verbose=false,
        debug_output=false
    )
    
    M = sdata.mparams.M
    N = sdata.chem_data.N_spt
    
    # Default is to cycle through states one at a time:
    if jperm == nothing
        jperm = circshift(collect(1:M),1)
    end
    
    for l=1:op.maxiter
        
        ShuffleStates!(sdata, perm=jperm)
        
        j_set = collect(1:n_twos)
        
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
            
            rH = CollectBlocks(
                sdata,
                sdata.ham_mpo,
                i1, # primed index
                i2, # unprimed index
                inv=true
            )
            
            rS = CollectBlocks(
                sdata,
                sdata.eye_mpo,
                i1, # primed index
                i2, # unprimed index
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
            
            T_tensor_list = [sdata.psi_list[i][p] * sdata.psi_list[i][p+1] for i=1:M]
            
            # Compile the one-hot tensor list:
            oht_list = [[T_tensor_list[i]] for i=1:M]
            for j in j_set
                oht_list[j] = OneHotTensors(T_tensor_list[j])
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
                
                H_tens = lH[bind]
                
                H_tens *= setprime(sdata.perm_ops[i1][p],2,plev=0)
                H_tens *= dag(setprime(setprime(sdata.ham_mpo[p],2,plev=0,tags="Site"),3,plev=1,tags="Site"),tags="Site")
                H_tens *= dag(setprime(setprime(sdata.perm_ops[i2][p],3,plev=0,tags="Site"),0,plev=1),tags="Site")
                
                H_tens *= setprime(sdata.perm_ops[i1][p+1],2,plev=0)
                H_tens *= dag(setprime(setprime(sdata.ham_mpo[p+1],2,plev=0,tags="Site"),3,plev=1,tags="Site"),tags="Site")
                H_tens *= dag(setprime(setprime(sdata.perm_ops[i2][p+1],3,plev=0,tags="Site"),0,plev=1),tags="Site")
                
                H_tens *= rH_list[bind][p]
                
                H_array = zeros(M_list[i1],M_list[i2])
                for k1=1:M_list[i1], k2=1:M_list[i2]
                    H_array[k1,k2] = scalar(dag(setprime(oht_list[i1][k1],1)) * H_tens * oht_list[i2][k2])
                end

                H_full[i10:i11, i20:i21] = H_array
                H_full[i20:i21, i10:i11] = conj.(transpose(H_full[i10:i11, i20:i21]))
                
                S_tens = lS[bind]
                
                S_tens *= setprime(sdata.perm_ops[i1][p],2,plev=0)
                S_tens *= dag(setprime(setprime(sdata.eye_mpo[p],2,plev=0,tags="Site"),3,plev=1,tags="Site"), tags="Site")
                S_tens *= dag(setprime(setprime(sdata.perm_ops[i2][p],3,plev=0,tags="Site"),0,plev=1),tags="Site")
                
                S_tens *= setprime(sdata.perm_ops[i1][p+1],2,plev=0)
                S_tens *= dag(setprime(setprime(sdata.eye_mpo[p+1],2,plev=0,tags="Site"),3,plev=1,tags="Site"), tags="Site")
                S_tens *= dag(setprime(setprime(sdata.perm_ops[i2][p+1],3,plev=0,tags="Site"),0,plev=1),tags="Site")
                
                S_tens *= rS_list[bind][p]
                
                S_array = zeros(M_list[i1],M_list[i2])
                for k1=1:M_list[i1], k2=1:M_list[i2]
                    S_array[k1,k2] = scalar(dag(setprime(oht_list[i1][k1],1)) * S_tens * oht_list[i2][k2])
                end
                
                S_full[i10:i11, i20:i21] = S_array
                S_full[i20:i21, i10:i11] = conj.(transpose(S_full[i10:i11, i20:i21]))
                
            end
            
            H_all, S_all = deepcopy(H_full), deepcopy(S_full)
            oht_all = deepcopy(oht_list)
            M_list_all = deepcopy(M_list)
            
            # Make a copy to revert to at the end if the energy penalty is violated:
            sdata_copy = copy(sdata)
                
            #println("Before discarding: M_list = $(M_list), discards = nothing")
            
            # Discard any states with Infs and NaNs first:
            H_full, S_full, M_list, discards1 = DiscardOverlapping(H_full, S_full, M_list, criterion="InfNaN")
            
            for j in j_set
                oht_list[j] = OneHotTensors(T_tensor_list[j], discards=discards1[j])
            end
            
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
            
            for j in j_set
                oht_list[j] = OneHotTensors(T_tensor_list[j], discards=discards[j])
            end
            
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
                do_swaps[i] = TestFSWAP2(T, linds, sdata.mparams.psi_maxdim)
                #end

            end

            # Enforce no swapping?
            if no_swap
                do_swaps = [false for i=1:M]
            end

            # Encode new FSWAPs into full H, S matrices:
            for i1 in j_set

                i10 = sum(M_list_all[1:i1-1])+1
                i11 = sum(M_list_all[1:i1])

                if do_swaps[i1]

                    # Construct FSWAP matrix:
                    fswap = BuildFermionicSwap(sdata.sites, p; dim=4);
                    fswap_mat = zeros(M_list_all[i1], M_list_all[i1])
                    for k1=1:M_list_all[i1], k2=1:M_list_all[i1]
                        fswap_mat[k1,k2] = scalar(oht_all[i1][k1] * fswap * dag(setprime(oht_all[i1][k2],1,tags="Site")))
                    end

                    for i2=1:M

                        i20 = sum(M_list_all[1:i2-1])+1
                        i21 = sum(M_list_all[1:i2])

                        # Left-mult all subblocks in row i2:
                        H_all[i10:i11,i20:i21] = fswap_mat * H_all[i10:i11, i20:i21]
                        S_all[i10:i11,i20:i21] = fswap_mat * S_all[i10:i11, i20:i21]

                        # Right-mult all subblocks in col i2:
                        H_all[i20:i21,i10:i11] = H_all[i20:i21, i10:i11] * fswap_mat
                        S_all[i20:i21,i10:i11] = S_all[i20:i21, i10:i11] * fswap_mat

                    end

                end

            end
            
            # Discard any states with Infs and NaNs first:
            H_full, S_full, M_list, discards1 = DiscardOverlapping(H_all, S_all, M_list_all, criterion="InfNaN")
            
            for j in j_set
                oht_list[j] = OneHotTensors(T_tensor_list[j], discards=discards1[j])
            end
            
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
            
            for j in j_set
                oht_list[j] = OneHotTensors(T_tensor_list[j], discards=discards[j])
            end
            
            E, C, kappa = SolveGenEig(
                H_full,
                S_full,
                thresh=op.sd_thresh,
                eps=op.sd_eps
            )

            #println("\n$(E[1])\n")

            inds_list = []
            for j in j_set
                linds = commoninds(T_tensor_list[j], sdata.psi_list[j][p])
                rinds = commoninds(T_tensor_list[j], sdata.psi_list[j][p+1])
                push!(inds_list, [linds, rinds])
            end

            # Do TripleGenEig on all states to lower energy:
            T_list, V_list, E_new = TripleGenEigM(
                H_full,
                S_full,
                oht_list,
                M_list,
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
            
            #println("\n $(do_replace)")
            
            #println("$(real(E_new) < real(E[1]) + op.sd_etol)\n")
            
            if (real(E_new) < real(sdata.E[1]) + op.sd_etol) && do_replace
                
                # Update params, orderings
                for j in j_set
                    
                    if do_swaps[j]

                        swap_counter += 1
                        sdata.ord_list[j][p:p+1] = reverse(sdata.ord_list[j][p:p+1])
                        
                        # Locally permute PMPO:
                        PSWAP!(
                            sdata.perm_ops[j], 
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
                
            else # Revert to previous parameters:
                
                for j in j_set
                    
                    U,S,V = svd(
                        T_tensor_list[j], 
                        commoninds(sdata.psi_list[j][p], T_tensor_list[j]),
                        alg="qr_iteration",
                        maxdim=sdata.mparams.psi_maxdim
                    )
                    
                    V_list[j] = U
                    T_list[j] = S*V
                    
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
            for j in j_set
                
                sdata.psi_list[j][p] = V_list[j]
                sdata.psi_list[j][p+1] = T_list[j]
                
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

                lH[bind] = UpdateBlock(
                    sdata,
                    sdata.ham_mpo,
                    lH[bind], 
                    p, 
                    i1,
                    i2
                )
                
                lS[bind] = UpdateBlock(
                    sdata,
                    sdata.eye_mpo,
                    lS[bind], 
                    p, 
                    i1,
                    i2
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
            
        for j in j_set # Make sure these states are normalized:
            normalize!(sdata.psi_list[j])
        end
        
        # Recompute H, S, E, C, kappa:
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
        
        for p=N:(-1):2
            
            orthogonalize!(sdata.psi_list[j], p)
            
            T_old = sdata.psi_list[j][p] * sdata.psi_list[j][p-1]
            
            psi_decomp = OneHotTensors(T_old)
            
            t_vec = [scalar(psi_decomp[k] * dag(T_old)) for k=1:length(psi_decomp)]
            t_vec += delta*normalize(randn(length(t_vec)))
            normalize!(t_vec)
            
            T_new = sum([t_vec[k] * psi_decomp[k] for k=1:length(psi_decomp)])
            
            """
            # Generate the "noise" term:
            pmpo = ITensors.ProjMPO(sdata.ham_list[j])
            ITensors.set_nsite!(pmpo,2)
            ITensors.position!(pmpo, sdata.psi_list[j], p)
            drho = noise*ITensors.noiseterm(pmpo,T_new,"left")
            """
            
            # Replace the tensors of the MPS:
            spec = ITensors.replacebond!(
                sdata.psi_list[j],
                p-1,
                T_new;
                maxdim=sdata.mparams.psi_maxdim,
                #eigen_perturbation=drho,
                ortho="right",
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


function SeedNoise2!(
        sdata::SubspaceProperties,
        delta::Float64;
        jset=nothing
    )
    
    N = sdata.chem_data.N_spt
    N_el = sdata.chem_data.N_el
    M = sdata.mparams.M
    maxdim = sdata.mparams.psi_maxdim
    
    if jset==nothing
        jset=collect(1:M)
    end
    
    for j in jset
        
        hf_occ = [FillHF(sdata.ord_list[j][p], N_el) for p=1:N]
        
        delta_mps = randomMPS(sdata.sites, hf_occ,linkdims=maxdim)
        
        #_, delta_mps = dmrg(MPO(sdata.sites, "I"), [deepcopy(sdata.psi_list[j])], deepcopy(sdata.psi_list[j]), sdata.dflt_sweeps, weight=2.0, outputlevel=0)
        
        normalize!(delta_mps)
        
        sdata.psi_list[j] += delta*delta_mps
        
        truncate!(sdata.psi_list[j], maxdim=maxdim)
        
        normalize!(sdata.psi_list[j])
        
    end
    
end

function GenEigPermute!(
        sdata::SubspaceProperties,
        op::OptimParameters,
        j::Int, # State to permute
        ord::Vector{Int}; # New ordering
        no_rev=false,
        verbose=false
    )
    
    N = sdata.chem_data.N_spt
    M = sdata.mparams.M
    
    verbose && println("\nPermuting state $(j); from $(sdata.ord_list[j]) -> $(ord):")
    
    # Move state j to the front of the list:
    perm = vcat([j], setdiff(1:M, j))
    ShuffleStates!(sdata, no_rev=no_rev, perm=perm)
    
    # Generate the FSWAP site positions:
    swap_pos = BubbleSort(sdata.ord_list[1], ord)
    
    # Iterate over site positions:
    for (l,p) in enumerate(swap_pos)
        
        # Orthogonalize to site p:
        orthogonalize!(sdata.psi_list[1], p)
        
        # Generate one-hot decomposition:
        T = sdata.psi_list[1][p] * sdata.psi_list[1][p+1]
        oht_list = [[ITensor(1.0)] for i=1:M]
        oht_list[1] = OneHotTensors(T)
        
        M_list = length.(oht_list)
        
        M_tot = sum(M_list)
        
        H_full = zeros(M_tot,M_tot)
        
        S_full = zeros(M_tot,M_tot)
        
        # Generate subspace matrices:
        for i=1:M
            
            if i==1
                
                psi_j, psi_i, hmpo1, hmpo2, smpo1, smpo2 = BlockSetup(sdata, 1, 1, p, p, 2, 2)
                
                H_tens = FullContract(psi_j, psi_i, mpo1=hmpo1, mpo2=hmpo2)
                
                H_array = zeros(M_list[1],M_list[1])
                for k1=1:M_list[1], k2=1:M_list[1]
                    H_array[k1,k2] = scalar(dag(setprime(oht_list[1][k1],1)) * H_tens * oht_list[1][k2])
                end
                
                H_full[1:M_list[1],1:M_list[1]] = H_array
                
                S_full[1:M_list[1],1:M_list[1]] = Matrix(I, (M_list[1], M_list[1]))
                
            else
                
                psi_j, psi_i, hmpo1, hmpo2, smpo1, smpo2 = BlockSetup(sdata, 1, i, p, p, 2, 0)
                
                H_tens = FullContract(psi_j, psi_i, mpo1=hmpo1, mpo2=hmpo2)
                
                H_array = zeros(M_list[1])
                for k=1:M_list[1]
                    H_array[k] = scalar(dag(setprime(oht_list[1][k],1)) * H_tens)
                end
                
                H_full[M_list[1]+i-1,1:M_list[1]] = H_array
                H_full[1:M_list[1],M_list[1]+i-1] = transpose(H_array)
                
                S_tens = FullContract(psi_j, psi_i, mpo1=smpo1, mpo2=smpo2)
                
                S_array = zeros(M_list[1])
                for k=1:M_list[1]
                    S_array[k] = scalar(dag(setprime(oht_list[1][k],1)) * S_tens)
                end
                
                S_full[M_list[1]+i-1,1:M_list[1]] = S_array
                S_full[1:M_list[1],M_list[1]+i-1] = transpose(S_array)
                
            end
            
        end
        
        H_full[M_list[1]+1:end,M_list[1]+1:end] = sdata.H_mat[2:end,2:end]
        S_full[M_list[1]+1:end,M_list[1]+1:end] = sdata.S_mat[2:end,2:end]
        
        # Encode new FSWAP:
        H_full, S_full = FSWAPModify(
            H_full, 
            S_full, 
            M_list, 
            oht_list, 
            sdata.sites, 
            vcat([2],[0 for i=2:M]), 
            [[p,p+1] for i=1:M], 
            vcat([true],[false for i=2:M])
        )
        
        # Drop overlapping states to control condition number:
        H_disc, S_disc, M_disc, oht_disc = DiscardOverlapping(
            H_full, 
            S_full, 
            M_list, 
            oht_list, 
            tol=op.sd_dtol,
            kappa_max=1e10
        )
        
        E, C, kappa = SolveGenEig(
            H_disc,
            S_disc,
            thresh=op.sd_thresh,
            eps=op.sd_eps
        )

        # Generate the initial T_i:
        T_init = []

        for i=1:M

            i0 = sum(M_disc[1:i-1])+1
            i1 = sum(M_disc[1:i])

            t_vec = normalize(C[i0:i1,1])
            T_i = sum([t_vec[k] * oht_disc[i][k] for k=1:M_disc[i]])

            push!(T_init, T_i)

        end
        
        linds = commoninds(T, sdata.psi_list[1][p])
        rinds = commoninds(T, sdata.psi_list[1][p+1])
        
        # Do TripleGenEig to lower energy:
        T_list, V_list, E_new = TripleGenEigM(
            H_full,
            S_full,
            oht_list,
            T_init,
            M_list,
            op,
            vcat([2], [0 for i=2:M]),
            [[linds, rinds]],
            sdata.mparams.psi_maxdim,
            nrep=20
        )
        
        # Replace site tensors and update site ordering:
        sdata.psi_list[1][p] = V_list[1]
        sdata.psi_list[1][p+1] = T_list[1]
        
        sdata.ord_list[1][p:p+1] = reverse(sdata.ord_list[1][p:p+1])
        
        GenHams!(sdata, j_set=[1])
        GenPermOps!(sdata, no_rev=no_rev)
        
        GenSubspaceMats!(sdata)
        SolveGenEig!(sdata)
        
        #println("\n$(E[1]) $(E_new) $(sdata.E[1])\n")
        
        # Print some output
        if verbose
            print("Swap: $(l)/$(length(swap_pos)); ")
            print("E_min = $(round(sdata.E[1], digits=5)); ") 
            print("Delta = $(round(sdata.E[1]-sdata.chem_data.e_fci+sdata.chem_data.e_nuc, digits=5)); ")
            print("kappa_full = $(round(cond(S_full), sigdigits=3)); ")
            print("kappa = $(round(sdata.kappa, sigdigits=3))     \r")
            flush(stdout)
        end
    
    end
    
    # Move state back to position j:
    ShuffleStates!(sdata, perm=invperm(perm))
    
    verbose && println("\nDone!\n")
    
end


"""
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
"""

