# Re-optimize the basis states to lower the total energy:
function ReOptimize(
        chemical_data,
        psi_list, 
        ham_list,
        ord_list;
        ovlp_weight=1e6,
        coeff_weight=0.01,
        loops=1,
        steps=4,
        sweeps=nothing,
        H_init=nothing, 
        S_init=nothing,
        perm_tol=1e-12,
        perm_maxdim=5000,
        state_tol=1e-12,
        state_maxdim=5000,
        thresh="none",
        eps=1e-9,
        spinpair=false,
        locdim=4,
        verbose=false,
        dmrg_opl=0
    )
    
    M = length(psi_list)
    
    sites = siteinds(psi_list[1])
    
    if H_init==nothing
        H_mat, S_mat = GenSubspaceMats(
            chemical_data, 
            sites, 
            ord_list, 
            psi_list,
            ham_list,
            perm_tol=perm_tol, 
            perm_maxdim=perm_maxdim, 
            spinpair=spinpair, 
            spatial=true, 
            singleham=false,
            verbose=false
        )
    else
        H_mat = H_init
        S_mat = S_init
    end
    
    E, C, kappa = SolveGenEig(
        H_mat, 
        S_mat, 
        thresh=thresh, 
        eps=eps
    )
    
    if verbose
        println("Initial energy: ", E[1])
        println("Condition number: ", kappa)
    end
    
    for l=1:loops
        
        if verbose
            println("\nRun $l of $(loops):\n")
        end
        
        for (j,psi) in enumerate(psi_list[2:end])
            
            j+=1
            
            hpsi_list = MPS[]
            opsi_list = MPS[]
            
            if verbose
                println("Optimizing state $(j-1) of $(M-1)...")
            end
            
            #println("j=$j")
            println(setdiff(collect(1:M), [j]))
            # Generate the MPS-MPO cost function states:
            for i in setdiff(collect(1:M), [j])
                
                #println("i=$i")
                
                perm_psi_i = Permute(
                    psi_list[i],
                    sites, 
                    ord_list[i], 
                    ord_list[j], 
                    tol=perm_tol, 
                    maxdim=perm_maxdim, 
                    spinpair=spinpair, 
                    locdim=locdim
                )
                
                hpsi_i = apply(
                    ham_list[j],
                    perm_psi_i,
                    tol=state_tol,
                    maxdim=state_maxdim
                )
                
                push!(opsi_list, perm_psi_i)
                
                push!(hpsi_list, hpsi_i)
                
            end
            
            for s=1:steps
            
                # Compile the coefficient weights:
                coeffs = [coeff_weight*2.0*C[j,1]*C[i,1] for i in setdiff(collect(1:M), [j])]

                # Re-optimize the state with the custom cost function:
                _, psi_j = dmrg_custom(
                    ham_list[j], 
                    hpsi_list, 
                    coeffs,
                    opsi_list,
                    ovlp_weight,
                    psi_list[j], 
                    sweeps, 
                    outputlevel=dmrg_opl
                )

                # Replace the existing state:
                psi_list[j] = deepcopy(psi_j)

                # Re-diagonalize:
                H_mat, S_mat = GenSubspaceMats(
                    chemical_data, 
                    sites, 
                    ord_list, 
                    psi_list,
                    ham_list,
                    perm_tol=perm_tol, 
                    perm_maxdim=perm_maxdim, 
                    spinpair=spinpair, 
                    spatial=true, 
                    singleham=false,
                    verbose=verbose
                )

                E, C, kappa = SolveGenEig(
                    H_mat, 
                    S_mat, 
                    thresh=thresh, 
                    eps=eps
                )
            
            end
            
            if verbose
                println("Done!")
            end
            
        end
        
        if verbose
            println("Run $l complete!")
            println("Ground state energy: $(E[1])")
            println("Condition number: ", kappa)
        end
        
    end
    
    return E, C, psi_list
    
end


# Re-optimize the basis states:
function ReOptimize(
        chemical_data,
        psi_list, 
        ham_list,
        ord_list,
        sweeps;
        ovlp_weight=1e6,
        loops=1,
        perm_tol=1e-12,
        perm_maxdim=5000,
        spinpair=false,
        locdim=4,
        verbose=false,
        dmrg_opl=0
    )
    
    M = length(psi_list)
    
    sites = siteinds(psi_list[1])
    
    for l=1:loops
        
        if verbose
            println("\nRun $l of $(loops):")
        end
        
        for (j,psi) in enumerate(psi_list[2:end])
            
            j+=1
            
            perm_psi_list = MPS[]
            
            if verbose && dmrg_opl==1
                println("\nOptimizing state $(j-1) of $(M-1)...")
            end
            
            # Generate the permuted states:
            for i in setdiff(collect(1:M), [j])
                
                perm_psi_i = Permute(
                    psi_list[i],
                    sites, 
                    ord_list[i], 
                    ord_list[j], 
                    tol=perm_tol, 
                    maxdim=perm_maxdim, 
                    spinpair=spinpair, 
                    locdim=locdim
                )
                
                push!(perm_psi_list, perm_psi_i)
                
            end

            # Re-optimize the state with the custom cost function:
            _, psi_j = dmrg_c(
                ham_list[j], 
                perm_psi_list,
                psi_list[j], 
                sweeps, 
                outputlevel=dmrg_opl,
                weight=ovlp_weight
            )

            # Replace the existing state:
            psi_list[j] = deepcopy(psi_j)
            
            if verbose && dmrg_opl==1
                println("Done!")
            elseif verbose
                print("Progress: [$(j-1)/$(M-1)] \r")
                flush(stdout)
            end
            
        end
        
        if verbose
            println("\nRun $l complete!")
        end
        
    end
    
    return psi_list
    
end


function ReOptimizeState(
        chemical_data,
        H_init,
        S_init,
        psi_j_in,
        ham_j,
        j_index,
        opsi_list,
        hpsi_list,
        sweeps;
        ovlp_weight=1e1,
        coeff_weight=1.0,
        steps=4,
        thresh="none",
        eps=1e-9,
        verbose=false,
        dmrg_opl=0
    )
    
    M = size(H_init, 1)
    
    i_indices = setdiff(collect(1:M), [j_index])
    
    H_mat = deepcopy(H_init)
    S_mat = deepcopy(S_init)
    
    psi_j = deepcopy(psi_j_in)
    
    # Solve the gen eig problem:
    E, C, kappa = SolveGenEig(
        H_mat, 
        S_mat, 
        thresh=thresh, 
        eps=eps
    )
    
    for s=1:steps
        
        # Generate the optimization coefficients:
        coeffs = [coeff_weight*2.0*C[j_index,1]*C[i_index,1] for i_index in i_indices]
        
        # Re-optimize the state with the custom cost function:
        _, psi_j = dmrg_custom(
            ham_j, 
            hpsi_list,
            coeffs,
            opsi_list,
            ovlp_weight,
            psi_j, 
            sweeps, 
            outputlevel=dmrg_opl
        )
        
        # Re-compile the H and S matrices:
        
        H_mat[j_index,j_index] = inner(psi_j', ham_j, psi_j)
        
        for i=1:(M-1)
            H_mat[i_indices[i],j_index] = inner(psi_j, hpsi_list[i])
            H_mat[j_index,i_indices[i]] = H_mat[i_indices[i],j_index]
            
            S_mat[i_indices[i],j_index] = inner(psi_j, opsi_list[i])
            S_mat[j_index,i_indices[i]] = S_mat[i_indices[i],j_index]
        end
        
        # Solve the gen eig problem:
        E, C, kappa = SolveGenEig(
            H_mat, 
            S_mat, 
            thresh=thresh, 
            eps=eps
        )
        
        if verbose
            println("SSOpt energy: $(E[1])")
        end
        
    end
    
    return psi_j
    
end


function Orthogonalize(
        psi_list,
        inds,
        ord_list,
        ham_list,
        sweeps;
        ovlp_weight=1.0,
        perm_tol=1e-12,
        perm_maxdim=5000,
        dmrg_opl=0
    )
    
    M = length(psi_list)
    
    for i in inds
        
        j_inds = setdiff(collect(1:M), [i])
        
        opsi_list = MPS[]
        
        for j in j_inds
            
            perm_psi_j = Permute(
                psi_list[j],
                sites,
                ord_list[j],
                ord_list[i],
                tol=perm_tol, 
                maxdim=perm_maxdim, 
                spinpair=false, 
                locdim=4   
            )
            
            push!(opsi_list, perm_psi_j)
            
        end
        
        _, psi_i = dmrg_c(
            ham_list[i], 
            opsi_list,
            psi_list[i], 
            sweeps, 
            outputlevel=dmrg_opl,
            weight=ovlp_weight
        )
        
        psi_list[i] = psi_i
        
    end
    
    return psi_list
    
end


# Re-optimize the basis states to lower the total energy:
function ReOptimizeCustom(
        chemical_data,
        psi_list_in, 
        ham_list,
        ord_list,
        sweeps;
        ovlp_weight=1e1,
        ortho_weight=1e1,
        coeff_weight=1.0,
        loops=1,
        steps=4,
        H_init=nothing, 
        S_init=nothing,
        perm_tol=1e-12,
        perm_maxdim=5000,
        state_tol=1e-12,
        state_maxdim=5000,
        thresh="none",
        eps=1e-9,
        spinpair=false,
        locdim=4,
        verbose=false,
        dmrg_opl=0
    )
    
    psi_list = deepcopy(psi_list_in)
    
    M = length(psi_list)
    
    sites = siteinds(psi_list[1])
    
    if H_init==nothing
        H_mat, S_mat = GenSubspaceMats(
            chemical_data, 
            sites, 
            ord_list, 
            psi_list,
            ham_list,
            perm_tol=perm_tol, 
            perm_maxdim=perm_maxdim, 
            spinpair=spinpair, 
            spatial=true, 
            singleham=false,
            verbose=false
        )
    else
        H_mat = H_init
        S_mat = S_init
    end
    
    E, C, kappa = SolveGenEig(
        H_mat, 
        S_mat, 
        thresh=thresh, 
        eps=eps
    )
    
    if verbose
        println("Initial energy: ", E[1])
        println("Condition number: ", kappa)
    end
    
    for l=1:loops
        
        if verbose
            println("\nRun $l of $(loops):\n")
        end
        
        new_psi_list = MPS[]
        
        for (j,psi) in enumerate(psi_list[2:end])
            
            j+=1
            
            psi_j = psi_list[j]
            
            hpsi_list = MPS[]
            opsi_list = MPS[]
            
            if verbose
                println("Optimizing state $(j-1) of $(M-1)...")
            end
            
            # Generate the MPS-MPO cost function states:
            for i in setdiff(collect(1:M), [j])
                
                perm_psi_i = Permute(
                    psi_list[i],
                    sites, 
                    ord_list[i], 
                    ord_list[j], 
                    tol=perm_tol, 
                    maxdim=perm_maxdim, 
                    spinpair=spinpair, 
                    locdim=locdim
                )
                
                hpsi_i = apply(
                    ham_list[j],
                    perm_psi_i,
                    tol=state_tol,
                    maxdim=state_maxdim
                )
                
                push!(opsi_list, perm_psi_i)
                
                push!(hpsi_list, hpsi_i)
                
            end
            
            psi_j = ReOptimizeState(
                chemical_data,
                H_mat,
                S_mat,
                psi_list[j],
                ham_list[j],
                j,
                opsi_list,
                hpsi_list,
                sweeps;
                ovlp_weight=ovlp_weight,
                coeff_weight=coeff_weight,
                steps=steps,
                thresh=thresh,
                eps=eps,
                verbose=verbose,
                dmrg_opl=dmrg_opl
            )
            
            if verbose
                println("Done!")
            end
            
            psi_list[j] = psi_j
            
            if verbose
                println("Re-orthogonalizing: ")
            end
            
            # Orthogonalize all but the just-optimized state and the anchor:
            ortho_inds = setdiff(collect(1:M), [1,j])
            
            psi_list = Orthogonalize(
                psi_list,
                ortho_inds,
                ord_list,
                ham_list,
                sweeps;
                ovlp_weight=ortho_weight,
                perm_tol=1e-12,
                perm_maxdim=5000,
                dmrg_opl=dmrg_opl
            )
            
            if verbose
                println("Done!")
            end
            
            # Re-diagonalize:
            H_mat, S_mat = GenSubspaceMats(
                chemical_data, 
                sites, 
                ord_list, 
                psi_list,
                ham_list,
                perm_tol=perm_tol, 
                perm_maxdim=perm_maxdim, 
                spinpair=spinpair, 
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
                println("Energy: $(E[1])")
            end
            
            push!(new_psi_list,psi_j)
            
        end
        
        #psi_list[2:end] = new_psi_list
        
        # Re-diagonalize:
        H_mat, S_mat = GenSubspaceMats(
            chemical_data, 
            sites, 
            ord_list, 
            psi_list,
            ham_list,
            perm_tol=perm_tol, 
            perm_maxdim=perm_maxdim, 
            spinpair=spinpair, 
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
            println("Run $l complete!")
            println("Ground state energy: $(E[1])")
            println("Condition number: ", kappa)
        end
        
    end
    
    return E, C, psi_list
    
end


# Re-optimize the basis states to lower the total energy:
function ReOptimizeAnchor(
        chemical_data,
        psi_list_in, 
        ham_list,
        ord_list,
        sweeps;
        ovlp_weight=1e1,
        ortho_weight=1e1,
        coeff_weight=1.0,
        loops=1,
        steps=4,
        H_init=nothing, 
        S_init=nothing,
        perm_tol=1e-12,
        perm_maxdim=5000,
        state_tol=1e-12,
        state_maxdim=5000,
        thresh="none",
        eps=1e-9,
        spinpair=false,
        locdim=4,
        verbose=false,
        dmrg_opl=0
    )
    
    psi_list = deepcopy(psi_list_in)
    
    M = length(psi_list)
    
    sites = siteinds(psi_list[1])
    
    if H_init==nothing
        H_mat, S_mat = GenSubspaceMats(
            chemical_data, 
            sites, 
            ord_list, 
            psi_list,
            ham_list,
            perm_tol=perm_tol, 
            perm_maxdim=perm_maxdim, 
            spinpair=spinpair, 
            spatial=true, 
            singleham=false,
            verbose=false
        )
    else
        H_mat = H_init
        S_mat = S_init
    end
    
    E, C, kappa = SolveGenEig(
        H_mat, 
        S_mat, 
        thresh=thresh, 
        eps=eps
    )
    
    if verbose
        println("Initial energy: ", E[1])
        println("Condition number: ", kappa)
    end
    
    for l=1:loops
        
        if verbose
            println("\nRun $l of $(loops):\n")
        end

        hpsi_list = MPS[]
        opsi_list = MPS[]

        if verbose
            println("Optimizing anchor state: ")
        end

        # Generate the MPS-MPO cost function states:
        for i=2:M

            perm_psi_i = Permute(
                psi_list[i],
                sites, 
                ord_list[i], 
                ord_list[1], 
                tol=perm_tol, 
                maxdim=perm_maxdim, 
                spinpair=spinpair, 
                locdim=locdim
            )

            hpsi_i = apply(
                ham_list[1],
                perm_psi_i,
                tol=state_tol,
                maxdim=state_maxdim
            )

            push!(opsi_list, perm_psi_i)

            push!(hpsi_list, hpsi_i)

        end

        anchor_state = ReOptimizeState(
            chemical_data,
            H_mat,
            S_mat,
            psi_list[1],
            ham_list[1],
            1,
            opsi_list,
            hpsi_list,
            sweeps;
            ovlp_weight=ovlp_weight,
            coeff_weight=coeff_weight,
            steps=steps,
            thresh=thresh,
            eps=eps,
            verbose=verbose,
            dmrg_opl=dmrg_opl
        )

        if verbose
            println("Done!")
        end

        psi_list[1] = anchor_state

        if verbose
            println("Re-orthogonalizing: ")
        end

        # Orthogonalize all but the just-optimized state and the anchor:
        ortho_inds = collect(2:M)

        psi_list = Orthogonalize(
            psi_list,
            ortho_inds,
            ord_list,
            ham_list,
            sweeps;
            ovlp_weight=ortho_weight,
            perm_tol=1e-12,
            perm_maxdim=5000,
            dmrg_opl=dmrg_opl
        )

        if verbose
            println("Done!")
        end
        
        # Re-diagonalize:
        H_mat, S_mat = GenSubspaceMats(
            chemical_data, 
            sites, 
            ord_list, 
            psi_list,
            ham_list,
            perm_tol=perm_tol, 
            perm_maxdim=perm_maxdim, 
            spinpair=spinpair, 
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
            println("Run $l complete!")
            println("Ground state energy: $(E[1])")
            println("Condition number: ", kappa)
        end
        
    end
    
    return E, C, psi_list
    
end


function ReplaceTensors(psi, p_index, tensor; tol=1e-12, maxdim=31)
    
    temp_inds = uniqueinds(psi[p_index],psi[p_index+1])
    
    U,S,V = svd(tensor,temp_inds,cutoff=tol,maxdim=maxdim,alg="divide_and_conquer")
    
    psi[p_index] = U
    psi[p_index+1] = S*V
    
    tnorm = norm(psi)
    
    psi[p_index+1] *= 1.0/tnorm
    
    return psi, tnorm
    
end


function ReplaceMatEls(H_mat, S_mat, psi_list, hpsi_list, j_index)
    
    M = length(psi_list)
    
    for i_index in setdiff(collect(1:M), [j_index])
        H_mat[i_index,j_index] = inner(psi_list[j_index], hpsi_list[i_index])
        H_mat[j_index,i_index] = H_mat[i_index,j_index]
        
        S_mat[i_index,j_index] = inner(psi_list[j_index], psi_list[i_index])
        S_mat[j_index,i_index] = S_mat[i_index,j_index]
    end
    
    return H_mat, S_mat
    
end


function UnlockMatEls(H_mat, S_mat, key_tensor, H_locks, S_locks, j_index, p_index)
    
    M = size(H_mat,1)
    
    for i_index in setdiff(collect(1:M), [j_index])
        
        H_mat[i_index,j_index] = scalar(dag(key_tensor) * H_locks[i_index])
        H_mat[j_index,i_index] = H_mat[i_index,j_index]
        
        S_mat[i_index,j_index] = scalar(dag(key_tensor) * S_locks[i_index])
        S_mat[j_index,i_index] = S_mat[i_index,j_index]
    end
    
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


function CostFunc(
        vars,
        sfac,
        t_inds,
        nz_inds,
        H_init, 
        S_init,
        #psi_in,
        #hpsi_list,
        #h_diag,
        H_locks,
        S_locks,
        D_locks,
        j_index,
        p_index;
        thresh="none",
        eps=1e-10,
        tol=1e-12,
        maxdim=31,
        verbose=false
    )
    
    H_mat = deepcopy(H_init)
    S_mat = deepcopy(S_init)
    
    key_tensor = ConstructTensor(vars, t_inds, nz_inds)
    
    # Re-scale the tensor:
    key_tensor *= 1.0/sfac
    
    #psi_list = deepcopy(psi_in)
            
    #psi_list[j_index], tnorm = ReplaceTensors(psi_list[j_index], p_index, x_tensor, tol=tol, maxdim=maxdim)
    
    # Re-normalize the tensor:
    S_jj_tensor = key_tensor
    for D_lock_S in D_locks[2:end]
        S_jj_tensor *= D_lock_S
    end
    S_jj_tensor *= dag(key_tensor)
    
    tnorm = sqrt(scalar(S_jj_tensor))
    
    key_tensor *= 1.0/tnorm

    #H_mat, S_mat = ReplaceMatEls(H_mat, S_mat, psi_list, hpsi_list, j_index)
    
    #H_mat[j_index,j_index] = scalar(dag(x_tensor) * ITensors.product(h_diag, x_tensor))
    
    #display(H_mat - H_init)
    
    H_mat, S_mat = UnlockMatEls(H_mat, S_mat, key_tensor, H_locks, S_locks, j_index, p_index)
    
    S_jj_tensor = key_tensor
    for D_lock_S in D_locks[2:end]
        S_jj_tensor *= D_lock_S
    end
    S_jj_tensor *= dag(key_tensor)
    S_mat[j_index,j_index] = scalar(S_jj_tensor)
    
    H_jj_tensor = D_locks[1] * key_tensor
    noprime!(H_jj_tensor)
    H_jj_tensor *= dag(key_tensor)
    
    H_mat[j_index, j_index] = scalar(H_jj_tensor)
    
    display(H_mat-H_init)
    display(S_mat-S_init)
    
    E, C, kappa = SolveGenEig(
        H_mat, 
        S_mat, 
        thresh=thresh, 
        eps=eps
    )
    
    return E[1]
    
end


function PartialTensors(psi2::MPS, ham::MPO, psi1::MPS; lr="left")
    
    N_sites = length(psi1)
    
    if lr=="left"
        ind = collect(1:N_sites)
        orthogonalize!(psi1, N_sites)
        orthogonalize!(psi2, N_sites)
        orthogonalize!(ham, N_sites)
    elseif lr=="right"
        ind = reverse(collect(1:N_sites))
        orthogonalize!(psi1, 1)
        orthogonalize!(psi2, 1)
        orthogonalize!(ham, 1)
    else
        throw(ArgumentError(lr, "Must be either left or right"))
    end
    
    H_tensor_list = ITensor[]
    S_tensor_list = ITensor[]
    
    H_tensor = psi1[ind[1]] 
    H_tensor *= ham[ind[1]]
    noprime!(H_tensor)
    H_tensor *= dag(psi2[ind[1]])
    
    S_tensor = psi1[ind[1]]
    S_tensor *= dag(psi2[ind[1]])
    
    push!(H_tensor_list, H_tensor)
    push!(S_tensor_list, S_tensor)
    
    for p=2:N_sites-2
        
        H_tensor *= psi1[ind[p]]
        H_tensor *= ham[ind[p]] 
        noprime!(H_tensor)
        H_tensor *= dag(psi2[ind[p]])
        
        S_tensor *= psi1[ind[p]]
        S_tensor *= dag(psi2[ind[p]])
        
        push!(H_tensor_list, H_tensor)
        push!(S_tensor_list, S_tensor)
        
    end
    
    return H_tensor_list, S_tensor_list
    
end


function UpdateBlocks(H_left_blocks, S_left_blocks, psi_list, ham, j_index, p_index)
    
    if p_index==1
        # Set blocks for first time:
        H_left_blocks = [psi_list[j_index][1] * ham[1] * psi_list[i_index][1] for i_index=1:M]
        S_left_blocks = [psi_list[j_index][1] * psi_list[i_index][1] for i_index=1:M]
    else
        # Update blocks after run:
        for i_index=1:M
            H_left_blocks[i_index] *= psi_list[j_index][p_index]
            H_left_blocks[i_index] *= ham[p_index]
            noprime!(H_left_blocks[i_index])
            H_left_blocks[i_index] *= dag(psi_list[i_index][p_index])
            
            S_left_blocks[i_index] *= psi_list[j_index][p_index]
            S_left_blocks[i_index] *= dag(psi_list[i_index][p_index])
        end
    end
    
    return H_left_blocks, S_left_blocks
    
end


function GetLockTensors(
        psi_list, 
        ham,
        H_left_blocks,
        S_left_blocks,
        H_right_blocks,
        S_right_blocks,
        j_index, 
        p_index
    )
    
    
    M = length(psi_list)
    N_sites = length(psi_list[j_index])
    
    H_locks = []
    S_locks = []
    
    orthogonalize!(ham, p_index)
    ham_twosite = ham[p_index] * ham[p_index+1]
    
    for i_index=1:M
        
        orthogonalize!(psi_list[i_index], p_index)
        
        S_lock_tensor = psi_list[i_index][p_index] * psi_list[i_index][p_index+1]
        H_lock_tensor = S_lock_tensor * ham_twosite
        noprime!(H_lock_tensor)
        
        if p_index!=1
            H_lock_tensor *= H_left_blocks[i_index]
            S_lock_tensor *= S_left_blocks[i_index]
        end
        
        if p_index!=N_sites-1
            q_index = length(H_right_blocks[i_index])+1-p_index
            H_lock_tensor *= H_right_blocks[i_index][q_index]
            S_lock_tensor *= S_right_blocks[i_index][q_index]
        end
        
        push!(H_locks, H_lock_tensor)
        push!(S_locks, S_lock_tensor)

    end
    
    return H_locks, S_locks
    
end


function GetDiagLockH(ham, j_index, p_index, H_left_blocks, H_right_blocks)
    
    N_sites = length(psi)
    
    D_lock = ham[p_index] * ham[p_index+1]
    
    if p_index!=1
        D_lock *= H_left_blocks[j_index][p_index]
    end
    if p_index!=N_sites-1
        q_index = length(H_right_blocks[j_index])+1-p_index
        D_lock *= H_right_blocks[j_index][q_index]
    end
    
    return D_lock
    
end


function NLOptimizeMPS(
        psi_in,
        ham_list,
        ord_list,
        j_index,
        H_mat,
        S_mat;
        sweeps=1,
        tol=1e-12,
        maxdim=31,
        perm_tol = 1e-12,
        perm_maxdim = 5000,
        thresh="none",
        eps=1e-10,
        verbose=false
    )
    
    psi_list = deepcopy(psi_in)
    
    M = length(psi_list)
    
    N_sites = length(psi_list[j_index])
    
    for i in setdiff(collect(1:M), [j_index])
        psi_list[i] = Permute(
            psi_list[i],
            sites, 
            ord_list[i], 
            ord_list[j_index], 
            tol=perm_tol, 
            maxdim=perm_maxdim, 
            spinpair=false, 
            locdim=4
        )   
        
    end
    
    hpsi_list = [apply(ham_list[j_index], psi_list[i_index]) for i_index=1:M]
    
    h_diag = ProjMPO(ham_list[j_index])
    ITensors.set_nsite!(h_diag, 2)
    
    for s=1:sweeps
        
        if verbose
            println("Sweep $(s) of $(sweeps): ")
        end
            
        # Pre-compute all of the right-blocks for each matrix element:
        H_right_blocks = []
        S_right_blocks = []
        for i_index=1:M
            H_right_blocks_i, S_right_blocks_i = PartialTensors(
                psi_list[j_index], 
                ham_list[j_index], 
                psi_list[i_index]; 
                lr="right"
            )
            push!(H_right_blocks, H_right_blocks_i)
            push!(S_right_blocks, S_right_blocks_i)
        end
        
        
        # Initialize the left-blocks for each matrix element:
        H_left_blocks = ITensor[]
        S_left_blocks = ITensor[]
        
        for p_index = 1:(N_sites-1)
            
            for i_index=1:M
                orthogonalize!(psi_list[i_index], p_index)
            end
                
            # Generate the lock tensors:
            H_locks, S_locks = GetLockTensors(
                psi_list,
                ham_list[j_index],
                H_left_blocks,
                S_left_blocks,
                H_right_blocks,
                S_right_blocks,
                j_index, 
                p_index
            )
            
            D_lock_H = GetDiagLockH(ham_list[j_index], j_index, p_index, H_left_blocks, H_right_blocks)
            
            D_locks = [D_lock_H]
            
            
            if p_index!=1
                push!(D_locks, S_left_blocks[j_index][p_index])
            end
            if p_index!=N_sites-1
                q_index = length(H_right_blocks[j_index])+1-p_index
                push!(D_locks, S_right_blocks[j_index][q_index])
            end
            
            #ITensors.position!(h_diag, psi_list[j_index], p_index)
            
            init_tensor = psi_list[j_index][p_index] * psi_list[j_index][p_index+1]

            nz_inds, t_inds = nzInds(init_tensor)
            
            n_nz = length(nz_inds)
            
            init_array = Array(init_tensor, t_inds)
            
            x0 = [init_tensor[nz_ind] for nz_ind in nz_inds]
            
            sfac = 0.999/maximum(abs.(x0))
            
            x0 .*= sfac
            
            # Cost function for the black-box optimizer:
            f(x) = CostFunc(
                x,
                sfac,
                t_inds,
                nz_inds,
                H_mat, 
                S_mat,
                #psi_list,
                #hpsi_list,
                #h_diag,
                H_locks,
                S_locks,
                D_locks,
                j_index,
                p_index,
                tol=tol,
                maxdim=maxdim,
                thresh=thresh,
                eps=eps,
                verbose=false
            )
            
            #println("TEST:", f(x0))
            
            res = bboptimize(f, x0; NumDimensions=n_nz, SearchRange = (-1.0, 1.0), MaxFuncEvals=100, TraceMode=:silent)
            
            x_opt = best_candidate(res)
            
            e_opt = best_fitness(res)
            
            if e_opt <= f(x0)
            
                x_tensor = ConstructTensor(x_opt, t_inds, nz_inds)
                
                x_tensor *= 1.0/sfac
            
                psi_list[j_index], tnorm = ReplaceTensors(psi_list[j_index], p_index, x_tensor, tol=tol, maxdim=maxdim)
                
                x_tensor *= 1.0/tnorm
                
                hpsi_list[j_index] = apply(ham_list[j_index], psi_list[j_index])

                H_mat, S_mat = ReplaceMatEls(H_mat, S_mat, psi_list, hpsi_list, j_index)
                
                #H_mat[j_index,j_index] = scalar(dag(x_tensor) * ITensors.product(h_diag, x_tensor))
                
                H_mat[j_index,j_index] = inner(psi_list[j_index],hpsi_list[j_index])
                
            end
            
            H_left_blocks, S_left_blocks = UpdateBlocks(
                H_left_blocks, 
                S_left_blocks,
                psi_list, 
                ham, 
                j_index, 
                p_index
            )
            
            if verbose
                print("Progress: [$(p_index)/$(N_sites-1)]\r")
                flush(stdout)
            end
            
        end
        
        E, C, kappa = SolveGenEig(
            H_mat, 
            S_mat, 
            thresh=thresh, 
            eps=eps
        )
        
        if verbose
            println("\nSweep $s complete!")
            println("Ground state energy: $(E[1])")
            println("Condition number: $(kappa)")
        end
        
    end
    
    return psi_list[j_index], H_mat, S_mat
    
end



# Carry out a full MPS-MPO DMRG procedure with a custom cost function:
function RunDMRGCustom(
        chemical_data, 
        sites, ord, H, 
        sweeps,
        cost_states,
        cost_weights; 
        spinpair=false, 
        spatial=false, 
        linkdims=31
    )
    
    if spinpair==true
        spnord = Spatial2SpinOrd(ord)
    else
        spnord = ord
    end
    
    if spatial==true
        hf_occ = [FillHF(spnord[i], chemical_data.N_el) for i=1:chemical_data.N_spt]
    else
        hf_occ = [FillHF(spnord[i], chemical_data.N_el) for i=1:chemical_data.N]
    end

    psi0 = randomMPS(sites, hf_occ, linkdims=linkdims)
    
    e_dmrg, psi = dmrg_custom(H, cost_states, cost_weights, psi0, sweeps, outputlevel=0, weight=weight)
    
    return psi, e_dmrg
    
end


# Off-diagonal overlap matrix element:
function BlockContract(psi1::MPS, psi2::MPS, p::Int)
    
    n = length(psi1)
    
    top_block = psi1[n] * psi2[n]
    for q=n-1:(-1):p+2
        top_block *= psi1[q] * psi2[q]
    end
    
    bottom_block = psi1[1] * psi2[1]
    for q=2:p-1
        bottom_block *= psi1[q] * psi2[q]
    end
    
    middle_block = psi2[p] * psi2[p+1]
    
    if p==1
        block_tensor = top_block * middle_block
    elseif p==n-1
        block_tensor = bottom_block * middle_block
    else
        block_tensor = top_block * middle_block * bottom_block
    end
    
    return block_tensor
    
end


# Off-diagonal Hamiltonian matrix element:
function BlockContract(psi1::MPS, ham::MPO, psi2::MPS, p::Int)
    
    n = length(psi1)
    
    top_block = psi1[n] * ham[n] * setprime(psi2[n],1)
    for q=n-1:(-1):p+2
        top_block *= psi1[q] * ham[q] * setprime(psi2[q],1)
    end
    
    bottom_block = psi1[1] * ham[1] * setprime(psi2[1],1)
    for q=2:p-1
        bottom_block *= psi1[q] * ham[q] * setprime(psi2[q],1)
    end
    
    middle_block = setprime(psi2[p],1) * setprime(psi2[p+1],1) * ham[p] * ham[p+1]
    
    if p==1
        block_tensor = top_block * middle_block
    elseif p==n-1
        block_tensor = bottom_block * middle_block
    else
        block_tensor = top_block * middle_block * bottom_block
    end
    
    return block_tensor
    
end


# Diagonal Hamiltonian element:
function BlockContract(psi1::MPS, ham::MPO, p::Int)
    
    n = length(psi1)
    
    top_block = psi1[n] * ham[n] * setprime(psi1[n],1)
    for q=n-1:(-1):p+2
        top_block *= psi1[q] * ham[q] * setprime(psi1[q],1)
    end
    
    bottom_block = psi1[1] * ham[1] * setprime(psi1[1],1)
    for q=2:p-1
        bottom_block *= psi1[q] * ham[q] * setprime(psi1[q],1)
    end
    
    middle_block = ham[p] * ham[p+1]
    
    if p==1
        block_tensor = top_block * middle_block
    elseif p==n-1
        block_tensor = bottom_block * middle_block
    else
        block_tensor = top_block * middle_block * bottom_block
    end
    
    return block_tensor
    
end
