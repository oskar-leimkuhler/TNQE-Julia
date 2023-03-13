function ScreenOrderings(chemical_data, sites, sweeps, M; maxiter=10, M_new=1, annealing=false, alpha=1.0, verbose=false)
    
    # Generate an initial set of orderings:
    ord_list = [randperm(chemical_data.N_spt) for i=1:M]
    
    # Generate a set of states:
    psi_list, ham_list = GenStates(
        chemical_data, 
        sites, 
        ord_list, 
        sweeps, 
        ovlp_opt=false,
        perm_tol=1E-10, 
        perm_maxdim=1000, 
        ham_tol=1E-8, 
        ham_maxdim=1000, 
        spinpair=false, 
        spatial=true, 
        singleham=false,
        verbose=false
    )
    
    H_mat, S_mat = GenSubspaceMats(
        chemical_data, 
        sites, 
        ord_list, 
        psi_list,
        ham_list,
        perm_tol=1E-10, 
        perm_maxdim=1000, 
        spinpair=false, 
        spatial=true, 
        singleham=false,
        verbose=true
    );
    
    # Solve the GenEig problem:
    E, C, kappa = SolveGenEig(H_mat, S_mat, thresh="inversion", eps=1e-8)
    e_old = E[1]
    kappa_old = kappa
    
    
    if verbose
        println("Screening states (batch size = $(M_new))")
    end
    
    # Repeat this loop until maxiter or convergence:
    for l=1:maxiter
        
        # Get a new random ordering:
        new_ord = [randperm(chemical_data.N_spt) for i=1:M_new]
        
        # Generate the new state:
        new_psi, new_ham = GenStates(
            chemical_data, 
            sites, 
            new_ord, 
            sweeps, 
            ovlp_opt=false,
            perm_tol=1E-10, 
            perm_maxdim=1000, 
            ham_tol=1E-8, 
            ham_maxdim=1000, 
            spinpair=false, 
            spatial=true, 
            singleham=false,
            verbose=false
        )
        
        # Extend the ordering lists:
        ext_psi_list = deepcopy(psi_list)
        ext_ham_list = deepcopy(ham_list)
        ext_ord_list = deepcopy(ord_list)
        
        # Expand the subspace matrices:
        H_exp, S_exp = deepcopy(H_mat), deepcopy(S_mat)
        
        for i=1:M_new
            H_exp, S_exp = ExpandSubspaceMats(H_exp, S_exp, new_psi[i], new_ham[i], new_ord[i], ext_psi_list, ext_ord_list)
            push!(ext_psi_list, new_psi[i])
            push!(ext_ham_list, new_ham[i])
            push!(ext_ord_list, new_ord[i])
        end
        
        # Test all of the reduced subspace matrices to see which performs best:
        ids_list = collect(combinations(1:size(H_exp,1),M_new))
        e_gnd_list = []
        
        for ids in ids_list
            
            H_red, S_red = deepcopy(H_exp), deepcopy(S_exp)
            
            for k=1:length(ids)
                kmod = sum([Int(ids[k]>ids[m]) for m=1:k-1])
                jpop = ids[k]-kmod
                H_red, S_red = ShrinkSubspaceMats(H_red,S_red,jpop)
            end
            
            # Solve the GenEig problem:
            E, C, kappa = SolveGenEig(H_red, S_red, thresh="inversion", eps=1e-8)
            e_gnd = minimum(filter(!isnan,real.(E)))
            push!(e_gnd_list, e_gnd)
            
        end
        
        # Find the minimum index:
        min_e_gnd, min_idx = findmin(e_gnd_list[1:end-1])
        
        # Accept with some probability:
        if annealing
            beta = l*alpha
            P = ExpProb(e_old, min_e_gnd, beta)
            accept = (rand(Float64)<P)
        else
            P = StepProb(e_old, min_e_gnd)
            accept = Bool(P)
        end
        
        min_ids = ids_list[min_idx]
        red_ids = setdiff(collect(1:length(ext_ord_list)),min_ids)
        
        if accept
            # Modify the states and subspace matrices:
            psi_list = ext_psi_list[red_ids]
            ham_list = ext_ham_list[red_ids]
            ord_list = ext_ord_list[red_ids]

            H_red, S_red = deepcopy(H_exp), deepcopy(S_exp)

            for k=1:length(min_ids)
                kmod = sum([Int(min_ids[k]>min_ids[m]) for m=1:k-1])
                jpop = min_ids[k]-kmod
                H_red, S_red = ShrinkSubspaceMats(H_red,S_red,jpop)
            end
            
            H_mat, S_mat = H_red, S_red
        end
        
        if verbose
            # Solve the GenEig problem:
            E, C, kappa = SolveGenEig(H_mat, S_mat, thresh="inversion", eps=1e-8)
            e_old = E[1]
            kappa_old = kappa
            min_eval = minimum(filter(!isnan,real.(E)))
            print("Progress: [$(l)/$(maxiter)]; min. eval = $(min_eval) \r")
            flush(stdout)
        end
        
    end
    
    if verbose
        println("\nDone!")
    end
    
    return psi_list, ham_list, ord_list
        
end