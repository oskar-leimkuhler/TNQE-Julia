# Functions to generate and manipulate subspace states and subspace matrices.


# Packages:
#


# Carry out a full MPS-MPO DMRG procedure for a given chemical data, site ordering, MPO Hamiltonian and sweep specification:
function RunDMRG(
        chemical_data, 
        sites, ord, H, 
        sweeps; 
        spinpair=false, 
        spatial=false, 
        ovlp_opt=false, 
        prev_states=[], 
        weight=1.0, 
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
    
    if ovlp_opt==true && size(prev_states,1)>0
        e_dmrg, psi = dmrg(H, prev_states, psi0, sweeps, outputlevel=0, weight=weight)
    else
        e_dmrg, psi = dmrg(H, psi0, sweeps, outputlevel=0)
    end
    
    return psi, e_dmrg
    
end


# Generate a list of states given a list of orderings, and prior states for overlap optimization
function GenStates(
        chemical_data,
        sites,
        ord_list,
        sweeps;
        ovlp_opt=false,
        weight=1.0,
        prior_states=[],
        prior_ords=[],
        perm_tol=1E-16,
        perm_maxdim=5000,
        ham_tol=1E-16,
        ham_maxdim=5000,
        spinpair=false,
        spatial=true,
        singleham=false,
        verbose=false
    )
    
    if spatial==true
        locdim=4
    else
        locdim=2
    end
    
    if singleham==true
        if verbose==true
            println("Generating initial Ham. OpSum/MPO:")
        end
        opsum = GenOpSum(chemical_data, ord_list[1])
        ham = MPO(opsum, sites, cutoff=ham_tol, maxdim=ham_maxdim);
    end
    
    num_priors = size(prior_states, 1)
    
    # Number of ref. states:
    M = size(ord_list, 1)
    
    psi_list = MPS[]
    ham_list = MPO[]
    
    # Initialize counter:
    c_tot = M
    c = 0
    if verbose
        println("Generating states:")
    end
    
    # Optimize the ref. states and fill in diagonals:
    for j=1:M
        
        if singleham==true
            perm_ham = PermuteMPO(
                ham, 
                sites, 
                ord_list[1], 
                ord_list[j], 
                tol=perm_tol, 
                maxdim=perm_maxdim, 
                locdim=locdim
            )
        else
            perm_opsum = GenOpSum(chemical_data, ord_list[j])
            perm_ham = MPO(perm_opsum, sites, cutoff=ham_tol, maxdim=ham_maxdim);
        end
        
        push!(ham_list, perm_ham)
        
        if ovlp_opt==true
            
            total_psi_list = vcat(prior_states,psi_list)
            total_ord_list = vcat(prior_ords,ord_list)
            
            # Minimize overlap with previous states (including the passed-in prior states)
            prev_states = MPS[]
            
            for i=1:num_priors+j-1
                new_prev = Permute(
                    total_psi_list[i], 
                    sites, 
                    total_ord_list[i],
                    total_ord_list[j], 
                    tol=perm_tol, 
                    maxdim=perm_maxdim, 
                    locdim=locdim
                )
                push!(prev_states, new_prev)
            end
            
            psi_j, H_jj = RunDMRG(
                chemical_data, 
                sites, 
                ord_list[j], 
                perm_ham, 
                sweeps, 
                spinpair=spinpair, 
                spatial=spatial, 
                ovlp_opt=true, 
                prev_states=prev_states, 
                weight=weight
            )
            
        else
            
            psi_j, H_jj = RunDMRG(
                chemical_data, 
                sites, ord_list[j], 
                perm_ham, 
                sweeps, 
                spinpair=spinpair, 
                spatial=spatial
            )
            
        end
        
        push!(psi_list, psi_j)
        
        c += 1
        
        if verbose==true
            print("Progress: [",string(c),"/",string(c_tot),"] \r")
            flush(stdout)
        end
        
    end
    
    if verbose
        println("")
        println("Done!")
    end
    
    return psi_list, ham_list
    
end


# Generate the subspace matrices H and S from a list of MPSs, MPOs and the corresponding orderings:
function GenSubspaceMats(
        chemical_data,
        sites,
        ord_list,
        psi_list,
        ham_list;
        perm_tol=1E-16, 
        perm_maxdim=5000,
        spinpair=false, 
        spatial=false, 
        singleham=false,
        verbose=false
    )
    
    if spatial==true
        locdim=4
    else
        locdim=2
    end
    
    # Number of ref. states:
    M = size(ord_list, 1)
    
    H_mat = zeros(Float64, M, M)
    S_mat = zeros(Float64, M, M)
    
    # Initialize counter:
    c_tot = Int((M^2+M)/2)
    c = 0
    
    # Get the off-diagonal contractions:
    if verbose
        println("computing matrix elements:")
    end
    
    for i=1:M
        
        for j=i:M
            
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
            
            S_mat[i,j] = inner(perm_psi_i, psi_list[j])
            S_mat[j,i] = S_mat[i,j]
            
            #println(siteinds(ham_list[j]))
            #println(siteinds(psi_list[j]))
            
            H_mat[i,j] = inner(perm_psi_i', ham_list[j], psi_list[j])
            H_mat[j,i] = H_mat[i,j]

            c += 1
            
        end
        
        if verbose==true
            print("Progress: [",string(c),"/",string(c_tot),"] \r")
            flush(stdout)
        end
        
    end
    
    if verbose
        println("")
        println("Done!")
    end
    
    return H_mat, S_mat
    
end


function ExpandSubspaceMats(H_in, S_in, new_psi, new_ham, new_ord, psi_list, ord_list)
    
    M = size(H_in,1)
    H_mat = zeros((M+1,M+1))
    S_mat = zeros((M+1,M+1))
    H_mat[1:M,1:M] = H_in
    S_mat[1:M,1:M] = S_in
    
    sites=siteinds(psi_list[1])
    
    for (i,psi) in enumerate(psi_list)
        
        perm_psi_i = Permute(
            psi_list[i],
            sites, 
            ord_list[i], 
            new_ord, 
            tol=1e-10, 
            maxdim=1000, 
            spinpair=false, 
            locdim=4
        )
        
        H_mat[M+1,i] = inner(new_psi',new_ham,perm_psi_i)
        H_mat[i,M+1] = H_mat[M+1,i]
        
        S_mat[M+1,i] = inner(new_psi,perm_psi_i)
        S_mat[i,M+1] = S_mat[M+1,i]
        
    end
    
    H_mat[M+1,M+1] = inner(new_psi',new_ham,new_psi)
    S_mat[M+1,M+1] = 1.0
    
    return H_mat, S_mat
    
end


function ShrinkSubspaceMats(H_in,S_in,jpop)
    
    M = size(H_in,1)
    H_mat = zeros((M-1,M-1))
    S_mat = zeros((M-1,M-1))
    
    H_mat[1:jpop-1,1:jpop-1] = H_in[1:jpop-1,1:jpop-1]
    H_mat[jpop:M-1,jpop:M-1] = H_in[jpop+1:M,jpop+1:M]
    H_mat[1:jpop-1,jpop:M-1] = H_in[1:jpop-1,jpop+1:M]
    H_mat[jpop:M-1,1:jpop-1] = H_in[jpop+1:M,1:jpop-1]
    
    S_mat[1:jpop-1,1:jpop-1] = S_in[1:jpop-1,1:jpop-1]
    S_mat[jpop:M-1,jpop:M-1] = S_in[jpop+1:M,jpop+1:M]
    S_mat[1:jpop-1,jpop:M-1] = S_in[1:jpop-1,jpop+1:M]
    S_mat[jpop:M-1,1:jpop-1] = S_in[jpop+1:M,1:jpop-1]
    
    return H_mat, S_mat
    
end


function ExpProb(E_0, E_1, beta)
    if E_1<=E_0
        P = 1
    else
        P = exp((E_0-E_1)*beta)
    end
    return P
end


function StepProb(E_0, E_1)
    if E_1<=E_0
        P = 1
    else
        P = 0
    end
    return P
end

function Fstun(E_0, E_1, gamma)
    return 1.0 - exp(gamma*(E_1-E_0))
end


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