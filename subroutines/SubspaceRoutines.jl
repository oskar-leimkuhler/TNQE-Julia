# Functions to generate and manipulate subspace states and subspace matrices.


# Carry out a full MPS-MPO DMRG procedure for a given chemical data, site ordering, MPO Hamiltonian and sweep specification:
function RunDMRG(
        chemical_data, 
        sites, ord, H, 
        sweeps;  
        ovlp_opt=false, 
        prev_states=[], 
        weight=1.0, 
        linkdims=31
    )
    
    if hastags(sites[1], "Electron") # Spatial orbital sites:
        hf_occ = [FillHF(ord[i], chemical_data.N_el) for i=1:chemical_data.N_spt]
    elseif hastags(sites[1], "Fermion") || hastags(sites[1], "Qubit") # Spin-orbital sites:
        hf_occ = [FillHFSporb(ord[i], chemical_data.N_el) for i=1:chemical_data.N]
    else
        println("Invalid site type!")
    end

    psi0 = randomMPS(sites, hf_occ, linkdims=linkdims)
    #psi0 = MPS(sites, hf_occ)
    
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
        singleham=false,
        denseify=false,
        verbose=false
    )
    
    if hastags(sites[1], "Electron") # Spatial orbital sites:
        locdim=4
    elseif hastags(sites[1], "Fermion") || hastags(sites[1], "Qubit") # Spin-orbital sites:
        locdim=2
    else
        println("Invalid site type!")
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
                ovlp_opt=true, 
                prev_states=prev_states, 
                weight=weight
            )
            
        else
            
            psi_j, H_jj = RunDMRG(
                chemical_data, 
                sites, ord_list[j], 
                perm_ham, 
                sweeps
            )
            
        end
        
        
        if denseify
            push!(ham_list, dense(perm_ham))
            push!(psi_list, dense(psi_j))
        else
            push!(ham_list, perm_ham)
            push!(psi_list, psi_j)
        end
        
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