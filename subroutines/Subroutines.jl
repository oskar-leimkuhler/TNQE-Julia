# Import modules:
using ITensors
using HDF5
using Random
using PyCall
#using Graphs
#using SimpleWeightedGraphs
using Plots
using GraphRecipes
using JuMP
import Ipopt
import ForwardDiff
using BlackBoxOptim
using Combinatorics

# Importing the other submodules:
include("./ChemData.jl")
include("./ChemRoutines.jl")
include("./GenEigRoutines.jl")
include("./GeometryHeuristics.jl")
include("./MutualInformation.jl")
include("./Permutations.jl")
include("./PlotRoutines.jl")
include("./Disentanglers.jl")
include("./Misc.jl")

# Custom iTensor functions:
include("./custom_itensor/projmpo_mps_c.jl")
include("./custom_itensor/proj_costfunc.jl")
include("./custom_itensor/dmrg_c.jl")

# Import Python modules, including the RunPySCF subroutines in Python in order to use PySCF:
py"""
import sys
import os
import configparser
wd = os.getcwd()
sys.path.append(wd+'/../subroutines/')
import RunPySCF
"""

# High-level subroutines go here

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
    
    println("Generating states:")
    
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
    
    println("")
    println("Done!")
    
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
    M = length(psi_list)
    
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
    
    # The sweep loop:
    for s=1:sweeps
        
        if verbose
            println("Sweep $(s) of $(sweeps): ")
        end

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
                print("Progress: [$(p)/$(N_sites-1)]\r")
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
            println("Run $(l) of $(loops):\n")
        end
        
        for j=1:M
            
            if verbose
                println("Optimizing state $(j) of $(M):")
            end
            
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
            
            if verbose
                println("Done!")
            end
            
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