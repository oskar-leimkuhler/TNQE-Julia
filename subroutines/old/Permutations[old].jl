function LeftRightSwap(sites, idx, link, dim)
    
    vecs = Matrix(I(dim^2))
    lvecs = reshape(vecs, (dim,dim,dim^2))
    rvecs = permutedims(lvecs, (2,1,3))
    
    left_swap = ITensor(lvecs, sites[idx]', sites[idx]'', link)
    
    right_swap = ITensor(rvecs, sites[idx+1]', sites[idx+1]'', link)
    
    return left_swap, right_swap
    
end


function SwapContract(site_tensor, swap, comb)
    temp_tensor = site_tensor*swap
    new_tensor = temp_tensor*comb
    setprime!(new_tensor, 1, plev=2)
    return new_tensor
end


function PermutationMPO(sites_in, ord1, ord2; tol=1e-12)
    
    dim=4
    
    iden = Matrix(I(dim))
    
    sites = removeqns(sites_in)
    
    #site_tensors = ITensor[]
    
    # Make MPO from site tensors
    pmpo = MPO(sites)
    
    # Fill with identity:
    
    comInd = Index(1,"Link") # Define a common index  
    
    for p=1:size(sites,1)
        
        if p==1
            j = Index(1,"Link")
            iden_array = zeros((dim, dim, 1))
            iden_array[:,:,1] = sparse(iden)
            #push!(site_tensors, ITensor(iden_array, sites[p], sites[p]', j))
            pmpo[p] = ITensor(iden_array, sites[p], sites[p]', j)
            comInd = j
        elseif p>1 && p<size(sites,1)
            i = comInd
            j = Index(1,"Link")
            iden_array = zeros((dim, dim, 1, 1))
            iden_array[:,:,1,1] = sparse(iden)
            #push!(site_tensors, ITensor(iden_array, sites[p], sites[p]', i, j))
            pmpo[p] = ITensor(iden_array, sites[p], sites[p]', i, j)
            comInd = j
        else
            i = comInd
            iden_array = zeros((dim, dim, 1))
            iden_array[:,:,1] = sparse(iden)
            #push!(site_tensors, ITensor(iden_array, sites[p], sites[p]', i))
            pmpo[p] = ITensor(iden_array, sites[p], sites[p]', i)
        end
        
    end
    
    # Apply swaps:
    swap_indices = BubbleSort(ord1, ord2)
    
    c_tot = size(swap_indices,1)
    c=0
    
    for idx in swap_indices
        
        # Contract to form a new site tensor
        #orthogonalize!(pmpo, idx, cutoff=tol)
        
        new_link = Index(dim^2,"Link")
        
        left_swap, right_swap = LeftRightSwap(sites, idx, new_link, dim)
        
        old_link = commoninds(pmpo[idx], pmpo[idx+1])
        
        comb = combiner(old_link, new_link; tags="Link")
        
        pmpo[idx] = SwapContract(pmpo[idx], left_swap, comb)
        
        pmpo[idx+1] = SwapContract(pmpo[idx+1], right_swap, comb)
        
        c += 1
        println("Progress: [",c,"/",c_tot,"]")
        println(maxlinkdim(pmpo))
        
    end
    
    return pmpo
    
end


# Applies a SWAP tensor to an MPO with specified accuracy tolerance:
function ApplySwapMPO(mpo, swap, side, idx, tol, maxdim, mindim)
    
    orthogonalize!(mpo,idx)
    
    setprime!(swap, 4, plev=1)
    setprime!(swap, 1, plev=0)
    setprime!(swap, 0, plev=4)
    
    if side=="bottom"
        # Apply bottom swap:
        setprime!(swap, 2, plev=1)
        setprime!(swap, 3, plev=0)
    end

    temp_tensor = (mpo[idx] * mpo[idx+1]) * swap
    
    # De-prime the temp_tensor:
    if side=="top"
        setprime!(temp_tensor, 0, plev=1)
    elseif side=="bottom"
        setprime!(temp_tensor, 2, plev=3)
    end
    
    temp_inds = uniqueinds(mpo[idx],mpo[idx+1])
    
    U,S,V = svd(temp_tensor,temp_inds,cutoff=tol,maxdim=maxdim,mindim=mindim,alg="qr_iteration")
    
    mpo[idx] = U
    mpo[idx+1] = S*V
    
    return mpo
    
end


# Permutes an MPO according to two sets of swap network indices provided:
function PermuteMPO(passed_mpo, sites, ordh, ord1, ord2; tol=1E-16, maxdim=5000, mindim=50, spinpair=false, locdim=4)
    
    # Copy MPO:
    mpo = deepcopy(passed_mpo)
    
    top_swap_indices = BubbleSort(ordh, ord1, spinpair=spinpair)
    bottom_swap_indices = BubbleSort(ordh, ord2, spinpair=spinpair)
    
    # Prime the indices:
    setprime!(mpo, 2, plev=1)
    
    for idx in top_swap_indices
        
        swap_tensor = BuildFermionicSwap(sites, idx, dim=locdim)
        mpo = ApplySwapMPO(mpo, swap_tensor, "top", idx, tol, maxdim, mindim)
        
    end
    
    for idx in bottom_swap_indices
        
        swap_tensor = BuildFermionicSwap(sites, idx, dim=locdim)
        mpo = ApplySwapMPO(mpo, swap_tensor, "bottom", idx, tol, maxdim, mindim)
        
    end
    
    setprime!(mpo, 1, plev=2)
    
    return mpo
    
end


function SparsePermutationMPO(sites_in, ord1, ord2; tol=1e-12)
    
    dim=4
    
    sites = removeqns(sites_in)
    
    #site_tensors = ITensor[]
    
    # Make MPO from site tensors
    pmpo = MPO(sites)
    
    # Fill with identity:
    
    comInd = Index([QN(0) => 1],"Link") # Define a common index  
    
    for p=1:size(sites,1)
        
        if p==1
            j = Index([QN(0) => 1],"Link")
            pmpo[p] = delta(sites[p], sites[p]', j)
            println("break")
            comInd = j
        elseif p>1 && p<size(sites,1)
            i = comInd
            j = Index([QN(0) => 1],"Link")
            pmpo[p] = delta(sites[p], sites[p]', i, j)
            comInd = j
        else
            i = comInd
            pmpo[p] = delta(sites[p], sites[p]', i)
        end
        
    end
    
    # Apply swaps:
    swap_indices = BubbleSort(ord1, ord2)
    
    c_tot = size(swap_indices,1)
    c=0
    
    for idx in swap_indices
        
        # Contract to form a new site tensor
        #orthogonalize!(pmpo, idx, cutoff=tol)
        
        new_link = Index([QN(0) => 1 for i=1:dim^2],"Link")
        
        left_swap, right_swap = LeftRightSwap(sites, idx, new_link, dim)
        
        old_link = commoninds(pmpo[idx], pmpo[idx+1])
        
        comb = combiner(dag(old_link), dag(new_link); tags="Link")
        
        pmpo[idx] = SwapContract(pmpo[idx], left_swap, comb)
        
        pmpo[idx+1] = SwapContract(pmpo[idx+1], right_swap, comb)
        
        c += 1
        println("Progress: [",c,"/",c_tot,"]")
        println(maxlinkdim(pmpo))
        
    end
    
    """
    # Make MPO from site tensors
    pmpo = MPO(sites)

    for (p, site_tensor) in enumerate(site_tensors)
        pmpo[p] = site_tensor
    end 
    """
    
    return pmpo
    
end
