# Functions for contructing and applying SWAP networks to permute the MPS indices

# Packages:
using ITensors


# Generates index positions for a bubble/insertion sorting network to rearrange sites:
function BubbleSort(ord1, ord2; spinpair=false)
    
    if spinpair==true
        ord1c = deepcopy(Spatial2SpinOrd(ord1))
        ord2c = deepcopy(Spatial2SpinOrd(ord2))
    else
        ord1c = deepcopy(ord1)
        ord2c = deepcopy(ord2)
    end
    
    N = size(ord2c,1)
    
    target_indices = [findall(x->x==i, ord2c)[1] for i=1:N]
    swap_indices = []
    
    p_list = vcat(collect(1:N-1), collect(N-2:(-1):1))
    
    for p in p_list
        for q=p:(-2):1
            if target_indices[ord1c[q+1]] - target_indices[ord1c[q]] < 0
                ord1cc = [ord1c[q], ord1c[q+1]]
                ord1c[q] = ord1cc[2]
                ord1c[q+1] = ord1cc[1]
                push!(swap_indices, q)
            end
        end
    end
    
    return swap_indices
    
end


# Builds the SWAP ITensor object between sites idx and idx+1:
function BuildSwap(sites, idx; dim=2)
    
    swap_array = zeros((dim,dim,dim,dim))
    
    for i=1:dim, j=1:dim, ip=1:dim, jp=1:dim
        swap_array[i,j,ip,jp] = Int(ip==j && jp==i)
    end

    swap = ITensor(swap_array, dag(sites[idx]),dag(sites[idx+1]),sites[idx]',sites[idx+1]')
    
    return swap
    
end


# Builds the fermionic SWAP ITensor object between sites idx and idx+1:
function BuildFermionicSwap(sites, idx; dim=2)
    
    fswap = [1 0 0 0;
             0 0 1 0;
             0 1 0 0;
             0 0 0 -1]
    
    i2 = Matrix(I(2))
    
    if dim==2
        swap_mat = fswap
    elseif dim==4
        swap_mat = kron(i2,kron(fswap,i2))*kron(fswap,fswap)*kron(i2,kron(fswap,i2)) 
    end
    
    swap_array = reshape(swap_mat, (dim,dim,dim,dim))

    swap = ITensor(swap_array, dag(sites[idx]),dag(sites[idx+1]),sites[idx]',sites[idx+1]')
    
    return swap
    
end


# Applies the SWAP tensor to the MPS with specified accuracy tolerance:
function ApplySwap(psi, swap, idx, tol, maxdim, mindim, alg)
    
    orthogonalize!(psi,idx)
    
    temp_tensor = (psi[idx] * psi[idx+1]) * swap
    noprime!(temp_tensor)
    
    temp_inds = uniqueinds(psi[idx],psi[idx+1])
    
    U,S,V = svd(temp_tensor,temp_inds,cutoff=tol,maxdim=maxdim,mindim=mindim,alg=alg)
    
    psi[idx] = U
    psi[idx+1] = S*V
    
    return psi
    
end


# Applies the same SWAP tensor to both sides of an MPO with specified accuracy tolerance:
function ApplySwapMPO(mpo, swap, idx, tol, maxdim, mindim)
    
    orthogonalize!(mpo,idx)
    
    # Apply top swap:
    setprime!(swap, 4, plev=1)
    setprime!(swap, 1, plev=0)
    setprime!(swap, 0, plev=4)
    
    temp_tensor = (mpo[idx] * mpo[idx+1]) * swap
    
    # Apply bottom swap:
    setprime!(swap, 2, plev=1)
    setprime!(swap, 3, plev=0)
    
    temp_tensor *= swap
    
    # De-prime the temp_tensor:
    setprime!(temp_tensor, 0, plev=1)
    setprime!(temp_tensor, 2, plev=3)
    
    temp_inds = uniqueinds(mpo[idx],mpo[idx+1])
    
    U,S,V = svd(temp_tensor,temp_inds,cutoff=tol,maxdim=maxdim,mindim=mindim,alg="qr_iteration")
    
    mpo[idx] = U
    mpo[idx+1] = S*V
    
    return mpo
    
end


# Permutes the MPS according to the swap network indices provided:
function Permute(passed_psi, sites, ord1, ord2; tol=1E-16, maxdim=5000, mindim=50, spinpair=false, locdim=2, verbose=false, alg="qr_iteration")
    
    # Copy psi:
    psi = deepcopy(passed_psi)
    
    swap_indices = BubbleSort(ord1, ord2, spinpair=spinpair)
    
    if verbose==true
        println("Permuting state: ")
    end
    
    for (i, idx) in enumerate(swap_indices)
        
        swap_tensor = BuildFermionicSwap(siteinds(psi), idx, dim=locdim)
        
        psi = ApplySwap(psi, swap_tensor, idx, tol, maxdim, mindim, alg)
        
        if verbose==true
            print("Progress: [",string(i),"/",string(length(swap_indices)),"] \r")
            flush(stdout)
        end
        
    end
    
    if verbose==true
        println("")
        println("Done!")
    end
    
    return psi
    
end


# Permutes an MPO according to the swap network indices provided:
function PermuteMPO(passed_mpo, sites, ord1, ord2; tol=1E-16, maxdim=5000, mindim=50, spinpair=false, locdim=2)
    
    # Copy MPO:
    mpo = deepcopy(passed_mpo)
    
    swap_indices = BubbleSort(ord1, ord2, spinpair=spinpair)
    
    # Prime the indices:
    setprime!(mpo, 2, plev=1)
    
    for idx in swap_indices
        
        swap_tensor = BuildFermionicSwap(sites, idx, dim=locdim)
        
        mpo = ApplySwapMPO(mpo, swap_tensor, idx, tol, maxdim, mindim)
        
    end
    
    setprime!(mpo, 1, plev=2)
    
    return mpo
    
end
