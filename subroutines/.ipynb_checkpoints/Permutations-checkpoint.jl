# Functions for contructing and applying SWAP networks to permute the MPS indices

# Packages:
using ITensors


# Returns an element of the SWAP tensor corresponding to i, j, i' and j':
function SwapElement(i,j,ip,jp)
    if ip==j && jp==i
        return 1
    else
        return 0
    end
end


# Builds the SWAP ITensor object between sites idx and idx+1:
function BuildSwap(sites, idx)
    
    swap = ITensor(sites[idx],sites[idx+1],sites[idx]',sites[idx+1]')
    
    for i=1:2
        for j=1:2
            for ip=1:2
                for jp=1:2
                    swap[i,j,ip,jp] = SwapElement(i,j,ip,jp)
                end
            end
        end
    end
    
    return swap
    
end


# Applies the SWAP tensor to the MPS with specified accuracy tolerance:
function ApplySwap(psi, swap, idx, tol, maxdim)
    
    orthogonalize!(psi,idx)
    
    temp_tensor = (psi[idx] * psi[idx+1]) * swap
    noprime!(temp_tensor)
    
    temp_inds = uniqueinds(psi[idx],psi[idx+1])
    
    U,S,V = svd(temp_tensor,temp_inds,cutoff=tol,maxdim=maxdim)
    
    psi[idx] = U
    psi[idx+1] = S*V
    
    return psi
    
end


# Generates index positions for a bubble/insertion sorting network to rearrange sites:
function BubbleSort(ord1, ord2; spatial=false)
    
    if spatial==true
        ord1c = Spatial2SpinOrd(ord1)
        ord2c = Spatial2SpinOrd(ord2)
    else
        ord1c = ord1
        ord2c = ord2
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
    
    println(ord1c)
    
    return swap_indices
    
end


# Permutes the MPS according to the swap network indices provided:
function Permute(psi, sites, ord1, ord2; tol=1E-16, maxdim=5000, spatial=false)
    
    swap_indices = BubbleSort(ord1, ord2, spatial=spatial)
    
    for idx in swap_indices
        
        swap_tensor = BuildSwap(sites, idx)
        
        psi = ApplySwap(psi, swap_tensor, idx, tol, maxdim)
        
    end
    
    return psi
    
end