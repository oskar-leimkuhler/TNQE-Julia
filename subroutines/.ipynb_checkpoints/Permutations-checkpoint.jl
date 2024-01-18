# Functions for contructing and applying SWAP networks to permute the MPS indices

# Packages:
using ITensors


# Globally declaring the identity operator for electron sites:
ITensors.op(::OpName"I",::SiteType"Electron") = [1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1]
ITensors.op(::OpName"SWAP",::SiteType"Electron") = [1 0 0 0; 0 0 1 0; 0 1 0 0; 0 0 0 1]
ITensors.op(::OpName"FSWAP",::SiteType"Electron") = [1 0 0 0; 0 0 1 0; 0 1 0 0; 0 0 0 -1]
ITensors.op(::OpName"CZ",::SiteType"Electron") = [1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 -1]


# Generates index positions for a bubble/insertion sorting network to rearrange sites:
function BubbleSort(ord1, ord2)
    
    N = size(ord1,1)
    
    aux = [0 for i=1:N]
    for i=1:N
        aux[ord2[i]] = i
    end
    
    targ = [aux[ord1[i]] for i=1:N]
    
    swap_indices = []
    p_list = vcat(collect(1:N-1), collect(N-2:(-1):1))
    
    for p in p_list
        for q=p:(-2):1
            if targ[q+1] < targ[q]
                targ[q:q+1] = [targ[q+1],targ[q]]
                push!(swap_indices, q)
            end
        end
    end
    
    return swap_indices
    
end


# Builds the SWAP ITensor object between sites idx and idx+1:
function BuildSwap(sites, idx; dim=2)
    
    swap = [1 0 0 0;
            0 0 1 0;
            0 1 0 0;
            0 0 0 1]
    
    i2 = Matrix(I(2))
    
    if dim==2
        swap_mat = swap
    elseif dim==4
        swap_mat = kron(i2,kron(swap,i2))*kron(swap,swap)*kron(i2,kron(swap,i2)) 
    end
    
    swap_array = reshape(swap_mat, (dim,dim,dim,dim))

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


# Applies the same SWAP tensor to one side of an MPO with specified accuracy tolerance:
function ApplySwapMPO(mpo, swap, idx, tol, maxdim, mindim)
    
    orthogonalize!(mpo,idx)
    
    # Apply swap to the primed indices:
    temp_tensor = (mpo[idx] * mpo[idx+1]) * swap
    
    # De-prime the temp_tensor:
    setprime!(temp_tensor, 1, plev=2)
    
    temp_inds = uniqueinds(mpo[idx],mpo[idx+1])
    
    U,S,V = svd(temp_tensor,temp_inds,cutoff=tol,maxdim=maxdim,mindim=mindim,alg="qr_iteration")
    
    mpo[idx] = U
    mpo[idx+1] = S*V
    
    return mpo
    
end


# Permutes the MPS according to the swap network indices provided:
function Permute(passed_psi, sites, ord1, ord2; tol=1E-16, maxdim=5000, mindim=1, locdim=4, verbose=false, alg="divide_and_conquer")
    
    # Copy psi:
    psi = deepcopy(passed_psi)
    
    swap_indices = BubbleSort(ord1, ord2)
    
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
    
    normalize!(psi)
    
    return psi
    
end


# Permutes an MPO according to the swap network indices provided:
function PermuteMPO(passed_mpo, sites, ord1, ord2; do_fswap=true, tol=1e-16, maxdim=5000, mindim=0, locdim=4)
    
    # Copy MPO:
    mpo = deepcopy(passed_mpo)
    
    # Prime one side of the MPO:
    setprime!(mpo, 2, plev=1)
    
    swap_indices = BubbleSort(ord1, ord2)
    
    for idx in swap_indices
        
        if do_fswap
            swap_tensor = BuildFermionicSwap(sites, idx, dim=locdim)
        else
            swap_tensor = BuildSwap(sites, idx, dim=locdim)
        end
        
        mpo = ApplySwapMPO(mpo, swap_tensor, idx, tol, maxdim, mindim)
        
    end
    
    setprime!(mpo, 1, plev=2)
    
    return mpo
    
end


function SlowPMPO(sites, ord1, ord2; do_fswap=true, no_rev=false, tol=1e-12, maxdim=2^16)
    
    # Generate an identity MPO:
    mpo = MPO(sites, "I")
    
    # Get the swap indices:
    swap_ind = BubbleSort(ord2, ord1)
    swap_rev = BubbleSort(reverse(ord2), ord1)
    
    rev_flag=false
    ords = [ord1, ord2]
    if length(swap_rev) < length(swap_ind) && !no_rev
        ords = [ord1, reverse(ord2)]
        rev_flag = true
    end 
    
    
    mpo = PermuteMPO(
        mpo, 
        sites, 
        ords[1],
        ords[2],
        tol=tol,
        maxdim=maxdim
    )
    
    if rev_flag && do_fswap
        ApplyPhases!(mpo, sites)
    end
    
    return mpo, rev_flag
    
end


function SWAPComponents(do_fswap)
    
    I_4 = Matrix(1.0I, 4, 4)
    I_2 = Matrix(1.0I, 2, 2)
    
    SWAP_4 = [
        1 0 0 0;
        0 0 1 0;
        0 1 0 0;
        0 0 0 1
    ]
    
    if do_fswap
        SWAP_4[4,4] = -1
    end

    SWAP_16 = kron(I_2,kron(SWAP_4,I_2))*kron(SWAP_4,SWAP_4)*kron(I_2,kron(SWAP_4,I_2))

    SWAP_16 = reshape(permutedims(reshape(SWAP_16, (4,4,4,4)), (1,3,2,4)), (16,16))

    F = svd(SWAP_16)

    lU = reshape(F.U, 4,4,16)
    rV = reshape(F.V, 16,4,4)

    qnvec = [
        QN(("Nf",0,-1),("Sz",0)) => 1,
        QN(("Nf",1,-1),("Sz",1)) => 1,
        QN(("Nf",1,-1),("Sz",-1)) => 1,
        QN(("Nf",2,-1),("Sz",0)) => 1,
        #---------------------------#
        QN(("Nf",-1,-1),("Sz",-1)) => 1,
        QN(("Nf",0,-1),("Sz",0)) => 1,
        QN(("Nf",0,-1),("Sz",-2)) => 1,
        QN(("Nf",1,-1),("Sz",-1)) => 1,
        #---------------------------#
        QN(("Nf",-1,-1),("Sz",1)) => 1,
        QN(("Nf",0,-1),("Sz",2)) => 1,
        QN(("Nf",0,-1),("Sz",0)) => 1,
        QN(("Nf",1,-1),("Sz",1)) => 1,
        #---------------------------#
        QN(("Nf",-2,-1),("Sz",0)) => 1,
        QN(("Nf",-1,-1),("Sz",1)) => 1,
        QN(("Nf",-1,-1),("Sz",-1)) => 1,
        QN(("Nf",0,-1),("Sz",0)) => 1
    ]
    
    return lU, rV, qnvec
    
end


# Applies reversal site-local phase tensors to an MPO:
function ApplyPhases!(mpo::MPO, sites)
    
    N = length(mpo)
    
    #Apply phases:
    phase_mat = [1 0 0 0; 
                 0 1 0 0;
                 0 0 1 0;
                 0 0 0 -1]
    
    # Local phases:
    for p=1:N

        mpo_ind = sites[p]
        #mpo_rev = sites[N-p+1]
        
        phase_gate = ITensor(phase_mat, setprime(dag(mpo_ind),2), setprime(mpo_ind,0))
        
        mpo[p] = mpo[p] * phase_gate
        setprime!(mpo[p], 0, plev=2)
        
    end
    
    # Global phase:
    mpo[1] *= -1.0
    
end


# Applies reversal site-local phase tensors to an MPS:
function ApplyPhases!(psi::MPS)
    
    N = length(psi)
    
    sites = siteinds(psi)
    
    #Apply phases:
    phase_mat = [1 0 0 0; 
                 0 1 0 0;
                 0 0 1 0;
                 0 0 0 -1]
    
    # Local phases:
    for p=1:N
        
        phase_gate = ITensor(phase_mat, setprime(sites[p],1), dag(sites[p]))
        
        psi[p] = psi[p] * phase_gate
        noprime!(psi[p])
        
    end
    
    # Global phase:
    psi[1] *= -1.0
    
end


function PSWAP!(mpo, p, lU, rV, qnvec; tol=1.0e-12, maxdim=2^16)
    
    psite = [siteinds(mpo, plev=0)[p][1], siteinds(mpo, plev=0)[p+1][1]]
    
    plink = commoninds(mpo[p], mpo[p+1])
    nlink = Index(qnvec, tags="nlink")
    
    combo = combiner(plink, nlink, tags="link,l=$(p)")
    
    lswap = ITensor(lU, dag(setprime(psite[1],2)), setprime(psite[1],1), nlink)
    rswap = ITensor(rV, dag(nlink), dag(setprime(psite[2],2)), setprime(psite[2],1))
    
    mpo[p] = mpo[p] * lswap * combo
    mpo[p+1] = mpo[p+1] * rswap * dag(combo)
    
    setprime!(mpo[p], 1, plev=2)
    setprime!(mpo[p+1], 1, plev=2)
    
    #truncate!(mpo, tol=tol, maxdim=maxdim)
    
end


# Generates a permutation MPO for the orderings provided with a reversed-order flag:
function FastPMPO(sites, ord1, ord2; do_fswap=true, no_rev=false, tol=1e-12, maxdim=2^16)
    
    lU, rV, qnvec = SWAPComponents(do_fswap)
    
    # Generate an identity MPO:
    mpo = MPO(sites, "I")
    
    # Get the swap indices:
    swap_ind = reverse(BubbleSort(ord1, ord2))
    swap_rev = reverse(BubbleSort(ord1, reverse(ord2)))
    
    rev_flag=false
    if length(swap_rev) < length(swap_ind) && !no_rev
        swap_ind = swap_rev
        rev_flag = true
    end 
    
    for idx in swap_ind
        PSWAP!(mpo, idx, lU, rV, qnvec, tol=tol, maxdim=maxdim)
    end
    
    if rev_flag && do_fswap
        ApplyPhases!(mpo, sites)
    end
    
    truncate!(mpo, cutoff=tol, maxdim=maxdim)
    
    return mpo, rev_flag
    
end


# return a reversed copy of an MPS:
function ReverseMPS(psi)
    
    N = length(psi)
    
    sites=siteinds(psi)
    
    psi2 = MPS(N)
    
    for p=1:N
        
        q=N-p+1
        
        si_p = sites[p]
        si_q = sites[q]
        
        Tq = deepcopy(psi[q])
        
        replaceind!(Tq, si_q, si_p)
        
        psi2[p] = Tq
        
    end
    
    return psi2
    
end


function ReverseMPO(mpo)
    
    N = length(mpo)
    
    sites=siteinds(mpo)
    
    mpo2 = MPO(N)
    
    for p=1:N
        
        q=N-p+1
        
        si_p = sites[p]
        si_q = sites[q]
        
        Tq = deepcopy(mpo[q])
        
        replaceind!(Tq, si_q[1], si_p[1])
        replaceind!(Tq, si_q[2], si_p[2])
        
        mpo2[p] = Tq
        
    end
    
    return mpo2
    
end

