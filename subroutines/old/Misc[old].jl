
function OnSiteSwap!(T, site; do_fswap=false)
    
    ids = vcat(site, uniqueinds(inds(T), site))
    
    TA = Array(T, ids)
    
    TA[2:3] = [TA[3],TA[2]]
    
    if do_fswap
        TA[4] *= -1.0
    end
    
    T=ITensor(TA,ids)
    
end


function QuickReverseMPS(psi)
    
    N = length(psi)
    
    psi2 = MPS(N)
    
    for p=1:N
        
        q=N-p+1
        
        Tq = deepcopy(psi[q])
        
        psi2[p] = Tq
        
    end
    
    return psi2
    
end


function QuickReverseMPO(mpo)
    
    N = length(mpo)
    
    mpo2 = MPO(N)
    
    for p=1:N
        
        q=N-p+1
        
        Tq = deepcopy(mpo[q])
        
        mpo2[p] = Tq
        
    end
    
    return mpo2
    
end


function ReverseMPS2(psi)
    
    maxdim = maxlinkdim(psi)
    sites = siteinds(psi)
    N = length(psi)
    
    psi = Permute(
        psi,
        sites,
        collect(1:N),
        reverse(collect(1:N))
    )
    
    truncate!(psi, maxdim=maxdim)
    normalize!(psi)
    
    return psi
    
end


function ReverseMPO2(mpo, sites)
    
    maxdim = maxlinkdim(mpo)
    sites = siteinds(mpo, plev=0)
    N = length(mpo)
    
    #println(siteinds(mpo))
    
    dag!(mpo)
    
    mpo = PermuteMPO(
        mpo,
        sites,
        collect(1:N),
        reverse(collect(1:N)),
        tol=1e-16
    )
    
    dag!(mpo)
    swapprime!(mpo, 0,1)
    
    #println(siteinds(mpo))
    
    mpo = PermuteMPO(
        mpo,
        sites,
        collect(1:N),
        reverse(collect(1:N)),
        tol=1e-16
    )
    
    swapprime!(mpo, 0,1)
    
    truncate!(mpo, maxdim=maxdim)
    
    return mpo
    
end
