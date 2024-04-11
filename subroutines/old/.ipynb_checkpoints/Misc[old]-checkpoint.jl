
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


# Truncate to bond dimension 2, using Givens rotations to reduce truncation errors:

function GivRotTrunc(psi_in)
    
    psi = deepcopy(psi_in)
    sites = siteinds(psi)
    N = length(psi)
    
    orthogonalize!(psi, 1)
    
    giv_rots = []
    
    for p=1:N-1
        
        T = psi[p] * psi[p+1]
        
        linds = commoninds(T, psi[p])
        
        f(theta) = GTruncErr(theta, T, 2, linds, sites, p; dim=2)
        
        res = Optim.optimize(f, [0.0], LBFGS())
        theta_opt = Optim.minimizer(res)
        
        G = BuildGivensRotation(sites, p, theta_opt[1]; dim=2)
        
        T *= G
        noprime!(T)
        
        U,S,V = svd(T, linds, maxdim=2)
        
        psi[p] = U
        
        psi[p+1] = S*V
        
        push!(giv_rots, G)
        
    end
    
    return psi, giv_rots
    
end

function ApplyRots(psi_in, giv_rots; tol=1e-13)
    
    psi = deepcopy(psi_in)
    N = length(psi)
    
    orthogonalize!(psi, 1)
    
    for p=1:N-1
        
        T = psi[p] * psi[p+1]
        
        linds = commoninds(T, psi[p])
        
        T *= giv_rots[p]
        noprime!(T)
        
        U,S,V = svd(T, linds, cutoff=tol)
        
        psi[p] = U
        
        psi[p+1] = S*V
        
    end
    
    return psi
    
end