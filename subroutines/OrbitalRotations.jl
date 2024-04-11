# Functions for contructing and selecting/optimizing pairwise orbital permutations/rotations


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

# Builds the rotation matrix:
function RotationMatrix(; dim=2, rotype="swap", theta=0.0)
    
    if rotype=="swap"
        
        swap = [1 0 0 0;
                0 0 1 0;
                0 1 0 0;
                0 0 0 1]
        
        mat1, mat2 = swap, swap
        
    elseif rotype=="fswap"
        
        fswap = [1 0 0 0;
                 0 0 1 0;
                 0 1 0 0;
                 0 0 0 -1]
        
        mat1, mat2 = fswap, fswap
        
    elseif rotype=="givens"
        
        swap = [1 0 0 0;
                0 0 1 0;
                0 1 0 0;
                0 0 0 1]
        
        c = cos(theta)
        s = sin(theta)

        grot = [1 0  0 0;
                0 c -s 0;
                0 s  c 0;
                0 0  0 1]
        
        mat1, mat2 = swap, grot
        
    end
    
    i2 = Matrix(I(2))
    
    if dim==2
        rot_mat = mat2
    elseif dim==4
        rot_mat = kron(i2,kron(mat1,i2))*kron(mat2,mat2)*kron(i2,kron(mat1,i2)) 
    end
    
    return rot_mat
    
end


# Builds the rotation ITensor object between sites idx and idx+1:
function RotationTensor(sites, idx; dim=2, rotype="fswap", theta=0.0)
    rot_mat = RotationMatrix(dim=dim, rotype=rotype, theta=theta)
    rot_array = reshape(rot_mat, (dim,dim,dim,dim))
    rot_tens = ITensor(rot_array, dag(sites[idx]),dag(sites[idx+1]),sites[idx]',sites[idx+1]')
    return rot_tens
end


# Determine whether inserting an FSWAP reduces the \\ 
# ...truncation error for the two-site tensor T:
function TestFSWAP2(T, linds, maxdim; crit="fidelity")
    
    site_inds = inds(T, tags="Site")
    fswap = RotationTensor(site_inds, 1; dim=4, rotype="fswap");
    
    do_comp = true
    do_swap = false
    
    try 
        U_test,S_test,V_test = svd(T, linds, maxdim=maxdim)
    catch err
        do_comp = false
    end
    
    if do_comp
        T_ = T * fswap
        noprime!(T_)

        U,S,V = svd(T, linds)
        sigma = diag(Array(S, inds(S)))
        m = minimum([length(sigma), maxdim])
        fid = sum(reverse(sort(sigma))[1:m].^2)
        vne = -sum(sigma .* log.(sigma))

        U_,S_,V_ = svd(T_, linds, maxdim=maxdim)
        sigma_ = diag(Array(S_, inds(S_))) 
        m_ = minimum([length(sigma_), maxdim])
        fid_ = sum(reverse(sort(sigma_))[1:m_].^2)
        vne_ = -sum(sigma_ .* log.(sigma_))
        
        if crit=="fidelity"
            do_swap = (fid_ > fid)
        elseif crit=="entropy"
            do_swap = (vne_ < vne)
        end
    end
    
    return do_swap
    
end


function BlockSiteEntropy(T, prime_inds)

    totdim = prod(dim.(prime_inds))
    
    Tprime = deepcopy(T)

    for prime_ind in prime_inds
        setprime!(Tprime, 1, id=id(prime_ind))
    end

    Trdm = Tprime * dag(T)
    
    ind_set = vcat(
        [dag(prime_ind) for prime_ind in prime_inds], 
        [prime_ind' for prime_ind in prime_inds]
    )

    #println(inds(Trdm))
    #println(ind_set)
    
    rdm = reshape(Array(Trdm, ind_set), (totdim, totdim))

    return vnEntropy(rdm)
    
end


# Determine whether inserting an FSWAP is favorable by \\
# ...computing the mutual information with the left/right blocks:
function TestFSWAP3(T, sites, links)
    
    # left/right block entropies:
    s_b = []
    
    for lr = 1:2
        if links[lr]==nothing
            push!(s_b, 0.0)
        else
            push!(s_b, BlockSiteEntropy(T, [links[lr]]))
        end
    end
    
    # p/p+1 site entropies:
    s_p = [BlockSiteEntropy(T, [sites[q]]) for q=1:2]
    
    # Block/site mutual information:
    i_bp = []
    
    for q=1:2, lr=1:2
        if links[lr]==nothing
            push!(i_bp, 0.0)
        else
            s_bp = BlockSiteEntropy(T, [sites[q], links[lr]])
            push!(i_bp, s_p[q] + s_b[lr] - s_bp)
        end
    end
    
    # Compute FSWAP favorability (heuristic):
    fav = i_bp[2] + i_bp[3] - i_bp[1] - i_bp[4] 
    
    #println("\n  $(i_bp)  $(fav > 0.0)  \n")
    
    return fav
    
end


function TestFSWAP4(T, sites, links)
    
    decomp = OneHotTensors(T)
    
    tvec = [scalar(T*dag(phid)) for phid in decomp]
    
    fav = 0.0
    
    # Sample in the vicinity of the solution vector:
    for k=1:10
        
        nvec = normalize(tvec + 0.02*randn(length(tvec)))
        
        nT = sum([nvec[k]*decomp[k] for k=1:length(decomp)])
        
        fav += TestFSWAP3(nT, sites, links)
        
    end
    
    return fav
    
end

function TestFSWAP5(sdata, j, p, T, linds)
    
    new_phi = deepcopy(sdata.phi_list[j])
    
    U,S,V = svd(T, linds)
    new_phi[p] = U
    new_phi[p+1] = S*V
    
    S1, S2, Ipq = MutualInformation(new_phi, sdata.ord_list[j], sdata.chem_data)
    
    # Use Ipq heuristic to decide whether to swap or not:
    
    fav = 0.0
    
    # Test with sites on the left:
    for q in setdiff(1:sdata.chem_data.N_spt, [p,p+1])
        if q < p
            fav -= Ipq[p,q]*abs(p-q)
            fav += Ipq[p+1,q]*abs(p+1-q)
        else
            fav += Ipq[p,q]*abs(p-q)
            fav -= Ipq[p+1,q]*abs(p+1-q)
        end
    end
    
    return fav
    
end


function GMutInf(theta, T, links, sites, p)
    
    site_inds = inds(T, tags="Site")
    G = RotationTensor(site_inds, 1; dim=4, rotype="givens", theta=theta[1])
    
    T = noprime(T*G)
    
    # left/right block entropies:
    s_b = []
    
    for lr = 1:2
        if links[lr]==nothing
            push!(s_b, 0.0)
        else
            push!(s_b, BlockSiteEntropy(T, [links[lr]]))
        end
    end
    
    # p/p+1 site entropies:
    s_p = [BlockSiteEntropy(T, [sites[p:p+1][q]]) for q=1:2]
    
    # Block/site mutual information:
    i_bp = []
    
    for (q, lr) in [(1,2), (2,1)]
        if links[lr]==nothing
            push!(i_bp, 0.0)
        else
            s_bp = BlockSiteEntropy(T, [sites[p:p+1][q], links[lr]])
            push!(i_bp, s_p[q] + s_b[lr] - s_bp)
        end
    end
    
    # Cost function:
    cost = i_bp[1] + i_bp[2]
    
    return cost
    
end


function GTruncErr(theta, T, maxdim, linds, sites, p; dim=4)
    
    site_inds = inds(T, tags="Site")
    G = RotationTensor(site_inds, 1; dim=4, rotype="givens", theta=theta[1])
    
    T = noprime(T*G)
    
    U,S,V = svd(T, linds)
    sigma = diag(Array(S, inds(S)))
    m = minimum([length(sigma), maxdim])
    fid = sum(reverse(sort(sigma))[1:m].^2)
    
    return abs(1.0-fid)
    
end


function GThetaOpt(T, sdata, i, p; heuristic="trunc", beta=0.1)
    
    # Get left and right link indices:
    linds = commoninds(T, sdata.phi_list[i][p])
    llink = uniqueinds(linds, commoninds(T, sdata.phi_list[i][p], tags="Site"))
    if length(llink)==0
        llink = [nothing]
    end

    rinds = commoninds(T, sdata.phi_list[i][p+1])
    rlink = uniqueinds(rinds, commoninds(T, sdata.phi_list[i][p+1], tags="Site"))
    if length(rlink)==0
        rlink = [nothing]
    end
    
    f1(theta) = GTruncErr(
        theta,
        T,
        sdata.mparams.mps_maxdim,
        linds,
        sdata.sites,
        p
    )
    
    f2(theta) = GMutInf(
        theta,
        T, 
        [llink[1], rlink[1]],
        sdata.sites,
        p
    )
    
    f3(theta) = (1.0-beta)*f1(theta) + beta*f2(theta)
    
    if heuristic=="trunc"
        
        #Non-linear optimization of the rotation angle theta:
        res = Optim.optimize(
            f1, 
            [0.0],
            LBFGS(),
            Optim.Options(
                iterations=10000,
                show_trace=false
            )
        )
        
    elseif heuristic=="mutinf"
        
        #Non-linear optimization of the rotation angle theta:
        res = Optim.optimize(
            f2, 
            [0.0],
            LBFGS(),
            Optim.Options(
                iterations=10000,
                show_trace=false
            )
            #NumDimensions=1, 
            #Method=:xnes,
            #SearchRange = (-pi, pi), 
            #MaxFuncEvals=1000,
            #TraceMode=:silent
            #TraceMode=:compact
        )
        
    elseif heuristic=="mix"
        
        #Non-linear optimization of the rotation angle theta:
        res = Optim.optimize(
            f3, 
            [0.0],
            LBFGS(),
            Optim.Options(
                iterations=10000,
                show_trace=false
            )
            #NumDimensions=1, 
            #Method=:xnes,
            #SearchRange = (-pi, pi), 
            #MaxFuncEvals=1000,
            #TraceMode=:silent
            #TraceMode=:compact
        )
        
    end
    
    theta_opt = Optim.minimizer(res)
    fid_opt = Optim.minimum(res)
    
    #theta_opt = best_candidate(res)
    #fid_opt = best_fitness(res)
    
    return theta_opt[1]
    
end