# Finds a set of orderings optimized according to the highest priority site chains:
# (n=1 for edges, n=2 for "elbows" etc.)
function PriorityChains(Ipq, M, n)
    
    N_sites = size(Ipq,1)
    
    Ind = zeros(Tuple([N_sites for d=1:n+1]))
    
    for inds in pairs(Ind)
        if length(Set(Tuple(inds[1])))==length(Tuple(inds[1])) # unique indices
            Ind[inds[1]] = sum([Ipq[inds[1][p],inds[1][p+1]] for p=1:n])
        end
    end
    
    priority_chains = []
    
    for i=1:M
        
        max_inds = Tuple(findmax(Ind)[2])
        
        push!(priority_chains, max_inds)
        
        Ind[CartesianIndex(Tuple(max_inds))] = 0.0
        Ind[CartesianIndex(Tuple(reverse(max_inds)))] = 0.0
        
    end
    
    return priority_chains
    
end


# Takes a distribution of states over chain sizes and returns a set of complete site orderings:
function PriorityOrderings(dist, Ipq; prior_ords=[], remove_duplicates=true, steps=1e4, swap_mult=3.0, tpow=5)
    
    ord_list = prior_ords
    
    for (n,Mn) in enumerate(dist)
    
        chain_list = PriorityChains(Ipq, Mn, n)

        chain_ords = []

        for chain in chain_list
            ord = SimulatedAnnealing(Ipq, constrained_edges=[chain[p:p+1] for p=1:n], steps=steps, swap_mult=swap_mult, tpow=tpow)
            push!(chain_ords, ord)
        end
        
        if remove_duplicates
            chain_ords = chain_ords[(!in).(chain_ords,Ref(ord_list))]
        end

        ord_list = vcat(ord_list, chain_ords)

    end
    
    return ord_list
    
end


# Finds all orderings that are k swaps away:
function KSwapOrderings(ord0, k; existing_ords=[ord0])
    
    if k==0
        permuted_ords = [ord0]
    else
        permuted_ords = []
        for i=1:length(ord0)-1
            new_ord = deepcopy(ord0)
            new_ord[i]=ord0[i+1]
            new_ord[i+1]=ord0[i]
            new_ords = KSwapOrderings(new_ord, k-1; existing_ords=existing_ords)
            permuted_ords = vcat(permuted_ords, new_ords[(!in).(new_ords,Ref(existing_ords))])
        end
        push!(existing_ords, ord0)
    end
        
    return permuted_ords
    
end



function BipartiteScreening(ord_list, bipartitions, entropies, maxdim; M_new=1, maxiter=10000, alpha=1.0, stun=false, gamma=10.0, zeta = 0.1, xi=0.3, verbose=false)
    
    M = length(ord_list)
    N_sites = length(ord_list[1])
    
    f = BipartiteFitness(ord_list, bipartitions, entropies, maxdim, zeta=zeta, xi=xi)
    f_best = f
    
    for s=1:maxiter
        
        #new_ords = [randperm(N_sites) for j=1:M_new]
        
        new_ords = [copy(ord_list[rand(1:M)]) for j=1:M_new]
        
        # Apply swaps randomly:
        for j=1:M_new
            swap_ind = rand(1:N_sites-1)
            new_ords[j][swap_ind:swap_ind+1]=reverse(new_ords[j][swap_ind:swap_ind+1])
        end
        
        new_ord_list = vcat(ord_list, new_ords)
        
        red_ord_lists = combinations(new_ord_list, M)
        
        for red_ords in red_ord_lists
            
            f_new =  BipartiteFitness(red_ords, bipartitions, entropies, maxdim, zeta=zeta, xi=xi)
            
            # Accept move with some probability
            beta = alpha*s
            
            if stun
                F_0 = Fstun(f_best, f, gamma)
                F_1 = Fstun(f_best, f_new, gamma)
                P = ExpProb(F_1, F_0, beta)
            else
                P = ExpProb(f, f_new, beta)
            end
            
            if f_new < f_best
                f_best = f_new
            end
            
            if rand()[1] < P
                
                ord_list = red_ords
                f = f_new
                
            end
            
        end
        
        if verbose && (div(s,1)>div((s-1),1))
            print("$(f)    \r")
            flush(stdout)
        end
        
    end
    
    return ord_list
end


# Information average-distance measure for a set of orderings:
function InfAvDist(ord_list, Ipq; eta=-2)
    
    N_sites = size(ord_list[1], 1)
    M = size(ord_list, 1)
    
    inf_dist = 0.0
    
    av_dists = zeros((N_sites, N_sites))
    
    for (j,ord) in enumerate(ord_list)
        
        inds = [findall(x->x==i, ord)[1] for i=1:N_sites]
        
        for p=1:N_sites
            for q=p+1:N_sites
                av_dists[p,q]+=float(abs(inds[p]-inds[q]))/float(M)
            end
        end
    end
    
    for p=1:N_sites
        for q=p+1:N_sites
            
            inf_dist += sign(eta)*Ipq[p,q]*(av_dists[p,q]^eta)
            
        end
    end
    
    return inf_dist
    
end


# Information minimum-distance measure for a set of orderings:
function InfMinDist(ord_list, Ipq; eta=-2)
    
    N_sites = size(ord_list[1], 1)
    M = size(ord_list, 1)
    
    inf_dist = 0.0
    
    min_dists = zeros((N_sites, N_sites))
    fill!(min_dists, float(N_sites))
    
    for (j,ord) in enumerate(ord_list)
        
        inds = [findall(x->x==i, ord)[1] for i=1:N_sites]
        
        for p=1:N_sites
            for q=p+1:N_sites
                if abs(inds[p]-inds[q]) < min_dists[p,q]
                    min_dists[p,q]=float(abs(inds[p]-inds[q]))
                end
            end
        end
    end
    
    for p=1:N_sites
        for q=p+1:N_sites
            
            inf_dist += sign(eta)*Ipq[p,q]*(min_dists[p,q]^eta)
            
        end
    end
    
    return inf_dist
    
end


# Selects the cost function and applies it to the ordering list:
function CostFunc(ord_list, Ipq, cost_func, eta, gamma)
    
    e = 0.0
    
    if cost_func=="standard"
        for ord in ord_list
            e += InfDist(ord, Ipq, eta=eta)
        end
        e *= 1.0/size(ord_list, 1)
        
    elseif cost_func=="avdist"
        e = InfAvDist(ord_list, Ipq, eta=eta)
        
    elseif cost_func=="mindist"
        e = InfMinDist(ord_list, Ipq, eta=eta)
        
    elseif cost_func=="mixdist"
        e += gamma*InfMinDist(ord_list, Ipq, eta=eta)
        e += (1.0-gamma)*InfAvDist(ord_list, Ipq, eta=eta)
        
    end
    
    return e
    
end



# Compute the "fitness" as a function of orderings, bipartitions, and ground-state entropies:
function BipartiteFitness(ord_list, bipartitions, entropies, maxdim; statevec=nothing, zeta=0.01, xi=0.4)
    
    cuts = length(bipartitions)
    M = length(ord_list)
    
    if statevec == nothing
        statevec = [1.0/sqrt(M) for j=1:M]
    end
    
    # Pre-compute some quantities for speed:
    ln4 = log(4)
    lnm = log(maxdim)
    
    fitness = 0.0
    
    for l=1:cuts
        
        k_l = 0
        
        count = 0
        
        A_sites, B_sites = bipartitions[l]
        cut = length(A_sites)
        
        N_p = minimum([length(A_sites), length(B_sites)])
        
        m_list = []
        k_j_list = []
        
        for j=1:M
            
            #n1 = length(setdiff(A_sites, ord_list[j][1:cut]))
            #n2 = length(setdiff(A_sites, ord_list[j][end-cut:end]))
            
            N_sites = length(ord_list[j])
            
            # Generate the sorted ordering
            ord_inA = []
            ord_inB = []
            for x in ord_list[j]
                if x in A_sites
                    push!(ord_inA, x)
                elseif x in B_sites
                    push!(ord_inB, x)
                end
            end
            ord_sorted = vcat(ord_inA, ord_inB)
            
            # Generate the permutation network(s)
            swap_network1 = BubbleSort(ord_list[j], ord_sorted)
            swap_network2 = BubbleSort(ord_list[j], reverse(ord_sorted))
            
            swap_networks = [swap_network1, swap_network2]
            cut_list = [cut, N_sites-cut]
            
            ppow_list = []
            #pdim_list = []
            
            # Try both sorted orderings and compare:
            for c=1:2
                
                swap_network = swap_networks[c]
                cutc = cut_list[c]
                
                # Generate the list of bond powers
                bpow_list = [0 for p=1:N_sites-1]

                #dim_list = [minimum([maxdim, 4^N_p]) for p=1:N_sites-1]

                # Apply the swap network to the list of bond powers
                for swap_site in swap_network
                    if swap_site != 1 && swap_site != N_sites-1
                        bpow_list[swap_site] = minimum([bpow_list[swap_site+1],bpow_list[swap_site-1]]) + 1
                        #dim_list[swap_site] = minimum([dim_list[swap_site+1], dim_list[swap_site-1]]) * 4
                    else
                        #dim_list[swap_site] = 4
                    end
                end
                
                push!(ppow_list, bpow_list[cutc])
                #push!(pdim_list, dim_list[cutc])
                
            end
            
            #ppow = minimum([bpow_list[cut], bpow_list[end-cut+1]])
            
            """
            if ppow > k_l
                k_l = ppow 
                count = 1
            elseif ppow == k_l
                count += 1
            end
            """
            
            #m_j = minimum([maxdim*4^ppow, 4^N_p])
            #m_j = maxdim*4^ppow
            #m_j = minimum(pdim_list)
            
            #push!(m_list, m_j)
            push!(k_j_list, minimum(ppow_list))
            
        end
        
        #xi_l *= 1.0/length(ord_list)
        """
        S_max1 = log(count) + log(maxdim) + k_l*log(4)
        S_max2 = minimum([length(A_sites),length(B_sites)])*log(4)
        S_max = minimum([S_max1,S_max2])
        """
        
        S_max1 = 0.0
        for j=1:M
            cj2 = statevec[j]^2
            S_max1 += cj2*( ln4*k_j_list[j] + lnm - log(cj2) )
        end
        #S_max2 = minimum([length(A_sites),length(B_sites)])*ln4
        #S_max = minimum([xi*S_max1,S_max2])
        S_max = S_max1
        
        S_est = entropies[l]
        
        fitness += maximum([0, S_est - zeta*S_max])
        
    end
    
    fitness *= 10.0/cuts
    
    return fitness
    
end



# Generates a set of random bipartitions of the sites:
function RandomBipartitions(N, n)
    
    bipartitions = []
    
    for l=1:n
        
        sites = randperm(N)
        
        cut = rand(1:N-1)
        
        #A_sites = sort(sites[1:cut])
        #B_sites = sort(sites[cut+1:end])
        A_sites = sites[1:cut]
        B_sites = sites[cut+1:end]
        
        push!(bipartitions, [A_sites,B_sites])
        
    end
    
    return bipartitions
    
end


# Generates all single-site bipartitions:
function SingleSiteBipartitions(N)
    bipartitions = []
    
    for p=1:N
        A_sites = [p]
        B_sites = setdiff(collect(1:N), A_sites)
        push!(bipartitions, [A_sites, B_sites])
    end
    
    return bipartitions
end


# Generates all two-site bipartitions:
function TwoSiteBipartitions(N)
    bipartitions = []
    
    for p=1:N
        for q=(p+1):N
            A_sites = [p,q]
            B_sites = setdiff(collect(1:N), A_sites)
            push!(bipartitions, [A_sites, B_sites])
        end
    end
    
    return bipartitions
end


# Generates all three-site bipartitions:
function ThreeSiteBipartitions(N)
    bipartitions = []
    
    for p=1:N
        for q=(p+1):N
            for r=(q+1):N
                A_sites = [p,q,r]
                B_sites = setdiff(collect(1:N), A_sites)
                push!(bipartitions, [A_sites, B_sites])
            end
        end
    end
    
    return bipartitions
end


# Generates a set of bipartitions with rotational symmetry around the site ring:
function SymmetricBipartitions(N, ns)
    
    bipartitions = []
    
    for i=1:Int(floor(N/2))
        
        nmax = factorial(N-1)/factorial(N-i) 
        
        for s=1:minimum([nmax,ns])
            
            sites = randperm(N)
            
            pattern = sites[1:i]
        
            for p=1:N
            
                A_sites = [mod1(p+q, N) for q in pattern]
                B_sites = setdiff(collect(1:N), A_sites)
            
                push!(bipartitions, [A_sites,B_sites])
                
            end
            
        end
        
    end
    
    return bipartitions
    
end


# Generates O(NlogN) "maximal" (i.e. split roughly 50-50 into A,B) \\
# ... bipartitions (by symmetry about the ring)
function LogarithmicBipartitions(N)
    
    bipartitions = []
    
    sites = collect(1:N)
    
    clN = ceil(log2(N))
    
    eclN = 2^clN
    
    for k=1:clN
        
        reps = 2^(k-1)
        
        rep_unit = vcat([true for i=1:reps], [false for j=1:reps])
        
        partition_string = reduce(vcat, [rep_unit for i=1:Int(eclN/(2*reps))])[1:N]
        
        for r=0:k
            rotated_string = circshift(partition_string,r)
            A_sites = sites[rotated_string]
            B_sites = sites[.!rotated_string]
            push!(bipartitions, [A_sites, B_sites])
        end
        
    end
    
    return bipartitions
    
end


# Computes the entanglement of the ground state over a given partition
# (approximated by DMRG)
function BipartiteEntanglement(
        partition, 
        psi, 
        ord; 
        tol=1e-8, 
        maxdim=512, 
        locdim=4
    )
    
    p_ord = vcat(partition[1],partition[2])
    cut = length(partition[1])
    
    perm_psi = Permute(
        psi, 
        siteinds(psi), 
        ord,
        p_ord, 
        tol=tol, 
        maxdim=maxdim, 
        locdim=locdim
    )
    
    truncate!(perm_psi, tol=1e-6)
    orthogonalize!(perm_psi,cut)
    
    temp_tensor = (perm_psi[cut] * perm_psi[cut+1])
    
    temp_inds = uniqueinds(perm_psi[cut],perm_psi[cut+1])
    
    U,S,V = svd(temp_tensor,temp_inds,cutoff=tol,maxdim=maxdim, alg="qr_iteration")
    
    sigma = Array(S.tensor)
    
    vn_entropy = -sum([sigma[j,j]^2*log(sigma[j,j]^2) for j=1:size(sigma,1)])
    
    return vn_entropy
    
end


function ChainMaxEls(A)
    
    min_el = minimum(A)
    
    max_vals, max_inds = findmax(A, dims=1)
    max_val, i = findmax(max_vals)
    p, q = Tuple(max_inds[i])
    
    # Set the rows p, q to be lower than the minimum element:
    A[p,:] .= min_el - 1.0
    A[q,:] .= min_el - 1.0
    
    tot = max_val
    
    for step = 1:size(A,1)-2
        # Look down the columns to find the maximal row index:
        max1, r1 = findmax(A[:,p])
        max2, r2 = findmax(A[:,q])
        
        if max1 > max2
            p = r1
            A[p,:] .= min_el - 1.0
            tot += max1
        else
            q = r2
            A[q,:] .= min_el - 1.0
            tot += max2
        end
        
    end
    
    
    return tot
    
end


# A "first-order" estimate of the block-entropy from \\
# (...) single-site entropies and two-site QMI:
function QMIBlockEntropy(partition, S_1, I_2)
    
    A_sites = partition[1]
    A_vol = length(A_sites)
    
    S_est = 0.0
    
    for p in A_sites
        S_est += S_1[p]
    end
    
    I_2A = zeros((A_vol, A_vol))
    for p=1:A_vol, q=1:A_vol
        I_2A[p,q] = I_2[A_sites[p], A_sites[q]]
    end
    
    # "Chain" the mutual information:
    I_sum = ChainMaxEls(I_2A)
    
    S_est -= I_sum
    
    return S_est
    
end


# Computes the entanglements over bipartitions
# of a (single-state) DMRG subspace
function ComputeBipartites(
        dmrg_subspace; 
        state=1,  
        verbose=false
    )
    
    if verbose
        println("Computing QMI...")
    end
    
    S1, S2, Ipq = MutualInformation(
        dmrg_subspace.psi_list[state], 
        dmrg_subspace.chem_data
    )
    
    if verbose
        println("Done!\n")
    end
    
    biparts1 = SingleSiteBipartitions(dmrg_subspace.chem_data.N_spt)
    biparts2 = TwoSiteBipartitions(dmrg_subspace.chem_data.N_spt)
    biparts3 = ThreeSiteBipartitions(dmrg_subspace.chem_data.N_spt)
    
    bipartitions = reduce(vcat, [biparts1, biparts2, biparts3])
    
    cuts = length(bipartitions)
    
    entropies = []
    for l=1:cuts
        
        #push!(entropies, BipartiteEntanglement(bipartitions[l], dmrg_subspace.psi_list[state], dmrg_subspace.ord_list[state]))
        
        #push!(entropies, QMIBlockEntropy(bipartitions[l], S1, Ipq))

        rdm_l = kRDM(dmrg_subspace.psi_list[state], bipartitions[l][1])
        push!(entropies, vnEntropy(rdm_l))
        
    end
    
    return bipartitions, entropies
    
end


# Compute the "fitness" as a function of orderings, bipartitions, and ground-state entropies:
function BipartiteFitness(
        ord_list, 
        bipartitions, 
        entropies, 
        maxdim; 
        statevec=nothing, 
        zeta=0.01
    )
    
    # Number of bipartitions ("AB-cuts")
    cuts = length(bipartitions)
    
    M = length(ord_list)
    
    if statevec == nothing
        statevec = [1.0/sqrt(M) for j=1:M]
    end
    
    # Pre-compute some quantities for speed:
    ln4 = log(4)
    lnm = log(maxdim)
    
    fitness = 0.0
    
    # Compute the fitness across each "cut" and add to the total:
    for l=1:cuts
        
        A_sites, B_sites = bipartitions[l]
        
        N_AB = [length(A_sites), length(B_sites)]
        
        # The "power" list (constants k_AB[j])
        k_AB = []
        
        for j=1:M
            
            N = length(ord_list[j])
            
            # Generate the sorted ordering
            ord_inA = []
            ord_inB = []
            for x in ord_list[j]
                if x in A_sites
                    push!(ord_inA, x)
                elseif x in B_sites
                    push!(ord_inB, x)
                end
            end
            ord_sort1 = vcat(ord_inA, ord_inB)
            ord_sort2 = vcat(ord_inB, ord_inA)
            
            # Generate the permutation networks:
            swap_network1 = BubbleSort(ord_list[j], ord_sort1)
            swap_network2 = BubbleSort(ord_list[j], ord_sort2)
            
            swap_networks = [swap_network1, swap_network2]
            
            # The "partition power" list:
            ppow_list = []
            
            # Try both sorted orderings and compare:
            for c=1:2
                
                swap_network = swap_networks[c]
                p_cut = N_AB[c]
                
                # Generate the list of "bond powers"
                bpow_list = [0 for p=1:N-1]

                # Apply the swap network to the list of bond powers
                for p_swap in swap_network
                    if p_swap != 1 && p_swap != N-1
                        bpow_list[p_swap] = minimum([bpow_list[p_swap+1],bpow_list[p_swap-1]]) + 1
                    end
                end
                
                push!(ppow_list, bpow_list[p_cut])
                
            end
            
            push!(k_AB, minimum(ppow_list))
            
        end
        
        # The maximal entropy calculation:
        S_max = 0.0
        for j=1:M
            cj2 = statevec[j]^2
            S_max += cj2*( ln4*k_AB[j] + lnm - log(cj2) )
        end
        
        #S_max = minimum([S_max, ln4*minimum(N_AB)])
        
        S_est = entropies[l]
        
        # The cost function calculation:
        fitness += maximum([0, S_est - zeta*S_max])
        #fitness += S_est*(1.0 - S_max/(ln4*minimum(N_AB)))
        
    end
    
    # Renormalize with respect to the number of bipartitions:
    fitness *= 10.0/cuts
    
    return fitness
    
end


function CompositeCostFunc(
        ord_list, 
        bipartitions, 
        entropies, 
        maxdim, 
        statevec, 
        gp
    )
    
    M = length(ord_list)
    
    if gp.wt==0.0
        cost=0.0
    else
        cost = BipartiteFitness(
            ord_list, 
            bipartitions[1], 
            entropies[1], 
            maxdim, 
            statevec=statevec, 
            zeta=gp.zeta_list[1]
        )

        cost -= gp.baseline[1]
        cost *= gp.wt
        cost *= 1.0/gp.baseline[1]
    end
    
    for j=1:length(ord_list)
        
        cost_add = BipartiteFitness(
            [ord_list[j]], 
            bipartitions[j], 
            entropies[j], 
            maxdim, 
            statevec=nothing, 
            zeta=gp.zeta_list[j]
        )
        
        cost_add -= gp.baseline[2]
        cost_add *= (1.0-gp.wt)
        cost_add *= 1.0/(gp.baseline[2]*M)
        
        cost += cost_add
        
    end
    
    return cost
    
end


# Optimize a set of orderings given a set of bipartite entropies:
function BipartiteAnnealing(
        ord_list, 
        bipartitions, 
        entropies, 
        maxdim, 
        gp; 
        statevec=nothing, 
        verbose=false
    )
    
    M = length(ord_list)
    
    if statevec==nothing
        if gp.init_statevec==[]
            statevec = [1.0/sqrt(M) for j=1:M]
        else
            statevec = gp.init_statevec
        end
    end
    
    zeta = gp.zeta_list[1]
    
    if gp.costfunc=="simple"
        f = BipartiteFitness(
            ord_list, 
            bipartitions, 
            entropies, 
            maxdim, 
            statevec=statevec, 
            zeta=zeta
        )
    else
        
        f = CompositeCostFunc(
            ord_list, 
            bipartitions, 
            entropies, 
            maxdim, 
            statevec, 
            gp
        )
        
    end
    
    N_sites = length(ord_list[1])
    f_best = f
    
    steps = floor(gp.a_maxiter/M)
    
    for s=1:gp.a_maxiter
        
        ord_listc = deepcopy(ord_list[(gp.anchor+1):end])
        
        # Try permuting each ordering in the list:
        for (j,ord) in enumerate(ord_listc)
            
            # Number of applied swaps to generate a new ordering (sampled from an exponential distribution):
            num_swaps = Int(ceil(gp.swap_mult*randexp()[1]))

            # Apply these swaps randomly:
            for swap=1:num_swaps
                swap_ind = rand(1:N_sites-1)
                ord_listc[j][swap_ind:swap_ind+1]=reverse(ord_listc[j][swap_ind:swap_ind+1])
            end
        end
        
        anchor_ords = [ord_list[j] for j=1:gp.anchor]
        ord_listc = vcat(anchor_ords, ord_listc)
        
        if gp.opt_statevec
            statevec += gp.delta*randn(M)
            normalize!(statevec)
        end
            
        if gp.costfunc=="simple"
            f_new = BipartiteFitness(
                ord_listc, 
                bipartitions, 
                entropies, 
                maxdim, 
                statevec=statevec, 
                zeta=zeta
            )
        else
            f_new = CompositeCostFunc(
                ord_listc, 
                bipartitions, 
                entropies, 
                maxdim, 
                statevec, 
                gp
            )
        end

        # Accept move with some probability
        beta = gp.a_alpha*s

        if gp.stun
            F_0 = Fstun(f_best, f, gp.a_gamma)
            F_1 = Fstun(f_best, f_new, gp.a_gamma)
            P = ExpProb(F_1, F_0, beta)
        else
            P = ExpProb(f, f_new, beta)
        end

        if f_new < f_best
            f_best = f_new
        end

        if rand()[1] < P
            ord_list = ord_listc
            f = f_new
        end

        if verbose && (div(s*M,100)>div((s-1)*M,100))
            print("$(f)    \r")
            flush(stdout)
        end
        
    end
    
    if gp.opt_statevec
        return ord_list, statevec
    else
        return ord_list
    end
    
end


# Genetic algorithm to optimize the geometry:
function BipartiteGenetics(
        M, 
        N_sites, 
        bipartitions, 
        entropies, 
        maxdim, 
        gp; 
        pop_in=nothing, 
        verbose=false
    )
    
    if gp.init_statevec==[]
        statevec = [1.0/sqrt(M) for j=1:M]
    else
        statevec = gp.init_statevec
    end
    
    zeta = gp.zeta_list[1]
    
    function f(population)
        
        if gp.costfunc=="simple"
        
            fitnesses = [BipartiteFitness(
                    population[p], 
                    bipartitions, 
                    entropies, 
                    maxdim, 
                    statevec=statevec,
                    zeta=zeta
                    ) for p=1:gp.maxpop]
            
        elseif gp.costfunc=="composite"
            
            fitnesses = [CompositeCostFunc(
                    population[p], 
                    bipartitions, 
                    entropies, 
                    maxdim, 
                    statevec,
                    gp
                    ) for p=1:gp.maxpop]
            
        end
        
        return fitnesses
    end
    
    if pop_in==nothing
        pop = [[randperm(N_sites) for j=1:M] for p=1:gp.maxpop]
    else
        pop = vcat(pop_in, [[randperm(N_sites) for j=1:M] for p=1:(gp.maxpop-length(pop_in))])
    end
    
    # Evaluate the fitnesses
    fits = f(pop)
    best_fit = minimum(fits)
    worst_fit = maximum(fits)
    
    for s=1:gp.g_maxiter
        
        # Stochastically select pairs and breed
        ballbag = []
        
        totballs = 0
        
        # Fill the ballbag:
        for p=1:gp.maxpop
            numballs = Int(10*floor(exp(-gp.g_alpha*(fits[p]-best_fit)/maximum([worst_fit-best_fit, 1e-6]))))
            ballbag = vcat(ballbag, [p for q=1:numballs])
            totballs += numballs
        end
        
        newpop = []
        
        # Divide new population into offspring and immigrants:
        n_offspr = maximum([Int(floor((1.0-gp.g_delta)*gp.maxpop))-1, 0])
        n_immigr = gp.maxpop-(n_offspr+1)
        
        for q=1:n_offspr
            
            # Choose parents (asexual reproduction allowed):
            p1 = ballbag[rand(1:totballs)]
            p2 = ballbag[rand(1:totballs)]
            
            parent1 = deepcopy(pop[p1])
            if gp.shufl
                parent2 = deepcopy(shuffle(pop[p2]))
            else
                parent2 = deepcopy(pop[p1])
            end
            
            # Breed the parents:
            child = parent1
            
            for j=1:M
                if rand()[1] < gp.g_beta
                    child[j] = parent2[j]
                end
            end
            
            # Apply random mutations
            # Try permuting each ordering in the list:
            for (j,ord) in enumerate(child)

                # Number of applied swaps to generate a new ordering (sampled from an exponential distribution):
                num_swaps = Int(ceil(gp.g_gamma*randexp()[1]))

                # Apply these swaps randomly:
                for swap=1:num_swaps
                    swap_ind = rand(1:N_sites-1)
                    ord[swap_ind:swap_ind+1]=reverse(ord[swap_ind:swap_ind+1])
                end
            end
            
            push!(newpop, child)
            
        end
        
        # Add immigrants:
        for q=1:n_immigr
            push!(newpop, [randperm(N_sites) for j=1:M])
        end
        
        # The fittest always survives:
        p_best = findmin(fits)[2]
        survivor = pop[p_best]
        
        # Replace the population:
        pop = vcat(newpop, [survivor])
        
        # Re-evaluate the fitnesses:
        fits = f(pop)
        best_fit = minimum(fits)
        worst_fit = maximum(fits)
        
        if verbose && (div(s,1)>div((s-1),1))
            print("$(best_fit)    \r")
            flush(stdout)
        end
        
    end
    
    # Select the fittest specimen:
    p_best = findmin(fits)[2]
    
    if gp.return_all
        return pop
    else
        ord_list = pop[p_best]
        return ord_list
    end
    
end


"""

# Optimize the state vector?
opt_statevec::Bool=false # Optimize Y/N
delta::Float64=0.3 # Step size
init_statevec::Vector{Float64}=[] # Init.

# Genetic alg. parameters:
maxpop::Int=40 # Population size
shufl=true # Shuffle the geometry order?
g_alpha::Float64=1.5 # Selective pressure
g_beta::Float64=0.5 # Degree of genetic mixing
g_gamma::Float64=0.1 # Random mutation rate
g_delta::Float64=0.3 # New random population (%)

# Entanglement entropy-based function patameters:
costfunc::String="simple" # "simple" or "composite"
zeta_list::Vector{Float64}=[0.03] # Entropy saturation parameters
wt::Float64=0.9 # Composite weight (ground vs. excited states)
baseline=[0.1,0.5] # Composite baseline parameters \\
# ... [multi-geom-ground, single-geom-excited]

"""