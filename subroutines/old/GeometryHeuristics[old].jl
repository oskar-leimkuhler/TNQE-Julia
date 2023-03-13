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