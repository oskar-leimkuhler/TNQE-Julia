# Functions to find MPS geometries:

# Packages:
# using LinearAlgebra


# Information distance measure for a given site ordering:
function InfDist(ord, Ipq; eta=-2)
    
    N_sites = size(ord, 1)
    
    # Index of each site:
    inds = [findall(x->x==i, ord)[1] for i=1:N_sites]
    
    inf_dist = 0.0
    
    for p=1:N_sites
        for q=p+1:N_sites
            inf_dist += sign(eta)*Ipq[p,q]*(float(abs(inds[p]-inds[q]))^eta)
        end
    end
    
    return inf_dist
    
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


# Runs a simulated annealing heuristic to find an approximately optimal geometry:
function SimulatedAnnealing(Ipq_in; M=1, cost_func="standard", gamma=0.5, eta=-2, constrained_edges=[], weight=1.0, swap_mult=3.0, steps=1e4, tpow=5, return_inf=false, greedy=false)
    
    N_sites = size(Ipq_in, 1)
    
    Ipq = deepcopy(Ipq_in)
    
    for edge in constrained_edges
        Ipq[edge[1],edge[2]] = weight
        Ipq[edge[2],edge[1]] = weight
    end
    
    temp = 1.0
    temp_step = temp/steps
    
    ord_list = [randperm(N_sites) for j=1:M]
    e = CostFunc(ord_list, Ipq, cost_func, eta, gamma)
    #e = InfDist(ord, Ipq)
    
    for s=1:steps
        
        # Try permuting each ordering in the list:
        for (j,ord) in enumerate(ord_list)
            
            # Number of applied swaps to generate a new ordering (sampled from an exponential distribution):
            num_swaps = Int(ceil(swap_mult*randexp()[1]))

            ord_listc = deepcopy(ord_list)

            # Apply these swaps randomly:
            for swap=1:num_swaps
                swap_ind = rand(1:N_sites-1)
                ord_listc[j][swap_ind:swap_ind+1]=reverse(ord_listc[j][swap_ind:swap_ind+1])
            end
            
            e_new = CostFunc(ord_listc, Ipq, cost_func, eta, gamma)
            #e_new = InfDist(ordc, Ipq)

            swap_prob = PolyProb(e, e_new, temp, tpow=tpow, greedy=greedy)

            if rand()[1] <= swap_prob
                ord_list = ord_listc
                e = e_new
            end
            
        end
        
        temp -= temp_step
        
    end
    
    if M==1 && return_inf
        return ord_list[1], e
    elseif return_inf
        return ord_list, e
    elseif M==1
        return ord_list[1]
    else
        return ord_list
    end
        
    
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
function BipartiteEntanglement(partition, psi, ord; tol=1e-8, maxdim=512, locdim=4)
    
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
    
    orthogonalize!(perm_psi,cut)
    
    temp_tensor = (perm_psi[cut] * perm_psi[cut+1])
    
    temp_inds = uniqueinds(perm_psi[cut],perm_psi[cut+1])
    
    U,S,V = svd(temp_tensor,temp_inds,cutoff=tol,maxdim=maxdim)
    
    sigma = Array(S.tensor)
    
    vn_entropy = -sum([sigma[j,j]^2*log(sigma[j,j]^2) for j=1:size(sigma,1)])
    
    return vn_entropy
    
end


# Computes the entanglements over bipartitions
# of a (single-state) DMRG subspace
function ComputeBipartites(dmrg_subspace; cuts=10, ns=2, verbose=false)
    
    mpm = dmrg_subspace.mparams
    cdata = dmrg_subspace.chem_data
    
    biparts1 = SingleSiteBipartitions(cdata.N_spt)
    biparts2 = TwoSiteBipartitions(cdata.N_spt)
    biparts3 = ThreeSiteBipartitions(cdata.N_spt)
    
    bipartitions = reduce(vcat, [biparts1, biparts2, biparts3])
    
    cuts = length(bipartitions)
    
    entropies = []
    for l=1:cuts
        push!(entropies, BipartiteEntanglement(bipartitions[l], dmrg_subspace.psi_list[1], dmrg_subspace.ord_list[1]))
    end
    
    return bipartitions, entropies
    
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
        S_max2 = minimum([length(A_sites),length(B_sites)])*ln4
        S_max = minimum([xi*S_max1,S_max2])
        
        S_est = entropies[l]
        
        fitness += maximum([0, S_est - zeta*S_max])
        
    end
    
    fitness *= 10.0/cuts
    
    return fitness
    
end


# Optimize a set of orderings given a set of bipartite entropies:
function BipartiteAnnealing(ord_list, bipartitions, entropies, maxdim; anchor=false, anum=1, maxiter=1000, swap_mult=3.0, alpha=1.0, stun=false, gamma=10.0, zeta = 0.01, xi=0.4, opt_statevec=false, delta=0.01, verbose=false)
    
    M = length(ord_list)
    
    statevec = [1.0/sqrt(M) for j=1:M]
    
    f = BipartiteFitness(ord_list, bipartitions, entropies, maxdim, statevec=statevec, zeta=zeta, xi=xi)
    N_sites = length(ord_list[1])
    f_best = f
    
    steps = floor(maxiter/M)
    
    for s=1:maxiter
        
        if anchor
            ord_listc = deepcopy(ord_list[(anum+1):end])
        else
            ord_listc = deepcopy(ord_list)
        end
        
        # Try permuting each ordering in the list:
        for (j,ord) in enumerate(ord_listc)
            
            # Number of applied swaps to generate a new ordering (sampled from an exponential distribution):
            num_swaps = Int(ceil(swap_mult*randexp()[1]))

            # Apply these swaps randomly:
            for swap=1:num_swaps
                swap_ind = rand(1:N_sites-1)
                ord_listc[j][swap_ind:swap_ind+1]=reverse(ord_listc[j][swap_ind:swap_ind+1])
            end
        end
        
        if anchor
            anchor_ords = [ord_list[j] for j=1:anum]
            ord_listc = vcat(anchor_ords, ord_listc)
        end
        
        if opt_statevec
            statevec += delta*randn(M)
            normalize!(statevec)
        end
            
        f_new = BipartiteFitness(ord_listc, bipartitions, entropies, maxdim, statevec=statevec, zeta=zeta, xi=xi)

        # Accept move with some probability
        beta = alpha*s

        if stun
            F_0 = Fstun(f_best, f, gamma)
            F_1 = Fstun(f_best, f_new, gamma)
            P = ExpProb(F_1, F_0, beta)
        else
            P = ExpProb(f, f_new, beta)
        end
        
        #println(f)
        #println(f_new)
        #println(P)

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
    
    if opt_statevec
        return ord_list, statevec
    else
        return ord_list
    end
    
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


# Genetic algorithm to optimize the geometry:
function BipartiteGenetics(M, N_sites, bipartitions, entropies, maxdim; maxiter=10000, maxpop=100, alpha=1.0, beta=0.5, gamma=0.1, zeta = 0.1, xi=0.3, shufl=false, verbose=false)
    
    function f(population)
        
        fitnesses = [BipartiteFitness(
                population[p], 
                bipartitions, 
                entropies, 
                maxdim, 
                zeta=zeta, 
                xi=xi
                ) for p=1:maxpop]
        
        return fitnesses
    end
    
    pop = [[randperm(N_sites) for j=1:M] for p=1:maxpop]
    
    # Evaluate the fitnesses
    fits = f(pop)
    best_fit = minimum(fits)
    worst_fit = maximum(fits)
    
    for s=1:maxiter
        
        # Stochastically select pairs and breed
        ballbag = []
        
        totballs = 0
        
        # Fill the ballbag:
        for p=1:maxpop
            numballs = Int(10*floor(exp(-alpha*(fits[p]-best_fit)/(worst_fit-best_fit))))
            ballbag = vcat(ballbag, [p for q=1:numballs])
            totballs += numballs
        end
        
        newpop = []
        
        for q=1:maxpop-1
            
            # Choose parents (asexual reproduction allowed):
            p1 = ballbag[rand(1:totballs)]
            p2 = ballbag[rand(1:totballs)]
            
            parent1 = deepcopy(pop[p1])
            if shufl
                parent2 = deepcopy(shuffle(pop[p2]))
            else
                parent2 = deepcopy(pop[p1])
            end
            
            # Breed the parents:
            child = parent1
            
            for j=1:M
                if rand()[1] < beta
                    child[j] = parent2[j]
                end
            end
            
            push!(newpop, child)
            
        end
        
        # Apply random mutations
        for q=1:maxpop-1
            
            # Try permuting each ordering in the list:
            for (j,ord) in enumerate(newpop[q])

                # Number of applied swaps to generate a new ordering (sampled from an exponential distribution):
                num_swaps = Int(ceil(gamma*randexp()[1]))

                # Apply these swaps randomly:
                for swap=1:num_swaps
                    swap_ind = rand(1:N_sites-1)
                    ord[swap_ind:swap_ind+1]=reverse(ord[swap_ind:swap_ind+1])
                end
            end
            
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
    
    ord_list = pop[p_best]
    
    return ord_list
    
end