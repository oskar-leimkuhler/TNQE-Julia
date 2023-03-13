# Functions to find MPS geometries:


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


# Runs a simulated annealing heuristic to find an approximately optimal geometry:
function InfDistAnnealing(
        Ipq_in; 
        M=1, 
        eta=-2, 
        constrained_edges=[], 
        weight=1.0, 
        swap_mult=3.0, 
        steps=1e4, 
        tpow=5, 
        return_inf=false, 
        greedy=false
    )
    
    N_sites = size(Ipq_in, 1)
    
    Ipq = deepcopy(Ipq_in)
    
    for edge in constrained_edges
        Ipq[edge[1],edge[2]] = weight
        Ipq[edge[2],edge[1]] = weight
    end
    
    temp = 1.0
    temp_step = temp/steps
    
    ord = randperm(N_sites)
    e = InfDist(ord, Ipq, eta=eta)
    
    for s=1:steps
        
            
        # Number of applied swaps to generate a new ordering \\
        # ... (sampled from an exponential distribution):
        num_swaps = Int(ceil(swap_mult*randexp()[1]))

        ord_c = deepcopy(ord)

        # Apply these swaps randomly:
        for swap=1:num_swaps
            swap_ind = rand(1:N_sites-1)
            ord_c[swap_ind:swap_ind+1]=reverse(ord_c[swap_ind:swap_ind+1])
        end

        e = InfDist(ord_c, Ipq, eta=eta)

        swap_prob = PolyProb(e, e_new, temp, tpow=tpow, greedy=greedy)

        if rand()[1] <= swap_prob
            ord = ord_c
            e = e_new
        end
        
        temp -= temp_step
        
    end
    
    if return_inf
        return ord, e
    else
        return ord
    end
        
    
end


# Parameters for the geometry heuristic algorithms:
@with_kw mutable struct GeomParameters 
    
    # Maximum number of iterations:
    a_maxiter::Int=1000 # Annealing iterations
    g_maxiter::Int=40 # Genetic alg. iterations
    
    # Annealing parameters:
    a_alpha::Float64=1e-1 # Sharpness of cutoff
    stun::Bool=true # Use stochastic tunnelling?
    a_gamma::Float64=1e2 # Stun parameter
    swap_mult::Float64=3.0 # "Swappiness" (i.e. step size)
    
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
    
    # Miscellaneous:
    anchor::Int=0 # Exclude some of the states?
    return_all::Bool=false # Return the full population?
    
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
function ComputeBipartites(
        dmrg_subspace; 
        state=1, 
        cuts=10, 
        ns=2, 
        verbose=false
    )
    
    mpm = dmrg_subspace.mparams
    cdata = dmrg_subspace.chem_data
    
    biparts1 = SingleSiteBipartitions(cdata.N_spt)
    biparts2 = TwoSiteBipartitions(cdata.N_spt)
    biparts3 = ThreeSiteBipartitions(cdata.N_spt)
    
    bipartitions = reduce(vcat, [biparts1, biparts2, biparts3])
    
    cuts = length(bipartitions)
    
    entropies = []
    for l=1:cuts
        push!(entropies, BipartiteEntanglement(bipartitions[l], dmrg_subspace.psi_list[state], dmrg_subspace.ord_list[state]))
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
            N_AB = [length(ord_inA), length(ord_inB)]
            
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
        
        S_est = entropies[l]
        
        # The cost function calculation:
        fitness += maximum([0, S_est - zeta*S_max])
        
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