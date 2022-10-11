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


# Returns a "temperature"-dependent acceptance probability P(e, e_new, T):
function AcceptanceProbability(e, e_new, temp; tpow=3, greedy=false)
    
    if e_new < e
        P=1.0
    elseif greedy==false
        P=temp^tpow
    else
        P=0.0
    end
    
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

            swap_prob = AcceptanceProbability(e, e_new, temp, tpow=tpow, greedy=greedy)

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