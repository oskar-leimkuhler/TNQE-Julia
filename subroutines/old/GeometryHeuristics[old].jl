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