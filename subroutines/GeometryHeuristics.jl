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
    
    bipartitions = SymmetricBipartitions(cdata.N_spt, ns)
    cuts = length(bipartitions)
    
    entropies = []
    for l=1:cuts
        push!(entropies, BipartiteEntanglement(bipartitions[l], dmrg_subspace.psi_list[1], dmrg_subspace.ord_list[1]))
    end
    
    return bipartitions, entropies
    
end


# Compute the "fitness" as a function of orderings, bipartitions, and ground-state entropies:
function BipartiteFitness(ord_list, bipartitions, entropies, maxdim; zeta=0.01)
    
    cuts = length(bipartitions)
    
    fitness = 0.0
    
    for l=1:cuts
        
        xi_l = 0
        
        count = 0
        
        A_sites, B_sites = bipartitions[l]
        cut = length(A_sites)
        
        for j=1:length(ord_list)
            
            n1 = length(setdiff(A_sites, ord_list[j][1:cut]))
            n2 = length(setdiff(A_sites, ord_list[j][end-cut:end]))
            
            n_swaps = minimum([n1, n2])
            
            if n_swaps > xi_l
                xi_l = n_swaps 
                count = 1
            elseif n_swaps == xi_l
                count += 1
            end
            
        end
        
        #xi_l *= 1.0/length(ord_list)
        
        S_max = log(count) + log(maxdim) + xi_l*log(16)
        S_est = entropies[l]
        
        fitness += maximum([0, S_est - zeta*S_max])
        
    end
    
    fitness *= 10.0/cuts
    
    return fitness
    
end


# Optimize a set of orderings given a set of bipartite entropies:
function BipartiteAnnealing(ord_list, bipartitions, entropies, maxdim; maxiter=1000, swap_mult=3.0, alpha=1.0, stun=false, gamma=10.0, zeta = 0.01, verbose=false)
    
    f = BipartiteFitness(ord_list, bipartitions, entropies, maxdim, zeta=zeta)
    N_sites = length(ord_list[1])
    f_best = f
    
    M = length(ord_list)
    
    steps = floor(maxiter/M)
    
    for s=1:maxiter
        
        ord_listc = deepcopy(ord_list)
        
        # Try permuting each ordering in the list:
        for (j,ord) in enumerate(ord_list)
            
            # Number of applied swaps to generate a new ordering (sampled from an exponential distribution):
            num_swaps = Int(ceil(swap_mult*randexp()[1]))

            # Apply these swaps randomly:
            for swap=1:num_swaps
                swap_ind = rand(1:N_sites-1)
                ord_listc[j][swap_ind:swap_ind+1]=reverse(ord_listc[j][swap_ind:swap_ind+1])
            end
        end
            
        f_new = BipartiteFitness(ord_listc, bipartitions, entropies, maxdim, zeta=zeta)

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
            ord_list = ord_listc
            f = f_new
        end

        if verbose && (div(s*M,100)>div((s-1)*M,100))
            print("$(f)    \r")
            flush(stdout)
        end
        
    end
    
    return ord_list
    
end