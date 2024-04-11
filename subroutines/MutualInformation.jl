# Functions to compute the mutual information and site ordering heuristics


# One- or two-orbital von neumann entropy function:
function vnEntropy(rdm; tol=1e-16)
    
    evals = eigvals(Hermitian(rdm))
    
    evals = evals[evals .> tol]
    
    entropy = 0.0
    
    for sig in evals
            entropy -= sig*log(sig)
    end
    
    return entropy
end


function kRDM(phi, p_list)
    
    k = length(p_list)
    
    d = dim(siteinds(phi)[1])
    
    phi2 = dag(phi)
    
    for p in p_list
        setprime!(phi2[p], 1, tags="Site")
    end
    
    T_rdm = ITensor(1.0)
    
    for p=1:length(phi2)
        
        T_rdm *= phi[p]
        T_rdm *= setprime(phi2[p], 1, tags="Link")
        
    end
    
    ids = vcat(inds(T_rdm, plev=0), inds(T_rdm, plev=1))
    
    A_rdm = reshape(Array(T_rdm, ids), (d^k, d^k))
    
    return A_rdm
    
end


# Computes the mutual information:
function MutualInformation(phi::MPS, ord, chemical_data; dim=4, spatial=true)
    
    orb = invperm(ord)
    
    if spatial==true
        N_sites = chemical_data.N_spt
    else
        N_sites = chemical_data.N
    end

    # One- and two-orbital entropies:
    S1 = zeros(N_sites)
    
    for p=1:N_sites
        
        rdm_p = kRDM(phi, [p])

        S1[ord[p]] = vnEntropy(rdm_p)
        
    end

    S2 = zeros((N_sites,N_sites))
    
    for p=1:N_sites
        for q=p+1:N_sites
            
            rdm_pq = kRDM(phi, [p,q])
            
            S2[ord[p],ord[q]] = vnEntropy(rdm_pq)
            S2[ord[q],ord[p]] = S2[ord[p],ord[q]]
        end
    end

    # The mutual information:
    Ipq = Array{Float64}(undef,N_sites,N_sites)
    
    for p=1:N_sites
        for q=1:N_sites
            if p==q
                Ipq[p,q] = 0.0
            else
                Ipq[p,q] = S1[p]+S1[q]-S2[p,q]
            end
        end
    end
    
    return S1, S2, Ipq
    
end


"""
# Computes the mutual information:
function MutualInformation(sdata::SubspaceProperties)
    
    orb = invperm(ord)
    
    if spatial==true
        N_sites = chemical_data.N_spt
    else
        N_sites = chemical_data.N
    end

    # One- and two-orbital entropies:
    S1 = zeros(N_sites)
    
    for p=1:N_sites
        
        rdm_p = kRDM(phi, [p])

        S1[ord[p]] = vnEntropy(rdm_p)
        
    end

    S2 = zeros((N_sites,N_sites))
    
    for p=1:N_sites
        for q=p+1:N_sites
            
            rdm_pq = kRDM(phi, [p,q])
            
            S2[ord[p],ord[q]] = vnEntropy(rdm_pq)
            S2[ord[q],ord[p]] = S2[ord[p],ord[q]]
        end
    end

    # The mutual information:
    Ipq = Array{Float64}(undef,N_sites,N_sites)
    
    for p=1:N_sites
        for q=1:N_sites
            if p==q
                Ipq[p,q] = 0.0
            else
                Ipq[p,q] = S1[p]+S1[q]-S2[p,q]
            end
        end
    end
    
    return S1, S2, Ipq
    
"""

# Parameters for the site ordering heuristic algorithm:
@with_kw mutable struct GeomParameters 
    
    # Maximum number of iterations:
    maxiter::Int=1000 # Annealing iterations
    
    # Annealing parameters:
    afunc::String="stun" # "exp", "stun", "step", or "poly"
    a_alpha::Float64=1e-1 # Sharpness of cutoff
    a_gamma::Float64=1e2 # Stun parameter
    swap_mult::Float64=3.0 # "Swappiness" (i.e. step size)
    a_tpow::Float64=5.0 # For the polyprob function
        
    # Information-distance parameters:
    cweight::Float64=1.0 # Constrained edge weight
    eta::Int=-2 # "eta" parameter in information distance function
    shrp::Float64=3.0 # Sharpness of "LogSumExp" function
    
    # Miscellaneous:
    anchor::Int=0 # Exclude some of the states?
    return_all::Bool=false # Return the full population?
    
end

function LogSumExp(alpha, x)
    
    xmax = maximum(x)
    
    if minimum(x) <= 0
        println("Error: xmin <= 0")
        return nothing
    end
    
    exptot = 0.0
    
    for j=1:length(x)
        exptot += exp(alpha*x[j]/xmax)
    end
    
    return xmax/alpha*log(exptot)
    
end

# Information distance measure for a given site ordering:
function InfDist(ord_list, Ipq; eta=-2, shrp=3.0)
    
    M = length(ord_list)
    N = length(ord_list[1])
    
    
    inds = []
    for ord in ord_list
        # Index of each site:
        push!(inds, [findall(x->x==p, ord)[1] for p=1:N])
    end
    
    inf_dist = 0.0
    
    for p=1:N
        for q=p+1:N
            
            nu_pq = [float(abs(inds[j][p]-inds[j][q]))^eta for j=1:M]
            
            inf_dist += sign(eta)*Ipq[p,q]*LogSumExp(shrp, nu_pq)
            
        end
    end
    
    return inf_dist
    
end

# Runs a simulated annealing heuristic to find an approximately optimal geometry:
function InfDistAnnealing(
        Ipq_in,
        M,
        gp; 
        ord_list=nothing,
        constrained_edges=[], 
        anchors=[],
        return_inf=false,
        verbose=false
    )
    
    N_sites = size(Ipq_in, 1)
    
    Ipq = deepcopy(Ipq_in)
    
    for edge in constrained_edges
        Ipq[edge[1],edge[2]] = gp.cweight
        Ipq[edge[2],edge[1]] = gp.cweight
    end
    
    if ord_list==nothing
        ord_list = [randperm(N_sites) for j=1:M]
    end
    e = InfDist(ord_list, Ipq, eta=gp.eta, shrp=gp.shrp)
    e_best = e
    
    for s=1:gp.maxiter
        
        ord_list_c = deepcopy(ord_list)
        
        for (j,ord) in enumerate(ord_list_c)
            if !(j in anchors)
                # Number of applied swaps to generate a new ordering \\
                # ... (sampled from an exponential distribution):
                num_swaps = Int(ceil(gp.swap_mult*randexp()[1]))

                # Apply these swaps randomly:
                for swap=1:num_swaps
                    swap_ind = rand(1:N_sites-1)
                    ord[swap_ind:swap_ind+1]=reverse(ord[swap_ind:swap_ind+1])
                end
            end
        end

        e_new = InfDist(ord_list_c, Ipq, eta=gp.eta, shrp=gp.shrp)
        
        temp = 1.0 - s/gp.maxiter
        beta = gp.a_alpha/temp
        
        if gp.afunc=="step"
            P = StepProb(e, e_new)
        elseif gp.afunc=="exp"
            P = ExpProb(e, e_new, beta)
        elseif gp.afunc=="stun"
            f0 = Fstun(e_best, e, gp.a_gamma)
            f1 = Fstun(e_best, e_new, gp.a_gamma)
            P = ExpProb(f1, f0, beta)
        elseif gp.afunc=="poly"
            P = PolyProb(e, e_new, temp, tpow=gp.tpow)
        else
            println("Invalid probability function!")
            return nothing
        end
        
        if e_new < e_best
            e_best = e_new
        end

        if rand()[1] <= P
            ord_list = ord_list_c
            e = e_new
        end
        
        if verbose && (div(s,100)>div((s-1),100))
            print("$(e)    \r")
            flush(stdout)
        end
        
    end
    
    if return_inf
        return ord_list, e
    else
        return ord_list
    end
        
    
end