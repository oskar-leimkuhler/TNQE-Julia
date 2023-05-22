# Functions to compute the mutual information for a given wavefunction ansatz

# Packages:
using LinearAlgebra


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


function kRDM(psi, p_list)
    
    k = length(p_list)
    
    d = dim(siteinds(psi)[1])
    
    psi2 = dag(psi)
    
    for p in p_list
        setprime!(psi2[p], 1, tags="Site")
    end
    
    T_rdm = ITensor(1.0)
    
    for p=1:length(psi2)
        
        T_rdm *= psi[p]
        T_rdm *= setprime(psi2[p], 1, tags="Link")
        
    end
    
    ids = vcat(inds(T_rdm, plev=0), inds(T_rdm, plev=1))
    
    A_rdm = reshape(Array(T_rdm, ids), (d^k, d^k))
    
    return A_rdm
    
end


# Computes the mutual information:
function MutualInformation(psi, ord, chemical_data; dim=4, spatial=true)
    
    orb = invperm(ord)
    
    if spatial==true
        N_sites = chemical_data.N_spt
    else
        N_sites = chemical_data.N
    end

    # One- and two-orbital entropies:
    S1 = zeros(N_sites)
    
    for p=1:N_sites
        
        rdm_p = kRDM(psi, [p])

        S1[ord[p]] = vnEntropy(rdm_p)
        
    end

    S2 = zeros((N_sites,N_sites))
    
    for p=1:N_sites
        for q=p+1:N_sites
            
            rdm_pq = kRDM(psi, [p,q])
            
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