# Functions to compute the mutual information for a given wavefunction ansatz

# Packages:
using LinearAlgebra


# Returns a list of relevant matrix element pairs for Hilbert space dimension 2 (fermion sites) or 4 (electron sites)
function OrbitalPairs(dim)
    
    pair_list = []
    ind_list = []
    
    if dim==2
        
        push!(pair_list, [1,1])
        push!(pair_list, [2,2],[2,3],[3,3])
        push!(pair_list, [4,4])
        
        ind_list = pair_list
        
    elseif dim==4
        
        push!(pair_list, [1,1])
        push!(pair_list, [1,6],[2,5],[6,1])
        push!(pair_list, [1,11],[3,9],[11,1])
        push!(pair_list, [6,6])
        push!(pair_list, [1,16],[2,15],[3,14],[4,13],[6,11],[7,10],[8,9],[11,6],[12,5],[16,1])
        push!(pair_list, [11,11])
        push!(pair_list, [6,16],[8,14],[16,6])
        push!(pair_list, [11,16],[15,12],[16,11])
        push!(pair_list, [16,16])
        
        push!(ind_list, [1,1])
        push!(ind_list, [2,2],[2,3],[3,3])
        push!(ind_list, [4,4],[4,5],[5,5])
        push!(ind_list, [6,6])
        push!(ind_list, [7,7],[7,8],[7,9],[7,10],[8,8],[8,9],[8,10],[9,9],[9,10],[10,10])
        push!(ind_list, [11,11])
        push!(ind_list, [12,12],[12,13],[13,13])
        push!(ind_list, [14,14],[14,15],[15,15])
        push!(ind_list, [16,16])
        
    else
        
        ArgumentError("Local Hilbert space dimension must be 2 or 4, not $dim")
        
    end
    
    return pair_list, ind_list
    
end


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


# Computes the mutual information:
function MutualInformation(psi_in, chemical_data; dim=4, spatial=true)
    
    if spatial==true
        N_sites = chemical_data.N_spt
    else
        N_sites = chemical_data.N
    end
    
    psi = dense(psi_in)
    
    pair_list, ind_list = OrbitalPairs(dim)
    
    # One-site local expectation vectors and two-site correlation matrices:
    expect_vecs = []
    corr_mats = []
    
    # One-orbital RDMs:
    for i=1:dim
        op_mat = zeros((dim,dim))
        op_mat[i,i] = 1.0
        push!( expect_vecs, expect(psi, op_mat) )
    end
    
    # Two-orbital RDMs:
    for pair in pair_list
        op_mat1 = reshape(Matrix(I(dim^2))[pair[1],:], (dim,dim))
        op_mat2 = reshape(Matrix(I(dim^2))[pair[2],:], (dim,dim))
        push!( corr_mats, correlation_matrix(psi, op_mat1, op_mat2) )
    end

    # One- and two-orbital entropies:
    S1 = zeros(N_sites)
    
    for p=1:N_sites
        
        rdm_p = zeros((dim,dim))
        
        for i=1:dim
            rdm_p[i,i] = expect_vecs[i][p]
        end

        S1[p] = vnEntropy(rdm_p)
        
    end

    S2 = zeros((N_sites,N_sites))
    
    for p=1:N_sites
        for q=p+1:N_sites
            
            rdm_pq = zeros((dim^2,dim^2))
            
            for (idx,pair) in enumerate(ind_list)
                rdm_pq[pair[1],pair[2]] = corr_mats[idx][p,q]
                rdm_pq[pair[2],pair[1]] = corr_mats[idx][p,q]
            end
            
            S2[p,q] = vnEntropy(rdm_pq)
            S2[q,p] = S2[p,q]
        end
    end

    # The mutual information:
    Ipq = Array{Float64}(undef,N_sites,N_sites)
    
    for p=1:N_sites
        for q=1:N_sites
            if p==q
                Ipq[p,q] = 0.0
            else
                Ipq[p,q] = 0.5*(S1[p]+S1[q]-S2[p,q])
            end
        end
    end
    
    return Ipq
    
end