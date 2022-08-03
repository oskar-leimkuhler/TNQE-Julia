# Functions to compute the mutual information for a given wavefunction ansatz

# Packages:
using LinearAlgebra


# Globally declaring the I-N operator:
ITensors.op(::OpName"ImN",::SiteType"Fermion") = [1 0; 0 0]
local_ops = ["ImN", "N"]


# One- or two-orbital von neumann entropy function:
function vnEntropy(rdm)
    
    evals = eigvals(rdm)
    
    entropy = 0.0
    
    for sig in evals
        entropy -= sig*log(sig)
    end
    
    return entropy
end


# Computes the mutual information:
function MutualInformation(psi, chemical_data)
    
    # One-site local expectation vectors and two-site correlation matrices:
    expect_vecs = []
    corr_mats = []

    for i=1:2
        push!( expect_vecs, expect(psi, local_ops[i]) )

        for j=i:2
            push!( corr_mats, correlation_matrix(psi, local_ops[i], local_ops[j]) )
        end
    end


    # One- and two-orbital entropies:
    S1 = Array{Float64}(undef, chemical_data.N)

    for p=1:chemical_data.N
        rdm_p = [expect_vecs[1][p] 0.0; 0.0 expect_vecs[2][p]]
        S1[p] = vnEntropy(rdm_p)
    end

    S2 = Array{Float64}(undef,chemical_data.N,chemical_data.N)

    for p=1:chemical_data.N
        for q=p+1:chemical_data.N
            rdm_pq = Diagonal([corr_mats[1][p,q], corr_mats[2][p,q], corr_mats[2][q,p], corr_mats[3][p,q]])
            S2[p,q] = vnEntropy(rdm_pq)
            S2[q,p] = S2[p,q]
        end
    end

    # The mutual information:
    Ipq = Array{Float64}(undef,chemical_data.N,chemical_data.N)

    for p=1:chemical_data.N
        for q=1:chemical_data.N
            if p==q
                Ipq[p,q] = 0.0
            else
                Ipq[p,q] = 0.5*(S1[p]+S1[q]-S2[p,q])
            end
        end
    end
    
    return Ipq
    
end