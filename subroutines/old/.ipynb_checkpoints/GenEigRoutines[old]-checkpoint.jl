# Modifies the generalized eigenvalue problem by including perturbing matrices:
# (only supports inversion currently)
function ModifiedGenEig(H_in, S_in, M_list, theta; eps=1E-12)
    
    H = Hermitian(H_in)
    S = Hermitian(S_in)
    
    F = svd(S)
    t = sum(F.S .> eps)

    sig_inv = [1.0/sig for sig in F.S[F.S .> eps]]

    S_inv = F.V[:,1:t] * Diagonal(sig_inv) * F.U[:,1:t]'
    
    M_f = theta[1]*S_inv*H
    
    for k=2:length(theta)
        M_f += theta[k]*M_list[k-1]
    end

    fact = eigen(M_f)
    E = fact.values
    C = fact.vectors

    kappa = maximum(F.S[F.S .> eps])/minimum(F.S[F.S .> eps])
    
    return E, C, kappa
    
end