# Functions for solving the generalized eigenvalue problem:

# Packages:
using LinearAlgebra


# Solve the generalized eigenvalue problem HC = SCE with a given thresholding procedure:
function SolveGenEig(H_in, S_in; thresh="none", eps=1e-12)
    
    H = Hermitian(H_in)
    S = Hermitian(S_in)
    M = size(S,1)
    
    if thresh=="none"
        
        # No thresholding:
        fact = eigen(H, S)
        E = fact.values
        C = fact.vectors
        
        kappa = cond(S)
        
    elseif thresh=="projection"
        
        # Projection-based thresholding:
        F = svd(S)
        t = sum(F.S .> eps)
        
        U = zeros(Float64, M, M)
        U[:,1:t] = F.U[:,1:t]

        H_thresh = transpose(U) * H * U
        S_thresh = transpose(U) * S * U
        
        fact = eigen(H_thresh, S_thresh)
        E = fact.values
        C = fact.vectors
        for i=1:M
            C[:,i] = C[:,i]/sqrt(transpose(C[:,i]) * S_thresh * C[:,i])
        end
        
        kappa = maximum(F.S[F.S .> eps])/minimum(F.S[F.S .> eps])
        
    elseif thresh=="inversion"
        
        # Inversion-based thresholding:
        F = svd(S)
        t = sum(F.S .> eps)

        sig_inv = [1.0/sig for sig in F.S[F.S .> eps]]
        
        S_inv = F.V[:,1:t] * Diagonal(sig_inv) * transpose(F.U[:,1:t])
        
        fact = eigen(S_inv * H)
        E = fact.values
        C = fact.vectors
        
        kappa = maximum(F.S[F.S .> eps])/minimum(F.S[F.S .> eps])
        
    end
    
    return E, C, kappa
    
end


function EigCondNums(E, C)
    
    eig_cond_nums = []
    
    for j=1:length(E)
        
        kappa_j = norm(C[:,j])^2/sqrt(E[j]^2+1)
        
        push!(eig_cond_nums, kappa_j)
        
    end
    
    return eig_cond_nums
    
end


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