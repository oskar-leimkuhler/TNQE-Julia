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
        
        U = F.U[:,1:t]

        H_thresh = transpose(U) * H * U
        S_thresh = transpose(U) * S * U
        
        fact = eigen(H_thresh, S_thresh)
        E_thresh = fact.values
        C_thresh = fact.vectors
        
        kappa = maximum(F.S[F.S .> eps])/minimum(F.S[F.S .> eps])
        
        E = zeros(Float64, M)
        E[1:t] = E_thresh
        
        C = zeros(Float64, M, M)
        C[:,1:t] = U * C_thresh
        
        for i=1:t
            C[:,i] = C[:,i]/sqrt(transpose(C[:,i]) * S * C[:,i])
        end
        
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
        
        for i=1:t
            C[:,i] = C[:,i]/sqrt(transpose(C[:,i]) * S * C[:,i])
        end
        
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
