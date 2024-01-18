# Functions for solving the generalized eigenvalue problem:

# Packages:
using LinearAlgebra


# Solve the generalized eigenvalue problem HC = SCE with a given thresholding procedure:
function SolveGenEig(H_in, S_in; thresh="none", eps=1e-12)
    
    H = Symmetric(H_in)
    S = Symmetric(S_in)
    M = size(S,1)
    
    # Check that S is positive semidefinite...
    # if not, set to the closest positive semidefinite matrix:
    if !isposdef(S)
        F = eigen(S)
        Lambda = deepcopy(F.values)
        for l=1:length(Lambda)
            if Lambda[l] < 1e-16
                Lambda[l] = 1e-16
            end
        end
        S = F.vectors * Diagonal(Lambda) * transpose(conj(F.vectors))
    end
    
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

        H_thresh = transpose(conj(U)) * H * U
        S_thresh = transpose(conj(U)) * S * U
        
        fact = eigen(H_thresh, S_thresh)
        E_thresh = fact.values
        C_thresh = fact.vectors
        
        kappa = maximum(F.S[F.S .> eps])/minimum(F.S[F.S .> eps])
        
        E = zeros(Float64, M)
        E[1:t] = E_thresh
        
        C = zeros(Float64, M, M)
        C[:,1:t] = U * C_thresh
        
        for i=1:t
            sfac = abs(transpose(conj(C[:,i])) * S * C[:,i])
            if sfac > 1e-16
                C[:,i] .*= (1.0/sqrt(sfac))
            end
        end
        
    elseif thresh=="inversion"
        
        # Inversion-based thresholding:
        F = svd(S)
        t = sum(F.S .> eps)

        sig_inv = [1.0/sig for sig in F.S[F.S .> eps]]
        
        S_inv = F.V[:,1:t] * Diagonal(sig_inv) * transpose(conj(F.U[:,1:t]))
        
        fact = eigen(S_inv * H)
        E = fact.values
        C = fact.vectors
        
        kappa = maximum(F.S[F.S .> eps])/minimum(F.S[F.S .> eps])
        
        for i=1:t
            sfac = abs(transpose(conj(C[:,i])) * S * C[:,i])
            if sfac > 1e-16
                C[:,i] .*= (1.0/sqrt(sfac))
            end
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
