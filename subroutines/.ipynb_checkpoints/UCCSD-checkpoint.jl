using SparseArrays
using LinearAlgebra
using Optim
using BlackBoxOptim
using Random
using FastExpm
using NLopt

# Jordan-Wigner operators constructed as kronecker products:
function JWOp(N, alist, eta_proj)
    
    # List of 2x2 matrices for each spin-orbital:
    pstr = [sparse([1 0; 0 1]) for r=1:N]
    
    # Creation and annihilation operators:
    fermop = [sparse([0 0; 1 0]), sparse([0 1; 0 0])]
    
    for d=1:2, p in alist[d]
        #a-dag/a on p
        pstr[p] = pstr[p] * fermop[d]
        # Pauli Z-string for fermionic antisymmetry:
        for t=1:p-1
            pstr[t] = pstr[t] * sparse([1 0; 0 -1])
        end 
    end
    
    return eta_proj * sparse(reduce(kron, pstr)) * eta_proj
    
end


# Trotterized uCCSD ansatz state in sparse format:
function tuCCSD(N, eta, T_list, eta_proj)
    
    N_spt = Int(N/2)
    
    N_occ = Int(eta/2)
    N_vir = N_spt-N_occ
    
    # The Hartree-Fock bitstring vector:
    hf_vec = vcat([1 for i=1:2*N_occ], [0 for a=1:2*N_vir])
    hf_str = reduce(*, string.(hf_vec))
    hf_row = parse(Int,hf_str;base=2)+1
    
    psi = spzeros(2^N)
    psi[hf_row] = 1.0
    
    # Singles terms
    cs = 0
    for p=1:N_vir, q=1:N_occ
        cs += 1
        for sigma=0:1
            t_op = T_list[1][cs]*JWOp(N,[[N_occ+2*p+sigma-1],[2*q+sigma-1]],eta_proj)
            t_ah = t_op - transpose(t_op)
            if norm(t_ah) > 1e-18
                 psi = transpose(fastExpm(t_ah)) * psi
            end
        end
    end
    
    # Doubles terms
    cd = 0
    for p=1:N_vir, q=p:N_vir, r=1:N_occ, s=r:N_occ
        cd += 1
        for sigma=0:1, tau=0:1
            t_op = T_list[2][cd]*JWOp(N,[[N_occ+2*p+sigma-1,N_occ+2*q+tau-1],[2*r+tau-1,2*s+sigma-1]],eta_proj)
            t_ah = t_op-transpose(t_op)
            if norm(t_ah) > 1e-18
                psi = transpose(fastExpm(t_ah)) * psi
            end
        end
    end
    
    return psi
    
end

function HMatrix(cdata; verbose=false)
    
    N = cdata.N
    
    N_spt = cdata.N_spt
    
    H_mat = sparse(zeros(2^N, 2^N))
    
    # Project onto the eta-subspace to save on computation:
    eta_vec = spzeros(2^N)
    for b=1:2^N
        eta_vec[b] = Int(sum(digits(b-1, base=2))==cdata.N_el)
    end
    eta_proj = sparse(diagm(eta_vec))
    
    # Singles terms
    for sigma=0:1, p=1:N_spt, q=1:N_spt
        if abs(cdata.h1e[p,q]) > 1e-15
            H_mat += cdata.h1e[p,q]*JWOp(N,[[2*p+sigma-1],[2*q+sigma-1]], eta_proj)
        end
    end
    
    verbose && println("\nComputing doubles terms:")
    
    # Doubles terms
    dcount = 1
    for sigma=0:1, tau=0:1, p=1:N_spt, q=1:N_spt, r=1:N_spt, s=1:N_spt
        if abs(cdata.h2e[p,s,q,r]) > 1e-15
            if sigma==tau
                if p!=q && r!=s
                    H_mat += 0.5*cdata.h2e[p,s,q,r]*JWOp(N,[[2*p+sigma-1,2*q+tau-1],[2*r+tau-1,2*s+sigma-1]], eta_proj)
                end
            else
                H_mat += 0.5*cdata.h2e[p,s,q,r]*JWOp(N,[[2*p+sigma-1,2*q+tau-1],[2*r+tau-1,2*s+sigma-1]], eta_proj)
            end
        end
        verbose && print("Progress: $(dcount)/$(4*N_spt^4)   \r")
        verbose && flush(stdout)
        dcount += 1
    end
    
    verbose && println("\nDone!\n")
    
    return H_mat, eta_proj
    
end

function uCCEval(x, N, eta, e_nuc, H_mat, eta_proj)
    
    N_spt = Int(N/2)
    N_occ = Int(eta/2)
    N_vir = N_spt - N_occ
    
    # Slice and reshape from input vector x:
    ns = N_vir*N_occ
    
    T1 = x[1:ns]
    
    T2 = x[ns+1:end]
    
    #T_list = [T1_a, T1_b, T2_aa, T2_ab, T2_ba, T2_bb];
    T_list = [T1, T2]
    
    # Construct Trotterized uCCSD unitary by fastExpm:
    psi = tuCCSD(N, eta, T_list, eta_proj)
    
    # The energy estimate:
    E_uCCSD = real(transpose(psi) * H_mat * psi) + e_nuc
    
    return E_uCCSD
    
end


function MP2Amps(cdata)
    
    N = cdata.N
    eta = cdata.N_el
    N_spt = Int(N/2)
    N_occ = Int(eta/2)
    N_vir = N_spt - N_occ
    
    h = cdata.h2e
    e = cdata.e_mo
    
    t = []
    
    mp2_en = 0.0
    
    for p=N_occ+1:N_spt, q=p:N_spt, r=1:N_occ, s=r:N_occ
        
        num = -(h[p,s,q,r] - h[q,s,p,r])
        denom = (e[p] + e[q] - e[r] - e[s])
        if abs(denom) > 1e-13
            push!(t, num/denom)
            mp2_en += num^2/(4*denom)
        end
        
    end
    
    return t, mp2_en
    
end


function uCCSDMinimize(chemical_data; maxiter=50, eps_init=1e-3, eps=0.0)
    
    H_mat, eta_proj = HMatrix(chemical_data)
    
    N = chemical_data.N
    eta = chemical_data.N_el
    N_spt = Int(N/2)
    N_occ = Int(eta/2)
    N_vir = N_spt - N_occ
    
    e_nuc = chemical_data.e_nuc
    
    #nparam = 2*N_vir*N_occ + 4*N_vir^2*N_occ^2
    nparam = N_vir*N_occ + N_vir^2*N_occ^2
    
    
    function f2(x, grad)
        
        E = uCCEval(
            x,
            N,
            eta,
            e_nuc,
            H_mat,
            eta_proj
        )
        
        print("$E\r")
        flush(stdout)
        
        return E
        
    end
    
    #hnorm = sqrt(norm(chemical_data.h1e, 2)^2 + norm(chemical_data.h2e, 2)^2)
    #hnorm = norm(chemical_data.h1e, 1) + norm(chemical_data.h2e, 1)
    
    eval_count = 0
    
    function f(x)
        
        eval_count += 1
    
        return uCCEval(
            x,
            N,
            eta,
            e_nuc,
            H_mat,
            eta_proj
        ) + eps*randn()
        
    end
    
    # Initial guess from MP2 amps:
    
    #t2 = reshape(permutedims(chemical_data.t2, [3,4,1,2]), (N_vir^2*N_occ^2))
    t2, _ = MP2Amps(chemical_data)
    x0 = vcat(zeros(N_vir*N_occ), 4.0*t2)
    nparam = length(x0)
    x0 = zeros(length(x0))
    
    #x0 = eps_init*randn(Float64, nparam)
    
    println(uCCEval(
        x0,
        N,
        eta,
        chemical_data.e_nuc,
        H_mat,
        eta_proj
    ))
    
    # Optimize!
    res = Optim.optimize(
        f, 
        x0,
        LBFGS(),
        Optim.Options(
            iterations=maxiter,
            show_trace=true
        )
    )
    
    x_opt = Optim.minimizer(res)
    e_opt = Optim.minimum(res)
    
    return e_opt, x_opt, eval_count
    
    """
    res = bboptimize(
        f, 
        x0,
        NumDimensions=nparam, 
        Method=:xnes,
        SearchRange = (-pi, pi), 
        MaxFuncEvals=1000, 
        TraceMode=:compact
    )
    
    x_opt = best_candidate(res)
    e_opt = best_fitness(res)

    """
    
    """
    opt = Opt(:LN_COBYLA, length(x0))
    #opt.lower_bounds = [-Inf, 0.]
    #opt.xtol_rel = 1e-4

    min_objective!(opt, (x,grad) -> f2(x, grad))

    (e_opt,x_opt,ret) = NLopt.optimize(opt, x0)
    numevals = opt.numevals # the number of function evaluations
    
    return e_opt, x_opt
    """
    
end