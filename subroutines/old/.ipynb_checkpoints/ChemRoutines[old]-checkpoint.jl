# A more exact inner product (using the exact Hamiltonian)
function ExactInner(psi1, psi2, mpo_list, coeff_list)
    
    exp_val = 0.0
    
    for (m, mpo) in enumerate(mpo_list)
        exp_val += coeff_list[m]*inner(psi1', mpo, psi2)
    end
    
    return exp_val
    
end


# Apply a Jordan-Wigner operator to the operator list:
function ApplyJordanWignerOp(op_list, op, f_op, idx)
    
    op_list[idx] *= op
    
    for k =1:idx-1
        op_list[k] *= f_op
    end
    
    return op_list
    
end


function SingleOpList(N, inds)
    
    # Start with identity on all sites:
    op_list = [[1.0 0.0; 0.0 1.0] for p=1:N]
    
    # Apply c† on site p:
    op_list[inds[1]] *= [0.0 0.0; 1.0 0.0]
    
    # JW string:
    for k=1:inds[1]-1
        op_list[k] *= [1.0 0.0; 0.0 -1.0]
    end
    
    # Apply c on site q:
    op_list[inds[2]] *= [0.0 1.0; 0.0 0.0]
    
    # JW string:
    for k=1:inds[2]-1
        op_list[k] *= [1.0 0.0; 0.0 -1.0]
    end
    
    return op_list
    
end


function DoubleOpList(N, inds)
    
    # Start with identity on all sites:
    op_list = [[1 0; 0 1] for p=1:N]
    
    # Apply c† on site p:
    op_list[inds[1]] *= [0 0; 1 0]
    
    # JW string:
    for k=1:inds[1]-1
        op_list[k] *= [1 0; 0 -1]
    end
    
    # Apply c† on site r:
    op_list[inds[3]] *= [0 0; 1 0]
    
    # JW string:
    for k=1:inds[3]-1
        op_list[k] *= [1 0; 0 -1]
    end
    
    # Apply c on site s:
    op_list[inds[4]] *= [0 1; 0 0]
    
    # JW string:
    for k=1:inds[4]-1
        op_list[k] *= [1 0; 0 -1]
    end
    
    # Apply c on site q:
    op_list[inds[2]] *= [0 1; 0 0]
    
    # JW string:
    for k=1:inds[2]-1
        op_list[k] *= [1 0; 0 -1]
    end
    
    return op_list
    
end


function OpList2MPO(op_list, sites, dim, N)
    
    hmpo = MPO(sites)

    comInd = Index(1,"Link") # Define a common index  

    for p=1:N # Run over the chain of size N

        siteind = sites[p]

        if p==1

            j = Index(1,"Link")

            Hp = zeros((dim, dim, 1))

            # Here set the values of the matrix elements of Hp 
            for k=1:2, l=1:2
                Hp[k,l,1] = op_list[p][k,l]
            end

            hmpo[p] = ITensor(Hp, siteind, siteind', j)
            comInd = j # Set the common index to the right index of Hp

        elseif p>1 && p<chemical_data.N

            i = comInd
            j = Index(1,"Link");
            
            Hp = zeros((dim, dim, 1, 1))

            # Here set the values of the matrix elements of Hp 
            for k=1:2, l=1:2
                Hp[k,l,1,1] = op_list[p][k,l]
            end

            hmpo[p] = ITensor(Hp, siteind, siteind', i, j)
            comInd = j # Set the common index to the right index of A 


        elseif p==chemical_data.N

            i = comInd
            Hp = zeros((dim, dim, 1))
            
            # Here set the values of the matrix elements of Hp 
            for k=1:2, l=1:2
                Hp[k,l,1] = op_list[p][k,l]
            end

            hmpo[p] = ITensor(Hp, siteind, siteind', i)

        end

    end 
    
    return hmpo
    
end


function ExactMPOs(chemical_data, sites)
    
    N_spt = chemical_data.N_spt
    h1e = chemical_data.h1e
    h2e = chemical_data.h2e
    
    mpo_list = MPO[]
    coeff_list = Float64[]
    
    # Singles terms:
    for p=1:N_spt, q=1:N_spt, σ=1:2
        
        ip = 2*(p-1)+σ
        iq = 2*(q-1)+σ
        
        op_list = SingleOpList(2*N_spt, [ip,iq])
        
        mpo = OpList2MPO(op_list, sites, 2*N_spt)
        
        push!(mpo_list, mpo)
        
        push!(coeff_list, h1e[p,q])
        
    end
        
    # Doubles terms:
    for p=1:N_spt, q=1:N_spt, r=1:N_spt, s=1:N_spt, σ=1:2, τ=1:2
        
        ip = 2*(p-1)+σ
        iq = 2*(q-1)+σ
        ir = 2*(r-1)+τ
        is = 2*(s-1)+τ
        
        if ip!=ir && is!=iq
            
            op_list = DoubleOpList(2*N_spt, [ip,iq,ir,is])
        
            mpo = OpList2MPO(op_list, sites, 2*N_spt)
            
            push!(mpo_list, mpo)
            
            push!(coeff_list, 0.5*h2e[p,q,r,s])
            
        end
        
    end
    
    return coeff_list, mpo_list
    
end



function ApplySingleSiteOp(psi, sites, op_name, idx)
    
    orthogonalize!(psi,idx)
    G = op(op_name, siteinds(psi)[idx])
    newA = psi[idx] * G
    noprime!(newA)
    psi[idx] = newA
    return psi
    
end


function ApplyHamString(psi_in, sites_in, op_string, op_inds; spatial=true)
    
    psi = deepcopy(psi_in)
    
    op_string = reverse(op_string)
    op_inds = reverse(op_inds)
    
    for (i, idx) in enumerate(op_inds)
        
        # Apply Jordan-Wigner strings:
        for k=idx-1:(-1):1
            psi = ApplySingleSiteOp(psi, sites, "F", k)
        end
        
        psi = ApplySingleSiteOp(psi, sites, op_string[i], idx)
        
    end
    
    return psi
    
end


function ExactInnerH(psi1, psi2, sites, chem_data, ord; tol=1E-12, spatial=true)
    
    N_spt = chem_data.N_spt
    h1e = chem_data.h1e
    h2e = chem_data.h2e
    
    orthogonalize!(psi1, 1)
    orthogonalize!(psi2, 1)
    
    # Initialize the expectation value:
    value = 0.0
    
    # Singles terms:
    for p=1:N_spt, q=1:N_spt
        
        cf = h1e[ord[p],ord[q]]

        if abs(cf) >= tol
            
            psi1_h = ApplyHamString(psi1, sites, ["c†↑","c↑"], [p,q], spatial=spatial)
            value += cf*inner(psi1_h, psi2)
            
            psi1_h = ApplyHamString(psi1, sites, ["c†↓","c↓"], [p,q], spatial=spatial)
            value += cf*inner(psi1_h, psi2)
            
        end
        
    end
    
    # Doubles terms:
    for p=1:N_spt, q=1:N_spt, r=1:N_spt, s=1:N_spt
        
        cf = 0.5*h2e[ord[p],ord[q],ord[r],ord[s]]
        
        if abs(cf) >= tol
            
            psi1_h = ApplyHamString(psi1, sites, ["c†↑","c†↓","c↓","c↑"], [p,r,s,q], spatial=spatial)
            value += cf*inner(psi1_h, psi2)
            
            psi1_h = ApplyHamString(psi1, sites, ["c†↓","c†↑","c↑","c↓"], [p,r,s,q], spatial=spatial)
            value += cf*inner(psi1_h, psi2)
            
            if p!=r && s!=q
                
                psi1_h = ApplyHamString(psi1, sites, ["c†↓","c†↓","c↓","c↓"], [p,r,s,q], spatial=spatial)
                value += cf*inner(psi1_h, psi2)
            
                psi1_h = ApplyHamString(psi1, sites, ["c†↑","c†↑","c↑","c↑"], [p,r,s,q], spatial=spatial)
                value += cf*inner(psi1_h, psi2)
                
            end
        end
    end
    
    return value
    
end



# Suppression localization factor:
function SLocFactor(sl_base, ord, p_inds)
    
    xj_set = [findall(x->x==p, ord)[1] for p in p_inds]
    
    dj = abs(maximum(xj_set) - minimum(xj_set))
    
    sl_exp = maximum([0, dj-1])
    
    sl_f = 1.0/sl_base^sl_exp
    
    return sl_f
    
end


# Generate the OpSum object from the Hamiltonian coefficients:
function GenOpSum(chem_data, ord; sloc=false, sl_base=2.0, tol=1e-14)
    
    N_spt = chem_data.N_spt
    h1e = chem_data.h1e
    h2e = chem_data.h2e
    
    ampo = OpSum()

    for p=1:N_spt, q=1:N_spt
        
        cf = h1e[ord[p],ord[q]]
        
        # Suppression-localization:
        if sloc
            cf *= SLocFactor(sl_base, ord, [p,q])
        end

        if abs(cf) >= tol
            ampo += cf,"c†↑",p,"c↑",q
            ampo += cf,"c†↓",p,"c↓",q
        end
        
    end

    for p=1:N_spt, q=1:N_spt, r=1:N_spt, s=1:N_spt
        
        cf = 0.5*h2e[ord[p],ord[q],ord[r],ord[s]]
        
        # Suppression-localization:
        if sloc
            cf *= SLocFactor(sl_base, ord, [p,q,r,s])
        end
        
        if abs(cf) >= tol
            ampo += cf,"c†↑",p,"c†↓",r,"c↓",s,"c↑",q
            ampo += cf,"c†↓",p,"c†↑",r,"c↑",s,"c↓",q
            if p!=r && s!=q
                ampo += cf,"c†↓",p,"c†↓",r,"c↓",s,"c↓",q
                ampo += cf,"c†↑",p,"c†↑",r,"c↑",s,"c↑",q
            end
        end
    end
    
    return ampo

end