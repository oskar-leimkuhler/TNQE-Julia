# Basic routines for constructing quantum chemical Hamiltonians and SCF states

# Packages:
using ITensors


# Globally declaring the c_dag and c operators in the JW representation:
ITensors.op(::OpName"1/2[X+iY]",::SiteType"Qubit") = [0 0; 1 0]
ITensors.op(::OpName"1/2[X-iY]",::SiteType"Qubit") = [0 1; 0 0]
ITensors.op(::OpName"N",::SiteType"Qubit") = [0 0; 0 1]
ITensors.op(::OpName"ImN",::SiteType"Qubit") = [1 0; 0 0]


# Fill spatial orbital i dependent on the electron number (HF filling):
function FillHF(i, nel)
    if 2*i <= nel
        return 4
    elseif 2*i-1 <= nel
        return 2
    else
        return 1
    end
end


# Fill spin-orbital i dependent on the electron number (HF filling):
function FillHFSporb(i, nel)
    if i <= nel
        return 2
    else
        return 1
    end
end


# Convert an ordering of spatial MOs to an ordering of (neighboring alpha-beta) spin-orbital MOs:
function Spatial2SpinOrd(spt_ord)
    spn_ord = Int[]
    
    for i in spt_ord
        push!(spn_ord, 2*i-1, 2*i)
    end
    
    return spn_ord 
end


# Generate the OpSum object from the Hamiltonian coefficients:
function GenOpSum(chem_data, ord; tol=1E-12)
    
    N_spt = chem_data.N_spt
    h1e = chem_data.h1e
    h2e = chem_data.h2e
    
    ampo = OpSum()

    for p=1:N_spt, q=1:N_spt
        
        cf = h1e[ord[p],ord[q]]

        if abs(cf) >= tol
            ampo += cf,"c†↑",p,"c↑",q
            ampo += cf,"c†↓",p,"c↓",q
        end
        
    end

    for p=1:N_spt, q=1:N_spt, r=1:N_spt, s=1:N_spt
        
        cf = 0.5*h2e[ord[p],ord[q],ord[r],ord[s]]
        
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


# Generate the OpSum object from the Hamiltonian coefficients:
function GenOpSumSporb(chem_data, ord; tol=1E-12)
    
    N_spt = chem_data.N_spt
    h1e = chem_data.h1e
    h2e = chem_data.h2e
    
    ampo = OpSum()

    for p=1:N_spt, q=1:N_spt, σ=1:2
        
        ip = 2*(p-1)+σ
        iq = 2*(q-1)+σ
        
        cf = h1e[ord[p],ord[q]]

        if abs(cf) >= tol
            ampo += cf,"c†",ip,"c",iq
        end
        
    end

    for p=1:N_spt, q=1:N_spt, r=1:N_spt, s=1:N_spt, σ=1:2, τ=1:2
        
        ip = 2*(p-1)+σ
        iq = 2*(q-1)+σ
        ir = 2*(r-1)+τ
        is = 2*(s-1)+τ
        
        cf = 0.5*h2e[ord[p],ord[q],ord[r],ord[s]]

        if iq!=is && ir!=ip && abs(cf) >= tol
            ampo += cf,"c†",ip,"c†",ir,"c",is,"c",iq
        end

    end
    
    return ampo
    
end


# Find the Jordan-Wigner indices and apply Z operators:
function AddJordanWignerOp(op_list, op, ind)
    
    push!(op_list, op)
    push!(op_list, ind)
    
    for i=1:ind-1
        push!(op_list, "Z")
        push!(op_list, i)
    end
    
    return op_list
    
end


# Generate the OpSum object from the Hamiltonian coefficients:
function GenOpSumJW(chem_data, ord; tol=1E-12)
    
    N_spt = chem_data.N_spt
    h1e = chem_data.h1e
    h2e = chem_data.h2e
    
    ampo = OpSum()

    for p=1:N_spt, q=1:N_spt, σ=1:2
        
        ip = 2*(p-1)+σ
        iq = 2*(q-1)+σ
        
        cf = h1e[ord[p],ord[q]]

        if abs(cf) >= tol
            
            op_list = Any[]
            push!(op_list, cf)

            op_list = AddJordanWignerOp(op_list, "1/2[X+iY]", ip)
            op_list = AddJordanWignerOp(op_list, "1/2[X-iY]", iq)

            ampo += Tuple(op_list)
            
        end
        
    end

    for p=1:N_spt, q=1:N_spt, r=1:N_spt, s=1:N_spt, σ=1:2, τ=1:2
        
        ip = 2*(p-1)+σ
        iq = 2*(q-1)+σ
        ir = 2*(r-1)+τ
        is = 2*(s-1)+τ
        
        cf = 0.5*h2e[ord[p],ord[q],ord[r],ord[s]]

        if iq!=is && ir!=ip && abs(cf) >= tol
            
            op_list = Any[]
            push!(op_list, cf)

            op_list = AddJordanWignerOp(op_list, "1/2[X+iY]", ip)
            op_list = AddJordanWignerOp(op_list, "1/2[X+iY]", ir)
            op_list = AddJordanWignerOp(op_list, "1/2[X-iY]", is)
            op_list = AddJordanWignerOp(op_list, "1/2[X-iY]", iq)

            ampo += Tuple(op_list)
            
        end

    end
    
    return ampo
    
end


# Hartree-Fock energy for a closed-shell configuration:
function ClosedShellEHF(chemical_data)
    
    e_hf = 0.0

    occ_orbs = collect(1:Int(chemical_data.N_el/2))

    for i in occ_orbs

        e_hf += 2.0*chemical_data.h1e[i, i]

    end

    for i in occ_orbs, j in occ_orbs

        e_hf += 2.0*chemical_data.h2e[i, i, j, j]

        e_hf -= 1.0*chemical_data.h2e[i, j, j, i]

    end

    return e_hf
    
end
