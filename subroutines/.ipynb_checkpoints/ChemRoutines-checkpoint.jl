# Basic routines for constructing quantum chemical Hamiltonians and SCF states

# Packages:
using ITensors


# Fill orbital i dependent on the electron number (HF filling):
function FillHF(i, nel)
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
function GenOpSum(chem_data, ord)
    
    N_spt = chem_data.N_spt
    h1e = chem_data.h1e
    h2e = chem_data.h2e
    
    ampo = OpSum()

    for p=1:N_spt
        for q=1:N_spt
            for σ=1:2
                ampo += h1e[ord[p],ord[q]],"c†",2*(p-1)+σ,"c",2*(q-1)+σ
            end
        end
    end

    for p=1:N_spt
        for q=1:N_spt
            for r=1:N_spt
                for s=1:N_spt
                    for σ=1:2
                        for τ=1:2
                            ampo += 0.5*h2e[ord[p],ord[q],ord[r],ord[s]],"c†",2*(p-1)+σ,"c†",2*(r-1)+τ,"c",2*(s-1)+τ,"c",2*(q-1)+σ
                        end
                    end
                end
            end
        end
    end
    
    return ampo
    
end
