# Miscellaneous functions

# Packages:
#


# Functions to print out useful information:

function PrintChemData(chemical_data)
    println("Molecule name: ", chemical_data.mol_name)
    println("Basis set: ", chemical_data.basis)
    println("Molecular geometry: ", chemical_data.geometry)
    println("RHF energy: ", chemical_data.e_rhf)
    println("FCI energy: ", chemical_data.e_fci)
end


function DisplayEvalData(chemical_data, H_mat, E, C, kappa)
    e_gnd = minimum(filter(!isnan,real.(E)))+chemical_data.e_nuc
    e_bsrf = minimum(diag(H_mat))+chemical_data.e_nuc

    println("Minimum eigenvalue: ", minimum(filter(!isnan,real.(E))))
    println("Condition number: ", kappa)

    println("FCI energy: ", chemical_data.e_fci)
    println("Final energy estimate: ", e_gnd)
    println("Best single ref. estimate: ", e_bsrf)

    println("Error: ", e_gnd - chemical_data.e_fci)
    println("BSRfE: ", e_bsrf - chemical_data.e_fci)
    println("Improvement: ", e_bsrf - e_gnd)
    println("Percentage error reduction: ", (e_bsrf - e_gnd)/(e_bsrf - chemical_data.e_fci)*100)

    kappa_list = EigCondNums(E, C)
    println("Eigenvalue condition numbers: ", round.(kappa_list, digits=4))
    
    e_corr = chemical_data.e_fci-chemical_data.e_rhf
    e_corr_dmrg = e_bsrf - chemical_data.e_rhf
    e_corr_tnqe = e_gnd - chemical_data.e_rhf
    pctg_dmrg = e_corr_dmrg/e_corr*100
    pctg_tnqe = e_corr_tnqe/e_corr*100
    println("Percent correlation energy with single-geometry DMRG: $pctg_dmrg")
    println("Percent correlation energy with multi-geometry TNQE: $pctg_tnqe")

    scatter(collect(1:length(C[:,1])), real.(C[:,1]),lw=2)
    hline!([0.0], lw=2)
end

###############################################################################

# Simulated annealing and stochastic tunnelling probability functions:

function ExpProb(E_0, E_1, beta)
    if E_1<=E_0
        P = 1
    else
        P = exp((E_0-E_1)*beta)
    end
    return P
end


function StepProb(E_0, E_1)
    if E_1<=E_0
        P = 1
    else
        P = 0
    end
    return P
end

# Returns a polynomial acceptance probability:
function PolyProb(e, e_new, temp; tpow=3, greedy=false)
    if e_new < e
        P=1.0
    elseif greedy==false
        P=temp^tpow
    else
        P=0.0
    end
    return P
end

function Fstun(E_0, E_1, gamma)
    return 1.0 - exp(gamma*(E_1-E_0))
end