# Importing the other submodules:
include("./ChemData.jl")
include("./ChemRoutines.jl")
include("./GenEigRoutines.jl")
include("./GeometryHeuristics.jl")
include("./MutualInformation.jl")
include("./Permutations.jl")

# High-level subroutines go here

# Carry out a full MPS-MPO DMRG procedure for a given chemical data, site ordering and sweep specification:
function RunDMRG(chemical_data, sites, ord, sweeps; mpo_cutoff=1E-16, mpo_maxdim=5000, pass_opsum=nothing, pass_H=nothing)
    
    if pass_H==nothing
        
        if pass_opsum==nothing
            opsum = GenOpSum(chemical_data, ord)
        else
            opsum = pass_opsum
        end
        
        H = MPO(opsum, sites, cutoff=mpo_cutoff);
        
    else
        
        H = pass_H
        
    end
    
    spnord = perm_spnord = Spatial2SpinOrd(perm_ord)
    
    hf_occ = [FillHF(spnord[i], chemical_data.N_el) for i=1:chemical_data.N]

    psi0 = randomMPS(sites, hf_occ)
    
    e_dmrg, psi = dmrg(H, psi0, sweeps)
    
    return H, psi, e_dmrg
    
end