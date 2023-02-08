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

##############################################################################


function OnSiteSwap!(T, site; do_fswap=false)
    
    ids = vcat(site, uniqueinds(inds(T), site))
    
    #println("")
    #display(ids)
    #println("")
    
    TA = Array(T, ids)
    
    TA[2:3] = [TA[3],TA[2]]
    
    if do_fswap
        TA[4] *= -1.0
    end
    
    T=ITensor(TA,ids)
    
end


# return a reversed copy of an MPS:

function ReverseMPS(psi)
    
    N = length(psi)
    
    sites=siteinds(psi)
    
    psi2 = MPS(N)
    
    for p=1:N
        
        q=N-p+1
        
        si_p = sites[p]
        si_q = sites[q]
        
        Tq = deepcopy(psi[q])
        
        replaceind!(Tq, si_q, si_p)
        
        """
        # Swap the on-site ordering:
        OnSiteSwap!(Tq, si_p)
        """
        
        psi2[p] = Tq
        
    end
    
    return psi2
    
end


function ReverseMPO(mpo)
    
    N = length(mpo)
    
    sites=siteinds(mpo)
    
    mpo2 = MPO(N)
    
    for p=1:N
        
        q=N-p+1
        
        si_p = sites[p]
        si_q = sites[q]
        
        Tq = deepcopy(mpo[q])
        
        replaceind!(Tq, si_q[1], si_p[1])
        replaceind!(Tq, si_q[2], si_p[2])
        
        """
        # Swap the on-site ordering:
        OnSiteSwap!(Tq, si_p[1])
        OnSiteSwap!(Tq, si_p[2])
        """
        
        mpo2[p] = Tq
        
    end
    
    return mpo2
    
end


function ReverseMPS2(psi)
    
    maxdim = maxlinkdim(psi)
    sites = siteinds(psi)
    N = length(psi)
    
    psi = Permute(
        psi,
        sites,
        collect(1:N),
        reverse(collect(1:N))
    )
    
    truncate!(psi, maxdim=maxdim)
    normalize!(psi)
    
    return psi
    
end


function ReverseMPO2(mpo, sites)
    
    maxdim = maxlinkdim(mpo)
    sites = siteinds(mpo, plev=0)
    N = length(mpo)
    
    #println(siteinds(mpo))
    
    dag!(mpo)
    
    mpo = PermuteMPO(
        mpo,
        sites,
        collect(1:N),
        reverse(collect(1:N)),
        tol=1e-16
    )
    
    dag!(mpo)
    swapprime!(mpo, 0,1)
    
    #println(siteinds(mpo))
    
    mpo = PermuteMPO(
        mpo,
        sites,
        collect(1:N),
        reverse(collect(1:N)),
        tol=1e-16
    )
    
    swapprime!(mpo, 0,1)
    
    truncate!(mpo, maxdim=maxdim)
    
    return mpo
    
end
