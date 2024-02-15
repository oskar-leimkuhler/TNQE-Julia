# Miscellaneous functions


# Functions to print out useful information:

function PrintChemData(chemical_data)
    println("Molecule name: ", chemical_data.mol_name)
    println("Basis set: ", chemical_data.basis)
    println("Molecular geometry: ", chemical_data.geometry)
    println("RHF energy: ", chemical_data.e_rhf)
    println("FCI energy: ", chemical_data.e_fci)
end


function DisplayEvalData(sdata)
    
    chemical_data = sdata.chem_data
    H_mat = sdata.H_mat
    E = sdata.E
    C = sdata.C
    kappa = sdata.kappa
    
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

    #scatter(collect(1:length(C[:,1])), real.(C[:,1]),lw=2)
    #hline!([0.0], lw=2)
end


function PrintFSWAPNetwork(ord1, ord2)
    
    # Calculate what the FSWAP network will look like:
    width = length(ord1)
    
    qubit_lines = []
    
    for p=1:(width-1)
        push!(qubit_lines, string(ord1[p])*"-")
        push!(qubit_lines, "  ")
    end
    
    push!(qubit_lines, string(ord1[width])*"-")
    
    swap_indices = BubbleSort(ord1, ord2)
    
    for swap_idx in swap_indices
        
        q = 2*swap_idx - 1
        
        maxl = maximum([length(qubit_lines[q]), length(qubit_lines[q+1]), length(qubit_lines[q+2])])
        
        for r in [q, q+2]
            if length(qubit_lines[r]) < maxl
                for d=1:(maxl-length(qubit_lines[r]))
                    qubit_lines[r] *= "-"
                end
            end
        end
        
        if length(qubit_lines[q+1]) < maxl
            for d=1:(maxl-length(qubit_lines[q+1]))
                qubit_lines[q+1] *= " "
            end
        end
        
        qubit_lines[q] *= "x-"
        qubit_lines[q+1] *= "| "
        qubit_lines[q+2] *= "x-"
        
    end
    
    depth = maximum([length(qubit_lines[q]) for q=1:(2*width-1)])
    
    for p=1:width
        q = 2*p-1
        if length(qubit_lines[q]) < depth
            for d=1:(depth-length(qubit_lines[q]))
                qubit_lines[q] *= "-"
            end
        end
        qubit_lines[q] *= string(ord2[p])
    end
    
    for p=1:width-1
        q = 2*p
        if length(qubit_lines[q]) < depth
            for d=1:(depth-length(qubit_lines[q]))
                qubit_lines[q] *= " "
            end
        end
        qubit_lines[q] *= " "
    end
    
    
    # Print the FSWAP network:
    for q=1:(2*width-1)
        println(qubit_lines[q])
    end
    
end


function PrintFSWAPNetworks!(sdata)
    
    for j1 = 1:sdata.mparams.M, j2 = (j1+1):sdata.mparams.M
        
        println("\n\n###### $(j1)-$(j2) ######")
        println("Max. linkdim = $(maxlinkdim(sdata.perm_ops[j1][j2-j1]))\n")
        
        PrintFSWAPNetwork(sdata.ord_list[j1], sdata.ord_list[j2])
        
    end
    
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
function PolyProb(e, e_new, temp; tpow=3)
    if e_new < e
        P=1.0
    else
        P=temp^tpow
    end
    return P
end

function Fstun(E_0, E_1, gamma)
    return 1.0 - exp(gamma*(E_1-E_0))
end

##############################################################################


# Count numerical nonzeros in a list of MPSs:
function CountNonZeros(psi_list)
    
    counter = 0
    
    for psi in psi_list
        
        for p=1:length(psi)

            T = Array(psi[p], inds(psi[p])) 

            counter += count(x->(abs(x)>1e-13), T)

        end
        
    end
    
    return counter
    
end


# Construct the summed matrix product state:
function SumMPS(
        sdata;
        final_ord=nothing,
        cutoff=1e-12
    )
    
    if final_ord==nothing
        final_ord = sdata.ord_list[1]
    end
    
    psi_list_copy = deepcopy(sdata.psi_list)
    
    # Permute everything into the ordering of the first MPS:
    for i=1:length(psi_list_copy)
        
        psi_list_copy[i] = Permute(
            psi_list_copy[i],
            sdata.sites,
            sdata.ord_list[i],
            final_ord
        )
        
        psi_list_copy[i] *= sdata.C[i,1]
        
    end
    
    summed_mps = add(Tuple(psi_list_copy)..., cutoff=cutoff)
    
    return summed_mps
    
end


##############################################################################


# Fetch job info from a config file:

mutable struct ConfigParams

    # Molecule and job info
    jobname::String
    nmol::Int
    mol_names::Vector{String}
    pyscf_paths::Vector{String}
    
    # Ansatze
    n_atz::Int
    atz_name::Vector{String}
    atz_M::Vector{Int}
    atz_m::Vector{Int}
    diff_ords::Vector{Bool}
    do_opt::Vector{Bool}
    init_type::Vector{Int}
    init_sweeps::Vector{Int}

    # Ipq calculation
    Ipq_calc::String
    Ipq_maxdim::Int
    Ipq_sweeps::Int

    # Geometry heuristics
    gp_optord::Int
    gp_multord::Int

    # Optimization
    rep_struct::Vector{Vector{Any}}

    # HDF5 output
    hdf5_out::Vector{String}
    
    
end


function FetchConfig(conf_path)

    conf = ConfParse("$(conf_path)/master.ini")
    parse_conf!(conf)

    # Get and store config parameters
    
    # Molecule and job info
    jobname = string(retrieve(conf, "info", "jobname"))
    nmol = parse(Int, retrieve(conf, "moldata", "nmol"))
    
    mol_names = [string(r) for r in retrieve(conf, "moldata", "mol_names")]
    mol_names = mol_names[mol_names .!= "end"]
    
    pyscf_paths = []
    for g=1:nmol
        path = string(retrieve(conf, "moldata", "pyscf_path$(g)"))
        push!(pyscf_paths, path)
    end
    
    
    # Ansatze
    n_atz = parse(Int, retrieve(conf, "ansatze", "natz"))
    
    atz_name = [string(r) for r in retrieve(conf, "ansatze", "atz_names")]
    atz_name = atz_name[atz_name .!= "end"]
    
    atz_M = [parse(Int, r) for r in retrieve(conf, "ansatze", "n_refs")]
    atz_M = atz_M[atz_M .!= -1]
    
    atz_m = [parse(Int, r) for r in retrieve(conf, "ansatze", "maxdim")]
    atz_m = atz_m[atz_m .!= -1]
    
    diff_ords = [parse(Int, r) for r in retrieve(conf, "ansatze", "diff_ords")]
    diff_ords = Bool.(diff_ords[diff_ords .!= -1])
    
    do_opt = [parse(Int, r) for r in retrieve(conf, "ansatze", "do_opt")]
    do_opt = Bool.(do_opt[do_opt .!= -1])
    
    n_sweeps = parse(Int, retrieve(conf, "ansatze", "n_sweeps"))
    
    # Construct the list of sweep objects
    swp_list = Sweeps[]
    for i=1:n_sweeps
        
        swp_conf = ConfParse("$(conf_path)/swp$(i).ini")
        parse_conf!(swp_conf)
        
        maxiter = parse(Int, retrieve(swp_conf, "params", "maxiter"))
        maxdim = parse(Int, retrieve(swp_conf, "params", "maxdim"))
        mindim = parse(Int, retrieve(swp_conf, "params", "mindim"))
        cutoff = parse(Float64, retrieve(swp_conf, "params", "cutoff"))
        noise = [parse(Float64, r) for r in retrieve(swp_conf, "params", "noise")]
        noise = noise[noise .!= -1.0]
        
        swp = Sweeps(maxiter)
        maxdim!(swp, maxdim)
        mindim!(swp, mindim)
        cutoff!(swp, cutoff)
        setnoise!(swp, noise...)
        
        push!(swp_list, swp)
        
    end
    
    init_type = [parse(Int, r) for r in retrieve(conf, "ansatze", "init_type")]
    init_type = init_type[init_type .!= -1]
    
    init_sweeps = [parse(Int, r) for r in retrieve(conf, "ansatze", "init_sweeps")]
    init_sweeps = init_sweeps[init_sweeps .!= -1]
    
    # Ipq calculation
    Ipq_calc = string(retrieve(conf, "mutinf", "ipq_calc"))
    Ipq_maxdim = parse(Int, retrieve(conf, "mutinf", "ipq_maxdim"))
    Ipq_sweeps = parse(Int, retrieve(conf, "mutinf", "ipq_sweeps"))

    # Geometry heuristics
    ngp = parse(Int, retrieve(conf, "ordering", "ngp"))
    
    # Construct the list of OptimParameters objects
    gp_list = GeomParameters[]
    for i=1:ngp
        
        gp_conf = ConfParse("$(conf_path)/gp$(i).ini")
        parse_conf!(gp_conf)
        
        maxiter = parse(Int, retrieve(gp_conf, "params", "maxiter"))

        afunc = string(retrieve(gp_conf, "params", "afunc"))
        a_alpha = parse(Float64, retrieve(gp_conf, "params", "a_alpha"))
        a_gamma = parse(Float64, retrieve(gp_conf, "params", "a_gamma"))
        swap_mult = parse(Float64, retrieve(gp_conf, "params", "swap_mult"))
        a_tpow = parse(Float64, retrieve(gp_conf, "params", "a_tpow"))

        cweight = parse(Float64, retrieve(gp_conf, "params", "cweight"))
        eta = parse(Int, retrieve(gp_conf, "params", "eta"))
        shrp = parse(Float64, retrieve(gp_conf, "params", "shrp"))

        anchor = parse(Int, retrieve(gp_conf, "params", "anchor"))
        return_all = Bool(parse(Int, retrieve(gp_conf, "params", "return_all")))
        
        gp = GeomParameters(
            # Maximum number of iterations:
            maxiter,
            # Annealing parameters:
            afunc,
            a_alpha,
            a_gamma,
            swap_mult,
            a_tpow,
            # Information-distance parameters:
            cweight,
            eta,
            shrp,
            # Miscellaneous:
            anchor,
            return_all
        )
        
        
        push!(gp_list, gp)
        
    end
    
    gp_optord = parse(Int, retrieve(conf, "ordering", "gp_optord"))
    gp_multord = parse(Int, retrieve(conf, "ordering", "gp_multord"))

    # Optimization
    nop = parse(Int, retrieve(conf, "optimize", "nop"))
    
    # Construct the list of OptimParameters objects
    op_list = OptimParameters[]
    for i=1:nop
        
        op_conf = ConfParse("$(conf_path)/op$(i).ini")
        parse_conf!(op_conf)
        
        maxiter = parse(Int, retrieve(op_conf, "params", "maxiter"))
        numloop = parse(Int, retrieve(op_conf, "params", "numloop"))
        numopt = parse(Int, retrieve(op_conf, "params", "numopt"))
        # Site decomposition parameters:
        noise = [parse(Float64, r) for r in retrieve(op_conf, "params", "noise")]
        noise = noise[noise .!= -1]
        
        delta = [parse(Float64, r) for r in retrieve(op_conf, "params", "delta")]
        delta = delta[delta .!= -1]
        
        theta = parse(Float64, retrieve(op_conf, "params", "theta"))
        ttol = parse(Float64, retrieve(op_conf, "params", "ttol"))
        
        swap_mult = parse(Float64, retrieve(op_conf, "params", "swap_mult"))
        
        # Generalized eigenvalue solver parameters:
        thresh = string(retrieve(op_conf, "params", "thresh"))
        eps = parse(Float64, retrieve(op_conf, "params", "eps"))
        # Site decomposition solver parameters:
        sd_method = string(retrieve(op_conf, "params", "sd_method"))
        sd_thresh = string(retrieve(op_conf, "params", "sd_thresh"))
        sd_eps = parse(Float64, retrieve(op_conf, "params", "sd_eps"))
        sd_penalty = parse(Float64, retrieve(op_conf, "params", "sd_penalty"))
        sd_swap_penalty = parse(Float64, retrieve(op_conf, "params", "sd_swap_penalty"))
        sd_dtol = parse(Float64, retrieve(op_conf, "params", "sd_dtol"))
        
        op = OptimParameters(
            maxiter,
            numloop,
            numopt,
            # Site decomposition parameters:
            noise,
            delta,
            theta,
            ttol,
            swap_mult,
            # Generalized eigenvalue solver parameters:
            thresh,
            eps,
            # Site decomposition solver parameters:
            sd_method,
            sd_thresh,
            sd_eps,
            sd_penalty,
            sd_swap_penalty,
            sd_dtol
        )
        
        push!(op_list, op)
        
    end
    
    nseq = parse(Int, retrieve(conf, "optimize", "nseq"))
    
    rep_struct = []
    
    for s=1:nseq
        
        nreps = parse(Int, retrieve(conf, "optimize", "nreps$(s)"))
        
        steps = [string(r) for r in retrieve(conf, "optimize", "steps$(s)")]
        steps = steps[steps .!= "end"]
        
        repop = [parse(Int, r) for r in retrieve(conf, "optimize", "repop$(s)")]
        steps = steps[steps .!= -1]
        
        for r=1:nreps
            
            for i=1:length(steps)
            
                repstep = [steps[i], repop[i]]
            
                push!(rep_struct, repstep)
                
            end
            
        end
        
    end

    # HDF5 output
    hdf5_out = [string(r) for r in retrieve(conf, "output", "output")]
    hdf5_out = hdf5_out[hdf5_out .!= "end"]
    
    conf_params = ConfigParams(
        # Molecule and job info
        jobname,
        nmol,
        mol_names,
        pyscf_paths,
        # Ansatze
        n_atz,
        atz_name,
        atz_M,
        atz_m,
        diff_ords,
        do_opt,
        init_type,
        init_sweeps,
        # Ipq calculation
        Ipq_calc,
        Ipq_maxdim,
        Ipq_sweeps,
        # Geometry heuristics
        gp_optord,
        gp_multord,
        # Optimization
        rep_struct,
        # HDF5 output
        hdf5_out
    )
    
    return conf_params, op_list, gp_list, swp_list
    
end