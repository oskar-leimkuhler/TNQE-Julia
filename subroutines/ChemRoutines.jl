# Basic routines for constructing quantum chemical Hamiltonians and SCF states


# Globally declaring the identity operator for electron sites:
ITensors.op(::OpName"I",::SiteType"Electron") = [1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1]
ITensors.op(::OpName"SWAP",::SiteType"Electron") = [1 0 0 0; 0 0 1 0; 0 1 0 0; 0 0 0 1]
ITensors.op(::OpName"FSWAP",::SiteType"Electron") = [1 0 0 0; 0 0 1 0; 0 1 0 0; 0 0 0 -1]
ITensors.op(::OpName"CZ",::SiteType"Electron") = [1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 -1]

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
function GenOpSum(chem_data, ord; tol=1e-14)
    
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



# Convert from PySCF alpha*beta format to 2^N bitstring array, then reshape to 4^N_spt array
function FCIArray(chem_data; sign_vec=nothing)
    
    fci_vec = chem_data.fci_vec
    fci_str = chem_data.fci_str
    fci_addr = chem_data.fci_addr
    N = chem_data.N
    N_spt = chem_data.N_spt
    N_el = chem_data.N_el
    N_a = Int(N_el/2)
    
    if sign_vec==nothing
        sign_vec = ones(size(fci_vec))
    end
    
    fci_vec .*= sign_vec
    
    dimvec = Tuple([4 for d=1:N_spt])
    
    fci_array = zeros(dimvec)
    
    blen = length(fci_str)
    
    for (da, str_a) in enumerate(fci_str), (db, str_b) in enumerate(fci_str)
        
        a_index = reverse(parse.( Int, split(lpad(str_a[3:end], N_spt,"0"), "") ) .+ 1)
        b_index = reverse(parse.( Int, split(lpad(str_b[3:end], N_spt,"0"), "") ) .+ 1)
        
        #permute!(a_index, ord)
        #permute!(b_index, ord)
        
        #println(a_index, b_index)
        
        full_index = ones(Int, length(a_index))
        
        for p=1:length(a_index)
            
            if a_index[p]==2 && b_index[p]==1
                full_index[p] = 2
            elseif a_index[p]==1 && b_index[p]==2
                full_index[p] = 3
            elseif a_index[p]==2 && b_index[p]==2
                full_index[p] = 4
            end
            
        end
        
        fci_array[full_index...] = fci_vec[da, db]
        
    end
    
    return fci_array;
    
end


function FCIMPS(chem_data; spin=0, verbose=false)
    
    verbose && println("Computing FCI MPS:")
    
    N = chem_data.N
    N_spt = chem_data.N_spt
    N_el = chem_data.N_el
    N_a = Int(N_el/2)
    
    # Generate the Hamiltonian MPO
    opsum = GenOpSum(chem_data, collect(1:N_spt))
    
    sites = siteinds("Electron", N_spt, conserve_qns=true)
    
    ham_mpo = MPO(
        opsum, 
        sites,
        cutoff=1e-16,
        maxdim=2^16
    )
    
    # Get the alpha/beta bitstring list:
    bstr_list = [[],[]]
    
    for (i, combo) in enumerate(combinations(collect(1:N_spt), N_a+spin))
        bstr = ones(Int, N_spt)
        for p=1:N_spt
            if p in combo
                bstr[p] = 2
            end
        end
        push!(bstr_list[1], bstr)
    end
    
    for (i, combo) in enumerate(combinations(collect(1:N_spt), N_a-spin))
        bstr = ones(Int, N_spt)
        for p=1:N_spt
            if p in combo
                bstr[p] = 2
            end
        end
        push!(bstr_list[2], bstr)
    end
    
    # Get the full bistring list:
    fbstr_list = []
    for bstr_a in bstr_list[1], bstr_b in bstr_list[2]
        
        full_bstr = ones(Int, N_spt)
        
        for p=1:N_spt
            if bstr_a[p]==2 && bstr_b[p]==1
                full_bstr[p] = 2
            elseif bstr_a[p]==1 && bstr_b[p]==2
                full_bstr[p] = 3
            elseif bstr_a[p]==2 && bstr_b[p]==2
                full_bstr[p] = 4
            end
        end
        
        push!(fbstr_list, full_bstr)
        
    end
    
    n_bstr = length(fbstr_list) 
    
    # Obtain the bitstring H-matrix
    H_fci = zeros((n_bstr, n_bstr))
    
    c = 0
    for i=1:n_bstr, j=i:n_bstr
        
        bmps_i = MPS(sites, fbstr_list[i])
        bmps_j = MPS(sites, fbstr_list[j])
        
        H_fci[i,j] = inner(bmps_i', ham_mpo, bmps_j)
        H_fci[j,i] = H_fci[i,j]
        
        c += 1
        if verbose && div(c, 100) > div(c-1, 100)
            print("Progress: $(c)/$(Int((n_bstr^2+n_bstr)/2))  \r")
            flush(stdout)
        end
        
    end
    
    # Diagonalize and obtain the ground state coefficients
    fact = eigen(H_fci)
    
    E = fact.values
    C = fact.vectors
    verbose && println("\n$(E[1]+chem_data.e_nuc)")
    fci_vec = C[:,1]
    
    # Construct the FCI array
    dimvec = Tuple([4 for d=1:N_spt])
    fci_array = zeros(dimvec)
    
    for (i, fbstr) in enumerate(fbstr_list)
        
        fci_array[fbstr...] = fci_vec[i]
        
    end
    
    # Convert to MPS
    
    fci_mps = MPS(
        fci_array, 
        sites, 
        cutoff=1e-16,
        maxdim=2^16
    )
    
    verbose && println("Done!")
    
    return fci_mps, ham_mpo
    
end

function FCIMPS_AllSpin(chem_data; verbose=false)
    
    verbose && println("Computing FCI MPS:")
    
    N = chem_data.N
    N_spt = chem_data.N_spt
    N_el = chem_data.N_el
    
    # Generate the Hamiltonian MPO
    opsum = GenOpSum(chem_data, collect(1:N_spt))
    
    sites = siteinds("Electron", N_spt, conserve_qns=true)
    
    ham_mpo = MPO(
        opsum, 
        sites,
        cutoff=1e-16,
        maxdim=2^16
    )
    
    # Get the full bistring list:
    fbstr_list = []
    
    for (i, combo) in enumerate(combinations(collect(1:N), N_el))
        fbstr = ones(Int, N_spt)
        for p=1:N_spt
            q = [2*p-1, 2*p]
            if q[1] in combo && q[2] in combo
                fbstr[p] = 4
            elseif q[1] in combo
                fbstr[p] = 2
            elseif q[2] in combo
                fbstr[p] = 3
            end
        end
        push!(fbstr_list, fbstr)
    end
    
    n_bstr = length(fbstr_list) 
    
    # Obtain the bitstring H-matrix
    H_fci = zeros((n_bstr, n_bstr))
    
    c = 0
    for i=1:n_bstr, j=i:n_bstr
        
        bmps_i = MPS(sites, fbstr_list[i])
        bmps_j = MPS(sites, fbstr_list[j])
        
        H_fci[i,j] = inner(bmps_i', ham_mpo, bmps_j)
        H_fci[j,i] = H_fci[i,j]
        
        c += 1
        if verbose && div(c, 100) > div(c-1, 100)
            print("Progress: $(c)/$(Int((n_bstr^2+n_bstr)/2))  \r")
            flush(stdout)
        end
        
    end
    
    # Diagonalize and obtain the ground state coefficients
    fact = eigen(H_fci)
    
    E = fact.values
    C = fact.vectors
    verbose && println("\n$(E[1]+chem_data.e_nuc)")
    fci_vec = C[:,1]
    
    # Construct the FCI array
    dimvec = Tuple([4 for d=1:N_spt])
    fci_array = zeros(dimvec)
    
    for (i, fbstr) in enumerate(fbstr_list)
        
        fci_array[fbstr...] = fci_vec[i]
        
    end
    
    # Convert to MPS
    
    fci_mps = MPS(
        fci_array, 
        sites, 
        cutoff=1e-16,
        maxdim=2^16
    )
    
    verbose && println("Done!")
    
    return fci_mps, ham_mpo
    
end