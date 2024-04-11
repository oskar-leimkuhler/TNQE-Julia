# Functions for generating unitary disentanglers to prepare an MPS on quantum hardware


# Pad state vector with zeros and reshape to 2x2x...x2 array
function PaddedArray(state_vector)
    
    dim = length(state_vector)
    num_qubits = Int(ceil( log(2, dim)))
    
    padded_state_vector = zeros(2^num_qubits)
    padded_state_vector[1:dim] = state_vector
    array_dims = [2 for i=1:num_qubits]
    reshaped_state_vector = reshape(padded_state_vector, Tuple(array_dims));
    
    return reshaped_state_vector
    
end


# Convert an input state vector to an MPS representation:
function StateVecMPS(state_vector; cutoff=1e-8, maxdim=1e3, sites_in=nothing)
    
    dim = length(state_vector)
    num_qubits = Int(ceil( log(2, dim)))
    if sites_in==nothing
        sites = siteinds(2,num_qubits)
    else
        sites = sites_in
    end
    
    # Constructing a filled-in, i.e. padded with zeros, and reshaped array:
    reshaped_state_vector = PaddedArray(state_vector)
    
    # Converting the array into an MPS with some fixed dimension cutoff parameters:
    @disable_warn_order mps = MPS(reshaped_state_vector,sites;cutoff=cutoff,maxdim=maxdim)
    
    return mps
    
end


# Get the overlap of an MPS with a single bitstring state:
function BitstringOverlap(mps, bitstring)
    V = ITensor(1.)
    for j=1:length(siteinds(mps))
        V *= (mps[j]*state(siteinds(mps)[j],bitstring[j]))
    end
    return scalar(V)
end


# Get the fidelity between an MPS and the full coefficient tensor:
function fidelity(mps, statevec)
    
    array = PaddedArray(statevec)
    
    dot_prod = 0.0
    for el in pairs(array)
        dot_prod += BitstringOverlap(mps, Tuple(el[1])) * el[2]
    end
    return dot_prod^2
end 


# Generate the first (single-qubit) gate:
function GenFirst(mps)
    N_sites = length(mps)
    ids = ( commoninds(mps[N_sites-1],mps[N_sites])[1], siteinds(mps)[N_sites] )
    coeffs = Array(mps[N_sites], (ids[2],ids[1]))
    gate = ITensor(coeffs, ids[2]', ids[2])
    return gate
end


# Generate a middle (two-qubit) gate at specified site:
function GenMid(mps, site)
    ids = ( commoninds(mps[site-1],mps[site])[1], commoninds(mps[site+1],mps[site])[1], siteinds(mps)[site] )
    coeffs = Array(mps[site], ids)
    coeff_vectors = [reshape(coeffs[j,:,:], (4,1)) for j=1:2]
    coeff_matrix = zeros((4,4))
    for j=1:2
        coeff_matrix[:,j] = coeff_vectors[j]
    end
    rank2_mat = coeff_matrix * transpose(coeff_matrix)
    U,S,V = svd(rank2_mat)
    kernel_vectors = [U[:,j] for j=3:4]
    kernel_coeffs = [reshape(kvec, (2,2)) for kvec in kernel_vectors]
    full_coeff_array = zeros((2,2,2,2))
    for j=1:2, k=1:2, l=1:2
        full_coeff_array[1,j,k,l] = coeffs[j,k,l]
        full_coeff_array[2,j,k,l] = kernel_coeffs[j][k,l]
    end
    ind1 = siteinds(mps)[site+1]
    ind2 = siteinds(mps)[site]
    gate = ITensor(full_coeff_array, ind1', ind2', ind1, ind2) 
    return gate
end

# Generate the last (two-qubit) gate:
function GenLast(mps)
    ids = ( commoninds(mps[2],mps[1])[1], siteinds(mps)[1] )
    coeffs = Array(mps[1], (ids[1],ids[2]))
    coeff_vector = reshape(coeffs, (4,1))
    rank1_mat = coeff_vector * transpose(coeff_vector)
    U,S,V = svd(rank1_mat)
    kernel_vectors = [U[:,j] for j=2:4]
    kernel_coeffs = [reshape(kvec, (2,2)) for kvec in kernel_vectors]
    full_coeff_array = zeros((2,2,2,2))
    for k=1:2, l=1:2
        full_coeff_array[1,1,k,l] = coeffs[k,l]
        full_coeff_array[1,2,k,l] = kernel_coeffs[1][k,l]
        full_coeff_array[2,1,k,l] = kernel_coeffs[2][k,l]
        full_coeff_array[2,2,k,l] = kernel_coeffs[3][k,l]
    end
    ind1 = siteinds(mps)[2]
    ind2 = siteinds(mps)[1]
    gate = ITensor(full_coeff_array, ind1', ind2', ind1, ind2)  
    return gate
end


function TruncOrthoMindim!(mps, maxdim, mindim)
    
    N = length(mps)
    
    orthogonalize!(mps, N)
    
    for p=N-1:(-1):1
        T = mps[p] * mps[p+1]
        linds = commoninds(T, mps[p])
        U,S,V = svd(T, linds, maxdim=maxdim, mindim=mindim)
        mps[p] = U*S
        mps[p+1] = V
    end
    
end


# Generate a single "staircase" layer of disentanglers:
function GenDLayer(mps_in)
    
    # Orthogonalize and "Optimally truncate":
    mps = deepcopy(mps_in)
    N_sites = length(mps)    
    TruncOrthoMindim!(mps,2,2)
    mps[1] *= 1.0/norm(mps)
    println(linkdims(mps))
    #println(inner(mps,mps_in))
    
    # Generate the disentanglers:
    
    new_disentangler_list = []
    
    # The first (single-qubit) gate:
    push!(new_disentangler_list, GenFirst(mps))
    
    # The middle (two-qubit) gates:
    for p=(N_sites-1):(-1):2
        push!(new_disentangler_list, GenMid(mps, p))
    end
    
    # The last (two_qubit) gate:
    push!(new_disentangler_list, GenLast(mps))
    
    return new_disentangler_list
    
end


# Generate a set of disentanglers from an input MPS with specified depth:
function GenerateDisentanglers(mps; depth=1, tol=1e-6, verbose=false)
    
    N_sites = length(mps)
    
    disentanglers = [[] for i=1:(N_sites+2*depth-1)]
    
    mps1 = deepcopy(mps)
    
    # Disentangler generation loop:
    if verbose
        println("Starting disentangler loop!")
        println("Initial bond dimension: ", maxlinkdim(mps1))
        println("----------------------------------")
    end
    
    for d=1:depth
        
        bdim = float(maxlinkdim(mps1))
        
        new_disentangler_list = GenDLayer(mps1)
        
        G1 = new_disentangler_list[1]
        
        d1 = 1+2*(d-1)
        
        push!(disentanglers[d1], [G1,N_sites,1])
        
        for p=2:N_sites
            
            Gp = new_disentangler_list[p]
            
            pp = N_sites-p+1
            
            dp = p+2*(d-1)
            
            push!(disentanglers[dp], [Gp,pp,2])
            
        end
        
        mps1 = ApplyDisentanglers(mps, disentanglers)
        
        if verbose
            println("Layer $d of $depth generated!")
            println("Bitstring fidelity: ", BitstringOverlap(mps1, [1 for i=1:N_sites])^2)
            println("Maxlinkdim: ", maxlinkdim(mps1))
        end
        
    end
    
    if verbose
        println("----------------------------------")
        println("Done!")
    end
    
    return disentanglers

end


# Apply a set of disentanglers to an input MPS:
function ApplyDisentanglers(mps_in, disentanglers; cutoff=1e-8)
    
    mps = deepcopy(mps_in)
    N_sites = length(mps)
    orthogonalize!(mps, 1)
    
    for dlist in disentanglers
    
        for G_inf in dlist

            G,p,n = G_inf

            if n==1
                phi = mps[p] * G
                noprime!(phi)
                mps[p] = phi
            else
                phi = mps[p] * mps[p+1] * G
                noprime!(phi)
                temp_inds = uniqueinds(mps[p],mps[p+1])
                U,S,V = svd(phi,temp_inds,cutoff=cutoff)
                mps[p] = U
                mps[p+1] = S*V
            end

        end
        
    end
    
    return mps
    
end


# Reverse the direction of a set of disentangler gates:
function ReverseDisentanglers(disentanglers_in)
    
    disentanglers = deepcopy(disentanglers_in)
    disentanglers = reverse(disentanglers)
    
    for dlist in disentanglers
        for G_inf in dlist
            swapprime!(G_inf[1], 0, 1)
        end
    end
    
    return disentanglers
    
end


# Generates a unitary matrix from the QR decomp. of an input matrix:
function GetUnitary(x)
    dim=Int(sqrt(length(x)))
    A = reshape(x, (dim,dim))
    F = qr(A)
    U = Matrix(F.Q)
    for i=1:dim
        U[:,i] *= sign(dot(U[:,i],A[:,i]))
    end
    return U
end


# Folds a 2x2 unitary gate into a 2x2 tensor, or a 4x4 unitary gate into a 2x2x2x2 tensor:
function GetTensor(U, ids)
    if length(ids)==2
        reshape(U,(2,2,2,2))
        G = ITensor(U,ids[2]',ids[1]',ids[2],ids[1])
    else
        G = ITensor(U,ids[1]',ids[1])
    end
    return G
end


# Applies a two-site gate to a matrix product state:
function ApplyTwoSite(mps,G,p;cutoff=1e-8)
    orthogonalize!(mps,p)
    phi = mps[p] * mps[p+1] * G
    noprime!(phi)
    temp_inds = uniqueinds(mps[p],mps[p+1])
    U,S,V = svd(phi,temp_inds,cutoff=cutoff)
    mps[p] = U
    mps[p+1] = S*V
    return mps
end


#### Here are a set of cost functions for optimizing disentanglers: ####

# For brick-wall gate generation:
function CostFuncBW(x, mps_in, p) 
    mps = deepcopy(mps_in)
    U = GetUnitary(x)
    ids = siteinds(mps)[p:p+1]
    G = GetTensor(U,ids)
    mps = ApplyTwoSite(mps,G,p)
    cost = 1.0 - abs(BitstringOverlap(mps, [1 for i=1:length(mps)]))
    return cost
end

# For one- or two-site gate re-optimization:
function CostFunc(x,ids,block_tensor;cutoff=1e-8)
    U = GetUnitary(x)
    G = GetTensor(U,ids)
    cost = scalar(G*block_tensor)
    return cost
end

#### ####


# Generate a brick-wall like structure of two-site disentanglers:
function BrickWallDisentanglers(mps_in; depth=1, verbose=false)
    
    mps = deepcopy(mps_in)
    
    odd_sites = filter(x->isodd(x), collect(1:length(mps)-1))
    even_sites = filter(x->iseven(x), collect(1:length(mps)-1))
    
    disentanglers=[]
    
    x0 = float(reshape(Matrix(I,4,4), (16)))
    #x0 = rand(Float64, 16)
    
    for d=1:depth
        
        # The odd layer:
        odd_list = []
        
        for p in odd_sites
            
            ids = siteinds(mps)[p:p+1]
            
            f(x) = CostFuncBW(x, mps, p)
            res = bboptimize(f, x0; NumDimensions=16, SearchRange=(-1.0,1.0), MaxFuncEvals=2000, TraceMode=:silent)
            x_opt = best_candidate(res)
            inf_opt = best_fitness(res)
            
            if inf_opt < f(x0)
                U = GetUnitary(x_opt)
                G = GetTensor(U,ids)
                mps = ApplyTwoSite(mps,G,p)
            else
                U = GetUnitary(x0)
                G = GetTensor(U,ids)
            end
            
            push!(odd_list, [G,p,2])
            
        end
        
        # The even layer:
        even_list = []
        
        for p in even_sites
            
            ids = siteinds(mps)[p:p+1]
            
            f(x) = CostFuncBW(x, mps, p)
            res = bboptimize(f, x0; NumDimensions=16, SearchRange=(-1.0,1.0), MaxFuncEvals=10000, TraceMode=:silent)
            x_opt = best_candidate(res)
            inf_opt = best_fitness(res)
            
            if inf_opt < f(x0)
                U = GetUnitary(x_opt)
                G = GetTensor(U,ids)
                mps = ApplyTwoSite(mps,G,p)
            else
                U = GetUnitary(x0)
                G = GetTensor(U,ids)
            end
            
            push!(even_list, [G,p,2])
        
        end
        
        if verbose
            println("Layer $d of $depth complete!")
            println("Bitstring fidelity: ", BitstringOverlap(mps, [1 for i=1:length(mps)])^2)
        end
        
        push!(disentanglers, odd_list)
        push!(disentanglers, even_list)
        
    end
    
    return disentanglers
        
end


# Re-optimize a pre-generated set of disentanglers:
function ReOptimizeDisentanglers(mps_in, disentanglers_in; loops=1, verbose=false, cutoff=1e-8)
    
    mps = deepcopy(mps_in)
    disentanglers = deepcopy(disentanglers_in)
    
    zero_mps = MPS(siteinds(mps),[1 for i=1:length(mps)])
    
    odd_sites = filter(x->isodd(x), collect(1:length(mps)-1))
    even_sites = filter(x->iseven(x), collect(1:length(mps)-1))
    
    for l=1:loops
        
        if verbose
            println("\nCommencing loop $l of $(loops):\n")
        end
        
        for (d,dlist) in enumerate(disentanglers)
            
            # Apply left-layers to the left:
            leftlayers = disentanglers[1:d-1]
            lmps = ApplyDisentanglers(mps,leftlayers,cutoff=cutoff)
            
            # Apply right-layers to the right:
            rightlayers = ReverseDisentanglers(disentanglers[d+1:end])
            rmps = ApplyDisentanglers(zero_mps,rightlayers,cutoff=cutoff)
            
            
            for (g,G_inf) in enumerate(dlist)
                
                # Apply gates in the same layer to the left:
                clmps = deepcopy(lmps)
                for cG_inf in dlist[setdiff(collect(1:length(dlist)),[g])]
                    if cG_inf[3]==2
                        clmps = ApplyTwoSite(clmps,cG_inf[1],cG_inf[2],cutoff=cutoff)
                    else
                        phi = clmps[cG_inf[2]] * cG_inf[1]
                        noprime!(phi)
                        clmps[cG_inf[2]] = phi
                    end
                end
                
                # Prime the appropriate tensors on the right:
                prmps = deepcopy(rmps)
                setprime!(prmps[G_inf[2]], 1, tags="Site")
                if G_inf[3]==2
                    setprime!(prmps[G_inf[2]+1], 1, tags="Site")
                end
                
                # "Pre-contract" the network:
                block_tensor = clmps[1] * prmps[1]
                for p=2:length(rmps)
                    block_tensor *= clmps[p] * prmps[p]
                end
                
                # Select the correct tensor indices:
                if G_inf[3]==2
                    ids = siteinds(clmps)[G_inf[2]:G_inf[2]+1]
                else
                    ids = [siteinds(clmps)[G_inf[2]]]
                end
                
                # Set up the optimization:
                f(x) = CostFunc(x,ids,block_tensor)
                
                x0 = reshape(Array(G_inf[1], inds(G_inf[1])),4^G_inf[3])

                #res = bboptimize(f, x0; NumDimensions=4^G_inf[3], SearchRange=(-1.1,1.1), MaxFuncEvals=8000, TraceMode=:silent)
                #x_opt = best_candidate(res)
                #inf_opt = best_fitness(res)
                
                res = Optim.optimize(f, x0, LBFGS())
                x_opt = Optim.minimizer(res)
                inf_opt = Optim.minimum(res)

                if inf_opt < f(x0)
                    U = GetUnitary(x_opt)
                    G = GetTensor(U,ids)
                    disentanglers[d][g] = [G,G_inf[2],G_inf[3]]
                end
                
            end
            
            if verbose
                test_mps = ApplyDisentanglers(mps, disentanglers,cutoff=cutoff)
                print("Layer $d of $(length(disentanglers)) complete; bitstring fidelity = $(BitstringOverlap(test_mps, [1 for i=1:length(mps)])^2)\r")
                flush(stdout)
            end
            
        end
        
        if verbose
            println("\nLoop complete!\n")
        end
        
    end
    
    return disentanglers
    
end