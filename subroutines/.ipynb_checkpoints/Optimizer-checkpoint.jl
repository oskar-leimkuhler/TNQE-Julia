# QPU-efficient generalized sweep algorithm with orbital rotations


# Optimizer function parameters data structure:
@with_kw mutable struct OptimParameters
    
    maxiter::Int=100 # Number of iterations
    numloop::Int=1 # Number of loops (not used)
    
    # Site decomposition parameters:
    delta::Vector{Float64}=[1e-3] # Size of Gaussian (QPU) noise in matrix elements
    noise::Vector{Float64}=[1e-5] # Size of DMRG two-site "noise" perturbation
    
    # Generalized eigenvalue solver parameters:
    thresh::String="inversion" # "none", "projection", or "inversion"
    eps::Float64=1e-8 # Singular value cutoff
    
    # Site decomposition (OHT expansion and GenEig) solver parameters:
    sd_method::String="geneig" # "geneig" or "triple_geneig"
    sd_thresh::String="inversion" # "none", "projection", or "inversion"
    sd_eps::Float64=1e-8 # Singular value cutoff
    sd_reps::Int=3 # Number of single-site decomp repetitions ("TripleGenEigM")
    sd_dtol::Float64=1e-4 # OHT-state overlap discard tolerance
    sd_etol::Float64=1e-4 # Energy penalty tolerance
    
end


# This function takes a one-site or two-site tensor and returns a \\
# ...list of one-hot tensors:
function OneHotTensors(T; discards=[])
    
    T_inds = inds(T)
    k = 1
    
    oht_list = []
    
    # Generate a list of index values for the one-hot states \\
    # ...of appropriate N_el, S_z symmetry:
    for c in CartesianIndices(Array(T, T_inds))
        c_inds = Tuple(c)
        ivs = []
        for i=1:length(T_inds)
            push!(ivs, T_inds[i]=>c_inds[i])
        end
        
        oht = onehot(Tuple(ivs)...)
        
        if (flux(oht)==flux(T))
            if !(k in discards)
                push!(oht_list, oht)
            end
            k += 1
        end
    end
    
    return oht_list
    
end


# Converts an OHT-MPS into a sparse vector:
function SparseVecOHT(phi, p, nsite, oht)
    if nsite==2
        phi_tens = reduce(*, reduce(vcat, [phi[1:p-1], [oht], phi[p+2:end]]));
    elseif nsite==1
        phi_tens = reduce(*, reduce(vcat, [phi[1:p-1], [oht], phi[p+1:end]]));
    else
        phi_tens = reduce(*, phi);
    end
    phi_vec = sparse(reshape(Array(phi_tens, siteinds(phi)), (4^length(phi))))
    return phi_vec
end

"""
# This is should be a little speedier than the naive implementation above ^^
function SparseVecOHTs(phi, p, nsite)
    
    if nsite==2
        T = phi[p]*phi[p+1]
        T_inds = siteinds(phi)[p:p+1]
    elseif nsite==1
        T = phi[p]
        T_inds = [siteinds(phi)[p]]
    elseif nsite==0
        return [ITensor(1.0)], [SparseVec(phi)]
    end

    N = length(phi)
    
    lb_vecs = []
    rb_vecs = []
    
    if p != 1 # Contract lblock
        lblock = reduce(*, phi[1:p-1])
        # Contract with a one-hot tensor for every value of the link index:
        lbi = uniqueinds(lblock, inds(lblock, tags="Site"))[1] # Non-site index
        for l=1:dim(lbi)
            lb = lblock * dag(onehot(lbi=>l))
            push!(lb_vecs, sparse(reshape(Array(lb, inds(lb)), (4^(p-1)))))
        end
        push!(T_inds, dag(lbi))
    end
    
    if p+nsite-1 != N # Contract rblock
        rblock = reduce(*, phi[p+nsite:end])
        # Contract with a one-hot tensor for every value of the link index:
        rbi = uniqueinds(rblock, inds(rblock, tags="Site"))[1] # Non-site index
        rb_vecs = []
        for r=1:dim(rbi)
            rb = rblock * dag(onehot(rbi=>r))
            push!(rb_vecs, sparse(reshape(Array(rb, inds(rb)), (4^(N-(p+nsite)+1)))))
        end
        push!(T_inds, dag(rbi))
    end
    
    oht_list = []
    vec_list = []
    
    I_4 = sparse(I, 4, 4)
    
    # Generate a list of index values for the one-hot states \\
    # ...of appropriate N_el, S_z symmetry:
    for c in CartesianIndices(Array(T, T_inds))
        c_inds = Tuple(c)
        ivs = []
        for i=1:length(T_inds)
            push!(ivs, T_inds[i]=>c_inds[i])
        end
        
        oht = onehot(Tuple(ivs)...)
        
        if (flux(oht)==flux(T))
            
            # Construct sparse vector:
            sitevec = kron(I_4[:,c_inds[1]], I_4[:,c_inds[2]])
            if p==1
                vec = reduce(kron, [sitevec, rb_vecs[c_inds[3]]])
            elseif p+nsite-1==N
                vec = reduce(kron, [lb_vecs[c_inds[3]], sitevec])
            else
                vec = reduce(kron, [lb_vecs[c_inds[3]], sitevec, rb_vecs[c_inds[4]]])
            end
            
            push!(oht_list, oht)
            push!(vec_list, normalize(vec))
        end
    end
    
    return oht_list, vec_list
    
end
"""


# Check if states are overlapping too much and discard if they are:
function DiscardOverlapping(H_in, S_in, M_in, oht_in; tol=0.01, kappa_max=1e10, verbose=false)
    
    H_full = deepcopy(H_in)
    S_full = deepcopy(S_in)
    
    M_list = deepcopy(M_in)
    
    M = length(M_list)
    M_tot = sum(M_list)
    
    oht_list = deepcopy(oht_in)
    
    # Iterate over the comp index objects in oht_list:
    for j=M:(-1):1
        
        M_j = M_list[j]
        
        # Iterate over the states in the comp index object:
        for k=M_j:(-1):1
            
            # j0 and j1, keeping track of discards:
            j0, j1 = sum(M_list[1:j-1])+1, sum(M_list[1:j])
            
            # The current column of S_full, keeping track of discards:
            col = j0 + k - 1
            
            do_discard = false
            
            if (M_list[j] > 1)
                
                # Check for any Infs or NaNs in that column:
                if (Inf in S_full[:,col]) || (true in isnan.(S_full[:,col])) || (Inf in H_full[:,col]) || (true in isnan.(H_full[:,col]))
                    do_discard = true
                end
                
            end
                
            if (j != M) && (M_list[j] > 1)
                
                # First check the overlap with the previous subspace is not too large:
                S_red = deepcopy(S_full[j1+1:M_tot,j1+1:M_tot])
                vphi = deepcopy(S_full[j1+1:M_tot,col])
                
                F = svd(S_red, alg=LinearAlgebra.QRIteration())
                rtol = sqrt(eps(real(float(oneunit(eltype(S_red))))))
                S_inv = zeros(length(F.S))
                for l=1:length(F.S)
                    if F.S[l] >= maximum(F.S)*rtol
                        S_inv[l] = 1.0/F.S[l]
                    else
                        S_inv[l] = 0.0
                    end
                end
                S_red_inv = transpose(F.Vt) * Diagonal(S_inv) * transpose(F.U) 

                sqnm = transpose(vphi) * S_red_inv * vphi

                # Also double-check that the condition number does not blow up:
                kappa_new = cond(S_full[col:M_tot,col:M_tot])
                
                # Mark the state for discarding:
                if (sqnm > 1.0-tol) || (kappa_new > kappa_max) || isnan(kappa_new) || (kappa_new == Inf)
                    do_discard=true
                end
                
            end
            
            if do_discard
                
                H_full = H_full[1:end .!= col, 1:end .!= col]
                S_full = S_full[1:end .!= col, 1:end .!= col]
                
                oht_list[j] = oht_list[j][1:end .!= k]
                
                M_list[j] -= 1
                M_tot -= 1
                
            end
            
        end # loop over k
        
    end # loop over j
            
    return H_full, S_full, M_list, oht_list
    
end


function FSWAPModify(H_in, S_in, M_list, oht_list, sites, nsite, q_set, do_swaps)
    
    H_full = deepcopy(H_in)
    S_full = deepcopy(S_in)
    
    M = length(M_list)
    
    for i1=1:M
        
        if nsite[i1]==2

            i10 = sum(M_list[1:i1-1])+1
            i11 = sum(M_list[1:i1])

            if do_swaps[i1]

                # Construct FSWAP matrix:
                fswap = RotationTensor(sites, q_set[i1][1]; dim=4, rotype="fswap")
                fswap_mat = zeros(M_list[i1], M_list[i1])
                for k1=1:M_list[i1], k2=1:M_list[i1]
                    fswap_mat[k1,k2] = scalar(oht_list[i1][k1] * fswap * dag(setprime(oht_list[i1][k2],1, tags="Site")))
                end

                for i2=1:M

                    i20 = sum(M_list[1:i2-1])+1
                    i21 = sum(M_list[1:i2])

                    # Left-mult all subblocks in row i2:
                    H_full[i10:i11,i20:i21] = fswap_mat * H_full[i10:i11, i20:i21]
                    S_full[i10:i11,i20:i21] = fswap_mat * S_full[i10:i11, i20:i21]

                    # Right-mult all subblocks in col i2:
                    H_full[i20:i21,i10:i11] = H_full[i20:i21, i10:i11] * fswap_mat
                    S_full[i20:i21,i10:i11] = S_full[i20:i21, i10:i11] * fswap_mat

                end

            end
            
        end

    end
    
    return H_full, S_full
    
end


function GivensModify(H_in, S_in, M_list, oht_list, sites, nsite, q_set, theta_opt)
    
    H_full = deepcopy(H_in)
    S_full = deepcopy(S_in)
    
    M = length(M_list)
    
    for i1=1:M
        
        if nsite[i1]==2

            i10 = sum(M_list[1:i1-1])+1
            i11 = sum(M_list[1:i1])

            # Construct Givens matrix:
            grot = RotationTensor(sites, q_set[i1][1]; dim=4, rotype="givens", theta=theta_opt[i1]);
            grot_mat = zeros(M_list[i1], M_list[i1])
            for k1=1:M_list[i1], k2=1:M_list[i1]
                grot_mat[k1,k2] = scalar(oht_list[i1][k1] * grot * dag(setprime(oht_list[i1][k2],1, tags="Site")))
            end

            for i2=1:M

                i20 = sum(M_list[1:i2-1])+1
                i21 = sum(M_list[1:i2])

                # Left-mult all subblocks in row i2:
                H_full[i10:i11,i20:i21] = transpose(grot_mat) * H_full[i10:i11, i20:i21]
                S_full[i10:i11,i20:i21] = transpose(grot_mat) * S_full[i10:i11, i20:i21]

                # Right-mult all subblocks in col i2:
                H_full[i20:i21,i10:i11] = H_full[i20:i21, i10:i11] * grot_mat
                S_full[i20:i21,i10:i11] = S_full[i20:i21, i10:i11] * grot_mat

            end
            
        end

    end
    
    return H_full, S_full
    
end


# Performs a sequence of alternating single-site decompositions \\
# ...to minimize truncation error from the two-site decomposition:
function TripleGenEigM(
        H_full,
        S_full,
        oht_list,
        T_init,
        M_list,
        op,
        nsite,
        ids,
        maxdim;
        nrep=op.sd_reps
    )
    
    M = length(M_list)
    
    # Generate the initial T_i:
    T_list = T_init
    
    E_new = 0.0 #E[1]
    
    T1_oht = Any[]
    V_list = Any[]
        
    T1_list = []
    
    # Probably a more compact way to write this...
    pos = zeros(Int64, M)
    c = 1
    for a=1:length(pos)
        if nsite[a]==2
            pos[a]=c
            c += 1
        end
    end
    
    for r=1:nrep, s=1:2
            
        T1_mats = []
        M1_list = Int[]

        T1_oht = Any[]
        V_list = Any[]

        for i=1:M
                
            if nsite[i]==2

                # Split by SVD and form single-site tensors
                linds = ids[pos[i]][s]
                #println(length(T_list))
                U, S, V = svd(T_list[i], linds, maxdim=maxdim, mindim=1, alg="qr_iteration")#, min_blockdim=1)
                push!(V_list, V)
                
                # Compute two-site tensors for the single-site one-hot states
                push!(T1_oht, OneHotTensors(U * S))
                push!(M1_list, length(T1_oht[end]))
                
                T1_twosite = [T1_oht[end][k] * V for k=1:M1_list[end]]

                # Form contraction matrix
                T1_mat = zeros(M1_list[i],M_list[i])
                for k1=1:M1_list[i], k2=1:M_list[i]
                    T1_mat[k1,k2] = scalar(T1_twosite[k1] * dag(oht_list[i][k2]))
                end

                push!(T1_mats, T1_mat)
                
            else
                
                push!(M1_list, M_list[i])
                push!(T1_mats, Matrix(I, M_list[i], M_list[i]))
                push!(T1_oht, oht_list[i])
                push!(V_list, 1.0)
                
            end

        end

        # Form reduced subspace H, S matrices
        H_red = zeros((sum(M1_list),sum(M1_list)))
        S_red = zeros((sum(M1_list),sum(M1_list)))

        for i=1:M, j=i:M

            i0 = sum(M_list[1:i-1])+1
            i1 = sum(M_list[1:i])
            j0 = sum(M_list[1:j-1])+1
            j1 = sum(M_list[1:j])

            i10 = sum(M1_list[1:i-1])+1
            i11 = sum(M1_list[1:i])
            j10 = sum(M1_list[1:j-1])+1
            j11 = sum(M1_list[1:j])

            H_red[i10:i11, j10:j11] = T1_mats[i] * H_full[i0:i1,j0:j1] * transpose(T1_mats[j])
            H_red[j10:j11, i10:i11] = transpose(H_red[i10:i11, j10:j11])

            S_red[i10:i11, j10:j11] = T1_mats[i] * S_full[i0:i1,j0:j1] * transpose(T1_mats[j])
            S_red[j10:j11, i10:i11] = transpose(S_red[i10:i11, j10:j11])

        end
        
        # Discard overlapping in the reduced space:
        H_red, S_red, M1_list, T1_oht = DiscardOverlapping(H_red, S_red, M1_list, T1_oht, tol=op.sd_dtol)

        # Diagonalize in reduced one-site space
        E_, C_, kappa_ = SolveGenEig(
            H_red,
            S_red,
            thresh=op.sd_thresh,
            eps=op.sd_eps
        )

        E_new = E_[1]

        T1_list = []

        # Convert coeffs back to one-site space
        for i=1:M

            i10 = sum(M1_list[1:i-1])+1
            i11 = sum(M1_list[1:i])

            t_vec = normalize(real.(C_[i10:i11,1]))
            #println(M1_list[i])
            T_i = sum([t_vec[k] * T1_oht[i][k] for k=1:M1_list[i]])
            push!(T1_list, T_i)
        end
        
        # Replace the two-site tensor for the next loop(!):
        for i=1:M
            if nsite[i] == 2
                T_list[i] = T1_list[i] * V_list[i]
            end
        end
        
    end
    
    return T1_list, V_list, E_new
    
end


# A "sweep" algorithm based on the two-site decomposition:
function TwoSiteBlockSweep!(
        sdata::SubspaceProperties,
        op::OptimParameters;
        nsite=nothing,
        jperm=nothing,
        rotype="fswap",
        verbose=false,
        return_calls=false
    )
    
    M = sdata.mparams.M
    N = sdata.chem_data.N_spt
    
    # Default is to just do twosite on everything
    if nsite==nothing
        nsite = [2 for i=1:M]
    end
    
    # Default is to cycle through states one at a time:
    if jperm==nothing
        jperm = circshift(collect(1:M),1)
    end
    
    tot_calls = 0
    
    for l=1:op.maxiter
        
        ShuffleStates!(sdata, perm=jperm)
        
        swap_counter = 0
        
        # Orthogonalize to site 1:
        for j=1:M
            orthogonalize!(sdata.phi_list[j], 1)
        end
        
        # Iterate over all bonds:
        for p=1:N-1
            
            # Compile the one-hot tensor list:
            oht_list = [[ITensor(1.0)] for i=1:M]
            
            T_tensor_list = []
            oht_list = []
            oht_vecs = []
            oht_hvecs = []
            
            for i=1:M
                if nsite[i]==0
                    push!(T_tensor_list, ITensor(1.0))
                    push!(oht_list, [ITensor(1.0)])
                    push!(oht_vecs, [normalize(sdata.G_list[i] * SparseVec(sdata.phi_list[i]))])
                    push!(oht_hvecs, [sdata.H_sparse * oht_vecs[end][1]])
                else
                    if nsite[i]==1
                        push!(T_tensor_list, sdata.phi_list[i][p])
                    elseif nsite[i]==2
                        push!(T_tensor_list, sdata.phi_list[i][p] * sdata.phi_list[i][p+1])
                    end
                    push!(oht_list, OneHotTensors(T_tensor_list[end]))
                    vecs = [normalize(sdata.G_list[i] * SparseVecOHT(sdata.phi_list[i], p, nsite[i], oht)) for oht in oht_list[end]]
                    hvecs = [sdata.H_sparse * vec for vec in vecs]
                    push!(oht_vecs, vecs)
                    push!(oht_hvecs, hvecs)
                end
            end
            
            M_list = [length(oht_list[i]) for i=1:M]
            M_tot = sum(M_list)
            
            # Construct the full H, S matrices:
            H_full = zeros(Float64, (M_tot, M_tot))
            S_full = zeros(Float64, (M_tot, M_tot))
            
            for i1=1:M, i2=i1:M
                
                i10, i11 = sum(M_list[1:i1-1])+1, sum(M_list[1:i1])
                i20, i21 = sum(M_list[1:i2-1])+1, sum(M_list[1:i2])
                
                H_array = zeros(M_list[i1],M_list[i2])
                S_array = zeros(M_list[i1],M_list[i2])
                
                for k1=1:M_list[i1], k2=1:M_list[i2]
                    
                    H_array[k1,k2] = transpose(oht_vecs[i1][k1]) * oht_hvecs[i2][k2]
                    S_array[k1,k2] = transpose(oht_vecs[i1][k1]) * oht_vecs[i2][k2]
                    
                end
                
                # Simulate noise:
                dH = randn((M_list[i1], M_list[i2]))
                dH *= op.delta[1]
                
                dS = randn((M_list[i1], M_list[i2]))
                dS *= op.delta[2]

                H_full[i10:i11, i20:i21] = H_array + dH
                H_full[i20:i21, i10:i11] = conj.(transpose(H_array + dH))
                
                S_full[i10:i11, i20:i21] = S_array + dS
                S_full[i20:i21, i10:i11] = conj.(transpose(S_array + dS))
                
                if i1 != i2 # Add QPU calls to total
                    tot_calls += 2*M_list[i1]*M_list[i2]
                end
                
            end
            
            # Make a copy to revert to at the end if the energy penalty is violated:
            sdata_copy = copy(sdata)
                
            H_all, S_all = deepcopy(H_full), deepcopy(S_full)
            oht_all = deepcopy(oht_list)
            M_list_all = deepcopy(M_list)
            
            # Discard overlapping states:
            H_full, S_full, M_list, oht_list = DiscardOverlapping(
                H_full, 
                S_full, 
                M_list, 
                oht_list, 
                tol=op.sd_dtol,
                kappa_max=1e10
            )
            
            E, C, kappa = SolveGenEig(
                H_full,
                S_full,
                thresh=op.sd_thresh,
                eps=op.sd_eps
            )
            
            # Initialize variables for FSWAPs/rotations:
            do_swaps = [false for i=1:M]
            theta_opt = [0.0 for i=1:M]
            
            if rotype=="fswap"
                
                # Test FSWAPS on each optimized two-site tensor:
                
                for i=1:M
                    if nsite[i]==2

                        i0 = sum(M_list[1:i-1])+1
                        i1 = sum(M_list[1:i])

                        t_vec = normalize(C[i0:i1,1])
                        T = sum([t_vec[k]*oht_list[i][k] for k=1:M_list[i]])

                        linds = commoninds(T, sdata.phi_list[i][p])
                        
                        do_swaps[i] = TestFSWAP2(T, linds, sdata.mparams.mps_maxdim, crit="fidelity")

                    end

                end
                
            elseif rotype=="givens"
                
                # Optimize rotation angles to minimize truncation error:
                
                for i=1:M
                    if nsite[i]==2

                        i0 = sum(M_list[1:i-1])+1
                        i1 = sum(M_list[1:i])

                        t_vec = normalize(C[i0:i1,1])
                        T = sum([t_vec[k]*oht_list[i][k] for k=1:M_list[i]])

                        linds = commoninds(T, sdata.phi_list[i][p])
                        
                        theta_opt[i] = GThetaOpt(T, sdata, i, p, heuristic="trunc", beta=0.1)

                    end
                end
            end
            
            # Modify the H, S matrices to encode the FSWAPs:
            if rotype=="fswap"
                H_all, S_all = FSWAPModify(
                    H_all, 
                    S_all, 
                    M_list_all, 
                    oht_all, 
                    sdata.sites, 
                    nsite, 
                    [[p,p+1] for i=1:M], 
                    do_swaps
                )
            elseif rotype=="givens"
                H_all, S_all = GivensModify(
                    H_all, 
                    S_all, 
                    M_list_all, 
                    oht_all, 
                    sdata.sites, 
                    nsite, 
                    [[p,p+1] for i=1:M], 
                    theta_opt
                )
            end
            
            # Discard overlapping states:
            H_full, S_full, M_list, oht_list = DiscardOverlapping(
                H_all, 
                S_all, 
                M_list_all, 
                oht_all, 
                tol=op.sd_dtol,
                kappa_max=1e10
            )
            
            E, C, kappa = SolveGenEig(
                H_full,
                S_full,
                thresh=op.sd_thresh,
                eps=op.sd_eps
            )
            
            # Generate the initial T_i:
            T_init = []

            for i=1:M

                i0 = sum(M_list[1:i-1])+1
                i1 = sum(M_list[1:i])

                t_vec = normalize(C[i0:i1,1])
                T_i = sum([t_vec[k] * oht_list[i][k] for k=1:M_list[i]])

                push!(T_init, T_i)

            end
            
            inds_list = []
            for j=1:M
                if nsite[j]==2
                    linds = commoninds(T_tensor_list[j], sdata.phi_list[j][p])
                    rinds = commoninds(T_tensor_list[j], sdata.phi_list[j][p+1])
                    push!(inds_list, [linds, rinds])
                end
            end

            # Do TripleGenEig on all states to lower energy:
            T_list, V_list, E_new = TripleGenEigM(
                H_all,
                S_all,
                oht_all,
                T_init,
                M_list_all,
                op,
                nsite,
                inds_list,
                sdata.mparams.mps_maxdim,
                nrep=20
            )
            
            if (real(E_new) > real(sdata.E[1]) + op.sd_etol)
                
                if rotype=="fswap"
                    # Revert to non-optimized starting guess (with FSWAPs applied):
                    fswap = RotationTensor(sdata.sites, p; dim=4, rotype="fswap")
                    for i=1:M
                        if do_swaps[i]
                            T_tensor_list[i] *= fswap 
                            noprime!(T_tensor_list[i], tags="Site")
                        end
                    end
                elseif rotype=="givens"
                    # Revert to non-optimized starting guess (with Givens rotations applied):
                    for i=1:M
                        grot = RotationTensor(sdata.sites, p; dim=4, rotype="givens", theta=theta_opt[i]);
                        T_tensor_list[i] *= grot
                        noprime!(T_tensor_list[i], tags="Site")
                    end
                end
                
                # Do TripleGenEig on all states to lower energy:
                T_list, V_list, E_new = TripleGenEigM(
                    H_all,
                    S_all,
                    oht_all,
                    T_tensor_list,
                    M_list_all,
                    op,
                    nsite,
                    inds_list,
                    sdata.mparams.mps_maxdim,
                    nrep=20
                )
                
            end

            do_replace = true
            for i=1:M
                if (NaN in T_list[i]) || (Inf in T_list[i])
                    do_replace = false
                end
            end
            
            if (real(E_new) < real(sdata.E[1]) + op.sd_etol) && do_replace
                
                # Update params, orderings/rotations:
                
                if rotype=="fswap"
                    
                    for j=1:M
                        if nsite[j]==2 && do_swaps[j]

                            swap_counter += 1
                            
                            I_4 = sparse([1 0 0 0;
                                          0 1 0 0;
                                          0 0 1 0;
                                          0 0 0 1])
                            
                            rmat = sparse(RotationMatrix(dim=4, rotype="fswap"))

                            sdata.G_list[j] *= reduce(kron, reverse(reduce(vcat, [[I_4 for q=1:p-1],[rmat],[I_4 for q=p+2:N]])))
                        end
                    end
                    
                elseif rotype=="givens"
                    
                    for j=1:M
                        if nsite[j]==2
                            
                            swap_counter += 1

                            I_4 = sparse([1 0 0 0;
                                          0 1 0 0;
                                          0 0 1 0;
                                          0 0 0 1])

                            rmat = sparse(RotationMatrix(dim=4, rotype="givens", theta=theta_opt[j]))
                            
                            sdata.G_list[j] *= reduce(kron, reverse(reduce(vcat, [[I_4 for q=1:p-1],[rmat],[I_4 for q=p+2:N]])))
                        end
                    end

                end
                
            else # Revert to previous parameters:
                
                for j=1:M
                    
                    if nsite[j]==2
                        U,S,V = svd(
                            T_tensor_list[j], 
                            commoninds(sdata.phi_list[j][p], T_tensor_list[j]),
                            alg="qr_iteration",
                            maxdim=sdata.mparams.mps_maxdim
                        )

                        V_list[j] = U
                        T_list[j] = S*V
                    elseif nsite[j]==1
                        
                        T_list[j] = T_tensor_list[j]
                        
                    end
                    
                end
                
            end

            # Regardless of replacement, update state:
            for j=1:M
                
                if nsite[j]==2
                    
                    #T_new = V_list[j] * T_list[j]
                    sdata.phi_list[j][p] = V_list[j]
                    sdata.phi_list[j][p+1] = T_list[j]
                    
                elseif nsite[j]==1
                    
                    T_new = T_list[j]*sdata.phi_list[j][p+1]
                    
                    # Replace the tensors of the MPS:
                    spec = ITensors.replacebond!(
                        sdata.phi_list[j],
                        p,
                        T_new;
                        maxdim=sdata.mparams.mps_maxdim,
                        #eigen_perturbation=drho,
                        ortho="left",
                        normalize=true,
                        svd_alg="qr_iteration"
                        #min_blockdim=1
                    )
                    
                elseif nsite[j]==0
                    #T_new = sdata.phi_list[j][p]*sdata.phi_list[j][p+1]
                end
                
                if nsite[j] > 0
                    
                    T_new = sdata.phi_list[j][p]*sdata.phi_list[j][p+1]

                    # Generate the DMRG "noise" term:
                    pmpo = ITensors.ProjMPO(sdata.H_mpo)
                    ITensors.set_nsite!(pmpo,2)
                    ITensors.position!(pmpo, sdata.phi_list[j], p)
                    drho = op.noise[1]*ITensors.noiseterm(pmpo,T_new,"left")

                    # Replace the tensors of the MPS:
                    spec = ITensors.replacebond!(
                        sdata.phi_list[j],
                        p,
                        T_new;
                        maxdim=sdata.mparams.mps_maxdim,
                        eigen_perturbation=drho,
                        ortho="left",
                        normalize=true,
                        svd_alg="qr_iteration"
                        #min_blockdim=1
                    )

                    # Make sure new state is normalized:
                    #normalize!(sdata.phi_list[j])
                    sdata.phi_list[j][p+1] *= 1.0/sqrt(norm(sdata.phi_list[j]))
                    
                end
                
            end
            
            GenSubspaceMats!(sdata)
            SolveGenEig!(sdata)
            
            # Double-check that the energy is not too high!
            if sdata.E[1] > sdata_copy.E[1] + op.sd_etol
                
                # Revert to previous subspace:
                copyto!(sdata, sdata_copy)
                
                for j=1:M
                    
                    if nsite[j] == 1 || nsite[j] == 2

                        T_j = sdata.phi_list[j][p]*sdata.phi_list[j][p+1]

                        spec = ITensors.replacebond!(
                            sdata.phi_list[j],
                            p,
                            T_j;
                            maxdim=sdata.mparams.mps_maxdim,
                            #eigen_perturbation=drho,
                            ortho="left",
                            normalize=true,
                            svd_alg="qr_iteration"
                            #min_blockdim=1
                        )

                        # Make sure new state is normalized:
                        #normalize!(sdata.phi_list[j])
                        sdata.phi_list[j][p+1] *= 1.0/sqrt(norm(sdata.phi_list[j]))
                        
                    end

                end
                
            end

            # Print some output
            if verbose
                print("Loop: ($(l)/$(op.maxiter)); ")
                print("Bond: $(p)/$(N-1); ")
                print("#swaps: $(swap_counter); ")
                print("E_min = $(round(sdata.E[1], digits=5)); ") 
                print("Delta = $(round(sdata.E[1]-sdata.chem_data.e_fci+sdata.chem_data.e_nuc, digits=5)); ")
                print("kappa_full = $(round(cond(S_full), sigdigits=3)); ")
                print("kappa = $(round(sdata.kappa, sigdigits=3))     \r")
                flush(stdout)
            end

        end # loop over p
            
        for j=1:M # Make sure these states are normalized:
            normalize!(sdata.phi_list[j])
        end
        
        # Recompute H, S, E, C, kappa:
        GenSubspaceMats!(sdata)
        SolveGenEig!(sdata)
        
        l += 1
        
    end # loop over j-pairs
    
    if verbose
        println("\nDone!\n")
    end
    
    if return_calls
        return tot_calls
    end
    
end
