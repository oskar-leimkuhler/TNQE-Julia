# Simple routines to contract MPS-MPO-MPS blocks:


# Update an inner-product block by contraction:
function UpdateBlock(
        block, 
        p,
        psi1, # primed indices of PMPO
        psi2, # unprimed indices of PMPO
        mpo1,
        mpo2
    )
    
    if mpo1==nothing && mpo2==nothing
        block *= psi2[p] * setprime(dag(psi1[p]),1, tags="Link")
    elseif mpo2==nothing
        block *= setprime(dag(psi1[p]),1) * mpo1[p] * psi2[p]
    elseif mpo1==nothing
        block *= setprime(dag(psi1[p]),1) * mpo2[p] * psi2[p]
    else
        Ax = setprime(mpo1[p],2,plev=0) * setprime(dag(psi1[p]),1)
        yB = psi2[p] * setprime(mpo2[p],2,plev=1,tags="Site")
        block *= Ax
        block *= yB
    end
    
    return block
    
end


function FullContract(
        psi1,
        psi2;
        mpo1=nothing,
        mpo2=nothing,
        combos=nothing,
        csites=nothing
    )
    
    block = ITensor(1.0)
    
    for p=1:length(psi1)
        
        block = UpdateBlock(block, p, psi1, psi2, mpo1, mpo2)
        
        if csites != nothing && p==csites[2]
            block *= combos[2]
        end
        if csites != nothing && p==csites[1]
            block *= setprime(dag(combos[1]),1)
        end
        
    end
    
    return block
    
end


# Collects partially contracted inner-product blocks \\
# (...) from two MPSs and up to two MPOs
function CollectBlocks(
        psi1,
        psi2;
        mpo1=nothing,
        mpo2=nothing,
        p0=length(psi1),
        p1=3,
        inv=false
    )
    
    p_block = ITensor(1.0)
    
    block_list = [p_block]
    
    for p = p0:sign(p1-p0):p1
        
        p_block = UpdateBlock(p_block, p, psi1, psi2, mpo1, mpo2)
        
        push!(block_list, deepcopy(p_block))
        
    end
    
    if inv
        return reverse(block_list)
    else
        return block_list
    end
    
end