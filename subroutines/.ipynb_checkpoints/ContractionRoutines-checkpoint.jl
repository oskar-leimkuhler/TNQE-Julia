# Simple routines to contract MPS-MPO-MPO-MPO-MPS blocks:

# Update an inner-product block by contraction:
function UpdateBlock(
        sdata,
        observable,
        block, 
        p,
        i, # primed index
        j # unprimed index
    )
    
    block *= setprime(dag(sdata.psi_list[i][p]),1)
    block *= setprime(sdata.perm_ops[i][p],2,plev=0)
    block *= dag(setprime(observable[p],2,plev=0,tags="Site"), tags="Site")
    block *= dag(swapprime(sdata.perm_ops[j][p],0,1,tags="Site"),tags="Site")
    block *= sdata.psi_list[j][p]
    
    return block
    
end


# Fully contract a matrix element block:
function FullContract(
        sdata,
        observable,
        i, # primed index
        j # unprimed index
    )
    
    block = ITensor(1.0)
    
    for p=1:length(sdata.sites)
        
        block = UpdateBlock(
            sdata,
            observable,
            block, 
            p,
            i, # primed index
            j # unprimed index
        )
        
    end
    
    return scalar(block)
    
end



# Collects partially contracted inner-product blocks
function CollectBlocks(
        sdata,
        observable,
        i, # primed index
        j, # unprimed index
        p0=length(sdata.sites),
        p1=3;
        inv=false
    )
    
    p_block = ITensor(1.0)
    
    block_list = [p_block]
    
    for p = p0:sign(p1-p0):p1
        
        p_block = UpdateBlock(
            sdata,
            observable,
            p_block, 
            p,
            i, # primed index
            j # Unprimed index
        )
        
        push!(block_list, deepcopy(p_block))
        
    end
    
    if inv
        return reverse(block_list)
    else
        return block_list
    end
    
end