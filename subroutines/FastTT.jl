function debug_func(arr)
    
    return abs.(arr) .> 1.0e-8
    
end


# A routine for fast construction of permutation MPOs using the FastTT factorization algorithm.

function FastTTPMPO(
        N, 
        ord1, 
        ord2, 
        isrev,
        sites; 
        vpos=N-1,
        maxdim=2^16,
        tol=1.0e-12
    )
    
    swap_inds = BubbleSort(ord1, ord2)
    
    array_list = FastTT.BubbleSort2TT(N, swap_inds, isrev, vpos, maxdim, tol)
    
    mpo = MPO(sites)
    
    comInd = Index(1,"Link") # Define a common index 

    for p=1:N
        
        siteind = sites[p]
        
        dtup = size(array_list[p])
        #display(array_list[p])
        # Drop numerical zeros:
        array_list[p][abs.(array_list[p]) .< 1e-12] .= 0.0
        
        #println(count(debug_func,  array_list[p]))

        if p==1
            
            j = Index(dtup[3],"Link")
            
            array_p = reshape(array_list[p], (4, 4, dtup[3]))
            
            mpo[p] = ITensor(array_p, dag(siteind), siteind', j)
            
            comInd = j # Set the common index to the right index of Hp
            
        elseif p==N
            
            i = comInd
            
            array_p = reshape(array_list[p], (dtup[1], 4, 4))
            
            mpo[p] = ITensor(array_p, i, dag(siteind), siteind')
            
        else
            
            i = comInd
            j = Index(dtup[3],"Link")
            
            array_p = reshape(array_list[p], (dtup[1], 4, 4, dtup[3]))
            
            mpo[p] = ITensor(array_p, i, dag(siteind), siteind', j)
            
            comInd = j # Set the common index to the right index of Hp
            
        end

    end
    
    return dense(mpo);
    
end


