# Functions for pre-plot data processing and plotting

# Packages:
using Plots
using GraphRecipes


# Generate a (directed) graph matrix for an MPS site ordering:
function GraphMat(ord; directed=true)
    
    N_sites = size(ord, 1)
    
    graph_mat = zeros((N_sites, N_sites))
    
    for i=1:N_sites-1
        graph_mat[ord[i],ord[i+1]] =1.0
    end
    
    if directed==false
        graph_mat += transpose(graph_mat)
    end
    
    return graph_mat
    
end


# Orbital graph plot:
function OrbitalGraphPlot(graph_mat; multiplier=2, mult_mat=nothing)
    
    if mult_mat!=nothing
        graph_mat .*= mult_mat
    end
    
    display(graphplot(graph_mat, 
        method=:circular,
        curves=false, 
        names=[lpad(string(i), 2, '0') for i=1:chemical_data.N_spt], 
        edgewidth=multiplier*graph_mat, 
        nodesize=0.2, 
        fontsize=8, 
        nodecolor=7, 
        nodeshape=:circle,
        linealpha=0.9,
        nodestrokealpha=0.0,
        edgecolor=:darkcyan
    ))
    
end


# 

