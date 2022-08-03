# Functions for preparing the chemical data structure (read in from the HDF5 output from PySCF)

# Packages:
using ITensors
using HDF5


# The chemical data structure:
struct ChemProperties
    mol_name::String
    basis::String
    geometry::String
    e_rhf::Float64
    e_fci::Float64
    e_nuc::Float64
    h1e
    h2e
    N_el::Int
    N::Int
    N_spt::Int
end


# Read in the chemical data from the HDF5 directory:
function ReadIn(fname)
    
    fid = h5open(fname, "r")
    
    mol_name = read(fid, "mol_name")
    basis = read(fid, "basis")
    geometries = read(fid, "geometries")
    
    cdata_list = ChemProperties[]

    for geometry in geometries
        
        grp = fid[geometry]
        
        e_rhf = read(grp, "e_rhf")
        e_fci = read(grp, "e_fci")
        h1e = read(grp, "h1e")
        h2e = read(grp, "h2e")
        N_el = read(grp, "nel")
        e_nuc = read(grp, "nuc")
        
        N_spt = size(h1e, 1)
        N = 2*N_spt
        
        new_cdata = ChemProperties(mol_name, basis, geometry, e_rhf, e_fci, e_nuc, h1e, h2e, N_el, N, N_spt)
        
        push!(cdata_list, new_cdata)
        
    end

    close(fid)
    
    return cdata_list
    
end