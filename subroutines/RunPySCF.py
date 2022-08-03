# Python functions to run HF and FCI calculations, and save the output in HDF5 format

# Packages:
import pyscf
from pyscf import fci, ao2mo
import numpy as np
import h5py
import datetime
import os
import configparser


# Parses the geometry specifications in the config file to generate a set of geometry strings to be passed to PySCF:
def ParseGeometries(config):
    
    mconf = config['MOLECULE PROPERTIES']
    gconf = config['GEOMETRIES']
    
    num_atoms = int(mconf['num_atoms'])
    
    # Atom property labels:
    altype = [f'atom{i}type' for i in range(1,num_atoms+1)]
    alx0 = [f'atom{i}x0' for i in range(1,num_atoms+1)]
    aly0 = [f'atom{i}y0' for i in range(1,num_atoms+1)]
    alz0 = [f'atom{i}z0' for i in range(1,num_atoms+1)]
    alx1 = [f'atom{i}x1' for i in range(1,num_atoms+1)]
    aly1 = [f'atom{i}y1' for i in range(1,num_atoms+1)]
    alz1 = [f'atom{i}z1' for i in range(1,num_atoms+1)]
    
    atom_types = [mconf[altype[i]] for i in range(num_atoms)]
    
    if gconf.getboolean('geometry_range')==True:
        
        # Implement the range functionality
        atom_positions = [ [float(gconf[alx0[i]]),float(gconf[aly0[i]]),float(gconf[alz0[i]])] for i in range(num_atoms)]
        
        geometry_string = ""
        
        for i in range(num_atoms):
            geometry_string += atom_types[i]
            
            for coord in range(3):
                geometry_string += " "
                geometry_string += str(atom_positions[i][coord])
                
            geometry_string += "; "
            
        geometries = [geometry_string]
    
    else:
        
        atom_positions = [ [float(gconf[alx0[i]]),float(gconf[aly0[i]]),float(gconf[alz0[i]])] for i in range(num_atoms)]
        
        geometry_string = ""
        
        for i in range(num_atoms):
            geometry_string += atom_types[i]
            
            for coord in range(3):
                geometry_string += " "
                geometry_string += str(atom_positions[i][coord])
                
            geometry_string += "; "
            
        geometries = [geometry_string]
        
    return geometries


# Runs the PySCF calculations and saves the output to a HDF5 file:
def RunPySCF(config):
    
    mol_name = config['MOLECULE PROPERTIES']['mol_name']
    basis = config['CALCULATION PARAMETERS']['basis']
    run_fci = config['CALCULATION PARAMETERS'].getboolean('run_fci')
    
    geometries = ParseGeometries(config)
    
    wd = os.getcwd()+"/../datasets/pyscf_data/"
    
    datestr = datetime.datetime.now()
    
    filename = wd + mol_name + "_" + basis + "_" + datestr.strftime("%m%d%y%%%H%M%S") + ".hdf5"
    
    with h5py.File(filename, 'w') as f:
        
        f.create_dataset("mol_name", data=mol_name)
        f.create_dataset("basis", data=basis)
        f.create_dataset("geometries", data=geometries)
    
        for geometry in geometries:

            mol_obj = pyscf.gto.M(atom=geometry, basis=basis)

            rhf_obj = pyscf.scf.RHF(mol_obj)
            e_rhf = rhf_obj.kernel()

            if run_fci==True:
                fci_obj = fci.FCI(rhf_obj)
                e_fci = fci_obj.kernel()[0]
            else:
                e_fci = "N/A"

            h1e = mol_obj.intor("int1e_kin") + mol_obj.intor("int1e_nuc")
            h2e = mol_obj.intor("int2e")

            scf_c = rhf_obj.mo_coeff

            h1e = scf_c.T @ h1e @ scf_c
            h2e = ao2mo.kernel(h2e, scf_c)

            grp = f.create_group(geometry)
            
            rhf_data = grp.create_dataset("e_rhf", data=e_rhf)
            fci_data = grp.create_dataset("e_fci", data=e_fci)
            h1e_data = grp.create_dataset("h1e", data=h1e)
            h2e_data = grp.create_dataset("h2e", data=h2e)
            nel_data = grp.create_dataset("nel", data=mol_obj.nelectron)
            nuc_data = grp.create_dataset("nuc", data=mol_obj.energy_nuc())