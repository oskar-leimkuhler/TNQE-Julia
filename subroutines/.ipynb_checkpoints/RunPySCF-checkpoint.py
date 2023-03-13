# Python functions to run HF and FCI calculations, and save the output in HDF5 format

# Packages:
import pyscf
from pyscf import fci, ao2mo, lo
from pyscf.tools import cubegen
import numpy as np
import h5py
import datetime
import os
import configparser


# Runs the PySCF calculations and saves the output to a HDF5 file:
def RunPySCF(config, gen_cubes=False):
    
    mol_name = config['MOLECULE PROPERTIES']['mol_name']
    mol_spin = config['MOLECULE PROPERTIES'].getint('mol_spin', fallback=0)
    mol_charge = config['MOLECULE PROPERTIES'].getint('mol_charge', fallback=0)
    basis = config['CALCULATION PARAMETERS']['basis']
    run_rohf = config['CALCULATION PARAMETERS'].getboolean('run_rohf', fallback=False)
    run_fci = config['CALCULATION PARAMETERS'].getboolean('run_fci', fallback=False)
    loc_orbs = config['CALCULATION PARAMETERS'].getboolean('loc_orbs', fallback=False)
    xyz_file = config['GEOMETRIES']['xyz_file']
    
    geometries = ["../configs/xyz_files/"+xyz_file]
    
    wd = os.getcwd()+"/../datasets/pyscf_data/"
    wd_o = os.getcwd()+"/../datasets/orbs/"
    
    datestr = datetime.datetime.now()
    
    filename = wd + mol_name + "_" + basis + "_" + datestr.strftime("%m%d%y%%%H%M%S") + ".hdf5"
    
    with h5py.File(filename, 'w') as f:
        
        f.create_dataset("mol_name", data=mol_name)
        f.create_dataset("basis", data=basis)
        f.create_dataset("geometries", data=geometries)
    
        for geometry in geometries:
            
            print(os.path.isfile(geometry))

            mol_obj = pyscf.gto.M(atom=geometry)
            mol_obj.basis = basis
            mol_obj.charge = mol_charge
            mol_obj.spin = mol_spin
            
            if run_rohf:
                rhf_obj = pyscf.scf.ROHF(mol_obj)
                e_rhf = rhf_obj.kernel()
            else:
                """
                mf = pyscf.scf.RHF(mol_obj)
                mf.conv_tol=1e-2
                mf.kernel()
                mo_init = mf.mo_coeff
                mocc_init = mf.mo_occ

                rhf_obj = pyscf.scf.RHF(mol_obj).newton()
                e_rhf = rhf_obj.kernel(mo_init, mocc_init)
                
                rhf_obj = pyscf.scf.UKS(mol_obj).newton()
                e_rhf = rhf_obj.kernel()
                """
                
                rhf_obj1 = pyscf.scf.RHF(mol_obj)
                rhf_obj1.DIIS = pyscf.scf.ADIIS
                #rhf_obj.conv_tol=1e-5
                e_rhf1 = rhf_obj1.kernel()
                
                mo_init = rhf_obj1.mo_coeff
                mocc_init = rhf_obj1.mo_occ
                
                rhf_obj = pyscf.scf.RHF(mol_obj).newton()
                #rhf_obj2.DIIS = pyscf.scf.EDIIS
                #rhf_obj.conv_tol=1e-5
                e_rhf = rhf_obj.kernel(mo_init,mocc_init)

            if run_fci:
                fci_obj = fci.FCI(rhf_obj)
                e_fci = fci_obj.kernel()[0]
            else:
                e_fci = "N/A"
                
            if loc_orbs:
                # C matrix stores the AO to localized orbital coefficients
                C = lo.pipek.PM(mol_obj).kernel(rhf_obj.mo_coeff)
                # Split-localization:
                """
                nocc = sum(rhf_obj.mo_occ>0.0)
                norb = len(rhf_obj.mo_occ)
                C = np.zeros((norb,norb))
                C[:,:nocc] = lo.pipek.PM(mol_obj).kernel(rhf_obj.mo_coeff[:,:nocc])
                C[:,nocc:] = lo.pipek.PM(mol_obj).kernel(rhf_obj.mo_coeff[:,nocc:])
                """
                
            h1e = mol_obj.intor("int1e_kin") + mol_obj.intor("int1e_nuc")
            h2e = mol_obj.intor("int2e")
            
            if loc_orbs:
                scf_c = C
            else:
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
            
            if gen_cubes:
                
                dirpath_o = wd_o + mol_name + "_" + basis + "_" + datestr.strftime("%m%d%y%%%H%M%S")
                
                os.mkdir(dirpath_o)
                
                for i in range(scf_c.shape[1]):
                    fstring = dirpath_o + '/' + str(i+1).zfill(3) + '.cube'
                    cubegen.orbital(mol_obj, fstring, scf_c[:,i])