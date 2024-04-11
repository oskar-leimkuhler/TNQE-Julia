# Python functions to run HF and FCI calculations, and save the output in HDF5 format

# Packages:
import pyscf
from pyscf import fci, ao2mo, lo, mp
from pyscf.tools import cubegen
import numpy as np
import h5py
import datetime
import os
import configparser


# Runs the PySCF calculations and saves the output to a HDF5 file:
def RunPySCF(config, gen_cubes=False, nosec=False):
    
    mol_name = config['MOLECULE PROPERTIES']['mol_name']
    mol_spin = config['MOLECULE PROPERTIES'].getint('mol_spin', fallback=0)
    mol_charge = config['MOLECULE PROPERTIES'].getint('mol_charge', fallback=0)
    basis = config['CALCULATION PARAMETERS']['basis']
    run_rohf = config['CALCULATION PARAMETERS'].getboolean('run_rohf', fallback=False)
    init_hispin = config['CALCULATION PARAMETERS'].getboolean('init_hispin', fallback=False)
    init_prev = config['CALCULATION PARAMETERS'].getboolean('init_prev', fallback=False)
    run_fci = config['CALCULATION PARAMETERS'].getboolean('run_fci', fallback=False)
    loc_orbs = config['CALCULATION PARAMETERS'].getboolean('loc_orbs', fallback=False)
    active_space = config['CALCULATION PARAMETERS'].getboolean('active_space', fallback=False)
    active_norb = config['CALCULATION PARAMETERS'].getint('active_norb', fallback=0)
    active_nel = config['CALCULATION PARAMETERS'].getint('active_nel', fallback=0)
    xyz_folder = config['GEOMETRIES']['xyz_folder']
    xyz_files = config['GEOMETRIES']['xyz_files'].split(",")
    
    geometries = ["../configs/xyz_files/"+xyz_folder+"/"+xyz_file+".xyz" for xyz_file in xyz_files]
    
    wd = os.getcwd()+"/../datasets/pyscf_data/"
    wd_o = os.getcwd()+"/../datasets/orbs/"
    
    datestr = datetime.datetime.now()
    
    if nosec:
        filename = wd + mol_name + "_" + basis + "_" + datestr.strftime("%m%d%y%%%H%M") + ".hdf5"
    else:
        filename = wd + mol_name + "_" + basis + "_" + datestr.strftime("%m%d%y%%%H%M%S") + ".hdf5"
    
    with h5py.File(filename, 'w') as f:
        
        f.create_dataset("mol_name", data=mol_name)
        f.create_dataset("basis", data=basis)
        f.create_dataset("geometries", data=geometries)
    
        # Initialize these for later:
        mo_init1 = None
        mocc_init1 = None
        
        for g, geometry in enumerate(geometries):
            
            print(os.path.isfile(geometry))

            mol_obj = pyscf.gto.M(atom=geometry, basis=basis)
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
                rhf_obj1.conv_tol=1e-10
                
                if init_prev and (g > 0):
                    
                    e_rhf1 = rhf_obj1.kernel(c0=mo_init1)
                
                else: 

                    if init_hispin:

                        mol_hispin = pyscf.gto.M(atom=geometry, basis=basis)
                        mol_hispin.basis = basis
                        mol_hispin.charge = mol_charge
                        mol_hispin.spin = mol_hispin.nelectron

                        rhf_obj2 = pyscf.scf.RHF(mol_hispin)
                        rhf_obj2.DIIS = pyscf.scf.ADIIS
                        rhf_obj2.conv_tol=1e-10
                        rhf_obj2.init_guess = 'huckel'
                        e_rhf2 = rhf_obj2.kernel()

                        mo_init1 = rhf_obj2.mo_coeff
                        
                        e_rhf1 = rhf_obj1.kernel(c0=mo_init1)

                    else:

                        rhf_obj1.init_guess = 'huckel'
                        e_rhf1 = rhf_obj1.kernel()

                mocc_init = rhf_obj1.mo_occ
                mo_init = rhf_obj1.mo_coeff
                
                rhf_obj = pyscf.scf.RHF(mol_obj).newton()
                #rhf_obj2.DIIS = pyscf.scf.EDIIS
                rhf_obj.conv_tol=1e-11
                e_rhf = rhf_obj.kernel(mo_init,mocc_init)
                
                if init_prev:
                    mo_init1 = rhf_obj.mo_coeff
                    mocc_init1 = rhf_obj.mo_occ

            if run_fci:
                
                fci_obj = fci.FCI(rhf_obj)
                
                fci_obj.conv_tol=1e-12
                
                e_fci, fci_vec = fci_obj.kernel()
                
                norb = np.shape(rhf_obj1.mo_coeff)[0]
                
                fci_str0 = fci.cistring.make_strings([*range(norb)], mol_obj.nelec[0])
                
                fci_str = [bin(x) for x in fci_str0]
                
                fci_addr = [fci.cistring.str2addr(norb, mol_obj.nelec[0], x) for x in fci_str0]
                
                #print(fci_str)
                
            else:
                
                e_fci, fci_vec, fci_str, fci_addr = "N/A", "N/A", "N/A", "N/A"
                
            
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
                
            #if active_space:
                
            mp2_obj = mp.MP2(rhf_obj).run()
            t2 = mp2_obj.t2
            
            h1e = mol_obj.intor("int1e_kin") + mol_obj.intor("int1e_nuc")
            h2e = mol_obj.intor("int2e")
            
            if loc_orbs:
                scf_c = C
            else:
                scf_c = rhf_obj.mo_coeff

            h1e = scf_c.T @ h1e @ scf_c
            h2e = ao2mo.kernel(h2e, scf_c)

            e_mo = rhf_obj.mo_energy
            
            grp = f.create_group(geometry)
            
            rhf_data = grp.create_dataset("e_rhf", data=e_rhf)
            fci_data = grp.create_dataset("e_fci", data=e_fci)
            fci_vecs = grp.create_dataset("fci_vec", data=fci_vec)
            fci_strs = grp.create_dataset("fci_str", data=fci_str)
            fci_addr = grp.create_dataset("fci_addr", data=fci_addr)
            h1e_data = grp.create_dataset("h1e", data=h1e)
            h2e_data = grp.create_dataset("h2e", data=h2e)
            t2_data = grp.create_dataset("t2", data=t2)
            e_mo = grp.create_dataset("e_mo", data=e_mo)
            nel_data = grp.create_dataset("nel", data=mol_obj.nelectron)
            nuc_data = grp.create_dataset("nuc", data=mol_obj.energy_nuc())
            
            if gen_cubes:
                
                dirpath_o = wd_o + mol_name + "_" + basis + "_" + datestr.strftime("%m%d%y%%%H%M%S")
                
                os.mkdir(dirpath_o)
                
                for i in range(scf_c.shape[1]):
                    fstring = dirpath_o + '/' + str(i+1).zfill(3) + '.cube'
                    cubegen.orbital(mol_obj, fstring, scf_c[:,i])