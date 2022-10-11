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