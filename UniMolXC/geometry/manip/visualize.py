from ase import Atoms
from ase.visualize import view
from ase.io import write

def build_ase_atoms(elem, coords, cell=None):
    '''
    Build an ASE Atoms object from element symbols and coordinates.
    
    Parameters
    ----------
    elem : str or list of str
        Element symbol or list of element symbols for each atom.
    coords : list of list of float
        List of coordinates for each atom. Warning: should provide
        the cartesian coordinates of the atoms.
    cell : list of float, optional
        Cell dimensions for periodic boundary conditions.
    
    Returns
    -------
    atoms : ase.Atoms
        ASE Atoms object.
    '''
    elem = elem if isinstance(elem, list) else [elem] * len(coords)
    if cell is None:
        atoms = Atoms(elem, positions=coords)
    else:
        atoms = Atoms(elem, positions=coords, cell=cell, pbc=True)
    return atoms

def visualize(atoms, fn=None, show=True):
    '''
    Visualize the atoms using ASE's view function.
    Optionally save the visualization to a file.
    '''
    if fn is not None:
        write(fn, atoms)
    if show:
        view(atoms)
        
