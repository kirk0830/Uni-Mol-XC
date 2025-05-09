'''

'''
import os
import unittest

import numpy as np
from unimol_tools import UniMolRepr

from UniMolXC.abacus.control import AbacusJob
from UniMolXC.geometry.cluster import clustergen
from UniMolXC.physics.database import convert_l_unit

def run(jobdir, 
        iat, 
        rc, 
        unimol_model='unimolv1',
        unimol_size='84m'):
    '''
    With one ABACUS jobdir, generate the UniMol representation
    of the structure.
    
    Parameters
    ----------
    jobdir : str
        The path to the ABACUS job directory.
    iat : list of int
        the index of atoms to be used as the center of the cluster.
    rc : list of float or float
        the cutoff radius of the cluster.
        If a list, the length of the list should be equal to the length of iat.
        If a float, the same cutoff radius will be used for all atoms.
    unimol_model : str
        The model name of UniMol.
    unimol_size : str
        The size of the UniMol model.
    
    Returns
    -------
    dict
        The UniMol representation of the structure.
    '''
    assert isinstance(iat, list)
    assert all(isinstance(i, int) for i in iat)
    assert isinstance(rc, (list, float, int))
    rc = rc if isinstance(rc, list) else [rc] * len(iat)
    assert len(iat) == len(rc)
    
    job = AbacusJob(jobdir)
    stru = job.read_stru(cache=True)
    
    # unpack the stru
    cell = np.array(stru['lat']['vec']) * \
        convert_l_unit(stru['lat']['const'], 
                       unit_from='bohr', 
                       unit_to='angstrom')
    
    elem = [[s['symbol']] * s['natom'] for s in stru['species']]
    elem = [item for sublist in elem for item in sublist]
    
    pos = np.array([atom['coord'] for s in stru['species'] for atom in s['atom']])
    factor = 1 if stru['coord_type'].startswith('Cartesian') else stru['lat']['const']
    factor *= 1 if 'angstrom' in stru['coord_type'] \
        else convert_l_unit(1, 'bohr', 'angstrom')
    pos = pos * factor
    pos = pos.reshape(-1, 3)
    
    # generate clusters
    clusters = [clustergen(pos=pos,
                           direct=False,
                           i=i,
                           rc=r,
                           cell=cell)
                for i, r in zip(iat, rc)]
    
    return UniMolRepr(data_type='molecule', 
                      model_name=unimol_model, 
                      model_size=unimol_size)\
           .get_repr(data={'atoms': [c[2] for c in clusters], 
                           'coordinates': [c[0] for c in clusters]}, 
                     return_atomic_reprs=True)

class TestAbacusToUniMolRepr(unittest.TestCase):
    def setUp(self):
        testfiles = os.path.dirname(__file__)
        testfiles = os.path.dirname(testfiles)
        self.testfiles = os.path.abspath(os.path.join(testfiles, 'testfiles'))
    
    def test_run(self):
        # from an unfinished ABACUS job
        jobdir = os.path.join(self.testfiles, 'scf-unfinished')
        result = run(jobdir, [0], 10)
        print(result)

if __name__ == '__main__':
    unittest.main()
