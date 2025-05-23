'''
Generate the UniMol representation of a structure by truncating
clusters around the atoms of interest.
'''
# in-built modules
import os
import unittest

# third-party modules
import numpy as np
try:
    from unimol_tools import UniMolRepr
except ImportError:
    raise ImportError('unimol_tools is not installed. '
                      'See https://github.com/deepmodeling/Uni-Mol/tree/main/unimol_tools '
                      'for more information.')

# local modules
from UniMolXC.abacus.control import AbacusJob
from UniMolXC.geometry.manip.cluster import clustergen

def generate_from_abacus(jobdir, 
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
    # simple sanity check
    assert isinstance(iat, list)
    assert all(isinstance(i, int) for i in iat)
    assert isinstance(rc, (list, float, int))
    rc = rc if isinstance(rc, list) else [rc] * len(iat)
    assert len(iat) == len(rc)
    
    job = AbacusJob(jobdir)
    job.read_stru(cache=True)
    
    # generate clusters
    clusters = [clustergen(cell=job.get_cell(unit='angstrom'),
                           elem=job.get_atomic_symbols(),
                           pos=job.get_atomic_positions(unit='angstrom'),
                           direct=False,
                           i=i,
                           rc=r)
                for i, r in zip(iat, rc)]
    
    return UniMolRepr(data_type='molecule', 
                      model_name=unimol_model, 
                      model_size=unimol_size)\
           .get_repr(data={'atoms': [c['elem'] for c in clusters], 
                           'coordinates': [c['pos'] for c in clusters]}, 
                     return_atomic_reprs=True)

class TestAbacusToUniMolRepr(unittest.TestCase):
    def setUp(self):
        testfiles = os.path.dirname(__file__)
        testfiles = os.path.dirname(testfiles)
        testfiles = os.path.dirname(testfiles)
        self.testfiles = os.path.abspath(os.path.join(testfiles, 'testfiles'))
    
    @unittest.skip('Skip due to the high time consumption of '
                   'downloading the UniMol model')
    def test_generate_from_abacus(self):
        # from an unfinished ABACUS job
        jobdir = os.path.join(self.testfiles, 'scf-finished')
        myrepr = generate_from_abacus(jobdir, [0], 4.0)
        self.assertTrue(isinstance(myrepr, dict))
        
        nframe_clsrepr, ndim_clsrepr = np.array(myrepr['cls_repr']).shape
        nframe_coords, nat_coords, nxyz = np.array(myrepr['atomic_coords']).shape
        self.assertEqual(nframe_clsrepr, nframe_coords)
        self.assertEqual(nxyz, 3)
        nframe_symbol, nat_symbol = np.array(myrepr['atomic_symbol']).shape
        self.assertEqual(nframe_symbol, nframe_coords)
        self.assertEqual(nat_symbol, nat_coords)
        nframe_atrepr, nat_atrepr, ndim_atrepr = np.array(myrepr['atomic_reprs']).shape
        self.assertEqual(nframe_atrepr, nframe_coords)
        self.assertEqual(nat_atrepr, nat_coords)
        self.assertEqual(ndim_atrepr, ndim_clsrepr)

if __name__ == '__main__':
    unittest.main()
