'''
Generate the DeePMD-kit representation of a structure with
the 'descriptor' of DeePMD model.
'''
# in-built modules
import os
import unittest

# third-party modules
import numpy as np
from ase.geometry import cell_to_cellpar, cellpar_to_cell
try:
    from deepmd.infer.deep_pot import DeepPot
except ImportError:
    raise ImportError('DeePMD-kit is not installed. '
        'See https://docs.deepmodeling.com/projects/deepmd/'
        'en/stable/getting-started/install.html#install-with-conda')

# local modules
from UniMolXC.abacus.control import AbacusJob

def generate_from_abacus(jobdir,
                         dpmodel,
                         head=None):
    '''
    With one ABACUS jobdir, generate the DeePMD representation
    of the structure.
    
    Parameters
    ----------
    jobdir : str
        The path to the ABACUS job directory.
    dpmodel : str
        The path to the DeePMD model file, or the instance of
        `deepmd.infer.deep_pot.DeepPot`.
    head : str, optional
        deprecated, the head of the DeePMD model.
    
    Returns
    -------
    np.ndarray
        The DeePMD representation of the structure, will be in
        shape of nframes x n_atoms x ?.
    '''
    # simple sanity check
    assert isinstance(jobdir, str)
    if isinstance(dpmodel, str):
        assert os.path.exists(dpmodel), f'{dpmodel} does not exist'
        dpmodel = DeepPot(dpmodel)
    else:
        assert isinstance(dpmodel, DeepPot)

    # convert the structure to the DPData format
    data = AbacusJob(jobdir).to_deepmd()
    
    type_map_model = dpmodel.get_type_map()
    atomtype = np.array([type_map_model.index(data['atom_names'][it_at])
                         for it_at in data['atom_types']])
    # the DeePMD model may crash for the lattice that all vectors
    # have fewer than two zeros, like fcc: [1, 1, 0], [1, 0, 1], [0, 1, 1]
    cells = np.array([cellpar_to_cell(cell_to_cellpar(c)).flatten() 
                      for c in data['cells']]).reshape(-1, 9)
    return dpmodel.eval_descriptor(
        coords=data['coords'],
        cells=cells,
        atom_types=atomtype
    )

class TestAbacusToDeePMDRepr(unittest.TestCase):
    def setUp(self):
        testfiles = os.path.dirname(__file__)
        testfiles = os.path.dirname(testfiles)
        testfiles = os.path.dirname(testfiles)
        self.testfiles = os.path.abspath(os.path.join(testfiles, 'testfiles'))
    
    # @unittest.skip('Skip this test due to the error raised by the DeePMD-kit')
    def test_generate_from_abacus(self):
        try:
            import deepmd
        except ImportError:
            self.skipTest('DeePMD-kit is not installed. '
                'See https://docs.deepmodeling.com/projects/deepmd/en/'
                'stable/getting-started/install.html#install-with-conda')

        fmodel = os.path.join(self.testfiles, 'dpa3-2p4-7m-mptraj.pth')
        jobdir = os.path.join(self.testfiles, 'scf-finished')
        # from an finished ABACUS job
        result = generate_from_abacus(jobdir, fmodel)
        nframe, nat, ndim = result.shape
        self.assertEqual(nframe, 1)
        self.assertEqual(nat, 2)
        self.assertEqual(ndim, 448) # what does the 448 mean?

if __name__ == '__main__':
    unittest.main()