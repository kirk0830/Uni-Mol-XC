'''
Generate the DeePMD-kit representation of a structure with
the 'descriptor' of DeePMD model.
'''

MODELS = {
    'DPA-1-OC2M': 'https://store.aissquare.com/models/4560acec-db9c-11ee-9b22-506b4b2349d8/OC_10M.pb',
}

import os
import unittest

import numpy as np
try:
    from deepmd.infer.deep_pot import DeepPot
except ImportError:
    raise ImportError('DeePMD-kit is not installed. '
                      'See https://docs.deepmodeling.com/projects/deepmd/en/stable/getting-started/install.html#install-with-conda')
from UniMolXC.abacus.control import AbacusJob

def generate(jobdir,
              fmodel,
              head=None):
    '''
    With one ABACUS jobdir, generate the DeePMD representation
    of the structure.
    
    Parameters
    ----------
    jobdir : str
        The path to the ABACUS job directory.
    fmodel : str
        The path to the DeePMD model file.
    '''
    # simple sanity check
    assert isinstance(jobdir, str)
    assert os.path.exists(fmodel), f'{fmodel} does not exist'
    
    # convert the structure to the DPData format
    data = AbacusJob(jobdir).to_deepmd()
        
    dpmodel = DeepPot(fmodel, head=head)
    type_map_model = dpmodel.get_type_map()
    type_transform = np.array([type_map_model.index(i)
                               for i in data['atom_names']])
    return dpmodel.eval_descriptor(
        coords=data['coords'],
        cells=data['cells'],
        atom_types=list(type_transform[data.data['atom_types']])
    )

class TestAbacusToDeePMDRepr(unittest.TestCase):
    def setUp(self):
        testfiles = os.path.dirname(__file__)
        testfiles = os.path.dirname(testfiles)
        testfiles = os.path.dirname(testfiles)
        self.testfiles = os.path.abspath(os.path.join(testfiles, 'testfiles'))
    
    @unittest.skip('Skip this test due to the error raised by the DeePMD-kit')
    def test_generate(self):
        try:
            import deepmd
        except ImportError:
            self.skipTest('DeePMD-kit is not installed. '
                          'See https://docs.deepmodeling.com/projects/deepmd/en/stable/getting-started/install.html#install-with-conda')
        try:
            import tensorflow
        except ImportError:
            self.skipTest('TensorFlow is not installed. '
                          'See https://www.tensorflow.org/install')
        fmodel = os.path.join(self.testfiles, 'OC_10M.pb')
        jobdir = os.path.join(self.testfiles, 'scf-finished')
        # from an finished ABACUS job
        result = generate(jobdir, fmodel)
        print(result)

if __name__ == '__main__':
    unittest.main()