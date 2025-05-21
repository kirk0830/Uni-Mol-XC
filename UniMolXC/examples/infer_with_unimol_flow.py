'''
Example/Integrated test
'''
import os
import shutil
import unittest

import numpy as np

from UniMolXC.network.trainer import XCParameterizationNetTrainer
from UniMolXC.network.utility.preprocess import build_dataset

class TestInferWithUniMolFlow(unittest.TestCase):
    def setUp(self):
        testfiles = os.path.dirname(__file__)
        testfiles = os.path.dirname(testfiles)
        self.testfiles = os.path.abspath(os.path.join(testfiles, 'testfiles'))

    def test_eval_unimol_from_file(self):
        mytrainer = XCParameterizationNetTrainer(
            model={'model_restart': 'XCPNTrainer-test'},
        )
        
        dataset = build_dataset(
            xdata=[os.path.join(self.testfiles, 'scf-unfinished')],
            ydata=[{'Zn': [1.0], 'Y': [5.0], 'S': [0.0]}],
            mode='abacus',
            cluster_truncation={'Zn': 5.0, 'Y': 3.0, 'S': 3.0},
            walk=False
        )
        
        res = mytrainer.inner_eval(dataset=dataset)
        shutil.rmtree('logs') # unimol produces this folder
        
        self.assertTrue(isinstance(res, np.ndarray))
        ref = [   1.255945,   1.2559448,   1.2559445, -0.05477616,  
                0.87933564,   1.2377898, -0.20508668, -0.20508668, 
               -0.20508716,   0.2683932, -0.12968668, -0.10768171, 
                -0.5796611, -0.62292475]
        for r, v in zip(ref, res.flatten()):
            self.assertAlmostEqual(r, v, places=5)

if __name__ == '__main__':
    unittest.main()