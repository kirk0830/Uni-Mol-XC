'''
Example/Integrated test
train the XCPNet model with a given label using the UniMolXC framework.
'''

import os
import unittest
import shutil

import numpy as np

from UniMolXC.network.kernel._xcnet import XCPNetImpl, build_dataset_from_abacus

class TestTrainWithXCPNetFlowGivenLabel(unittest.TestCase):
    def setUp(self):
        testfiles = os.path.dirname(__file__)
        testfiles = os.path.dirname(testfiles)
        self.testfiles = os.path.abspath(os.path.join(testfiles, 'testfiles'))

    def test_train_xcpnet_from_scratch(self):
        
        # Step 1: prepare the dataset
        dataset = build_dataset_from_abacus(
            xdata=[os.path.join(self.testfiles, 'scf-unfinished')],
            ydata=[np.random.rand(20)]
        )
        
        x, y = dataset
        _, _, ndim = x[0].shape
        
        # Step 2: instantiate the XCPNet model
        model = XCPNetImpl(
            elem=['Zn', 'Y', 'S'],
            ndim=ndim,
            nhidden=[64, 64, 64],
            nparam=20
        )
        
        # Step 3: train the model
        model.train()
        
        


if __name__ == '__main__':
    unittest.main()
