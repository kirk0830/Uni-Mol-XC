'''
Example/Integrated test
train the XCPNet model with a given label using the UniMolXC framework.
'''

import os
import unittest
import shutil

import numpy as np

from UniMolXC.network.kernel._xcnet import XCParameterizationNet, \
    build_dataset_from_abacus

class TestTrainWithXCPNetFlowGivenLabel(unittest.TestCase):
    def setUp(self):
        testfiles = os.path.dirname(__file__)
        testfiles = os.path.dirname(testfiles)
        self.testfiles = os.path.abspath(os.path.join(testfiles, 'testfiles'))

    def test_train_xcpnet_from_scratch(self):
        
        # Step 1: prepare the dataset
        dataset = build_dataset_from_abacus(
            folders=[os.path.join(self.testfiles, 'scf-finished')],
            labels=[[-1.0,]] # one job, one frame, one energy label
        )
                
        # Step 2: instantiate the XCPNet model
        trainer = XCParameterizationNet(
            elem=['Si'],
            model_size={'ndim': 448,
                        'nparam': 1,
                        'nhidden': [20, 20, 20]}
        )
        
        # Step 3: train the model
        with self.assertRaises(NotImplementedError):
            # what is not implemented yet is the read_exc_terms() function
            # in the AbacusJob class
            trainer.train(
                data=dataset,
                epochs=2,
                batch_size=5,
                prefix='test_xcpnet_flow_given_label'
            )

if __name__ == '__main__':
    unittest.main()
