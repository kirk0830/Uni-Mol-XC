'''
Example/Integrated test
'''

import os
import unittest
import shutil

from UniMolXC.network.trainer import XCParameterizationNetTrainer
from UniMolXC.network.utility.preprocess import build_dataset

class TestTrainWithUniMolFlowGivenLabel(unittest.TestCase):

    def setUp(self):
        testfiles = os.path.dirname(__file__)
        testfiles = os.path.dirname(testfiles)
        self.testfiles = os.path.abspath(os.path.join(testfiles, 'testfiles'))

    def test_train_unimol_from_scratch(self):
        
        # Step 1: instantiate the trainer
        mytrainer = XCParameterizationNetTrainer(
            model={'model_name': 'unimolv1',
                   'model_size': '84m'}
            )
        
        # Step 2: prepare the dataset
        # NOTE: the cluster center atom always has coordiantes (0,0,0)
        #       therefore it is possible to identify the atom type of
        #       the cluster center atom by checking the coordinates:
        # ```python
        # center_typ = [atoms[i]
        # for atoms, coords in zip(dataset['atoms'], dataset['coordinates'])
        # for i, c in enumerate(coords) if np.allclose(c, [0.0, 0.0, 0.0])]]
        # ```
        dataset = build_dataset(
            xdata=[os.path.join(self.testfiles, 'scf-unfinished')],
            ydata=[{'Zn': [1.0], 'Y': [5.0], 'S': [0.0]}],
            mode='abacus',
            cluster_truncation={'Zn': 5.0, 'Y': 3.0, 'S': 3.0},
            walk=False
        )
        
        # Step 3: train the model and get predictions
        mytrainer.inner_train_unimol_net(
            dataset=dataset,
            inner_epochs=2,
            inner_batchsize=5,
            prefix='test'
        )
        # print(mytrainer.inner_eval(dataset=dataset))
        self.assertTrue(os.path.exists('XCPNTrainer-test'))
        self.assertTrue(all([os.path.exists(os.path.join('XCPNTrainer-test', f)) for f in \
            ['config.yaml', 'cv.data', 'metric.result', 
             'model_0.pth', 'model_1.pth', 'model_2.pth', 'model_3.pth', 'model_4.pth',
             'target_scaler.ss']]))

    def test_train_unimol_restart(self):
        
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
        
        # Step 3: train the model and get predictions
        mytrainer.inner_train_unimol_net(
            dataset=dataset,
            inner_epochs=2,
            inner_batchsize=5,
            prefix='test'
        )
        # print(mytrainer.inner_eval(dataset=dataset))
        self.assertTrue(os.path.exists('XCPNTrainer-test'))
        self.assertTrue(all([os.path.exists(os.path.join('XCPNTrainer-test', f)) for f in \
            ['config.yaml', 'cv.data', 'metric.result', 
             'model_0.pth', 'model_1.pth', 'model_2.pth', 'model_3.pth', 'model_4.pth',
             'target_scaler.ss']]))

        # Clean up
        shutil.rmtree('XCPNTrainer-test')
        shutil.rmtree('logs')

if __name__ == '__main__':
    unittest.main()
