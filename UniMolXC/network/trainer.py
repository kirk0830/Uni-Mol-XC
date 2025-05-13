# built-in modules
import os
import logging
import unittest
import time

# third-party modules
import numpy as np

# local modules
from UniMolXC.abacus.control import AbacusJob
from UniMolXC.geometry.manip.cluster import clustergen
from UniMolXC.network.kernel.xcpnet import XCParameterizationNet
try:
    from UniMolXC.network.kernel._unimol import UniMolRegressionNet
    from UniMolXC.geometry.repr._unimol import generate as unimol_repr
except ImportError:
    raise ImportError('unimol_tools is not installed. '
                      'See https://github.com/deepmodeling/Uni-Mol/'
                      'tree/main/unimol_tools '
                      'for more information.')
try:
    from UniMolXC.geometry.repr._deepmd import generate as deepmd_desc
except ImportError:
    raise ImportError('DeePMD-kit is not installed. '
                      'See https://docs.deepmodeling.com/projects/'
                      'deepmd/en/stable/getting-started/install.html'
                      '#install-with-conda')
    
class XCParameterizationNetTrainer:
    '''
    the trainer for XCParameterizationNet.
    
    The trainning task of XCParameterizationNet is very computationally
    costly in general, because it requires the output of XC functionals
    parameterization always to be consistent with the energy calculated
    by Density Functional Theory level, Self-Consistent Field (SCF)
    energy. To reduce the computational cost, we use a two-step training
    strategy: 
    
    1. with a initial guess on the parameterization of XC, we perform
       once the DFT-SCF, obtain the energy components corresponding to
       each parameter to be trained.
    2. with those energy components, we train the XCParameterizationNet
       such that the loss function which takes the energy difference with
       the exact (label) value is minimized, as such, we can obtain a
       new set of parameters for XC functionals. It should be noted that
       the stop of training of this step needs a arbitrary threshold,
       in the following implementation, we name it as `inner_train_thr`.
       Also we set the `inner_train_max_iter` to limit the maximum
       number of iterations for this step.
    3. with the newly output parameters, we perform the DFT-SCF again, 
       get energy components again, and repeat the above two steps until
       the loss function converges to a certain threshold.
    '''
    def __init__(self, 
                 model):
        '''
        instantiate a trainer for XCParameterizationNet.
        
        Parameters
        ----------
        model : XCParameterizationNet or dict
            the XCParameterizationNet model to be trained, or the dict
            that contains the parameters of the model (the way to employ
            other models)
        '''
        if isinstance(model, dict):
            # the case that utilize the third-party model
            assert 'model_name' in model
            self.model = UniMolRegressionNet(
                model_name=model.get('model_name', 'unimolv1'),
                model_size=model.get('model_size', '84m'))
        else:
            assert isinstance(model, XCParameterizationNet)
            self.model = model

        self.coef_traj = None
        self.loss_traj = None
    
    @staticmethod
    def build_dataset_from_unimol_custom_structure(xdata, ydata):
        '''
        build the dataset in the case that the data arranged like
        UniMol custom structure. See the UniMol documentation for
        more details:
        https://unimol.readthedocs.io/en/latest/examples.html
        
        Parameters
        ----------
        xdata : dict
            the dict should contain the following keys:
            - 'atoms', a list of list of str, contains element symbol
            - 'coordinates', a list of np.array, for each element, 
               contains the Cartesian coordinates of atoms, the number
               of lines should be identical with the corresponding
               element in 'atoms'
        ydata : array-like
            the labels of the dataset.
        
        Returns
        -------
        xdata : dict
            standardized xdata (now we design the `from_unimol_custom_structure`
            as the standardized one)
        ydata : array-like
            as input
        '''
        return xdata, ydata
    
    @staticmethod
    def build_dataset_from_abacus(folders, ydata, truncation=None):
        '''
        build the dataset in the case that the data arranged like
        ABACUS jobdir. The function will walk through the directories
        '''
        assert len(folders) > 0
        assert len(folders) == len(ydata)

        jobs = [AbacusJob(f) for f in folders]
        for job in jobs:
            job.read_stru(cache=True)
        if truncation is None:
            assert isinstance(ydata, (list, np.ndarray))
            assert len(jobs) == len(ydata)
            return {'atoms': [j.get_atomic_symbols() for j in jobs],
                    'coordinates': [j.get_atomic_positions() for j in jobs]},\
                    ydata
                    
        # the case that requires the clustergen, ydata should be either
        # a dict whose keys are element symbols, or a list of list of float
        assert isinstance(truncation, (list, dict))
        if isinstance(truncation, dict):
            truncation = [truncation] * len(jobs)
        assert all(isinstance(t, dict) for t in truncation)
        assert len(truncation) == len(jobs)
        assert len(truncation) == len(ydata)
        
        clusters = [[clustergen(cell=j.get_cell(unit='angstrom'),
                                elem=j.get_atomic_symbols(),
                                pos=j.get_atomic_positions(unit='angstrom'),
                                direct=False,
                                i=iat,
                                rc=t[e]) 
                     for iat, e in enumerate(j.get_atomic_symbols()) if e in t]
                     # I am not sure what should be the behavior of this function
                     # when there is no element being defined in the truncation
                     # presently the job will be skipped.
                     for t, j in zip(truncation, jobs)]
        return {'atoms': [c[2] for j in clusters for c in j],
                'coordinates': [c[0] for j in clusters for c in j]},\
                [y for y, j in zip(ydata, clusters) for _ in range(len(j))]
    
    @staticmethod
    def build_dataset_to_coordinates(xstandard, ystandard):
        '''
        build the dataset to the one accepted by the UniMol
        '''
    
    @staticmethod
    def build_dataset(xdata, 
                      ydata,
                      mode='coordinates',
                      cluster_truncation=None,
                      walk=False):
        '''
        build the dataset for training
        
        Parameters
        ----------
        xdata : str or list of str or dict or list of np.ndarray
            For the case of being str or list of str:
            the root directory of the dataset. If a list, the list
            should contain the paths to the directories of the dataset.
            For the case of being dict:
            the dict should contain the following keys:
            - 'atoms', a list of list of str, contains element symbol
            - 'coordinates', a list of np.array, for each element, 
               contains the Cartesian coordinates of atoms, the number
               of lines should be identical with the corresponding
               element in 'atoms'
            For the case of being a list of np.ndarray:
            mostly it is the representation
        ydata : array-like
            the labels of the dataset. The labels should be in the
            same order as the directories in `root`.
        mode : str
            the mode of the 'x' data, which can be 'coordinates',
            'unimol-repr' or 'deepmd-desc'. The default is
            'coordinates', which means the input data is the
            coordinates of the atoms.
        cluster_truncation : list of dict or dict
            the truncation radius of the cluster, which is compulsory
            when the mode is 'unimol-repr'. For the case 'mode' is
            'coordinates' and when 'cluster_truncation' is not None,
            the function will generate the clusters. For the case
            that assigned as dict, the dict should contain all the
            elements of interest as key, and a float for each as the
            truncation radius. For the case that assigned as list,
            it should have the same length as `xdata`.
        walk : bool
            whether to walk through the directories in `root` to
            find the files. The default is False, which means the
            function will only look for the files in the directories
            in `root`. If set to True, the function will walk through
            all the subdirectories in `root` to find the files.
            The default is False.
    
        Returns
        -------
        '''
        xstandard, ystandard = None, None
        if isinstance(xdata, list):
            if all(isinstance(x, np.ndarray) for x in xdata):
                raise NotImplementedError('Not implemented yet')
            elif all(isinstance(x, str) for x in xdata):
                pass # the case that there are multiple directories
            else:
                raise ValueError('xdata should be a list of str '
                                 'or list of np.ndarray')
        elif isinstance(xdata, str):
            xdata = [xdata]
            pass # the case that there is only one directory
        elif isinstance(xdata, dict):
            # the case that xdata contains two keys, 'atoms' and 'coordinates'
            assert 'atoms' in xdata
            assert 'coordinates' in xdata
        else:
            raise TypeError('xdata should be: str, list of str, '
                            'dict or list of np.ndarray')
        if mode not in ['coordinates', 'unimol-repr', 'deepmd-desc']:
            raise ValueError('mode should be: coordinates, unimol-repr '
                             'or deepmd-desc')
        if mode != 'coordinates':
            raise NotImplementedError('Currently only the mode "coordinates" '
                                      'has been implemented')
        if not isinstance(cluster_truncation, (list, dict, type(None))):
            raise TypeError('cluster_truncation should be: list, dict '
                            'or None')
        if walk:
            raise NotImplementedError('The recursive walk mode to find '
                                      'the files is not implemented yet')

        xstandard, ystandard = \
            XCParameterizationNetTrainer.build_dataset_from_abacus(
                folders=xdata,
                ydata=ydata,
                truncation=cluster_truncation
            )
        return xstandard|{'target': ystandard}
    
    def _inner_train_unimol_regression_net(self,
                                           dataset,
                                           inner_epochs=10,
                                           inner_batchsize=16,
                                           prefix=None):
        # a single training step
        prefix = prefix or time.strftime("%Y%m%d-%H%M%S")
        self.model.train(data=dataset,
                         epochs=inner_epochs,
                         batch_size=inner_batchsize,
                         save_path=f'XCPNTrainer-{prefix}')
        return self.model.eval(data=dataset)
    
    def _inner_train_xc_parameterization_net(self,
                                             dataset,
                                             inner_thr=None,
                                             inner_maxiter=1e5,
                                             save_model=None):
        raise NotImplementedError('The inner training of XCParameterizationNet '
                                  'is not implemented yet')           
    
    def train(self,
              dataset,
              dft_machine,
              outer_thr=None,
              outer_thrtyp='energy',
              inner_epochs=10,
              inner_batchsize=16,
              inner_thr=None,
              inner_maxiter=1e5,
              inner_prefix=None
              ):
        '''
        train the XCParameterizationNet model.
        
        Parameters
        ----------
        dataset : any
            see function XCParameterizationNet.build_dataset for more
            information
        dft_machine : str
            the DFT machine to be used for training.
        outer_thr : float
            the threshold for determining the convergence of the whole
            training.
        outer_thrtyp : str
            the type of the threshold for determining the convergence of
            the whole training. The default is 'energy', which means
            the threshold is based on the total energy. The other options
            are:  
            `energies`: the threshold is based on the energy components
            `components`: the threshold is based on the parameters
        inner_epochs : int
            the number of epochs for the inner training. The default is 10.
        inner_batchsize : int
            the batch size for the inner training. The default is 16.
        inner_thr : float, optional
            the threshold for the inner training. If set as None, means
            the inner training is not expected to be stopped by setting
            a threshold, which brings the one-step training, that is,
            the parameters of XC and energy components are updated
            simultaneously.
        inner_maxiter : int, optional
            the maximum number of iterations for the inner training.
            The default is 1e5.
        inner_prefix : str, optional
            the prefix to name the model. The default is None, which means
            the model will be named as a random UUID. The model will be
            saved in the current directory.
        
        Returns
        -------
        
        '''
        pass
        
class XCParameterizationNetTrainerTest(unittest.TestCase):
    '''
    the test for XCParameterizationNetTrainer
    '''
    def setUp(self):
        testfiles = os.path.dirname(__file__)
        testfiles = os.path.dirname(testfiles)
        self.testfiles = os.path.abspath(os.path.join(testfiles, 'testfiles'))
    
    # acrymon: build_dataset_from_abacus: bdfa
    def test_bdfa_simplest(self):
        folders = [os.path.join(self.testfiles, 'scf-finished')]
        ydata = [0.0] # assume it is the Hubbard U exact value of Silicon
        
        # the case that do not require the clustergen
        xdata, ydata = XCParameterizationNetTrainer.build_dataset_from_abacus(
            folders=folders,
            ydata=ydata
        )
        # there would be a direct output of the atomic coordinates
        self.assertTrue(isinstance(xdata, dict))
        self.assertTrue('atoms' in xdata)
        self.assertTrue('coordinates' in xdata)
        self.assertTrue(isinstance(xdata['atoms'], list))
        self.assertTrue(all(isinstance(job, list) and \
                        all(isinstance(elem, str) for elem in job)
                            for job in xdata['atoms'] ))
        # the label will not be duplicated because there is only one structure
        self.assertListEqual(ydata, [0.0])

    def test_bdfa_clustergen(self):
        folders = [os.path.join(self.testfiles, 'scf-finished')]
        ydata = [0.0] # assume it is the Hubbard U exact value of Silicon
        
        # the case that requires the clustergen
        # the truncation radius is 5 Angstrom, and there are two Si atoms
        # in the system, so the output should be two clusters
        xdata, ydata = XCParameterizationNetTrainer.build_dataset_from_abacus(
            folders=folders,
            ydata=ydata,
            truncation=[{'Si': 5.0}] # the truncation radius as 5 Angstrom
        )
        self.assertTrue(isinstance(xdata, dict))
        self.assertTrue('atoms' in xdata)
        self.assertTrue('coordinates' in xdata)
        self.assertTrue(isinstance(xdata['atoms'], list))
        self.assertTrue(len(xdata['atoms']) == 2) # two clusters
        self.assertTrue(len(xdata['coordinates']) == 2) # two clusters
        # check the correspondence of length between the atoms and coordinates
        self.assertTrue(all(len(x) == len(y) 
                            for x, y in zip(xdata['atoms'], xdata['coordinates'])))

    def test_bdfa_multielem(self):
        folders = [os.path.join(self.testfiles, 'scf-finished'),
                   os.path.join(self.testfiles, 'scf-unfinished')]
        ydata = [[0.0], [1.0, 5.0, 0.0]]
        
        xdata, ydata = XCParameterizationNetTrainer.build_dataset_from_abacus(
            folders=folders,
            ydata=ydata
        )
        self.assertTrue(isinstance(xdata, dict))
        self.assertTrue('atoms' in xdata)
        self.assertTrue('coordinates' in xdata)
        self.assertTrue(isinstance(xdata['atoms'], list))
        self.assertTrue(len(xdata['atoms']) == 2) # two structures, no cluster generated
        self.assertTrue(len(xdata['coordinates']) == 2)
        # check the correspondence of length between the atoms and coordinates
        self.assertTrue(all(len(x) == len(y) 
                            for x, y in zip(xdata['atoms'], xdata['coordinates'])))
        # check the correspondence of length between the labels and structures
        self.assertTrue(len(ydata) == 2)
        self.assertListEqual(ydata, [[0.0], [1.0, 5.0, 0.0]])
    
    def test_bdfa_multielem_allrc(self):
        folders = [os.path.join(self.testfiles, 'scf-finished'),
                   os.path.join(self.testfiles, 'scf-unfinished')]
        ydata = [[0.0], [1.0, 5.0, 0.0]]
        
        xdata, ydata = XCParameterizationNetTrainer.build_dataset_from_abacus(
            folders=folders,
            ydata=ydata,
            truncation={'Si': 3.0, 'Y': 5.0, 'Zn': 5.0, 'S': 3.0}
        )
        self.assertTrue(isinstance(xdata, dict))
        self.assertTrue('atoms' in xdata)
        self.assertTrue('coordinates' in xdata)
        self.assertTrue(isinstance(xdata['atoms'], list))
        # because this time we set the truncation radius for all the elements
        # so the output should be many clusters. The number of atoms of the
        # first structure is 2, and the second is 14, therefore we should have
        # 2 + 14 = 16 clusters
        self.assertTrue(len(xdata['atoms']) == 16)
        self.assertTrue(len(xdata['coordinates']) == 16)
        # check the correspondence of length between the atoms and coordinates
        self.assertTrue(all(len(x) == len(y) 
                            for x, y in zip(xdata['atoms'], xdata['coordinates'])))
        # check the correspondence of length between the labels and structures
        self.assertTrue(len(ydata) == 16)
        # will have two for the first structure and 14 for the second
        self.assertListEqual(ydata, [[0.0]]*2 + [[1.0, 5.0, 0.0]]*14)

    def test_bdfa_multielem_partrc(self):
        folders = [os.path.join(self.testfiles, 'scf-finished'),
                   os.path.join(self.testfiles, 'scf-unfinished')]
        ydata = [[0.0], [1.0, 5.0, 0.0]]
        
        # we do not include the Si and S because DFT+U is not
        # intended for them
        xdata, ydata = XCParameterizationNetTrainer.build_dataset_from_abacus(
            folders=folders,
            ydata=ydata,
            truncation={'Zn': 5.0, 'Y': 3.0}
        )
        self.assertTrue(isinstance(xdata, dict))
        self.assertTrue('atoms' in xdata)
        self.assertTrue('coordinates' in xdata)
        self.assertTrue(isinstance(xdata['atoms'], list))
        # we only define the truncation radius for Zn and Y, therefore only
        # the clusters of these two elements will be generated
        # there are 2 Zn and 4 Y, therefore we should have 6 clusters
        self.assertTrue(len(xdata['atoms']) == 6)
        self.assertTrue(len(xdata['coordinates']) == 6)
        # check the correspondence of length between the atoms and coordinates
        self.assertTrue(all(len(x) == len(y) 
                            for x, y in zip(xdata['atoms'], xdata['coordinates'])))
        # check the correspondence of length between the labels and structures
        self.assertTrue(len(ydata) == 6)
        self.assertListEqual(ydata, [[1.0, 5.0, 0.0]]*6)

    @unittest.skip('This is an integrated test, it runs well, so skip it')
    def test_inner_train_unimol_regression_net(self):
        
        mytrainer = XCParameterizationNetTrainer(
            model={'model_name': 'unimolv1',
                   'model_size': '84m'})
        dataset = mytrainer.build_dataset(
            xdata=[os.path.join(self.testfiles, 'scf-unfinished')],
            ydata=[[1.0, 5.0, 0.0]],
            mode='coordinates',
            cluster_truncation={'Zn': 5.0, 'Y': 3.0, 'S': 3.0},
            walk=False
        )
        res = mytrainer._inner_train_unimol_regression_net(
            dataset=dataset,
            inner_epochs=2,
            inner_batchsize=5,
            prefix='test'
        )
        print(res)

if __name__ == '__main__':
    unittest.main()