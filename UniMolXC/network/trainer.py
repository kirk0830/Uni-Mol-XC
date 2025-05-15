'''
Title
-----
Local chemical environment incorperated parameterizing XC functionals 
enabled by machine-learning

Concept
-------
The training of machine-learning parameterization on Density Functional
eXchange-Correlation here only considers re-parameterize the XC
functionals with known analytical form. What will be solved by the
machine-learning technique is how to find the best parameters for the
energy components in the XC functionals chosen. This work is expected
to parameterize the XC functionals based on the local chemical
environment, which means the mapping from the local chemical
environment to the parameterization will be established and that is
what be learned by the neural network.

Our ultimate goal is to minimize the energy error with respect to one
golden standard to a negligible level as much as possible, and
ideally the loss function would be the absolute difference (or its 
quadratic form) between the exact value and the energy predicted by
a parameterized XC. 

For simplicity, the parameterization practically done here is linearly
combining the energy components, whose coefficients are the parameters 
to be learned and being related with the local chemical environment. 
For the loss function, a simple implementation can be found in the file 
`UniMolXC/network/kernel/classical.py`, the function `loss`.

However, what distincts with common machine learning tasks is the label
is not known: exact values of parameters are not known, but they can
be found during the training process, somehow similar with the 
classical way. For more information, please refer to the leading 
annotation of the file `UniMolXC/network/kernel/classical.py`.

We should note that, it is obvious that the energy components are 
actually the functionals of the parameters, which means during the 
training process, in principle one should always keep the parameters 
and the energy components consistent. 

However, this manner is quite computationally costly, so an alternative
strategy is to train the parameters with fixed energy components. 
After some epochs, the energy components will be updated with the
newly trained parameters, so the loss function of the inner training
will be changed, as a result. This process will be repeated until the
loss converges to a certain threshold.

In following codes, we will call the training of parameters with fixed
energy components as inner training, counterpartly, the outer loop
will be called the outer training.

Backend
-------
In principle the inner training can be done with any machine-learning
model, but presently a POC is implemented by incorperating with the 
UniMol, a quite efficient deep-learning framework that can predict the
properties of molecule, but the drawback is its loss function is not 
quite readily to be modified. 

For UniMol backend, the training is not proceed in a real two-fold
manner, instead, it will be proceed in two steps:

1. find the coefficients of the energy components in a given pool in
   the classical way
2. use the coefficients as label to train UniMol in one-shot

For the backend built by our own, the training is proceed in the way
introduced above.
'''

# built-in modules
import os
import logging
import unittest
import time
import shutil

# third-party modules
import numpy as np

# local modules
from UniMolXC.abacus.control import AbacusJob
from UniMolXC.geometry.manip.cluster import clustergen
from UniMolXC.network.kernel.xcpnet import XCParameterizationNet
try:
    from UniMolXC.network.kernel._unimol import UniMolRegressionNet
    from UniMolXC.geometry.repr._unimol import generate_from_abacus as unimol_repr
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
        self.model_backend = None
        if isinstance(model, dict):
            # the case that utilize the third-party model
            self.model = UniMolRegressionNet(
                model_name=model.get('model_name', 'unimolv1'),
                model_size=model.get('model_size', '84m'),
                model_restart=model.get('model_restart')
            )
            self.model_backend = 'unimol'
        else:
            assert isinstance(model, XCParameterizationNet)
            self.model = model
            self.model_backend = 'xcpnet'

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
        
        Parameters
        ----------
        folders : str or list of str
            the root directory of the dataset. If a list, the list
            should contain the paths to the directories of the dataset.
        ydata : list of float, list of list of float or list of dict
            the labels of the dataset. 
            The case of being the list of float corresponds to the
            label is a scalar value; 
            The case of being the list of list of float corresponds
            to the label is a list of float; 
            The case of being the list of dict corresponds to the
            label is element-wise, and the value can also be either
            a scalar or a list of float.
        truncation : list of dict or dict
            the truncation radius of the cluster for each element, 
            can be specified one for all folders, or one for each.
            
        Returns
        -------
        xdata : dict
            the standardized xdata, which contains the following
            keys:
            - 'atoms', a list of list of str, contains element symbol
            - 'coordinates', a list of np.array, for each element,
               contains the Cartesian coordinates of atoms, the number
               of lines should be identical with the corresponding
               element in 'atoms'
        ydata : array-like
            the labels of the dataset. The labels should be in the 
            same order as xdata.
        '''
        # sanity check
        # folders
        assert all(isinstance(f, str) for f in folders)
        
        # ydata
        assert isinstance(ydata, list)
        # assert y in ydata: int|float|list|dict
        assert all(isinstance(y, (int, float)) for y in ydata) or \
               all(isinstance(y, list) for y in ydata) or \
               all(isinstance(y, dict) for y in ydata)
        # assert y in ydata: list[list[int|float]]
        if isinstance(ydata[0], list):
            assert all(isinstance(yi, (int, float)) \
                for y in ydata for yi in y)
        # assert y in ydata: list[dict[str, int|float|list[int|float]]]
        if isinstance(ydata[0], dict):
            assert all(isinstance(k, str) and isinstance(v, (int, float, list)) \
                for y in ydata for k, v in y.items())
            assert all(isinstance(v, (int, float)) \
                for y in ydata for v in y.values()) or \
                   all(isinstance(v, list) and \
                       all(isinstance(vi, (int, float)) for vi in v) \
                for y in ydata for v in y.values())
        # convert the ydata when ydata is list[int|float] to list[list[int|float]]
        if isinstance(ydata[0], (int, float)):
            ydata = [[y] for y in ydata]

        # finally
        assert len(folders) == len(ydata) > 0
        
        # instantiate the AbacusJob
        myjobs = [AbacusJob(f) for f in folders]
        for myjob in myjobs:
            myjob.read_stru(cache=True)
            
        # the case that do not require the clustergen
        if truncation is None:
            if isinstance(ydata, dict):
                ydata = list(ydata.values())
            temp = [(j.get_atomic_symbols(), j.get_atomic_positions()) for j in myjobs]
            return dict(zip(['atoms', 'coordinates'], list(zip(*temp)))), ydata

        # the case that requires the clustergen
        # assert truncation: list[dict] or dict
        assert isinstance(truncation, (list, dict))
        truncation = [truncation] * len(myjobs) \
            if isinstance(truncation, dict) else truncation
        # assert truncation: list[dict]
        assert all(isinstance(t, dict) for t in truncation)
        # for each job, the truncation and ydata should be defined
        assert len(truncation) == len(myjobs) == len(ydata)
        
        # let the clusters indexed by [ijob][icluster]
        clusters = [[clustergen(cell=myjob.get_cell(unit='angstrom'),
                                elem=myjob.get_atomic_symbols(),
                                pos =myjob.get_atomic_positions(unit='angstrom'),
                                i=iat, 
                                rc=truncate[elem])
                     for iat, elem in enumerate(myjob.get_atomic_symbols()) \
                         if (elem in truncate and isinstance(y, list)) or \
                            (elem in truncate and elem in y)]
                     for truncate, myjob, y in zip(truncation, myjobs, ydata)]

        # make sure the clusters are not empty
        assert sum(len(c) for c in clusters) > 0

        # reorganize the ydata
        if isinstance(ydata[0], list): # either duplicate for each cluster
            ydata = [y for y, j in zip(ydata, clusters) for _ in range(len(j))]
        elif isinstance(ydata[0], dict): # or element-wise prepare for each cluster
            ydata = [y[c['center_typ']] for y, j in zip(ydata, clusters) for c in j]
        else: # be careful
            raise TypeError('ydata should have been list[list[int|float]] or '
                            'list[dict[str, int|float|list[int|float]]]')

        temp = [(c['elem'], c['pos']) for j in clusters for c in j]
        return dict(zip(['atoms', 'coordinates'], list(zip(*temp)))), ydata
    
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
        return self.model.train(data=dataset,
                                epochs=inner_epochs,
                                batch_size=inner_batchsize,
                                save_path=f'XCPNTrainer-{prefix}')
    
    def _inner_train_xc_parameterization_net(self,
                                             dataset,
                                             inner_thr=None,
                                             inner_maxiter=1e5,
                                             save_model=None):
        raise NotImplementedError('The inner training of XCParameterizationNet '
                                  'is not implemented yet')           
    
    def _inner_eval(self, dataset):
        if self.model is None:
            raise RuntimeError('No backend model is assigned')
        if self.model_backend != 'unimol':
            raise NotImplementedError('The evaluation employing XCParameterizationNet '
                                      'as the inner training kernel is not implemented yet')
        return self.model.eval(data=dataset)
    
    def train(self,
              dataset,
              outer_fdft,
              outer_floss,
              outer_thr=None,
              outer_maxiter=50,
              inner_guess=None,
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
        outer_fdft : callable
            the function for calculating the energy components with
            given parameters. The function must have two and only
            two arguments:
            1. the parameters to be trained, which is a list of list
               of float, the first order index is the index of the
               prediction, and the second order index is the index
               of the energy components.
            2. the indexing from the parameters first order index to
               the practical dft cases to calculate the energy components.
            This function should return a list of list of float, each
            list indexed by the first order index is the dft cases,
            and the second order index is the energy components.
            ```python
            def f(c, index):
                # sanity check (suggested)
                assert isinstance(c, list)
                assert all(isinstance(i, list) for i in c)
                assert all(all(isinstance(cij, float) 
                           for cij in ci) for ci in c)
                assert isinstance(index, list)
                assert all(isinstance(i, int) for i in index)
                
                return [np.mean(calculate_exc_components(ci, jobdir[i]))
                            for i, ci in zip(index, c)]
            ```
        outer_floss : callable
            the function for scalarizing the prediction from inner net,
            so as to estimate the convergence of the outer training. 
            The function must have two and only two arguments:
            1. the output of the inner training that will be in the 
               form of a list of list of float.
            2. the energy components or the difference of the energy
               components between the fixed during training and the
               exact value. 
            The function should return a float.
            ```python
            def f(c, e):
                # sanity check (suggested)
                assert isinstance(c, list)
                assert all(isinstance(i, list) for i in c)
                assert all(all(isinstance(cij, float) 
                           for cij in ci) for ci in c)
                assert isinstance(e, list)
                assert all(isinstance(i, float) for i in e)
                return np.mean([np.dot(e[i], c) 
                                for i, c in zip(index, c)])
            ```
        outer_thr : float
            the threshold for determining the convergence of the whole
            training.
        outer_maxiter : int
            the maximum number of iterations for the outer training.
            Default is 50. If the `outer_thr` is achieved, the
            training will be stopped earlier than this number.
        inner_guess : list of list of float
            the initial guess for the parameters to be trained. This may
            provide a more reasonable set of energy components, so that
            the loss function of the outer loop may be minimized more
            efficiently. If set as not None, then firstly there will be
            series of DFT calculations with the initial guess parameters
            performed. If set as None, no DFT will be performed before the
            first run of inner training. NOTE: users are obliged to check
            the rationality of the length of the initial guess parameters.
            The default is None.
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
        if True: # self.model_backend != 'unimol':
            raise NotImplementedError('The training employing XCParameterizationNet '
                                      'as the inner training kernel is not implemented yet')
        loss = np.inf if inner_guess is None else outer_floss(
            c = self._inner_train_unimol_regression_net(dataset=dataset,
                                                        inner_epochs=inner_epochs,
                                                        inner_batchsize=inner_batchsize,
                                                        prefix=inner_prefix), 
            e = outer_fdft(inner_guess, np.arange(len(inner_guess)))
        )
        
        
            
        

        
        
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
        self.assertTrue(isinstance(xdata['atoms'], tuple))
        self.assertTrue(all(isinstance(job, list) and \
                        all(isinstance(elem, str) for elem in job)
                            for job in xdata['atoms'] ))
        # the label will not be duplicated because there is only one structure
        self.assertListEqual(ydata, [[0.0]])

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
        self.assertTrue(isinstance(xdata['atoms'], tuple))
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
        self.assertTrue(isinstance(xdata['atoms'], tuple))
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
        self.assertTrue(isinstance(xdata['atoms'], tuple))
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
        self.assertTrue(isinstance(xdata['atoms'], tuple))
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

    @unittest.skip('Example/Integrated test: start the inner training '
                   'of unimol backend skipped.')
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
        dataset = mytrainer.build_dataset(
            xdata=[os.path.join(self.testfiles, 'scf-unfinished')],
            ydata=[{'Zn': [1.0], 'Y': [5.0], 'S': [0.0]}],
            mode='coordinates',
            cluster_truncation={'Zn': 5.0, 'Y': 3.0, 'S': 3.0},
            walk=False
        )
        
        # Step 3: train the model and get predictions
        mytrainer._inner_train_unimol_regression_net(
            dataset=dataset,
            inner_epochs=2,
            inner_batchsize=5,
            prefix='test'
        )
        # print(mytrainer._inner_eval(dataset=dataset))
        self.assertTrue(os.path.exists('XCPNTrainer-test'))
        self.assertTrue(all([os.path.exists(os.path.join('XCPNTrainer-test', f)) for f in \
            ['config.yaml', 'cv.data', 'metric.result', 
             'model_0.pth', 'model_1.pth', 'model_2.pth', 'model_3.pth', 'model_4.pth',
             'target_scaler.ss']]))

    @unittest.skip('Example/Integrated test: restart the inner training'
                   ' from saved model of unimol backend skipped.')
    def test_train_unimol_restart(self):
        
        mytrainer = XCParameterizationNetTrainer(
            model={'model_restart': os.path.join(self.testfiles, 'XCPNTrainer-test')},
        )

        dataset = mytrainer.build_dataset(
            xdata=[os.path.join(self.testfiles, 'scf-unfinished')],
            ydata=[{'Zn': [1.0], 'Y': [5.0], 'S': [0.0]}],
            mode='coordinates',
            cluster_truncation={'Zn': 5.0, 'Y': 3.0, 'S': 3.0},
            walk=False
        )
        
        # Step 3: train the model and get predictions
        mytrainer._inner_train_unimol_regression_net(
            dataset=dataset,
            inner_epochs=2,
            inner_batchsize=5,
            prefix='test'
        )
        # print(mytrainer._inner_eval(dataset=dataset))
        self.assertTrue(os.path.exists('XCPNTrainer-test'))
        self.assertTrue(all([os.path.exists(os.path.join('XCPNTrainer-test', f)) for f in \
            ['config.yaml', 'cv.data', 'metric.result', 
             'model_0.pth', 'model_1.pth', 'model_2.pth', 'model_3.pth', 'model_4.pth',
             'target_scaler.ss']]))

    @unittest.skip('Example/Integrated test: instantiate the unimol backend'
                   ' model and call eval function skipped, because the model'
                   ' files are too large to upload. The model is produced '
                   'by the unittest `test_train_unimol_from_scratch`')
    def test_eval_unimol_from_file(self):
        mytrainer = XCParameterizationNetTrainer(
            model={
                'model_restart': os.path.join(self.testfiles, 
                                              'XCPNTrainer-test')
                },
        )
        
        dataset = mytrainer.build_dataset(
            xdata=[os.path.join(self.testfiles, 'scf-unfinished')],
            ydata=[{'Zn': [1.0], 'Y': [5.0], 'S': [0.0]}],
            mode='coordinates',
            cluster_truncation={'Zn': 5.0, 'Y': 3.0, 'S': 3.0},
            walk=False
        )
        
        res = mytrainer._inner_eval(dataset=dataset)
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