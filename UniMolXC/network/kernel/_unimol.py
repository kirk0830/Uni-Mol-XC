# in-built modules
import os
import unittest

# third-party modules
try:
    from unimol_tools import MolTrain, MolPredict
except ImportError:
    raise ImportError('unimol_tools is not installed. '
                      'See https://github.com/deepmodeling/Uni-Mol/'
                      'tree/main/unimol_tools for more information.')

# local modules
from UniMolXC.abacus.control import AbacusJob
from UniMolXC.geometry.manip.cluster import clustergen

def build_dataset_from_abacus(folders, labels, cluster_truncation=None):
    '''
    build the dataset in the case that the data arranged like
    ABACUS jobdir. The function will walk through the directories
    
    Parameters
    ----------
    folders : str or list of str
        the root directory of the dataset. If a list, the list
        should contain the paths to the directories of the dataset.
    labels : list of float, list of list of float or list of dict
        the labels of the dataset. 
        The case of being the list of float corresponds to the
        label is a scalar value; 
        The case of being the list of list of float corresponds
        to the label is a list of float; 
        The case of being the list of dict corresponds to the
        label is element-wise, and the value can also be either
        a scalar or a list of float.
    cluster_truncation : list of dict or dict
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
    labels : array-like
        the labels of the dataset. The labels should be in the 
        same order as xdata.
    '''
    # sanity check
    # folders
    assert all(isinstance(f, str) for f in folders)
    
    # labels
    assert isinstance(labels, list)
    # assert y in labels: int|float|list|dict
    assert all(isinstance(y, (int, float)) for y in labels) or \
            all(isinstance(y, list) for y in labels) or \
            all(isinstance(y, dict) for y in labels)
    # assert y in labels: list[list[int|float]]
    if isinstance(labels[0], list):
        assert all(isinstance(yi, (int, float)) \
            for y in labels for yi in y)
    # assert y in labels: list[dict[str, int|float|list[int|float]]]
    if isinstance(labels[0], dict):
        assert all(isinstance(k, str) and isinstance(v, (int, float, list)) \
            for y in labels for k, v in y.items())
        assert all(isinstance(v, (int, float)) \
            for y in labels for v in y.values()) or \
                all(isinstance(v, list) and \
                    all(isinstance(vi, (int, float)) for vi in v) \
            for y in labels for v in y.values())
    # convert the labels when labels is list[int|float] to list[list[int|float]]
    if isinstance(labels[0], (int, float)):
        labels = [[y] for y in labels]

    # finally
    assert len(folders) == len(labels) > 0
    
    # instantiate the AbacusJob
    myjobs = [AbacusJob(f) for f in folders]
    for myjob in myjobs:
        myjob.read_stru(cache=True)
        
    # the case that do not require the clustergen
    if cluster_truncation is None:
        if isinstance(labels, dict):
            labels = list(labels.values())
        temp = [(j.get_atomic_symbols(), j.get_atomic_positions()) for j in myjobs]
        return dict(zip(['atoms', 'coordinates'], list(zip(*temp)))), labels

    # the case that requires the clustergen
    # assert cluster_truncation: list[dict] or dict
    assert isinstance(cluster_truncation, (list, dict))
    cluster_truncation = [cluster_truncation] * len(myjobs) \
        if isinstance(cluster_truncation, dict) else cluster_truncation
    # assert truncation: list[dict]
    assert all(isinstance(t, dict) for t in cluster_truncation)
    # for each job, the truncation and labels should be defined
    assert len(cluster_truncation) == len(myjobs) == len(labels)
    
    # let the clusters indexed by [ijob][icluster]
    clusters = [[clustergen(cell=myjob.get_cell(unit='angstrom'),
                            elem=myjob.get_atomic_symbols(),
                            pos =myjob.get_atomic_positions(unit='angstrom'),
                            i=iat, 
                            rc=truncate[elem])
                    for iat, elem in enumerate(myjob.get_atomic_symbols()) \
                        if (elem in truncate and isinstance(y, list)) or \
                        (elem in truncate and elem in y)]
                    for truncate, myjob, y in zip(cluster_truncation, myjobs, labels)]

    # make sure the clusters are not empty
    assert sum(len(c) for c in clusters) > 0

    # reorganize the labels
    if isinstance(labels[0], list): # either duplicate for each cluster
        labels = [y for y, j in zip(labels, clusters) for _ in range(len(j))]
    elif isinstance(labels[0], dict): # or element-wise prepare for each cluster
        labels = [y[c['center_typ']] for y, j in zip(labels, clusters) for c in j]
    else: # be careful
        raise TypeError('labels should have been list[list[int|float]] or '
                        'list[dict[str, int|float|list[int|float]]]')

    temp = [(c['elem'], c['pos']) for j in clusters for c in j]
    return dict(zip(['atoms', 'coordinates'], list(zip(*temp)))), labels

class TestBuildUnimolDataset(unittest.TestCase):
    def setUp(self):
        testfiles = os.path.dirname(__file__)
        testfiles = os.path.dirname(testfiles)
        testfiles = os.path.dirname(testfiles)
        self.testfiles = os.path.abspath(os.path.join(testfiles, 'testfiles'))
    
    def test_abacus_simplest(self):
        folders = [os.path.join(self.testfiles, 'scf-finished')]
        ydata = [0.0] # assume it is the Hubbard U exact value of Silicon
        
        # the case that do not require the clustergen
        xdata, ydata = build_dataset_from_abacus(
            folders=folders,
            labels=ydata
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

    def test_abacus_clustergen(self):
        folders = [os.path.join(self.testfiles, 'scf-finished')]
        ydata = [0.0] # assume it is the Hubbard U exact value of Silicon
        
        # the case that requires the clustergen
        # the truncation radius is 5 Angstrom, and there are two Si atoms
        # in the system, so the output should be two clusters
        xdata, ydata = build_dataset_from_abacus(
            folders=folders,
            labels=ydata,
            cluster_truncation=[{'Si': 5.0}] # the truncation radius as 5 Angstrom
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

    def test_abacus_multiple_element(self):
        folders = [os.path.join(self.testfiles, 'scf-finished'),
                   os.path.join(self.testfiles, 'scf-unfinished')]
        ydata = [[0.0], [1.0, 5.0, 0.0]]
        
        xdata, ydata = build_dataset_from_abacus(
            folders=folders,
            labels=ydata
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
    
    def test_abacus_multiple_element_all_rc(self):
        folders = [os.path.join(self.testfiles, 'scf-finished'),
                   os.path.join(self.testfiles, 'scf-unfinished')]
        ydata = [[0.0], [1.0, 5.0, 0.0]]
        
        xdata, ydata = build_dataset_from_abacus(
            folders=folders,
            labels=ydata,
            cluster_truncation={'Si': 3.0, 'Y': 5.0, 'Zn': 5.0, 'S': 3.0}
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

    def test_abacus_multiple_element_part_rc(self):
        folders = [os.path.join(self.testfiles, 'scf-finished'),
                   os.path.join(self.testfiles, 'scf-unfinished')]
        ydata = [[0.0], [1.0, 5.0, 0.0]]
        
        # we do not include the Si and S because DFT+U is not
        # intended for them
        xdata, ydata = build_dataset_from_abacus(
            folders=folders,
            labels=ydata,
            cluster_truncation={'Zn': 5.0, 'Y': 3.0}
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

class UniMolRegressionNet:
    '''
    the wrapper of regression model from UniMol. 
    Unimol is a deep learning model for predicting molecular properties
    with molecule's SMILES representation, or with the atomic coordiates.
    '''
    def __init__(self,
                 model_name='unimolv1',
                 model_size='84m',
                 model_restart=None):
        '''
        instantiate an UniMol regression model interface.
        
        Parameters
        ----------
        model_name : str
            the name of the UniMol model to be used. Available options
            are: 'unimolv1', 'unimolv2'. Default is 'unimolv1'.
        model_size : str
            the size of the UniMol model to be used. Available options
            are: '84m', '164m', '310m', '570m', '1.1B'. Default is '84m'.
            To use the larger models, make sure you have enough memory
            and CPU/GPU resources.
        model_restart : str, optional
            the path to the model checkpoint to be loaded. If None,
            a new model will be trained from scratch. Default is None.

        Notes
        -----
        The kernel UniMol model will not be allocated soon after this
        function is called. The model will be allocated only when the
        function `train` is called.
        '''
        # the kernel
        self.model = None
        
        # either from scratch or from a restart
        assert (None not in [model_name, model_size]) or \
               (model_restart is not None)

        # from scratch
        self.model_name = model_name
        self.model_size = model_size
        self.model_restart_from = None
        
        # from file
        if model_restart is not None:
            self.model = MolPredict(load_model=model_restart)
            self.model_restart_from = model_restart
    
    def train(self,
              data,
              metrics='mse',
              epochs=10,
              batch_size=16,
              save_path=None,
              **kwargs):
        '''
        trigger the training process of the UniMol multilabel 
        regression model. We strongly recommend users to assign
        appropriate values to the parameters `epochs`, `batch_size`
        and `metrics` to get a good model and do not let them
        to be the default values. Other parameters are optional
        and will also be passed to the training process.
        
        Parameters
        ----------
        data : dict or list of str
            the training data. For the use in this package, please
            refer to the function in file network/train.py:
            build_dataset_from_abacus.
        metrics : str, optional
            the way to scalarize the loss. Default is 'mse', which
            supports both scalar and vector labels. To see other
            options, please see annotation of the function
            unimol_tools.MolTrain.__init__.
        epochs : int, optional
            the number of epochs to train. Default is 10.
        batch_size : int, optional
            the batch size of training. Default is 16.
        save_path : str, optional
            the path to save the model. Default is None, which means
            the model will not be saved. If you want to save the model,
            please provide a valid path.
        **kwargs : dict
            other parameters to be passed to the training process.
            For the full list of parameters, please refer to the
            function unimol_tools.MolTrain.__init__.
        '''
        # distinct with the original UniMol, we force the user
        # to provide the save_path for the model to be saved.
        assert save_path is not None
        
        # we check the parameters list of the constructor of
        # unimol_tools.MolTrain, and we only keep the parameters
        # closely related to the training process.
        unimol_mol_train_instantiate_selected_kwargs = [
            'learning_rate', 'early_stopping', 'split', 
            'split_group_col', 'kfold', 'remove_hs', 'smiles_col', 
            'target_cols', 'target_col_prefix', 'target_anomaly_check', 
            'smiles_check', 'target_normalize', 'max_norm', 'use_cuda', 
            'use_amp', 'use_ddp', 'use_gpu', 'freeze_layers', 
            'freeze_layers_reversed'
        ]
        # extract the parameters from the kwargs
        train_param = {k: v for k, v in kwargs.items() 
                       if k in unimol_mol_train_instantiate_selected_kwargs}
        pred = MolTrain(task='multilabel_regression',
                        data_type='molecule',
                        epochs=epochs,
                        batch_size=batch_size,
                        metrics=metrics,
                        model_name=self.model_name,
                        model_size=self.model_size,
                        save_path=save_path,
                        load_model_dir=self.model_restart_from,
                        **train_param).fit(data=data)

        # cache the model trained
        self.model = MolPredict(load_model=save_path)        
        return pred
        
    def eval(self, data):
        '''
        trigger the evaluation process of the UniMol multilabel 
        regression model.
        '''
        if self.model is None:
            raise ValueError('The model is not trained yet. '
                             'Please call the `train` function first.')
        return self.model.predict(data=data)
