'''
build the dataset for the model training
'''
import os
import unittest

import numpy as np

from UniMolXC.abacus.control import AbacusJob
from UniMolXC.geometry.manip.cluster import clustergen

def build_dataset(xdata,
                   ydata,
                   walk=False,
                   mode='abacus',
                   **kwargs):
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
            the mode of the 'x' data, which can be 'abacus',
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
        if mode not in ['abacus', 'unimol-repr', 'deepmd-desc']:
            raise ValueError('mode should be: coordinates, unimol-repr '
                             'or deepmd-desc')
        if mode != 'abacus':
            raise NotImplementedError('Currently only the mode "abacus" '
                                      'has been implemented')
        if walk:
            raise NotImplementedError('The recursive walk mode to find '
                                      'the files is not implemented yet')

        builder = {'abacus': _build_from_abacus}

        xstandard, ystandard = builder[mode](
            folders=xdata,
            labels=ydata,
            **kwargs
        )
        return xstandard|{'target': ystandard}

def _build_from_abacus(folders, 
                       labels, 
                       cluster_truncation=None):
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

class NetworkDataManagerTest(unittest.TestCase):
    def setUp(self):
        testfiles = os.path.dirname(__file__)
        testfiles = os.path.dirname(testfiles)
        testfiles = os.path.dirname(testfiles)
        self.testfiles = os.path.abspath(os.path.join(testfiles, 'testfiles'))
    
    def test_abacus_simplest(self):
        folders = [os.path.join(self.testfiles, 'scf-finished')]
        ydata = [0.0] # assume it is the Hubbard U exact value of Silicon
        
        # the case that do not require the clustergen
        xdata, ydata = _build_from_abacus(
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

    def test_abacus_clustergen(self):
        folders = [os.path.join(self.testfiles, 'scf-finished')]
        ydata = [0.0] # assume it is the Hubbard U exact value of Silicon
        
        # the case that requires the clustergen
        # the truncation radius is 5 Angstrom, and there are two Si atoms
        # in the system, so the output should be two clusters
        xdata, ydata = _build_from_abacus(
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

    def test_abacus_multiple_element(self):
        folders = [os.path.join(self.testfiles, 'scf-finished'),
                   os.path.join(self.testfiles, 'scf-unfinished')]
        ydata = [[0.0], [1.0, 5.0, 0.0]]
        
        xdata, ydata = _build_from_abacus(
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
    
    def test_abacus_multiple_element_all_rc(self):
        folders = [os.path.join(self.testfiles, 'scf-finished'),
                   os.path.join(self.testfiles, 'scf-unfinished')]
        ydata = [[0.0], [1.0, 5.0, 0.0]]
        
        xdata, ydata = _build_from_abacus(
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

    def test_abacus_multiple_element_part_rc(self):
        folders = [os.path.join(self.testfiles, 'scf-finished'),
                   os.path.join(self.testfiles, 'scf-unfinished')]
        ydata = [[0.0], [1.0, 5.0, 0.0]]
        
        # we do not include the Si and S because DFT+U is not
        # intended for them
        xdata, ydata = _build_from_abacus(
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

