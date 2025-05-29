'''
build the dataset for the model training
'''
# in-built modules
import os
import unittest

# third-party modules
import numpy as np

# local modules
from UniMolXC.network.kernel._unimol import \
    build_dataset_from_abacus as abacus_to_unimol_dataset
from UniMolXC.network.kernel._xcnet import \
    build_dataset_from_abacus as abacus_to_xcnet_dataset
    
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
    return abacus_to_unimol_dataset(
        folders=folders,
        labels=labels,
        cluster_truncation=cluster_truncation
    )

def uniform_random_selector(nsample, ntotal, nfold, seed=None):
    '''
    A generator returning the list of indices of the selected samples.
    
    Parameters
    ----------
    nsample : int
        the number of samples to be selected
    ntotal : int
        the total number of samples
    nfold : int
        the number of times one sample will be selected
    seed : int, optional
        the seed for the random number generator. The default is None.
        If None, the random number generator will be initialized
        with the current time.
    
    Details
    -------
    with a uniform random manner, select the `nsample` from `ntotal`
    samples by `ntimes` times, for each sample, it will be sampled
    `nfold` times, and the seed is used to control the random
    number generator. We have: `ntotal` * `nfold` = `nsample` * `ntimes`
    when we collect all generated indices.
    
    Yield
    -----
    list of int
        the list of indices of the selected samples.
    '''
    pass

class NetworkDataManagerTest(unittest.TestCase):

    def test_uniform_random_selector(self):
        nsample = 3
        ntotal = 10
        ntimes = 5
        nfold = 2
        seed = 42
        # 10 samples, select 3 samples each time, and for each sample
        # being selected 2 times
        
        res = list(uniform_random_selector(nsample, ntotal, ntimes, nfold, seed))
        for r in res:
            print(r)
        
        self.assertEqual(len(res), ntimes)
        self.assertTrue(all(len(r) == nsample for r in res))
        merged = np.array(res).flatten()
        unique, count = np.unique(merged, return_counts=True)
        # the number of unique samples should be equal to the number of
        # samples
        self.assertEqual(len(unique), ntotal)
        self.assertTrue(all(c == nfold for c in count))

if __name__ == '__main__':
    unittest.main()