'''
In brief
--------
this file places the implementation of the classical
way to parameterize the eXchange-Correlation functional.

More specifically, it implements the linear regression,
the classical Minnesota functional production manner.

Algorithm
---------
With a given training set (a pool of molecules), several
energy terms are calculated initially with one arbitrary 
method, then they are combined with "parameters". Those
parameters will be optimized.

The energy obtained by the above combination will 
participate in the loss/penalty function. Usually the 
"distance" with experimental value, like formation 
enthalpy, ..., will be used as the loss. The parameters' 
optimal values are then obtained by minimizing the loss 
function.

Once the set of parameters is obtained, the energy terms 
are re-evaluated with the newly obtained parameters. 
Then the process is repeated until convergence is reached.
'''
# built-in modules
import os
import unittest

# third-party modules
import numpy as np
from scipy.optimize import minimize

# local modules
from UniMolXC.abacus.control import AbacusJob

def loss(e0, e, coefs):
    '''
    calculate the loss function
    
    Parameters
    ----------
    e0 : float or array-like
        the reference value
    e : array-like
        the energy terms
    coefs : array-like
        the coefficients of the energy terms
        
    Returns
    -------
    float
        the loss value
    '''
    assert len(e) == len(coefs)
    e = np.array(e); coefs = np.array(coefs)
    
    if not isinstance(e0, (int, float)):
        assert len(e0) == len(e)
        e0 = np.array(e0)
        return np.sum((e0 - e * coefs)**2) # element-wise
    
    return np.sum((e0 - np.dot(e, coefs))**2)

def dloss(e0, e, coefs):
    '''
    calculate the gradient of the loss function
    
    Parameters
    ----------
    e0 : float or array-like
        the reference value
    e : array-like
        the energy terms
    coefs : array-like
        the coefficients of the energy terms
        
    Returns
    -------
    array-like
        the gradient of the loss value
    '''
    assert len(e) == len(coefs)
    e = np.array(e); coefs = np.array(coefs)
    if not isinstance(e0, (int, float)):
        assert len(e0) == len(e)
        e0 = np.array(e0)
    return -2 * np.dot((e0 - np.dot(e, coefs)), e)

def calc_coefs(e0, e, coefs0=None, bound=None):
    '''
    calculate the coefficients of the energy terms by minimizing
    the loss function
    
    Parameters
    ----------
    e0 : float or array-like
        the reference value
    e : array-like
        the energy terms
    coefs0 : array-like, optional
        the initial coefficients of the energy terms
    bound : list of tuples, optional
        the bounds of the coefficients of the energy terms
        [(min1, max1), (min2, max2), ...]
        if None, the coefficients are not bounded
        
    Returns
    -------
    array-like
        the coefficients of the energy terms
    '''
    # sanity checks
    if coefs0 is None:
        coefs0 = np.ones(len(e))
    assert len(e) == len(coefs0)
    assert all(isinstance(c, (int, float)) for c in coefs0)
    
    if bound is None:
        bound = [(None, None) for _ in range(len(e))]
    assert len(e) == len(bound)
    assert all([len(b) == 2 for b in bound])
    
    # minimize the loss function
    res = minimize(loss, coefs0, args=(e0, e), jac=dloss, bounds=bound)
    if not res.success:
        raise ValueError('Optimization failed: {}'.format(res.message))
    return res.x

def build_training_set(jobdir):
    '''
    build the training set from the jobdir
    
    Parameters
    ----------
    jobdir : str of list of str
        the directory of the job or a list of directories
    
    Returns
    -------
    list of AbacusJob
        the training set
    '''
    jobdir = [jobdir] if isinstance(jobdir, str) else jobdir
    assert all(isinstance(d, str) for d in jobdir)
    
    return [AbacusJob(d) for d in jobdir]



class TestNetworkClassicalParameterizationKernel(unittest.TestCase):
    
    def test_loss(self):
        e0 = 1.0
        e = np.array([1.0, 2.0, 3.0])
        coefs = np.array([1.0, 2.0, 3.0])
        self.assertAlmostEqual(loss(e0, e, coefs), 13**2)

        e0 = [1.0, 2.0, 3.0]
        e = np.array([1.0, 2.0, 3.0])
        coefs = np.array([1.0, 2.0, 3.0])
        self.assertAlmostEqual(loss(e0, e, coefs), 2**2 + (2*3)**2)

    def test_dloss(self):
        e0 = 1.0
        e = np.array([1.0, 2.0, 3.0])
        coefs = np.array([1.0, 2.0, 3.0])
        self.assertTrue(np.allclose(dloss(e0, e, coefs), 
                                    -2 * np.dot((e0 - np.dot(e, coefs)), e)))
        
if __name__ == '__main__':
    unittest.main()
