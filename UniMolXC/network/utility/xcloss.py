'''
this file places the implementation of the loss function for
fitting XC functional

any functions starts with `t` is the implementation with
torch Tensor, and any functions starts with `d` is the derivative
of the loss function with respect to the coefficients
'''
import unittest

import numpy as np
import torch as th
from torch import Tensor

from UniMolXC.utility.easyassert import loggingassert

def minnesota(coef, e, eref):
    '''
    calculate the XC loss function of minnesota flavor.
    
    Loss = sum_i (e_ref[i] - sum_j (e[i][j] * coef[j]))^2
    , in which i indexes the different structures, and j
    indexes the different energy terms.
    
    Parameters
    ----------
    coef : list of float or np.ndarray of float in 1-d
        the coefficients of the energy terms
    e : list of list of float or np.ndarray of float in 2-d
        the energy terms obtained from dft calculations,
        the first index is the job index (different 
        structures), the second index is the coefficients
        of the energy terms
    eref : list of float or np.ndarray of float in 1-d
        the reference value, e.g., the formation enthalpy,
        total energy, etc.
        
    Returns
    -------
    float
        the loss value
    '''
    e    = np.array(e)
    coef = np.array(coef)
    eref = np.array(eref)
    
    loggingassert(e.ndim == 2, 
        'the energy terms must be a 2D array, e[ijob][icoef] -> float')
    loggingassert(coef.ndim == 1,
        'the coefficients must be a 1D array, coef[icoef] -> float')
    loggingassert(eref.ndim == 1,
        'the reference value must be a 1D array, eref[icoef] -> float')
    
    njob, ncoef = e.shape
    loggingassert(njob == len(eref),
        'inconsistent number of jobs and reference values, each job should'
        f' have one reference value: {njob} != {len(eref)}')
    loggingassert(ncoef == len(coef),
        'inconsistent number of coefficients and energy terms:'
        f' {ncoef} != {len(coef)}')
    
    # calculate the loss value
    return np.mean([(erefi - np.dot(ei, coef))**2 for ei, erefi in zip(e, eref)])

def tminnesota(coef: Tensor, e: Tensor, eref: Tensor) -> Tensor:
    '''this is the PyTorch overload of function minnesota which
    benefits from the autograd feature of PyTorch. See the 
    docstring of minnesota() for the details'''
    # check the shape of the input
    loggingassert(e.ndim == 2,
        'the energy terms must be a 2D array, e[ijob][icoef] -> float')
    loggingassert(coef.ndim == 1,
        'the coefficients must be a 1D array, coef[icoef] -> float')
    loggingassert(eref.ndim == 1,
        'the reference value must be a 1D array, eref[icoef] -> float')
    njob, ncoef = e.shape
    loggingassert(njob == len(eref),
        'inconsistent number of jobs and reference values, each job should'
        f' have one reference value: {njob} != {len(eref)}')
    loggingassert(ncoef == len(coef),
        'inconsistent number of coefficients and energy terms:'
        f' {ncoef} != {len(coef)}')
    # calculate the loss value
    return th.mean(
        (eref - th.matmul(e, coef.view(-1, 1)).view(-1))**2
    )

def dminnesota(coef, e, eref):
    '''
    calculate the gradient of the XC loss function of minnesota flavor
    
    Parameters
    ----------
    coef : list of float or np.ndarray of float in 1-d
        the coefficients of the energy terms
    e : list of list of float or np.ndarray of float in 2-d
        the energy terms obtained from dft calculations,
        the first index is the job index (different 
        structures), the second index is the coefficients
        of the energy terms
    eref : list of float or np.ndarray of float in 1-d
        the reference value, e.g., the formation enthalpy,
        total energy, etc.
        
    Returns
    -------
    np.ndarray of float in 1-d
        the gradient of the loss function with respect to
        the coefficients
    '''
    e    = np.array(e)
    coef = np.array(coef)
    eref = np.array(eref)
    
    loggingassert(e.ndim == 2, 
        'the energy terms must be a 2D array, e[ijob][icoef] -> float')
    loggingassert(coef.ndim == 1,
        'the coefficients must be a 1D array, coef[icoef] -> float')
    loggingassert(eref.ndim == 1,
        'the reference value must be a 1D array, eref[icoef] -> float')
    
    njob, ncoef = e.shape
    loggingassert(njob == len(eref),
        'inconsistent number of jobs and reference values, each job should'
        f' have one reference value: {njob} != {len(eref)}')
    loggingassert(ncoef == len(coef),
        'inconsistent number of coefficients and energy terms:'
        f' {ncoef} != {len(coef)}')
    
    # calculate the gradient of the loss function
    return -2 * np.mean([ei * (erefi - np.dot(ei, coef))
                         for ei, erefi in zip(e, eref)], axis=0)

def perdew(coef, e, eref):
    raise NotImplementedError('Perdew flavor loss function is not'
                              ' implemented yet.')

def tperdew(coef: Tensor, e: Tensor, eref: Tensor) -> Tensor:
    '''this is the PyTorch overload of function perdew which
    benefits from the autograd feature of PyTorch. See the 
    docstring of perdew() for the details'''
    raise NotImplementedError('Perdew flavor loss function is not'
        ' implemented yet.')

def dperdew(coef, e, eref):
    raise NotImplementedError('Perdew flavor loss function derivative'
                              ' of loss function is not implemented yet.')

class TestXCLoss(unittest.TestCase):
    
    def test_minnesota(self):
        eref = [1.0]
        e = np.array([[1.0, 2.0, 3.0]])
        coef = np.array([1.0, 2.0, 3.0])
        self.assertAlmostEqual(minnesota(coef, e, eref), 13**2)

    def test_dminnesota(self):
        eref = [1.0]
        e = np.array([[1.0, 2.0, 3.0]])
        coef = np.array([1.0, 2.0, 3.0])
        self.assertTrue(np.allclose(dminnesota(coef, e, eref), 
            -2 * np.dot((eref - np.dot(e, coef)), e)))
    
    def test_tminnesota(self):
        eref = [1.0]
        e = np.array([[1.0, 2.0, 3.0]])
        coef = np.array([1.0, 2.0, 3.0])
        
        teref = th.tensor(eref, requires_grad=True)
        te = th.tensor(e, requires_grad=True)
        tcoef = th.tensor(coef, requires_grad=True)
        loss = tminnesota(tcoef, te, teref)
        # in accord with the numpy version
        self.assertAlmostEqual(loss.item(), minnesota(coef, e, eref))
        # check the gradient
        loss.backward()
        # in accord with the numpy version
        self.assertTrue(np.allclose(
            tcoef.grad.numpy(), dminnesota(coef, e, eref)
        ))
    
if __name__ == '__main__':
    unittest.main()