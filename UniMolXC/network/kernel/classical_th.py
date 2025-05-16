'''
The PyTorch overload of loss functions
'''
# built-in modules
import unittest

# third-party modules
import torch as th

def minnesota(e0: th.Tensor, 
              e: th.Tensor, 
              coefs: th.Tensor):
    '''
    calculate the XC loss function of minnesota flavor, this
    is the PyTorch overload of the function. Original version
    is in UniMolXC/network/kernel/classical.py
    
    Parameters
    ----------
    e0 : int, float or torch.Tensor
        the reference value
    e : torch.Tensor
        the energy terms
    coefs : torch.Tensor
        the coefficients of the energy terms
        
    Returns
    -------
    float
        the loss value
    '''
    assert len(e) == len(coefs)
    assert all(isinstance(x, th.Tensor) for x in [e0, e, coefs])
    
    if e0.ndim == e.ndim:
        assert e0.shape == e.shape == coefs.shape, \
            f'e0 shape {e0.shape}, e shape {e.shape}, ' \
            f'coefs shape {coefs.shape} do not match'
        return th.sum((e0 - e * coefs)**2) # element-wise
    
    return th.sum((e0 - th.dot(e, coefs))**2)

def perdew(e0: th.Tensor, 
           e: th.Tensor, 
           coefs: th.Tensor):
    raise NotImplementedError('Perdew flavor loss function is not'
                              ' implemented yet.')

def head_gordon(e0: th.Tensor, 
                e: th.Tensor, 
                coefs: th.Tensor):
    raise NotImplementedError('Martin Head-Gordon flavor loss '
                              'function is not implemented yet.')

class TestXCClassicalParameterizationTorchKernel(unittest.TestCase):
    '''
    Test the XC classical parameterization kernel in PyTorch
    '''
    def test_minnesota(self):
        e0 = 1.0
        e = th.tensor([0.5, 0.5])
        coefs = th.tensor([1.0, 1.0])
        loss = minnesota(e0, e, coefs)
        self.assertAlmostEqual(loss.item(), 0)
        
    def test_minnesota_autodiff(self):
        e0 = th.tensor(1.0, requires_grad=True)
        e = th.tensor([0.5, 0.5], requires_grad=True)
        coefs = th.tensor([1.0, 1.5], requires_grad=True)
        loss = minnesota(e0, e, coefs)
        loss.backward()
        
        self.assertAlmostEqual(e0.grad.item(), -0.5)
        self.assertTrue(th.allclose(e.grad, th.tensor([0.5, 0.75])))
        self.assertTrue(th.allclose(coefs.grad, th.tensor([0.25, 0.25])))

if __name__ == '__main__':
    unittest.main()