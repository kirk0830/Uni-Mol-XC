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
import unittest
import time

# third-party modules
import numpy as np

# local modules
from UniMolXC.network.kernel._xcnet import XCParameterizationNet
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
    
    def inner_train_unimol_net(self,
                                           dataset,
                                           inner_epochs=10,
                                           inner_batchsize=16,
                                           prefix=None,
                                           **kwargs):
        # a single training step
        prefix = prefix or time.strftime("%Y%m%d-%H%M%S")
        return self.model.train(data=dataset,
                                epochs=inner_epochs,
                                batch_size=inner_batchsize,
                                save_path=f'XCPNTrainer-{prefix}',
                                **kwargs)
    
    def inner_train_xcp_net(self,
                                             dataset,
                                             inner_thr=None,
                                             inner_maxiter=1e5,
                                             save_model=None):
        raise NotImplementedError('The inner training of XCParameterizationNet '
                                  'is not implemented yet')           
    
    def inner_eval(self, dataset):
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
            c = self.inner_train_unimol_net(dataset=dataset,
                                                        inner_epochs=inner_epochs,
                                                        inner_batchsize=inner_batchsize,
                                                        prefix=inner_prefix), 
            e = outer_fdft(inner_guess, np.arange(len(inner_guess)))
        )
        
