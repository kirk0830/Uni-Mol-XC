'''
Title
-----
Local chemical environment incorperated parameterizing XC functionals 
enabled by machine-learning

Concept
-------
The training of machine-learning parameterization on Density Functional
eXchange-Correlation (XC) here only considers re-parameterize the XC
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
and the energy components consistent with each other. 

However, this manner is quite computationally costly, so an alternative
strategy is to train the parameters with fixed energy components. 
After some epochs, the energy components will be updated with the
newly trained parameters, so the loss function of the inner training
will be changed, as a result. This process will be repeated until the
loss converges to a certain threshold.

In following codes, we will call the training of parameters with fixed
energy components as `inner training`, counterpartly, the outer loop
during which the XC energy terms are updated will be called the 
`outer training`.

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
import time

# third-party modules
import numpy as np

# local modules
from UniMolXC.network.kernel._xcnet import XCParameterizationNet
from UniMolXC.network.kernel._xcnet import \
    build_dataset_from_abacus as abacus_to_xcpnet_dataset
from UniMolXC.network.kernel._unimol import UniMolRegressionNet
from UniMolXC.network.kernel._unimol import \
    build_dataset_from_abacus as abacus_to_unimol_dataset
from UniMolXC.network.utility.xcfit import fit as classical_xc_fit
from UniMolXC.network.utility.xcfit import build_ener_calculator
from UniMolXC.network.utility.xcfit import calc_ener as calc_exc_terms
from UniMolXC.network.utility.xcloss import minnesota, dminnesota, tminnesota

class XCParameterizationNetTrainer:

    def __init__(self, 
                 model: dict):
        '''
        instantiate a trainer for XCParameterizationNet.
        
        Parameters
        ----------
        model : dict
            the basic configuration of the backend model
        '''
        self.model_backend = model.get('model_backend', 'xcpnet')
        assert self.model_backend in ['xcpnet', 'unimol']
        assert isinstance(model, dict)
        
        # instantiate the model backend
        if self.model_backend == 'unimol':
            # the case that utilize the third-party model
            self.model = UniMolRegressionNet(
                model_name=model.get('model_name', 'unimolv1'),
                model_size=model.get('model_size', '84m'),
                model_restart=model.get('model_restart')
            )
        else:
            self.model = XCParameterizationNet(
                elem=model.get('elem', None),
                model_size=model.get('model_size', 
                                     {'ndim': 448, 
                                      'nparam': 1, 
                                      'nhidden': [240, 240, 240]}),
                model_restart=model.get('model_restart')
            )

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
    
    def train_unimol_net(self,
                         outer_e_ref,  # the reference energies
                         outer_e_init, # the initial exc components
                         outer_dft_prototyp_dir, # xc training set folders
                         outer_dft_recipe_name,  # the adjustable parameter name
                         outer_dft_run_option,   # the run option for the DFT jobs
                         outer_e_read_func,      # the function to read the energy from log
                         outer_loss_func=minnesota,
                         outer_dloss_func=dminnesota,
                         outer_loss_thr=None,
                         outer_maxiter=10,
                         label_init=None,
                         label_thr=None,
                         inner_epochs=10,
                         inner_batchsize=16,
                         inner_clustergen_scheme=None,
                         prefix=None,
                         **kwargs):
        # raise NotImplementedError('The overall training covering the process of '
        #     'the label generation and the training of UniMol Multilabel Regression '
        #     ' Net is not properly implemented yet. Presently, the training data is '
        #     'not properly constructed, due to all structures share the same suite '
        #     'of XC functional coefficients, so all clusters generated will also have'
        #     ' the identical labels. This will drive the model to act to be a '
        #     'constant function, which is not the expected behavior. A strategy to '
        #     'mitigate this issue is to randomly select a subset of systems to '
        #     'generate the labels, and then use the corresponding structures to '
        #     'generate clusters and train the model. How good will this be is not'
        #     ' tested yet. The function for randomly selecting a subset of systems '
        #     'is not implemented yet, see the function uniform_random_selector in '
        #     'file network/utility/preprocess.py.\n'
        #     'Another strategy would be manually impose some numerical noise to the
        #     'labels, while this is not tested, neither.')
        
        # one-shot label generation, but will be time-consuming
        label, loss_labelgen = classical_xc_fit(
            eref=outer_e_ref, e_init=outer_e_init,
            dft_prototyp_dir=outer_dft_prototyp_dir,
            f_xc_loss=outer_loss_func,
            keyword_coef=outer_dft_recipe_name,
            jobrun_option=outer_dft_run_option,
            f_ener_reader=outer_e_read_func,
            df_xc_loss=outer_dloss_func,
            coef_init=label_init, coef_thr=label_thr,
            loss_thr=outer_loss_thr,
            maxiter=outer_maxiter,
            remove_jobdir_after_run=kwargs.get('remove_jobdir_after_run', True),
        )
        print(f'The label is generated with error: {loss_labelgen:>10.4e}', flush=True)
        dataset = abacus_to_unimol_dataset(
            xdata=outer_dft_prototyp_dir,
            ydata=[label for _ in range(len(outer_dft_prototyp_dir))],
            cluster_truncation=inner_clustergen_scheme
        )
        return self.inner_train_unimol_net(
            dataset=dataset,
            inner_epochs=inner_epochs,
            inner_batchsize=inner_batchsize,
            prefix=prefix,
            **kwargs
        )
    
    def inner_train_xcp_net(self,
                            dataset,
                            inner_epochs=10,
                            inner_batchsize=16,
                            prefix=None,
                            **kwargs):
        self.model.train(data=dataset,
                         epochs=inner_epochs,
                         batch_size=inner_batchsize,
                         f_loss=tminnesota,
                         save_path=f'XCPNTrainer-{prefix}',
                         **kwargs)

    def train_xcp_net(self,
                      outer_e_ref,
                      outer_e_init,
                      outer_dft_prototyp_dir,
                      outer_dft_recipe_name,
                      outer_dft_run_option,
                      outer_e_read_func,
                      outer_maxiter=10,
                      label_init=None,
                      label_thr=None,
                      inner_epochs=5,
                      inner_batchsize=16,
                      inner_descgen=None,
                      prefix=None,
                      **kwargs):
        '''
        the xcpnet is for handling a more complicated case, where the
        energy terms of XC are the functionals of the inner train label
        (the XC coefficients).
        
        Algorithm
        ---------
        1. with given outer_e_ref and outer_e_init, generate the dataset
           and train the xcpnet model with some epochs
        2. after the inner training finished, get the coefficients and
           perform DFT calculations with the new set of coefficients,
           obtain the new outer_e_ref and outer_e_init
        3. repeat the step 1 and 2 until the coefficients converges
        4. save the trained model
        '''
        coefs_ = [0.0 for _ in outer_dft_prototyp_dir]
        dcoef = np.inf
        outer_jiter = 0
        
        msg = f'{"ITER":>4s} {"Convergence":>15s} {"TIME/s":>10s}'
        print('', flush=True)
        print(msg, flush=True)
        
        time_ = time.time()
        # generate the initial dataset
        dataset = abacus_to_xcpnet_dataset(
            xdata=outer_dft_prototyp_dir,
            ydata=outer_e_ref,
            f_descmodel=inner_descgen
        )
        print(f'{"Data initialization":>20s} {time.time() - time_:>10.4f}', flush=True)
        time_ = time.time()
        
        while (outer_jiter <= outer_maxiter) and \
              (True if label_thr is None else dcoef > label_thr):
            # step 1: generate the dataset and train the xcpnet model

            self.inner_train_xcp_net(
                dataset=dataset,
                inner_epochs=inner_epochs,
                inner_batchsize=inner_batchsize,
                prefix=prefix,
                **kwargs
            ) # it would be better to train in a fine-tune manner
            
            # step 2: get the coefficients and perform DFT calculations
            coefs = self.model.eval(data=dataset)
            dataset['e'] = [calc_exc_terms(
                job,
                outer_dft_run_option,
                {outer_dft_recipe_name: coef},
                f_ener_reader=outer_e_read_func,
                remove_after_run=kwargs.get('remove_jobdir_after_run', True))
            for job, coef in zip(build_ener_calculator(outer_dft_prototyp_dir), coefs)]
            
            # step 3: calculate the loss
            dcoef = np.mean([np.linalg.norm(coef - coef_) 
                             for coef, coef_ in zip(coefs, coefs_)])
            coefs_ = coefs.copy()
            
            # print the information
            msg = f'{outer_jiter:>4d} {dcoef:>15.4e} {time.time() - time_:>10.4f}'
            print(msg, flush=True)
            time_ = time.time()
            
            # update the loop control variables
            outer_jiter += 1
        
    
    def inner_eval(self, dataset):
        if self.model is None:
            raise RuntimeError('No backend model is assigned')
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
            the dataset to be used for training.
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
        
