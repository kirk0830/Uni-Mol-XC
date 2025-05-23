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
import time
import uuid
import shutil
import unittest
import logging
from typing import Callable

# third-party modules
import numpy as np
from scipy.optimize import minimize

# local modules
from UniMolXC.abacus.control import AbacusJob
from UniMolXC.utilities.easyassert import loggingassert

def calc_coef(eref, e, loss_func, dloss_func, coef_init=None, bound=None):
    '''
    calculate the coefficients of the energy terms by minimizing
    the loss function
    
    Parameters
    ----------
    eref : float or array-like
        the reference value
    e : np.ndarray
        the energy terms in 2D, the first dimension is the job index,
        the second dimension is the coefficients of the energy terms
    loss_func : callable
        the loss function to be minimized, must place the variable to
        minimize 
    dloss_func : callable
        the gradient of the loss function
    coef_init : np.ndarray, optional
        the energy coefficients to be optimized.
        if provided, the shape should be identical with `e`
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
    assert isinstance(e, np.ndarray)
    assert e.ndim == 2, 'e must be a 2D array-like object'
    
    _, ncoef = e.shape
    if coef_init is None:
        coef_init = np.random.rand(ncoef)
    assert coef_init.ndim == 1
    assert len(coef_init) == ncoef
    
    if bound is None:
        bound = [(None, None) for _ in range(ncoef)]
    assert len(bound) == ncoef
    assert all([len(b) == 2 for b in bound])
    
    # minimize the loss function
    res = minimize(loss_func, 
                   coef_init, 
                   args=(e, eref), 
                   jac=dloss_func, 
                   bounds=bound)
    if not res.success:
        raise ValueError(f'Optimization failed: {res.message}')
    return res.x

def build_ener_calculator(prototyp_dir):
    '''
    build the energy terms calculator from the prepared ABACUS jobdir
    
    Parameters
    ----------
    jobdir : str of list of str
        the directory of the job or a list of directories
    
    Returns
    -------
    list of AbacusJob
        the well-managed ABACUS job objects, whose `build_derived`
        method can be called to build new job with updated parameters
    '''
    if not isinstance(prototyp_dir, list):
        prototyp_dir = [prototyp_dir]
    assert all(isinstance(prototyp, str) for prototyp in prototyp_dir)
    
    proto = [AbacusJob(d) for d in prototyp_dir]
    for j in proto:
        _ = j.read_stru(cache=True)
        _ = j.read_kpt(cache=True)
    return proto

def calc_ener(job: AbacusJob, 
              jobrun_option: dict, 
              kw_new: dict, 
              f_ener_reader: Callable,
              remove_after_run: bool = True):
    '''
    calculate the energy terms for one abacus job
    '''
    assert isinstance(job, AbacusJob)
    assert isinstance(jobrun_option, dict)
    assert 'command' in jobrun_option
    assert isinstance(kw_new, dict)
    
    jobdir_new = f'XCPNTrainerClassical-{str(uuid.uuid4().hex)}'
    newjob = job.build_derived(jobdir=jobdir_new,
                               dftparam=kw_new,
                               instatiate=True)
    newjob.run(**jobrun_option)
    # wait for the job to finish
    
    e = f_ener_reader(newjob.path)
    if remove_after_run:
        shutil.rmtree(newjob.path)
    return e

def _fit_kernel(eref,
                e_init,
                f_xc_loss,
                keyword_coef,
                prototyp_dir,
                jobrun_option,
                f_ener_reader,
                df_xc_loss=None,
                coef_init=None,
                coef_thr=None,
                loss_thr=None,
                maxiter=10,
                remove_jobdir_after_run=True):
    '''see function `fit` for details'''
    # initialize
    eterms = np.array(e_init)
    _, ncoef = eterms.shape
    loss = np.inf
    coef = np.zeros(ncoef)
    coef_ = coef_init
    dcoef = np.inf
    
    jiter = 0
    time_ = time.time()
    
    msg = f'{"ITER":>4s} {"LOSS":>10s} {"Convergence":>15s} {"TIME/s":>10s}'
    print('', flush=True)
    print(msg, flush=True)
    logging.info(msg)
    while jiter <= maxiter and \
        (True if coef_thr is None else dcoef > coef_thr) and \
        (True if loss_thr is None else loss > loss_thr):
        
        # minimize the loss function to get the coefficients
        coef = calc_coef(eref=eref, 
                         e=eterms, 
                         loss_func=f_xc_loss,
                         dloss_func=df_xc_loss,
                         coef_init=coef_)
        
        # re-calculate the energy terms with the new coefficients
        eterms = np.array([calc_ener(job,
                                     jobrun_option,
                                     {keyword_coef: coef.tolist()},
                                     f_ener_reader,
                                     remove_jobdir_after_run)
                           for job in build_ener_calculator(prototyp_dir)])
        
        # calculate the loss value
        loss = f_xc_loss(coef, eterms, eref)
        
        # calculate the change of coefficients
        dcoef = np.linalg.norm(coef - coef_)
        coef_ = coef.copy()
        
        # print the information
        msg = f'{jiter:>4d} {loss:>10.6f} {dcoef:>15.6f} {time.time() - time_:>10.2f}'
        print(msg, flush=True)
        logging.info(msg)
        time_ = time.time()
        
        # update the loop control variables
        jiter += 1
    
    msg = f'Fitting finished after {jiter} iterations.'
    print(msg, flush=True)
    logging.info(msg)
    
    return coef, loss

def fit(eref,
        e_init,
        f_xc_loss,
        keyword_coef,
        prototyp_dir,
        jobrun_option,
        f_ener_reader,
        df_xc_loss=None,
        coef_init=None,
        coef_thr=None,
        loss_thr=None,
        maxiter=10,
        remove_jobdir_after_run=True,
        flog=None):
    '''
    fit the target reference energies by iteratively optimizing
    the coefficients of the energy terms. After the optimization
    being converged, the energy terms will be re-evaluated
    with the newly obtained coefficients. This process will be
    proceed till convergence is reached.
    
    Parameters
    ----------
    eref : array-like
        the reference values, e.g., the formation enthalpy
    e_init : array-like
        the initial energy terms, e.g., the energy terms
        calculated with one arbitrary method, e.g., DFT, CCSD, etc.
    f_xc_loss : callable
        the loss function to be minimized, e.g., minnesota
    keyword_coef : str
        the keyword of the coefficients in the abacus job
        input file, e.g., 'xc_coef'
    prototyp_dir : str or list of str
        the directory of the prototyped abacus job, or a list
        of directories. The `eref` should be those energy 
        components calculated with these prototyped jobs at
        one level of theory
    jobrun_option : dict
        the options for running the abacus job, e.g., 
        {'command': 'abacus run'}. For more details, please
        refer to the `AbacusJob.run` method.
    f_ener_reader : callable
        the function to read the energy terms from the
        abacus job. It should take the job directory as the
        only input and return the energy terms as a numpy
        array
    df_xc_loss : callable, optional
        the gradient of the loss function, if None, the
        gradient will be calculated numerically. If provided,
        it should take the same arguments as `f_xc_loss`
        and return the gradient of the loss function
        with respect to the coefficients
    coef_init : array-like, optional
        the initial coefficients of the energy terms, if None,
        random values will be used
    coef_thr : float, optional
        the threshold of the coefficients. The convergence
        would be evaluated by calculating the norm-2
        distance of the coefficients in two consecutive
        iterations.
    loss_thr : float, optional
        the threshold of the loss function. Once the loss value
        decreases to smaller than `loss_thr`, this criterion is
        satisfied. NOTE: At least one of the two thresholds 
        (`coef_thr` or `loss_thr`) should be provided, 
        otherwise, the function will raise a ValueError.
    maxiter : int, optional
        the maximum number of iterations, default is 10.
        If the convergence is reached within given steps, the
        fitting will stop eariler.
    remove_jobdir_after_run : bool, optional
        whether to remove the job directory after the job
        is finished. Default is True.
    flog : str, optional
        the file name of the log file, if None, the log will be
        set to `xcparameterization-YYYYMMDD-HHMMSS.log` in the
        current directory. If set to -1, the log file will not
        be created. Default is None.
    
    Returns
    -------
    tuple
        the optimized coefficients of the energy terms, and
        the loss value of the last iteration
    '''
    loggingassert(flog is None or isinstance(flog, str) or flog == -1,
        'illegal value assigned for `flog`: must be None, a string or -1')
    if flog is None:
        flog = f'xcparameterization-{time.strftime("%Y%m%d-%H%M%S")}.log'
    if flog != -1:
        logging.basicConfig(filename=flog, level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')
        logging.info('Start fitting the XC functional parameters.')
    
    # sanity checks
    loggingassert(isinstance(eref, (list, np.ndarray)),
                  'eref must be a scalar')
    eref = np.array(eref)
    loggingassert(eref.ndim == 1,
                  'eref must be a 1D array-like object')
    
    loggingassert(isinstance(e_init, (list, np.ndarray)),
                  'e_init must be a list or a numpy array')
    e_init = np.array(e_init)
    loggingassert(e_init.ndim == 2,
                  'e_init must be a 2D array-like object')
    njob, ncoef = e_init.shape
    loggingassert(njob > 0 and ncoef > 0,
                  'e_init must contain at least one job and one coefficient')

    loggingassert(isinstance(f_xc_loss, Callable),
                  'f_xc_loss must be a callable function')
    
    loggingassert(isinstance(keyword_coef, str),
                  'keyword_coef must be a string')
    
    loggingassert(isinstance(prototyp_dir, (str, list)),
                  'prototyp_dir must be a string or a list of strings')
    if isinstance(prototyp_dir, str):
        prototyp_dir = [prototyp_dir]
    loggingassert(len(prototyp_dir) > 0,
                  'prototyp_dir must contain at least one directory')
    loggingassert(all(isinstance(d, str) for d in prototyp_dir),
                  'prototyp_dir must be a string or a list of strings')
    loggingassert(len(prototyp_dir) == njob,
                  'the length of prototyp_dir must be equal to the number of jobs')

    loggingassert(isinstance(jobrun_option, dict),
                  'jobrun_option must be a dictionary')    
    loggingassert('command' in jobrun_option,
                  'jobrun_option must contain the "command" key')
    
    loggingassert(isinstance(f_ener_reader, Callable),
                  'f_ener_reader must be a callable function')    
        
    if coef_init is not None:
        loggingassert(isinstance(coef_init, (list, np.ndarray)),
                      'coef_init must be a list or a numpy array')
        coef_init = np.array(coef_init)
        loggingassert(coef_init.ndim == 1,
                      'coef_init must be a 1D array-like object')
        loggingassert(len(coef_init) == ncoef,
                      'the length of coef_init must be equal to the '
                      'number of coefficients')

    loggingassert(not all([x is None for x in [coef_thr, loss_thr]]),
                  'at least one of coef_thr or loss_thr must be provided')    
    if coef_thr is not None:
        loggingassert(isinstance(coef_thr, (int, float)),
                      'coef_thr must be a number')
        loggingassert(coef_thr > 0,
                      'coef_thr must be a positive number')
    if loss_thr is not None:
        loggingassert(isinstance(loss_thr, (int, float)),
                      'loss_thr must be a number')
        loggingassert(loss_thr > 0,
                      'loss_thr must be a positive number')

    loggingassert(isinstance(maxiter, (int, float)),
                  'maxiter must be a positive integer')
    loggingassert(maxiter > 0,
                  'maxiter must be a positive integer')
    
    loggingassert(isinstance(remove_jobdir_after_run, bool),
                  'remove_jobdir_after_run must be a boolean value')

    myfit = _fit_kernel(eref=eref, 
                        e_init=e_init,
                        f_xc_loss=f_xc_loss,
                        keyword_coef=keyword_coef,
                        prototyp_dir=prototyp_dir,
                        jobrun_option=jobrun_option,
                        f_ener_reader=f_ener_reader,
                        df_xc_loss=df_xc_loss,
                        coef_init=coef_init,
                        coef_thr=coef_thr,
                        loss_thr=loss_thr,
                        maxiter=maxiter,
                        remove_jobdir_after_run=remove_jobdir_after_run)
    if flog != -1:
        logging.info('Fitting finished.')
        logging.shutdown()
    return myfit

class TestXCClassicalParameterizationKernel(unittest.TestCase):

    def setUp(self):
        testfiles = os.path.dirname(__file__)
        testfiles = os.path.dirname(testfiles)
        testfiles = os.path.dirname(testfiles)
        self.testfiles = os.path.abspath(os.path.join(testfiles, 'testfiles'))
    
    def test_fit(self):
        from UniMolXC.network.utilities.xcloss import minnesota, dminnesota
        
        eref = [1.0]
        e_init = np.array([[1.0, 2.0, 3.0]])
        f_xc_loss = minnesota
        df_xc_loss = dminnesota
        keyword_coef = 'mlxc_placeholder'
        prototyp_dir = [os.path.join(self.testfiles, 'scf-unfinished')]
        jobrun_option = {'command': 'echo "Unittest of function UniMolXC/'
                                    'network/kernel/classical.py:test_fit:'
                                    ' Running ABACUS job"',
                         'reload_after_run': False}
        f_ener_reader = lambda x: np.array([0.5, 2.7, 3.2])
        
        coef, loss = fit(eref, e_init, f_xc_loss, keyword_coef,
                         prototyp_dir, jobrun_option,
                         f_ener_reader, coef_thr=1e-2,
                         coef_init=np.array([1.0, 1.0, 1.0]),
                         df_xc_loss=df_xc_loss, 
                         remove_jobdir_after_run=True,
                         flog=-1)
        
        self.assertEqual(len(coef), e_init.shape[1])
        self.assertTrue(isinstance(loss, float))
        self.assertAlmostEqual(loss, 0.0, places=2)
        self.assertAlmostEqual(np.dot(coef, [0.5, 2.7, 3.2]), eref[0], places=2)
    
if __name__ == '__main__':
    unittest.main()
