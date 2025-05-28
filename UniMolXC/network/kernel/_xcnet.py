'''
the XCNet utilies the DeePMD descriptor, which is more
feasible to catch information that is not local.
'''

# built-in modules
import os
import re
import unittest
import shutil
import time
import json
from typing import Callable

# third-party modules
import numpy as np
from torch import nn
import torch
import torch.optim as optim
import torch.nn.functional as F
try:
    from deepmd.infer.deep_pot import DeepPot
except ImportError:
    raise ImportError('DeePMD-kit is not installed. '
        'See https://docs.deepmodeling.com/projects/deepmd/'
        'en/stable/getting-started/install.html#install-with-conda')
    
# local modules
from UniMolXC.network.utility.xcloss import tminnesota
from UniMolXC.geometry.repr._deepmd import \
    generate_from_abacus as descgen
from UniMolXC.utility.easyassert import \
    loggingassert as lgassert

def build_dataset_from_abacus(folders,
                              labels,
                              f_descmodel=None):
    '''
    build the dataset for training from the ABACUS
    folders and labels.
    
    Parameters
    ----------
    folders : list of str
        the list of ABACUS folders, each folder should contain
        the output files of the ABACUS calculation.
    labels : list of list of float
        the list of labels for each folder, each label should
        be a list of floats, which will be used as the target
        values for the training.
    f_descmodel : str
        the path to the DeePMD model file, if None,
        the function will use the default DeePMD model in 
        folder `UniMolXC/testfiles/dpa3-2p4-7m-mptraj.pth`.
        
    Returns
    -------
    descriptors : list of torch.Tensor
        the list of descriptors for each folder, each descriptor
        is a torch.Tensor of shape (nat, ndim), where nat is the
        number of atoms in the structure, and ndim is the
        number of dimensions of the descriptor.
    labels : list of torch.Tensor
        the list of labels for each folder, each label is a
        torch.Tensor of shape (ncoef,), where ncoef is the
        number of coefficients in the label.
    '''
    # sanity check
    # folders
    assert all(isinstance(f, str) for f in folders)
    
    # labels
    assert isinstance(labels, list)
    assert len(labels) == len(folders)
    ncoef = len(labels[0])
    assert all(len(l) == ncoef for l in labels)
    
    if f_descmodel is None:
        testfiles = os.path.dirname(__file__)
        testfiles = os.path.dirname(testfiles)
        testfiles = os.path.dirname(testfiles)
        f_descmodel = os.path.join(
            testfiles, 'testfiles', 'dpa3-2p4-7m-mptraj.pth')
    assert os.path.exists(f_descmodel), \
        f'{f_descmodel} does not exist, please provide a valid DeePMD model file'
    
    # instantiate the DeePMD model here to reduce the time
    # cost on iteratively loading the model
    dpmodel = DeepPot(f_descmodel)
    
    # generate the descriptors for each folder
    return [torch.tensor(descgen(f, dpmodel), 
                         dtype=torch.float32,
                         requires_grad=True)
            for f in folders], labels

class TestBuildXCPNetDataset(unittest.TestCase):
    def setUp(self):
        testfiles = os.path.dirname(__file__)
        testfiles = os.path.dirname(testfiles)
        testfiles = os.path.dirname(testfiles)
        self.testfiles = os.path.abspath(os.path.join(testfiles, 'testfiles'))
    
    def test_abacus_simplest(self):
        folders = [os.path.join(self.testfiles, 'scf-finished')]
        ydata = [[1.0, 2.0, 3.0]]  # mock labels
        desc, labels = build_dataset_from_abacus(folders, ydata)
        self.assertEqual(len(desc), 1) # one folder
        self.assertEqual(len(labels), 1) # one label
        # becauase the default model for generating the descriptor is used,
        # we know the length of descriptor for atom is 448
        self.assertEqual(desc[0].shape, (1, 2, 448)) 
        # 1 frame, 2 atoms, 448 dimensions
        self.assertListEqual(labels[0], [1.0, 2.0, 3.0])

class LinearResidualNet(nn.Module):
    '''
    a basic implementation of a linear residual network.
    '''
    def __init__(self, 
                 ndim, 
                 nhidden=[240, 240, 240], 
                 nfeatures=1):
        '''
        instantiate an LinearResidualNet object. 

        Parameters
        ----------
        ndim : int
            the input dimension, should be the number of dimension
            of the descriptor of one atom
        nhidden : list of int, optional
            number of neurons in each hidden layer. NOTE: if two
            adjacent layers have the same dimension, a residual 
            block is used, by default [240, 240, 240], which means
            3 hidden layers with 240 neurons each, connected by
            residual blocks
        nfeatures : int, optional
            number of features to output, by default 1
        '''
    
        super(LinearResidualNet, self).__init__()
        self.ndim = ndim
        self.nfeatures = nfeatures
        self.ninout = [ndim] + nhidden + [nfeatures]
        
        self.layers = nn.ModuleList()
        for n1, n2 in zip(self.ninout[:-1], self.ninout[1:]):
            self.layers.append(nn.Linear(n1, n2))
        
    def forward(self, x):
        '''
        forward propagation
        
        Parameters
        ----------
        x : torch.Tensor
            input tensor of shape (batch_size, ndim)
        
        Returns
        -------
        torch.Tensor
            output tensor of shape (batch_size, nparam)
        '''
        for i, layer in enumerate(self.layers[:-1]):
            y = F.relu(layer(x))
            if self.ninout[i] == self.ninout[i + 1]:
                y = y + x
            x = y
        return self.layers[-1](x)  # final layer without residual
    
class TestLinearResidualNet(unittest.TestCase):
    def test_init(self):
        model = LinearResidualNet(10, [20, 30], nfeatures=2)
        self.assertEqual(model.ndim, 10)
        self.assertEqual(model.nfeatures, 2)
        self.assertEqual(len(model.layers), 3)
        # from 10 to 20, 20 to 30, and 30 to 2
        self.assertEqual(model.layers[0].in_features, 10)
        self.assertEqual(model.layers[0].out_features, 20)
        self.assertEqual(model.layers[1].in_features, 20)
        self.assertEqual(model.layers[1].out_features, 30)
        self.assertEqual(model.layers[2].in_features, 30)
        self.assertEqual(model.layers[2].out_features, 2)

    def test_forward(self):
        model = LinearResidualNet(10, [20, 30], nfeatures=2)
        print('\n'
              'Test the forward() method of LinearResidualNet... '
              'Run for 10 times to check the consistency'
              , flush=True)
        for i in range(10):
            x = torch.randn(5, 10)  # batch size of 5
            output = model(x)
            self.assertEqual(output.shape, (5, 2))
            # only three layers (10 -> 20, 20 -> 30, 30 -> 2),
            # we calculate here by hands
            y1 = F.relu(model.layers[0](x))
            y2 = F.relu(model.layers[1](y1))
            y3 = model.layers[2](y2)
            self.assertTrue(torch.allclose(y3, output, atol=1e-6))
            print(f'{i+1}, ', end='', flush=True)
        print('done.', flush=True)
        print('Test the forward() method of LinearResidualNet finished.',
              flush=True)

    def test_residual_net(self):
        model = LinearResidualNet(10, [10, 10], nfeatures=1)
        print('\n'
              'Test the residual connection in LinearResidualNet... '
              'Run for 10 times to check the consistency'
              , flush=True)
        for i in range(10):
            x = torch.randn(5, 10)  # batch size of 5
            output = model(x)        
            self.assertEqual(output.shape, (5, 1))
            # only three layers, we calculate here by hands
            y1 = F.relu(model.layers[0](x))
            y1 = y1 + x
            y2 = F.relu(model.layers[1](y1))
            y2 = y2 + y1
            y3 = model.layers[2](y2)
            self.assertTrue(torch.allclose(y3, model(x), atol=1e-6))
            print(f'{i+1}, ', end='', flush=True)
        print('done.', flush=True)
        print('Test the residual connection in LinearResidualNet finished.',
              flush=True)

    def test_backward(self):
        model = LinearResidualNet(10, [20, 30], nfeatures=2)
        x = torch.randn(5, 10, requires_grad=True)
        output = model(x)
        loss = torch.sum(output)
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, (5, 10))

    def test_train(self):
        ''' test the training process of the LinearResidualNet
        '''
        print('\n'
              'Testing training process of LinearResidualNet...', 
              flush=True)
        model = LinearResidualNet(10, [20, 30], nfeatures=2)
        x = torch.randn(5, 10, requires_grad=True)
        y = torch.randn(5, 2, requires_grad=True)  # target output
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        model.train()  # set the model to training mode
        print(f'{"IEPOCH":<10} {"LOSS":<10}', flush=True)
        print('-' * 21, flush=True)
        for iepoch in range(10):  # 10 epochs
            optimizer.zero_grad()  # zero the gradients
            output = model(x)  # forward pass
            loss = criterion(output, y)  # calculate loss
            print(f'{iepoch:>10} {loss.item():>10.4f}', flush=True)
            loss.backward()  # backward pass
            optimizer.step()  # update parameters
        
        print('Training process of LinearResidualNet finished.',
              flush=True)

class XCPNetImpl(nn.Module):
    '''
    a machine-learning-based prediction of the parameterization 
    on XC functionals that with analytical form. 
    
    This net contains several sub-network and one for each atom 
    type.
    
    Each time this net accepts a trajectory of one structure,
    output corresponding number of sets of XC parameters. For
    different structures (different chemical composition, 
    different number of atoms, etc.), this model now is designed
    to be re-trained.
    
    This network construction is referred from the Deep potential
    paper: 
    
    [1] Han J, Zhang L, Car R. 
        Deep potential: A general representation of a many-body 
        potential energy surface[J]. 
        arXiv preprint arXiv:1707.01478, 2017.

    [2] Zhang L, Han J, Wang H, et al. 
        Deep potential molecular dynamics: a scalable model with 
        the accuracy of quantum mechanics[J]. 
        Physical review letters, 2018, 120(14): 143001.
    '''

    def __init__(self, 
                 elem: list,
                 ndim: int=448,
                 nhidden: list=[240, 240, 240],
                 nparam: int=1,
                 f_loss: Callable|None=None):
        '''
        instantiate the XCPNetImpl object.
        
        Parameters
        ----------
        elem : list of str
            the overall list of element symbols, e.g. ['H', 'O', 'C']
            the number of elements will determine the number of
            LinearResidualNet instances in the network. 
            NOTE: this list should always contain all possible
            elements throughout the training and testing
            processes, otherwise the type map will not be correct!
            For example, if the system1 has elements ['H', 'O']
            and the system2 has elements ['H', 'C'], the elem
            should be ['H', 'O', 'C'], and no extra elements should
            appear later.
        ndim : int
            the input dimension, should be the number of dimension
            of the descriptor of one atom, default is 448, which
            corresponds to the case of DPA3 MPtraj branch model
        nhidden : list of int, optional
            parameter for the sub-network for each element. The 
            number of neurons in each hidden layer. NOTE: if two
            adjacent layers have the same dimension, a residual 
            block is used, by default [240, 240, 240], which means
            3 hidden layers with 240 neurons each, connected by
            residual blocks
        nparam : int, optional
            number of XC parameters, by default 1
        xc_loss : callable, optional
            XC loss function, by default None
            if None, the XC loss function is set to `tminnesota`
        restart_from : str, optional
            the path to the file to restart the model from, by default None
        '''

        super(XCPNetImpl, self).__init__()
        self.elem = elem
        self.ndim = ndim
        self.nparam = nparam
        self.f_loss = f_loss if f_loss is not None else tminnesota
        # instantiate all sub-networks for each atom type
        self.xcatomtyp_nets = nn.ModuleList([
            LinearResidualNet(ndim, nhidden, nparam)
            for _ in elem
        ])

    def encode_type_map(self, elem):
        '''
        convert the list of element symbols to a list of indices.
        NOTE: this encoding may vary from network instance, so 
        this is a method of the XCPNetImpl class, instead of a
        static method.
        
        Parameters
        ----------
        elem : list of str
            the list of element symbols, e.g. ['H', 'O', 'C'].
            For example, if the system is the water molecule H2O,
            and the allocated LinearResidualNet instances are
            ['H', 'O'], the type map will be [0, 0, 1] for the
            water molecule, where 0 is the index for 'H' and 1
            is the index for 'O'. NOTE: even if it is common case
            in which there are not only one atoms with the same
            symbol, they should all appear in the type map!
            
        Returns
        -------
        list of int
            a list of indices 
        '''
        return [self.elem.index(e) for e in elem]

    def forward(self, 
                x, 
                type_map):
        '''
        the forward propagation of the XCPNetImpl for ONE
        structure.
        
        Parameters
        ----------
        x : torch.Tensor
            input tensor of shape (batch_size, nat, ndim). The 
            so-called `batch_size` here always corresponds to the 
            number of trajectory frames.
        type_map : list of int
            a list of atom types for each atom in the input tensor,
            should be of length nat, where nat is the number of atoms
            in the input tensor. It is assumed all frames share the
            identical type_map.
            
        Returns
        -------
        torch.Tensor
            output tensor of shape (batch_size, nparam)
            the XC parameters for each atom type in the input tensor.
            For different atoms with the same type, the
            XC parameters are averaged.
        '''
        assert x.ndim == 3, \
            'input tensor x should be of shape (batch_size, nat, ndim)'
        _, nat, ndim = x.shape
        assert len(type_map) == nat, \
            'type_map should be of length nat, where nat is the number of atoms in x'
        assert all(isinstance(i, int) for i in type_map), \
            'type_map should be a list of integers representing atom types'
        assert all(0 <= i < len(self.elem) for i in type_map), \
            'type_map should contain valid indices for the element list'
        assert ndim == self.ndim, \
            f'input tensor x should have ndim={self.ndim}, but got {ndim}'

        y = torch.stack([
            self.xcatomtyp_nets[ityp].forward(x[:, i, :]) # x in shape (batch_size, nat, ndim)
            for i, ityp in enumerate(type_map)
        ], dim=1) # in shape (batch_size, nat, nparam)
        return torch.mean(y, dim=1) # is taking average over atoms a good idea?

    def loss(self,
             y,
             e,
             eref):
        '''
        calculate the loss of the XCPNetImpl on one
        structure.
        
        Parameters
        ----------
        y : torch.Tensor
            the output of forward() method, a tensor of shape
            (batch_size, nparam) containing the XC parameters
            calculated on all frames of the input tensor.
        e : torch.Tensor
            a tensor of shape (batch_size, nparam) containing the 
            energy terms calculated on all frames of the input tensor.
        eref : torch.Tensor
            a tensor of shape (batch_size,) containing the reference
            energy for each frame of the input tensor.
            
        Returns
        -------
        float
            the loss value, which is the XC loss function value
            calculated by the XC loss function defined in f_loss
        '''
        loss_batch = torch.vstack([self.f_loss(coef, e, eref) for coef in y])
        return torch.mean(loss_batch)

    def __savetorch__(self, fmodel):
        torch.save(self.state_dict(), fmodel)

    def save(self, path, timestamp=None):
        '''
        save the model to a path
        
        Parameters
        ----------
        path : str
            the path to save the model
        timestamp : str, optional
            the timestamp to append to the filename. If not provided,
            will use the timestring of the current time.
        '''
        mystate = {'elem': self.elem, 'save_path': path}
        os.makedirs(path) # does not allow to overwrite the directory
        
        timestamp = timestamp or time.strftime('%Y%m%d%H%M%S')
        ftorch = os.path.join(path, f'xcpnet.{timestamp}.pth')
        mystate['ftorch'] = ftorch
        self.__savetorch__(ftorch)

        with open(os.path.join(path, 'xcpnet.json'), 'w') as f:
            json.dump(mystate, f, indent=4)
    
    def __loadtorch__(self, fmodel):
        if not os.path.exists(fmodel):
            raise FileNotFoundError(f'{fmodel} does not exist')
        self.load_state_dict(torch.load(fmodel))

    @staticmethod
    def __calcnetdim__(torch_model):
        '''calculate the dimension of the network'''
        indices = [re.match(r'^xcatomtyp_nets\.(\d+)\.layers\.(\d+)\.weight$', 
                            k) for k in torch_model.keys()]
        indices = [tuple(map(int, m.groups())) for m in indices if m is not None]
        indices = sorted(indices, key=lambda x: (x[0], x[1]))
        # indices is a list of tuples (atom_type_index, layer_index)
        nelem = max(i for i, _ in indices) + 1
        nlayer = len([j for i, j in indices if i == 0])
        
        # sanity checks
        assert nlayer > 0
        assert len(indices) == nelem * nlayer 
        # assert the number of layers is the same for each atom type
        
        sizes = [tuple(torch_model[f'xcatomtyp_nets.0.layers.{i}.weight'].shape)
                 for i in range(nlayer)]
        ndim = sizes[0][1]  # the input dimension is the first layer's input size
        nparam = sizes[-1][0]
        nhidden = [s[1] for s in sizes[1:]]

        return dict(zip(['nelem', 'ndim', 'nparam', 'nhidden'],
                        [nelem, ndim, nparam, nhidden]))

    @staticmethod
    def load(path):
        '''instantiate the XCPNetImpl object from a path'''
        with open(os.path.join(path, 'xcpnet.json'), 'r') as f:
            mystate = json.load(f)
        
        state = XCPNetImpl.__calcnetdim__(torch.load(mystate['ftorch']))
        nelem, ndim, nparam, nhidden = state.values()
        
        assert nelem == len(mystate['elem'])
        return XCPNetImpl(elem=mystate['elem'],
                          ndim=ndim,
                          nhidden=nhidden,
                          nparam=nparam)

    @staticmethod
    def build_xcpnet(elem: list,
                     model_size: dict = {'ndim': 448, 
                                         'nparam': 1, 
                                         'nhidden': [240, 240, 240]},
                     f_loss: Callable = None,
                     model_restart=None):
        '''build the XCPNetImpl object
        
        Parameters
        ----------
        elem : list of str
            the list of element symbols, e.g. ['H', 'O', 'C']
        model_size : dict, optional
            the size of the model, by default {'ndim': 448, 
                                               'nparam': 1, 
                                               'nhidden': [240, 240, 240]}
        f_loss : Callable, optional
            the loss function to use, by default None, which means
            the XC loss function is set to `tminnesota`
        model_restart : str, optional
            the path to the file to restart the model from, by default None
        
        Returns
        -------
        XCPNetImpl
            the XCPNetImpl object
        '''
        if model_restart is not None:
            return XCPNetImpl.load(model_restart)
        else:
            return XCPNetImpl(elem=elem,
                              ndim=model_size['ndim'],
                              nhidden=model_size['nhidden'],
                              nparam=model_size['nparam'],
                              f_loss=f_loss)

class TestXCPNetImpl(unittest.TestCase):
    def setUp(self):
        testfiles = os.path.dirname(__file__)
        testfiles = os.path.dirname(testfiles)
        testfiles = os.path.dirname(testfiles)
        self.testfiles = os.path.abspath(os.path.join(testfiles, 'testfiles'))
        self.scheduled_delete = []

    def tearDown(self):
        for f in self.scheduled_delete:
            if os.path.isfile(f):
                os.remove(f)
            else:
                shutil.rmtree(f)
        self.scheduled_delete = []

    def test_init(self):
        model = XCPNetImpl(['H', 'O', 'C'], 
                                      ndim=10, 
                                      nhidden=[20, 20, 20],
                                      nparam=2)
        self.assertEqual(model.elem, ['H', 'O', 'C'])
        self.assertEqual(model.ndim, 10)
        self.assertEqual(model.nparam, 2)
        self.assertEqual(len(model.xcatomtyp_nets), 3)
        self.assertTrue(all(isinstance(net, LinearResidualNet) 
                            for net in model.xcatomtyp_nets))
        self.assertTrue(all(net.ninout == [10, 20, 20, 20, 2]
                            for net in model.xcatomtyp_nets))
        self.assertTrue(callable(model.f_loss))

    def test_encode_type_map(self):
        model = XCPNetImpl(['H', 'O', 'C'])
        type_map = model.encode_type_map(['H', 'O', 'C', 'H'])
        self.assertEqual(type_map, [0, 1, 2, 0])
        
        # test with an element not in the list
        with self.assertRaises(ValueError):
            model.encode_type_map(['H', 'O', 'N'])
        # 'N' is not in the list ['H', 'O', 'C']
        # so it should raise a ValueError

    def test_forward(self):
        model = XCPNetImpl(['H', 'O'], ndim=10, nparam=2)
        
        # mock descriptor for 5 batch size, 3 atoms, 10 dimensions
        x = torch.randn(5, 3, 10, requires_grad=True)
        
        # mock type map for 3 atoms (2 H and 1 O)
        type_map = [0, 0, 1]
        
        # forward one structure
        y = model.forward(x, type_map)
        
        # checks
        self.assertEqual(y.shape, (5, 2)) # 5 batches, 2 parameters
        self.assertTrue(isinstance(y, torch.Tensor))
        # check if the output is a tensor with requires_grad
        self.assertTrue(y.requires_grad)
        
    def test_loss(self):
        model = XCPNetImpl(['H', 'O'], ndim=10, nparam=2)
        
        # mock the output of forward() method
        y = torch.randn(5, 2, requires_grad=True)  # 5 structures, 2 parameters
        
        # mock energy terms and reference energy
        e = torch.randn(5, 2, requires_grad=True)  # 5 structures, 2 parameters
        eref = torch.randn(5, requires_grad=True)  # 5 reference energies
        
        # calculate loss
        loss = model.loss(y, e, eref)
        
        # check if the loss is a scalar
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.shape, ())
 
    def test_backward(self):
        model = XCPNetImpl(['H', 'O'], ndim=10, nparam=2)
        
        # mock descriptor for 5 batch size, 3 atoms, 10 dimensions
        x = torch.randn(5, 3, 10, requires_grad=True)
        
        # mock type map for 3 atoms (2 H and 1 O)
        type_map = [0, 0, 1]
        
        # mock energy terms and reference energy
        e = torch.randn(5, 2, requires_grad=True)  # 5 structures, 2 parameters
        eref = torch.randn(5, requires_grad=True)  # 5 reference energies
        
        # calculate loss
        y = model.forward(x, type_map)
        loss = model.loss(y, e, eref)
        
        # backward propagation
        loss.backward()
        
        # check if the gradients are computed
        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, (5, 3, 10))

    def test_train(self):
        ''' test the training process of the XCPNetImpl
        '''
        print('\n'
              'Testing training process of XCPNetImpl...', 
              flush=True)
        model = XCPNetImpl(['H', 'O'], ndim=10, nparam=2)
        
        # mock descriptor for 5 batch size, 3 atoms, 10 dimensions
        x = torch.randn(5, 3, 10, requires_grad=True)
        
        # mock type map for 3 atoms (2 H and 1 O)
        type_map = [0, 0, 1]
        
        # mock energy terms and reference energy
        e = torch.randn(5, 2, requires_grad=True)
        eref = torch.randn(5, requires_grad=True)
        
        # configure the optimizer
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        model.train() # set the model to training mode
        print(f'{"IEPOCH":<10} {"LOSS":<10}', flush=True)
        print('-' * 21, flush=True)
        for iepoch in range(10):
            optimizer.zero_grad()
            y = model.forward(x, type_map)
            loss = model.loss(y, e, eref)
            print(f'{iepoch:>10} {loss.item():>10.4f}', flush=True)
            loss.backward()
            optimizer.step()
            
        print('Training process of XCPNetImpl finished.',
              flush=True)

    def test_save(self):
        model = XCPNetImpl(['H', 'O'], ndim=10, nparam=2)
        path_backup = os.path.join(self.testfiles, 'my-temporary-xcpnet')
        self.scheduled_delete.append(path_backup)
        model.save(path_backup)
        self.assertTrue(os.path.exists(path_backup))
        self.assertEqual(len(os.listdir(path_backup)), 2)
        self.assertTrue(os.path.exists(os.path.join(path_backup, 'xcpnet.json')))

    def test_load(self):
        model = XCPNetImpl(['H', 'O'], ndim=10, nparam=2)
        path_backup = os.path.join(self.testfiles, 'my-temporary-xcpnet')
        self.scheduled_delete.append(path_backup)
        model.save(path_backup)
        
        # load the model from the saved path
        loaded_model = XCPNetImpl.load(path_backup)
        
        self.assertTrue(isinstance(loaded_model, XCPNetImpl))
        
        # check if the loaded model has the same attributes
        self.assertEqual(loaded_model.elem, ['H', 'O'])
        self.assertEqual(loaded_model.ndim, 10)
        self.assertEqual(loaded_model.nparam, 2)
        self.assertEqual(len(loaded_model.xcatomtyp_nets), 2)

class XCParameterizationNet:
    '''
    the trainer of XCPNetImpl, also the train-wrapped "inner train" kernel
    of the whole XC parameterization network. This is a counterpart of the
    Uni-Mol tools API (see UniMolXC/network/kernel/_unimol.py).
    '''
    def __init__(self,
                 elem: list=None,
                 model_size: dict = {'ndim': 448, 
                                     'nparam': 1, 
                                     'nhidden': [240, 240, 240]},
                 model_restart=None):
        # the kernel
        self.model = None

        # model size
        self.elem = elem
        self.ndim = model_size.get('ndim', 448)
        self.nhidden = model_size.get('nhidden', [240, 240, 240])
        self.nparam = model_size.get('nparam', 1)

    @staticmethod
    def split_train_validation(data, ratio=0.8):
        '''
        split the training data into training set and validation set with
        given ratio. 
        
        Parameters
        ----------
        data : dict
            the training data, see function train() for the details.
        
        ratio : float, optional
            the ratio of the training set size to the whole dataset size, 
            by default 0.8, which means 80% of the data will be used for 
            training and 20% for validation.
        
        Returns
        -------
        tuple
            a tuple of two dictionaries, the first one is the training set
            and the second one is the validation set. Each dictionary
            contains the same keys as the input data.
        '''
        def sort_then_merge(index):
            '''sort the indices by the first order, then merge all
            the indices of the same structure together, like from
            [[0, 0], [0, 1], [0, 2], [1, 1]] to [[0, 1, 2], [1]]'''
            index = torch.tensor(index, dtype=torch.long)
            sorted_indices = torch.argsort(index[:, 0])
            index = index[sorted_indices]
            temp = index[:, 0]
            ifirst, _, counts = torch.unique(temp, return_inverse=True, return_counts=True)
            iaccum = torch.cat([torch.tensor([0]), torch.cumsum(counts, dim=0)])
            return ifirst.tolist(), [index[iaccum[i]:iaccum[i + 1], 1].tolist()
                                     for i in range(len(ifirst))]
        
        # randomly select indices for training set for each structure
        n = sum([len(d) for d in data['desc']])
        indexing = [[i, j] for i, d in enumerate(data['desc'])
                    for j in range(len(d))]  # (structure index, batch index)
        # reshuffle the indices
        indexing = torch.tensor(indexing, dtype=torch.int64)[torch.randperm(n)]
        # split the indices into training and validation sets
        ntrain = int(n * ratio)
        
        imol, ibatch = sort_then_merge(indexing[:ntrain, :].tolist())
        trainset = {
            'atoms': [data['atoms'][i]   for i    in imol],
            'desc':  [data['desc'][i][j] for i, j in zip(imol, ibatch)],
            'e':     [data['e'][i][j]    for i, j in zip(imol, ibatch)],
            'eref':  [data['eref'][i][j] for i, j in zip(imol, ibatch)]
        }
        imol, ibatch = sort_then_merge(indexing[ntrain:, :].tolist())
        validset = {
            'atoms': [data['atoms'][i]   for i    in imol],
            'desc':  [data['desc'][i][j] for i, j in zip(imol, ibatch)],
            'e':     [data['e'][i][j]    for i, j in zip(imol, ibatch)],
            'eref':  [data['eref'][i][j] for i, j in zip(imol, ibatch)]
        }
        return trainset, validset

    def train(self,
              data,
              epochs=10,
              batch_size=16,
              f_loss=None,
              save_path=None,
              **kwargs):
        '''
        trigger the training process of the XC parameterization net. 
        We strongly recommend users to assign appropriate values to
        the parameters `epochs`, `batch_size` and `f_loss` to get a
        good model and do not let them to be the default values. 
        Other parameters are optional and will also be passed to the
        training process.
        
        Parameters
        ----------
        data : dict
            the training data, should contain the following keys:
            
            - 'atoms': list of list of str
            
               the element symbols for each structure, should be a list
               of lists, where each list contains the element symbols
               of the atoms in the structure, e.g. [['H', 'O'], ['C', 'H']]
               
            - 'desc': list of torch.Tensor
            
               the counterpart of the `coordinates` in the Uni-mol API.
               should be a list of tensors with shape (batch_size, nat, ndim), 
               where `batch_size` can be the length of molecular dynamics 
               simulation trajectory, `nat` is the number of atoms in the 
               structure and `ndim` is the number of dimensions of the 
               descriptor. For different element, the dimensions can be 
               different
               
            - 'e': list of torch.Tensor
            
               should be a list of tensors with shape (batch_size, ncoef).
               For each "row" in the tensor, it contains the energy terms
               calculated on the structure.
               
            - 'eref': list of torch.Tensor
            
               should be a list of tensors with shape (batch_size,).
               For each "row" in the tensor, it contains the reference
               energy for the structure.
               
        epochs : int, optional
            the number of epochs for training, by default 10
            
        batch_size : int, optional
            the batch size for training, by default 16
            
        f_loss : callable, optional
            the loss function for training, by default None,
            which means the tminnesota loss function will be used
            
        save_path : str, optional
            the path to save the trained model, by default None,
            which means the model will not be saved
            
        **kwargs : dict
            other parameters for the training process, such as
            learning rate, optimizer, etc. These parameters will
            be passed to the training process.
        '''
        # allocate the trainable model
        self.model = XCPNetImpl(elem=self.elem, 
                                ndim=self.ndim, 
                                nhidden=self.nhidden, 
                                nparam=self.nparam,
                                f_loss=f_loss)       

        # split data into train and validation sets
        trainset, validset = XCParameterizationNet.split_train_validation(
            data=data, ratio=kwargs.get('train_data_split_ratio', 0.8))
        
        optimizer = optim.Adam(self.model.parameters(), lr=kwargs.get('lr', 0.01))
        print('\n'
              f'{"EPOCH":<10} {"Train loss":<15} {"Validation loss":<15}', flush=True)
        print('-' * (10+1+15+1+15), flush=True)
        for iepoch in range(epochs):
            # train
            self.model.train() # switch to training mode
            train_loss = 0.0
            for atoms, desc, e, eref in zip(*trainset.values()): # loop over systems
                type_map = self.model.encode_type_map(atoms)
                # if there are batches more than the batch size, we need to
                # split the data into batches, otherwise, we can just use 
                # the whole data
                nbatch = len(desc)
                if nbatch > batch_size:
                    for i in range(0, nbatch, batch_size):
                        lo, hi = i, min(i + batch_size, nbatch)
                        x = desc[lo:hi]
                        y = self.model.forward(x, type_map)
                        loss = self.model.loss(y, e[lo:hi], eref[lo:hi])
                        optimizer.zero_grad()
                        loss.backward(retain_graph=True)
                        optimizer.step()
                        train_loss += loss.item() * (hi - lo)
                else:
                    x = desc
                    y = self.model.forward(x, type_map)
                    loss = self.model.loss(y, e, eref)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item() * nbatch
            train_loss /= len(trainset['desc'])
            
            # validate
            self.model.eval()
            valid_loss = 0.0
            with torch.no_grad():
                for atoms, desc, e, eref in zip(*validset.values()):
                    type_map = self.model.encode_type_map(atoms)
                    x = desc
                    y = self.model.forward(x, type_map)
                    loss = self.model.loss(y, e, eref)
                    valid_loss += loss.item() * len(desc)
            valid_loss /= len(validset['desc'])
            print(f'{iepoch:>10} {train_loss:>15.4f} {valid_loss:>15.4f}', flush=True)

class TestXCParameterizationNet(unittest.TestCase):
    def setUp(self):
        self.testfiles = os.path.abspath(os.path.join(
            os.path.dirname(__file__), 'testfiles'))
        self.scheduled_delete = []

    def tearDown(self):
        for f in self.scheduled_delete:
            if os.path.isfile(f):
                os.remove(f)
            else:
                shutil.rmtree(f)
        self.scheduled_delete = []

    def test_split_train_validation(self):
        # mock data
        data = {
            'atoms': [['H', 'H', 'O'], ['C', 'H', 'H', 'H', 'H']],
            'desc': [torch.randn(5, 3, 10), torch.randn(5, 5, 10)],
            'e': [torch.randn(5, 1), torch.randn(5, 1)],
            'eref': [torch.randn(5), torch.randn(5)]
        }
        
        trainset, validset = XCParameterizationNet.split_train_validation(data, ratio=0.8)
        # check the keys
        self.assertEqual(set(trainset.keys()), set(validset.keys()))
        self.assertEqual(set(trainset.keys()), {'atoms', 'desc', 'e', 'eref'})
        
        # there are two systems, so the length of atoms should be no larger than 2
        self.assertLessEqual(len(trainset['atoms']), 2)
        self.assertLessEqual(len(trainset['desc']),  2)
        self.assertLessEqual(len(trainset['e']),     2)
        self.assertLessEqual(len(trainset['eref']),  2)
        # check the iterability of the data
        n = len(trainset['atoms'])
        self.assertTrue(all(n == len(v) for v in trainset.values()))
        self.assertEqual(n, len(list(zip(*trainset.values()))))
        
        self.assertLessEqual(len(validset['atoms']), 2)
        self.assertLessEqual(len(validset['desc']),  2)
        self.assertLessEqual(len(validset['e']),     2)
        self.assertLessEqual(len(validset['eref']),  2)
        # check the iterability of the data
        n = len(validset['atoms'])
        self.assertTrue(all(n == len(v) for v in validset.values()))
        self.assertEqual(n, len(list(zip(*validset.values()))))
        
        # there are in total 10 frames, 8 for training and 2 for validation
        self.assertEqual(sum(len(d)    for d    in trainset['desc']), 8)
        self.assertEqual(sum(len(e)    for e    in trainset['e']),    8)
        self.assertEqual(sum(len(eref) for eref in trainset['eref']), 8)
        self.assertEqual(sum(len(d)    for d    in validset['desc']), 2)
        self.assertEqual(sum(len(e)    for e    in validset['e']),    2)
        self.assertEqual(sum(len(eref) for eref in validset['eref']), 2)
    
    def test_train(self):
        # mock data: water and methane
        data = {
            'atoms': [['H', 'H', 'O'], ['C', 'H', 'H', 'H', 'H']],
            'desc': [torch.randn(5, 3, 10, requires_grad=True), 
                     torch.randn(5, 5, 10, requires_grad=True)],
            'e': [torch.randn(5, 1, requires_grad=True), 
                  torch.randn(5, 1, requires_grad=True)],
            'eref': [torch.randn(5, requires_grad=True), 
                     torch.randn(5, requires_grad=True)]
        }
        
        # train
        trainer = XCParameterizationNet(elem=['H', 'O', 'C'], 
                                        model_size={'ndim': 10, 
                                                    'nparam': 1, 
                                                    'nhidden': [20, 20, 20]})
        trainer.train(data, epochs=20, batch_size=2, f_loss=tminnesota)
        # check if the model is instantiated
        self.assertIsNotNone(trainer.model)
        self.assertTrue(isinstance(trainer.model, XCPNetImpl))
        # check if the model has the correct attributes
        self.assertEqual(trainer.model.elem, ['H', 'O', 'C'])
        self.assertEqual(trainer.model.ndim, 10)
        self.assertEqual(trainer.model.nparam, 1)
        self.assertEqual(len(trainer.model.xcatomtyp_nets), 3)
        self.assertTrue(all(isinstance(net, LinearResidualNet) 
                            for net in trainer.model.xcatomtyp_nets))
        
if __name__ == '__main__':
    # run unit tests
    unittest.main(exit=True)
    