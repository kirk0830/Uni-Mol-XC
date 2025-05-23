# built-in modules

# third-party modules
from torch import nn
import torch
import torch.optim as optim
import torch.nn.functional as F

# local modules
from UniMolXC.network.utility.xcloss import tminnesota

class ResidualLinear(nn.Module):
    '''
    a simple residual block
    '''
    def __init__(self, ndim_in, ndim_out):
        '''
        instantiate a ResidualLinear object
        
        Parameters
        ----------
        ndim_in : int
            input dimension
        ndim_out : int
            output dimension
        '''
        super(ResidualLinear, self).__init__()
        self.fc1 = nn.Linear(ndim_in, ndim_out)
        self.fc2 = nn.Linear(ndim_out, ndim_out)
    
    def forward(self, x):
        '''
        forward propagation
        
        Parameters
        ----------
        x : torch.Tensor
            input tensor of shape (batch_size, ndim_in)
        
        Returns
        -------
        torch.Tensor
            output tensor of shape (batch_size, ndim_out)
        '''
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        return out + x  # residual connection

class XCParameterizationNet(nn.Module):
    '''
    a machine learning based parameterization of 
    exchange-correlation functionals
    '''
    def __init__(self, 
                 ndim, 
                 nhidden=[240, 240, 240], 
                 nparam=1,
                 xc_loss=None):
        '''
        instantiate a XCParameterizationNet object
        
        Parameters
        ----------
        ndim : int
            input dimension
        nhidden : list of int, optional
            number of neurons in each hidden layer. NOTE: if two
            adjacent layers have the same dimension, a residual 
            block is used, by default [240, 240, 240], which means
            3 hidden layers with 240 neurons each, connected by
            residual blocks
        nparam : int, optional
            number of XC parameters, by default 1
        xc_loss : callable, optional
            XC loss function, by default None
            if None, the XC loss function is set to tminnesota
        '''
    
        super(XCParameterizationNet, self).__init__()
        self.ndim = ndim
        self.nparam = nparam
        self.xc_loss = xc_loss if xc_loss is not None else tminnesota
        self.nhidden = [ndim] + nhidden + [nparam]
        
        self.layers = nn.ModuleList()
        for i in range(len(self.nhidden) - 1):
            if self.nhidden[i] == self.nhidden[i + 1]:
                self.layers.append(ResidualLinear(self.nhidden[i], self.nhidden[i + 1]))
            else:
                self.layers.append(nn.Linear(self.nhidden[i], self.nhidden[i + 1]))
        
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
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        x = self.layers[-1](x)
        return x
    
    def loss(self, coef, e, eref):
        '''
        compute the XC loss
        
        Parameters
        ----------
        coef : torch.Tensor
            XC coefficients of shape (batch_size, nparam)
        e : torch.Tensor
            XC energy of shape (batch_size, 1)
        eref : torch.Tensor
            reference XC energy of shape (batch_size, 1)
        
        Returns
        -------
        torch.Tensor
            XC loss of shape (batch_size, 1)
        '''
        return self.xc_loss(coef, e, eref)
    
    def train(self, 
              data, 
              epochs, 
              batch_size, 
              optimizer='Adam', 
              lr=1e-3,
              save_path=None):
        '''
        train the model itself
        
        Parameters
        ----------
        data : dict
            training data, must contain 'descriptor', 'e' and 'eref' 
            fields, in which the 'descriptor' field is an np.ndarray
            in shape of (nstructure, natoms, ndim), the 'e' is an
            np.ndarray in shape of (nstructure, ncoef), and the 'eref'
            is an np.ndarray in shape of (nstructure, )
        epochs : int
            number of training epochs
        batch_size : int
            batch size
        optimizer : str, optional
            optimizer name, by default 'Adam'
        lr : float, optional
            learning rate, by default 1e-3
        save_path : str, optional
            path to save the model, by default None
        '''
        # convert data to torch tensors
        descriptor = torch.tensor(data['descriptor'], dtype=torch.float32)
        e = torch.tensor(data['e'], dtype=torch.float32)
        eref = torch.tensor(data['eref'], dtype=torch.float32)
        # create dataset and dataloader
        dataset = torch.utils.data.TensorDataset(descriptor, e, eref)
        dataloader = torch.utils.data.DataLoader(dataset, 
                                                 batch_size=batch_size, 
                                                 shuffle=True)
        # create optimizer
        if optimizer == 'Adam':
            optimizer = optim.Adam(self.parameters(), lr=lr)
        else:
            raise ValueError(f'Unsupported optimizer: {optimizer}')
        
        # training loop
        for epoch in range(epochs):
            for i, (x_batch, e_batch, eref_batch) in enumerate(dataloader):
                # forward pass
                coef = self.forward(x_batch)
                # compute loss
                loss = self.loss(coef, e_batch, eref_batch)
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')
        
        # save the model
        if save_path is not None:
            torch.save(self.state_dict(), save_path)
            print(f'Model saved to {save_path}')
        else:
            print('Model not saved')
    