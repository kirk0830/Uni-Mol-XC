# built-in modules

# third-party modules
from torch import nn

# local modules
from UniMolXC.network.utilities.xcloss import tminnesota

class XCParameterizationNet(nn.Module):
    '''
    a machine learning based parameterization of 
    exchange-correlation functionals
    '''
    def __init__(self, 
                 ndim, 
                 nhidden=[128, 128], 
                 nparams=1,
                 xc_loss=None):
        '''
        instantiate a XCParameterizationNet object
        
        
        Parameters
        ----------
        ndim : int
            dimension of the input features
        nhidden : list of int
            the number of neurons in each hidden layer,
            default is [128, 128]. NOTE: if there are two
            adjacent layers with the same number of neurons,
            a residual connection will be added to avoid
            vanishing gradient
        nparams : int
            number of output parameters, default is 1
        xc_loss : callable
            the loss function to be used for training. This
            function must take only the model output as
            input, and return a scalar loss value.
        '''
        super(XCParameterizationNet, self).__init__()
        self.ndim = ndim
        self.nhidden = nhidden
        self.nparams = nparams

        # Define the neural network layers. However, if there are two 
        # adjacent layers with the same number of neurons, use 
        # residual connection to avoid vanishing gradient
        layers = []
        in_features = ndim
        for i, n in enumerate(nhidden):
            layers.append(nn.Linear(in_features, n))
            layers.append(nn.ReLU())
            if i > 0 and n == nhidden[i-1]:
                layers.append(nn.ReLU())
            in_features = n
        layers.append(nn.Linear(in_features, nparams))
        layers.append(nn.ReLU())
        
        # Create the model
        self.model = nn.Sequential(*layers)
        # Set the loss function
        assert callable(xc_loss), "xc_loss must be a callable function"
        self.xc_loss = xc_loss

    def forward(self, x):
        '''
        forward propagation
        
        Parameters
        ----------
        x : torch.Tensor
            input tensor of shape (batch_size, ndim), should be
            the features of the system, e.g., atomic representation
            from UniMol on truncated cluster, or DeePMD descriptors
        
        Returns
        -------
        torch.Tensor
            output tensor of shape (batch_size, nparams)
        '''
        return self.model(x)
    
    def loss(self, y):
        '''
        calculate the loss function
        
        Parameters
        ----------
        y : torch.Tensor
            output tensor of shape (batch_size, nparams), should be
            the output of the model, i.e., the parameters to be optimized
            
        Returns
        -------
        torch.Tensor
            the loss value
        '''
        return self.xc_loss(y)
    
    