from torch import nn

class XCParameterizationNet(nn.Module):
    '''
    a machine learning based parameterization of 
    exchange-correlation functionals
    '''
    def __init__(self, ndim, nhidden=[128, 128], nparams=1):
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
            output tensor of shape (batch_size, nparams)
        '''
        return self.model(x)
    
    