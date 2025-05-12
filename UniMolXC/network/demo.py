from torch import nn

class MachineLearningScaledDensityFunctional(nn.Module):
    '''
    machine learning functional that with the atomic
    representation as input, outputs the scaling coefficients
    of energy terms
    '''

    def __init__(self, 
                 ndim_repr, 
                 ncoefs, 
                 nhidden=[128, 128], 
                 activation='relu'):
        '''
        instantiate the machine learning functional network
        
        Parameters
        ----------
        ndim_repr : int
            number of dimensions of the atomic representation
        ncoefs : int
            number of coefficients to be predicted. This is upon the
            density functional form. For example, the Hubbard U
            is the only one parameter, so ncoefs = 1. For the
            M06-L functional, there would be more than 20 coefficients.
        nhidden : list of int
            the number of neurons in the hidden layers
        activation : str
            the activation function to be used. The default is 'relu'.
            Other options are 'tanh', 'sigmoid', etc.
        '''
        # sanity check
        assert isinstance(ndim_repr, int) and ndim_repr > 0
        assert isinstance(ncoefs, int) and ncoefs > 0
        assert isinstance(nhidden, list) and len(nhidden) > 0
        assert all(isinstance(i, int) and i > 0 for i in nhidden)
        assert isinstance(activation, str) and activation in ['relu', 'tanh', 'sigmoid']
        
        # set parameters
        self.ndim_repr = ndim_repr
        self.ncoefs = ncoefs
        self.nhidden = nhidden
        self.activation = {'relu': nn.ReLU(),
                           'tanh': nn.Tanh(),
                           'sigmoid': nn.Sigmoid()}[activation]
        
        # define the network
        layers = []
        layers.append(nn.Linear(self.ndim_repr, self.nhidden[0]))
        layers.append(self.activation)
        for i in range(1, len(self.nhidden)):
            layers.append(nn.Linear(self.nhidden[i-1], self.nhidden[i]))
            layers.append(self.activation)
        layers.append(nn.Linear(self.nhidden[-1], self.ncoefs))
        self.network = nn.Sequential(*layers)
        self.network.apply(self._init_weights)
        
    