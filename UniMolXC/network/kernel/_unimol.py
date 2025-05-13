
# third-party modules
import numpy as np
try:
    from unimol_tools import MolTrain, MolPredict
except ImportError:
    raise ImportError('unimol_tools is not installed. '
                      'See https://github.com/deepmodeling/Uni-Mol/'
                      'tree/main/unimol_tools for more information.')

class UniMolRegressionNet:
    '''
    the wrapper of regression model from UniMol. 
    Unimol is a deep learning model for predicting molecular properties
    with molecule's SMILES representation, or with the atomic coordiates.
    '''
    def __init__(self,
                 model_name='unimolv1',
                 model_size='84m'):
        '''
        instantiate an UniMol regression model interface.
        
        Parameters
        ----------
        model_name : str
            the name of the UniMol model to be used. Available options
            are: 'unimolv1', 'unimolv2'. Default is 'unimolv1'.
        model_size : str
            the size of the UniMol model to be used. Available options
            are: '84m', '164m', '310m', '570m', '1.1B'. Default is '84m'.
            To use the larger models, make sure you have enough memory
            and GPU resources.
            
        Notes
        -----
        The kernel UniMol model will not be allocated soon after this
        function is called. The model will be allocated only when the
        function `train` is called.
        '''
        self.model = None
        self.model_name = model_name
        self.model_size = model_size
        self.save_path = None
    
    def train(self,
              data,
              metrics='mse',
              epochs=10,
              batch_size=16,
              save_path=None):
        '''
        trigger the training process of the UniMol multilabel 
        regression model.
        '''
        assert save_path is not None
        self.save_path = save_path
        self.model = MolTrain(task='multilabel_regression',
                              data_type='molecule',
                              epochs=epochs,
                              batch_size=batch_size,
                              metrics=metrics,
                              model_name=self.model_name,
                              model_size=self.model_size,
                              save_path=save_path)
        self.model.fit(data=data)

    def eval(self, data):
        '''
        trigger the evaluation process of the UniMol multilabel 
        regression model.
        '''
        if self.model is None or self.save_path is None:
            raise ValueError('The model is not trained yet. '
                             'Please call the `train` function first.')
        return MolPredict(load_model=self.save_path).predict(data=data)
