
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
                 model_size='84m',
                 model_restart=None):
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
            and CPU/GPU resources.
        model_restart : str, optional
            the path to the model checkpoint to be loaded. If None,
            a new model will be trained from scratch. Default is None.

        Notes
        -----
        The kernel UniMol model will not be allocated soon after this
        function is called. The model will be allocated only when the
        function `train` is called.
        '''
        # the kernel
        self.model = None
        
        # either from scratch or from a restart
        assert (None not in [model_name, model_size]) or \
               (model_restart is not None)

        # from scratch
        self.model_name = model_name
        self.model_size = model_size
        self.model_restart_from = None
        
        # from file
        if model_restart is not None:
            self.model = MolPredict(load_model=model_restart)
            self.model_restart_from = model_restart
    
    def train(self,
              data,
              metrics='mse',
              epochs=10,
              batch_size=16,
              save_path=None,
              **kwargs):
        '''
        trigger the training process of the UniMol multilabel 
        regression model. We strongly recommend users to assign
        appropriate values to the parameters `epochs`, `batch_size`
        and `metrics` to get a good model and do not let them
        to be the default values. Other parameters are optional
        and will also be passed to the training process.
        
        Parameters
        ----------
        data : dict or list of str
            the training data. For the use in this package, please
            refer to the function in file network/train.py:
            build_dataset_from_abacus.
        metrics : str, optional
            the way to scalarize the loss. Default is 'mse', which
            supports both scalar and vector labels. To see other
            options, please see annotation of the function
            unimol_tools.MolTrain.__init__.
        epochs : int, optional
            the number of epochs to train. Default is 10.
        batch_size : int, optional
            the batch size of training. Default is 16.
        save_path : str, optional
            the path to save the model. Default is None, which means
            the model will not be saved. If you want to save the model,
            please provide a valid path.
        **kwargs : dict
            other parameters to be passed to the training process.
            For the full list of parameters, please refer to the
            function unimol_tools.MolTrain.__init__.
        '''
        # distinct with the original UniMol, we force the user
        # to provide the save_path for the model to be saved.
        assert save_path is not None
        
        # we check the parameters list of the constructor of
        # unimol_tools.MolTrain, and we only keep the parameters
        # closely related to the training process.
        unimol_mol_train_instantiate_selected_kwargs = [
            'learning_rate', 'early_stopping', 'split', 
            'split_group_col', 'kfold', 'remove_hs', 'smiles_col', 
            'target_cols', 'target_col_prefix', 'target_anomaly_check', 
            'smiles_check', 'target_normalize', 'max_norm', 'use_cuda', 
            'use_amp', 'use_ddp', 'use_gpu', 'freeze_layers', 
            'freeze_layers_reversed'
        ]
        # extract the parameters from the kwargs
        train_param = {k: v for k, v in kwargs.items() 
                       if k in unimol_mol_train_instantiate_selected_kwargs}
        pred = MolTrain(task='multilabel_regression',
                        data_type='molecule',
                        epochs=epochs,
                        batch_size=batch_size,
                        metrics=metrics,
                        model_name=self.model_name,
                        model_size=self.model_size,
                        save_path=save_path,
                        load_model_dir=self.model_restart_from,
                        **train_param).fit(data=data)

        # cache the model trained
        self.model = MolPredict(load_model=save_path)        
        return pred
        
    def eval(self, data):
        '''
        trigger the evaluation process of the UniMol multilabel 
        regression model.
        '''
        if self.model is None:
            raise ValueError('The model is not trained yet. '
                             'Please call the `train` function first.')
        return self.model.predict(data=data)
