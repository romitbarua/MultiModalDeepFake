import numpy as np

NO_PROB_MODELS = ['svm']
LABEL_TYPES = ['binary', 'multiclass']

def checkParams(model_managers: tuple):
    
    assert len(model_managers) == 2, 'Please provide exactly two model managers'
    
    model_manager_1 = model_managers[0]
    model_manager_2 = model_managers[1]
    assert model_manager_1.model_name not in NO_PROB_MODELS, f'{model_manager_1.model_name} does not have probabilities'
    

def predCorrelation(model_managers: tuple, label_type='multiclass'):
    
    checkParams(model_managers)
    matrix_1 = model_managers[0].y_prob
    matrix_2 = model_managers[1].y_prob

    corr_coef = np.corrcoef(matrix_1.T, matrix_2.T)
    
    idx_ar = np.arange(matrix_1.shape[1], 2*matrix_1.shape[1]-1, 1)
    idx_ar = idx_ar.reshape(idx_ar.shape[0], -1)
    
    corrs = np.take_along_axis(corr_coef, idx_ar, axis=1)
    return corrs
    
    