import numpy as np

def create_nl_feature(X):
    '''
    Create additional features and add it to the dataset.
    
    Returns:
        X_new - (N, d + num_new_features) array with 
                additional features added to X such that it
                can classify the points in the dataset.
    '''
    # NotImplementedError    
    X2 = X.copy()
    for i in range(2, 6):
        X2 = np.concatenate((X2, np.power(X, i)), axis=1)
    
    return X2