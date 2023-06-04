import numpy as np
import time

def vectorized_dct(X: np.ndarray) -> np.ndarray:
    '''
    @params
        X : np.float64 array of size(m,n)
    return np.float64 array of size(m,n)
    '''
    # TODO
    return None
    # END TODO

def relevance_one(D: np.ndarray, Q: np.ndarray) -> np.ndarray:
    '''
    @params
        D : n x w x k numpy float64 array 
            where each n x w slice represents a document
            with n vectors of length w 
        Q : m x w numpy float64 array
            which represents a query with m vectors of length w

    return np.ndarray of shape (k,) of docIDs sorted in descending order by relevance score
    '''
    # TODO
    return None
    # END TODO

def relevance_two(D: np.ndarray, Q: np.ndarray) -> np.ndarray:
    '''
    @params
        D : n x w x k numpy float64 array 
            where each n x w slice represents a document
            with n vectors of length w 
        Q : m x w numpy float64 array
            which represents a query with m vectors of length w

    return np.ndarray of shape (k,) of docIDs sorted in descending order by relevance score
    '''
    # TODO
    return None
    # END TODO