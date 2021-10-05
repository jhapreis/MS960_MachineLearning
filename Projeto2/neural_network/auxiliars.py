import numpy as np


# =============================================================================
def transform_thetas_to_array(thetas):
    '''
    '''
    
    for i in range(len(thetas)):
        thetas[i] = np.array(thetas[i])

    return(thetas)



# =============================================================================
def print_list(lista):
    '''
    '''

    for i in range(len(lista)):
        print(lista[i])
    


# =============================================================================
def convert_1D_to_column(array):
    '''
    '''

    if len(array.shape) == 1: # array is one-dimensional
        array = array.reshape(   (len(array),1)   )

    return(array)


