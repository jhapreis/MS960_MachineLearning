import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time
from datetime import timedelta

from Projeto1.exercicio_1.gd.gradient_descendent import normalization



# =============================================================================
def multiple_linear_function(thetas, x_data):

    number_of_terms = len(thetas)

    thetas = np.array([thetas]).reshape(1, number_of_terms)

    x_0_matrix = np.ones( (1,x_data.shape[1]) )

    x_data = np.append( x_0_matrix, x_data, axis=0 )

    y_data = np.dot(thetas, x_data)

    return(y_data)



# =============================================================================
def multiple_residual_function(x_data, y_data, coefficients):
    '''
    '''

    residuals = multiple_linear_function(thetas=coefficients, x_data=x_data) - y_data

    data_size = residuals.shape[1]
    
    cost_function = 1/(2*data_size) * sum(residuals[0]**2)

    return(cost_function)



# =============================================================================
def multiple_gradient_residual_function(x_data, y_data, coefficients):
    '''
    '''

    residuals = multiple_linear_function(thetas=coefficients, x_data=x_data) - y_data
    data_size = residuals.shape[1]

    x_0_matrix = np.ones( (1,x_data.shape[1]) )
    x_data     = np.append( x_0_matrix, x_data, axis=0 ).T

    gradient   = np.dot(residuals,x_data) / data_size

    return(gradient)

    

# =============================================================================
def Multiple_Gradient_Descendent(x_data_init, y_data_init, initial_guess, learning_rate, min_residual=1E-3, max_tries=100, normalize=1):

    if normalize == 1:
        x_data, y_data = multiple_normalization(x_data_init, y_data_init)
    else:
        x_data, y_data = x_data_init, y_data_init


    thetas   = initial_guess
    residual = multiple_residual_function(x_data=x_data, y_data=y_data, coefficients=thetas)
    tries    = 0

    residuals_values = [residual]

    while (tries <= max_tries) and (residual > min_residual):

        gradient = multiple_gradient_residual_function(x_data, y_data, thetas)

        thetas = thetas - learning_rate*gradient[0]
        
        residual = multiple_residual_function(x_data=x_data, y_data=y_data, coefficients=thetas)
        tries    += 1
        residuals_values.append(residual)

    if residual <= min_residual:
        print(f'\n   Success! The residual is under the minimal value.\nAfter {tries} steps.\n\n')
    elif tries > max_tries:
        print(f'\n   Failed! The number of tries was exceeded.\nThe residual is {residual:e}.\n\n')

    if normalize == 1:
        coefficients = undo_multiple_normalization(x_data_init, y_data_init, thetas)
    else:
        coefficients = thetas

    coefficients     = pd.DataFrame(coefficients, columns=['values'], index=[('theta_'+str(i)) for i in range(coefficients.shape[0])])
    residuals_values = pd.DataFrame(residuals_values, columns=['residuals'])

    return( coefficients, residuals_values  )



# =============================================================================
def polynomial_matrice(x_data, thetas):
    '''
    '''

    number_of_terms = len(thetas)

    # i = 1
    data_matrix = np.array( x_data )

    # i = 2, ..., n
    for i in range( 2,number_of_terms ):

        x_matrix = np.array( x_data**i )
        data_matrix = np.append( data_matrix, x_matrix, axis=0 )

    return(data_matrix)


 
# =============================================================================
def polynomial_function(thetas, x_data):
    '''
    P(x) = SUM a_i*x**i, from i=0 to i=n; n+1 terms
    '''
    
    number_of_terms = len(thetas)

    thetas = np.array(thetas).reshape(1, number_of_terms)

    # i = 0
    data_matrix = np.array([ [1 for i in range( len(x_data) )] ])

    # i = 1
    x_matrix = np.array([ x_data ])
    data_matrix = np.append( data_matrix, x_matrix, axis=0 )

    # i = 2, ..., n
    for i in range(2,number_of_terms):

        x_matrix = np.array( [x_data**i] )
        data_matrix = np.append( data_matrix, x_matrix, axis=0 )

    P_x = np.dot(thetas, data_matrix)[0]

    return(P_x)



# =============================================================================
def multiple_normalization(x_data, y_data):
    '''
    This function is built to...

    Parameters
    ----------
    x_data:
        Entries are on row-like input. [ [x_1], [x_2], ..., [x_n] ]
    y_data:
        Row vector; one-dimensional array
    '''

    y_data_norm = (y_data - y_data.min()) / (y_data.max() - y_data.min())

    x_min   = x_data.min(axis=1).reshape(x_data.shape[0], 1)
    x_max   = x_data.max(axis=1).reshape(x_data.shape[0], 1)
    delta_x = x_max - x_min

    x_data_norm = (x_data - x_min) / delta_x

    return(x_data_norm, y_data_norm)



# =============================================================================
def undo_multiple_normalization(x_data, y_data, thetas):
    '''
    This function is built to...

    Parameters
    ----------
    x_data:
        Entries are on row-like input. [ [x_1], [x_2], ..., [x_n] ]
    y_data:
        Row vector; one-dimensional array
    '''

    y_min   = y_data.min()
    y_max   = y_data.max()
    delta_y = y_max - y_min

    x_min   = x_data.min(axis=1).reshape(x_data.shape[0], 1)
    x_max   = x_data.max(axis=1).reshape(x_data.shape[0], 1)
    delta_x = x_max - x_min

    thetas_1_to_n = thetas[1:].reshape(x_data.shape[0],1)
    theta_0 = delta_y*( thetas[0] - sum(thetas_1_to_n*x_min/delta_x) ) + y_min
    theta_i = delta_y*( thetas_1_to_n/delta_x ) # i >= 1

    coefficients = np.append(theta_0, theta_i)

    return(coefficients)


