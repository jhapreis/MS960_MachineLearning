import numpy as np
import pandas as pd

from Projeto2.neural_network.errors import dimensional_error, matrix_multiplication_error, single_dimension_error

from Projeto2.neural_network.insiders import *






# =============================================================================
def generate_thetas(number_input, number_output, limit=1):
    '''
    returns: thetas that connect one layer to the following; 
        theta_XY --> theta that arrives on the Yth cell, starting from the Xth cell
    '''

    thetas = initiate_constants(   (number_input, number_output) , limit=limit   ) 

    thetas.index   = ['theta_x'+str(i) for i in range(thetas.shape[0])]
    thetas.columns = [str(i+1) for i in range(thetas.shape[1])]

    return(thetas)



# =============================================================================
def classification_matrix(df_label, correspondent_labels):
    '''
    '''

    classification_matrix = pd.DataFrame()

    for i in correspondent_labels:
        column = (df_label == i)
        classification_matrix = pd.concat([classification_matrix, column], axis=1)

    classification_matrix *= 1 # True or False --> as type int
    classification_matrix = classification_matrix.T
    classification_matrix.index = ['label_'+str(i+1) for i in range(classification_matrix.shape[0])]

    return(classification_matrix)



# =============================================================================
def neural_net_dimension(x_data, y_data, number_of_layers, multiplier):
    '''
    '''

    dimensions = number_of_layers*[0]

    dimensions[0]  = x_data.shape[0] # first layer
    dimensions[-1] = y_data.shape[0] # last layer

    for i in range( 1, len(dimensions) - 1 ): # middle layers
        dimensions[i] = multiplier*dimensions[0]

    return(dimensions)



# =============================================================================
def activation_values(x_data, thetas):
    '''
    x_data: matrix with entries on the columns
    thetas: 

    returns ...
    '''

    x_data_with_bias = add_x0_column(x_data.T).T

    activation = sigmoid_function(x_data_with_bias, thetas)

    return (activation)
    


# =============================================================================
def gradient_layer(y_data, activations, thetas, lambda_value):
    '''
    '''

    deltas = Delta_layer(y_data, activations, thetas)

    grad = []

    for i in range( len(thetas) ):

        delta = deltas[i].to_numpy() / activations[i].shape[1] # delta = delta / n
        theta = thetas[i].to_numpy()

        dimensional_error(theta.shape, delta.shape)

        grad_bias     = (delta[0]).reshape( 1, len(delta[0]) ) # theta_ij --> j  = 0
        grad_not_bias = delta[1:] + lambda_value*theta[1:]     # theta_ij --> j != 0

        grad_layer = np.concatenate([grad_bias, grad_not_bias], axis=0)

        grad_layer = pd.DataFrame(grad_layer, index=thetas[i].index, columns=thetas[i].columns)

        grad.append(grad_layer)

    return(grad)


# =============================================================================
def cost_function_sigmoid(x_data, classification_matrix, coefficients):
    '''
    This function is built to...

    Parameters
    ----------
    x_data:
        Entries are on column-like input. [ [x_1], [x_2], ..., [x_n] ]
    y_data:
        .
    coefficients:
        .
    '''

    data_size = x_data.shape[1]

    sigmoid = sigmoid_function(x_data=x_data, coefficients=coefficients)

    zero_term = classification_matrix * np.log(sigmoid)
    one_term  = (1 - classification_matrix) * np.log(1 - sigmoid)

    residual_individual = -1*(zero_term + one_term) / data_size

    residual = np.array( residual_individual.sum(axis=1) ).reshape( len(residual_individual), 1 )

    return(residual)



# =============================================================================
def test_gradient():
    pass
    


# =============================================================================
def derivative_value_layer():
    pass





