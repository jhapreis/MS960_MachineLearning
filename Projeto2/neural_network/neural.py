import numpy as np
import pandas as pd

from Projeto2.neural_network.errors import dimensional_error, matrix_multiplication_error, single_dimension_error, check_equal_values

from Projeto2.neural_network.insiders import *



def thetas_layers(dimensions, limit):
    '''
    '''
    thetas = []
    for i in range( len(dimensions)-1 ):
        theta = generate_thetas(dimensions[i]+1, dimensions[i+1], limit=limit) #include bias on the starting layer
        thetas.append(theta)

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
def neural_net_dimension(x_data, y_data, number_of_layers, multiplier=1, additional=0):
    '''
    '''

    dimensions = number_of_layers*[0]

    dimensions[0]  = x_data.shape[0] # first layer
    dimensions[-1] = y_data.shape[0] # last layer

    for i in range( 1, len(dimensions) - 1 ): # middle layers have the same size
        dimensions[i] = multiplier*dimensions[0] + additional

    return(dimensions)
    


# =============================================================================
def activation_layer(x_data, classification_matrix, thetas):
    '''
    '''

    activations = [x_data]           # calculate for the first layer

    for i in range( len(thetas) ): # calculate for the others layers
        activation = activation_values(activations[-1], thetas[i])
        activations.append(activation)

    activations[-1].index = classification_matrix.index

    return(activations)



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
def cost_function_sigmoid(activations, classification_matrix):
    '''
    This function is built to...

    Parameters
    ----------
    
    '''

    dimensional_error(activations[-1].shape, classification_matrix.shape)

    number_of_elements = classification_matrix.shape[1]
    last_activation = activations[-1]

    residuals = classification_matrix*np.log(last_activation) + (1 - classification_matrix)*np.log(1 - last_activation)

    cost = -1/number_of_elements * residuals.sum(axis=1)

    cost = pd.DataFrame(cost, columns=['cost'])

    return(cost)



# =============================================================================
def update_thetas(thetas, gradient, learning_rate):
    '''
    '''

    for i in range(len(thetas)):

        delta_theta = learning_rate*gradient[i]

        check_equal_values(thetas[i], delta_theta)

        thetas[i] -= delta_theta
    
    return(thetas)



# =============================================================================
def test_gradient():
    pass
    


# =============================================================================
def derivative_value_layer():
    pass



# =============================================================================
def check_cost_decreasing(total_costs):
    '''
    '''

    boolean = pd.DataFrame()

    for i in range(total_costs.shape[1]-1):

        _ = ( total_costs[total_costs.columns[i+1]] <= total_costs[total_costs.columns[i]] )

        boolean[f"{i+1}_{i+2}"] = _

    if ( np.all(boolean) != True ):

        not_decreasing = []

        _ = np.all(boolean, axis=0)

        for i in range(_.shape[0]):
            if _.iloc[i] != True:
                not_decreasing.append(_.iloc[i])
        
        msg = f"\n\n   Not decreading on {not_decreasing}\n\n"

    else:
        msg = "\n\n   All costs are decreasing in every single step.\n\n"
        
    print(msg)


