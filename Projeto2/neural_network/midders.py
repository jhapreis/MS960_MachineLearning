import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

from Projeto2.neural_network.errors import dimensional_error, matrix_multiplication_error, single_dimension_error, check_equal_values
from Projeto2.neural_network.insiders import *



# =============================================================================
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

        grad_bias     = (delta[0]).reshape( 1, len(delta[0]) )     # theta_ij --> j  = 0
        
        if lambda_value == 0:
            grad_not_bias = delta[1:]                              # theta_ij --> j != 0
        else:
            grad_not_bias = delta[1:] + lambda_value*theta[1:]     # theta_ij --> j != 0

        grad_layer = np.concatenate([grad_bias, grad_not_bias], axis=0)

        grad_layer = pd.DataFrame(grad_layer, index=thetas[i].index, columns=thetas[i].columns)

        grad.append(grad_layer)

    return(grad)



# =============================================================================
def cost_function_sigmoid(activations, classification_matrix, thetas, lambda_value=0):
    '''
    This function is built to...

    Parameters
    ----------
    
    '''

    dimensional_error(activations[-1].shape, classification_matrix.shape)

    number_of_elements = classification_matrix.shape[1]

    cost_not_regular = classification_matrix*np.log(activations[-1]) + (1 - classification_matrix)*np.log(1 - activations[-1])
    cost_not_regular = cost_not_regular.sum(axis=1)     # sum for all costs that derive from a data

    if lambda_value == 0:
        cost_regular = np.zeros(cost_not_regular.shape) # cost_regular = 0 
    else:
        cost_regular = ((thetas[-1])**2).sum(axis=0).T  # sum all thetas that arrive on the same cell
        cost_regular.index = cost_not_regular.index     # label with same name, in order to sum element-wise
        dimensional_error(cost_not_regular.shape, cost_regular.shape)


    """
    cost = -1/m*cost_regular + lambda/2m*cost_not_regular
    """
    cost = -1/number_of_elements*cost_not_regular + lambda_value/(2*number_of_elements)*cost_regular 
    dimensional_error(cost_regular.shape, cost.shape)

    cost = pd.DataFrame( cost, columns=['cost'] ) 

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
def normalize_data(df):

    df = pd.DataFrame( normalize(df), columns=df.columns, index=df.index )

    return df



# =============================================================================
def test_gradient(gradient, thetas, activations_neural, classification_matrix, lambda_value=1, step=1E-4, tolerance=1E-4):
    """
        All parameters are for the entire neural network

        Method: calculate thetas+/-epsilon and, for every element on the starting-layer, 
    switch the correspondent line on the array and calculate the correspondent costs and numerical-gradients.

        Applies the test only on the last layer, using the results from activations[-2]
    """
    
    dimensional_error(thetas[-1].shape, gradient[-1].shape)

    last_thetas = thetas[-1]
    
    last_thetas_plus_step  = thetas[-1] + step
    last_thetas_minus_step = thetas[-1] - step

    num_grad_total = pd.DataFrame()

    for i in range( gradient[-1].shape[0] ):


        last_thetas_plus      = pd.concat( [last_thetas[0:i], last_thetas_plus_step[i:i+1] , last_thetas[i+1:]] , axis=0 )

        last_thetas_minus     = pd.concat( [last_thetas[0:i], last_thetas_minus_step[i:i+1], last_thetas[i+1:]] , axis=0 )

        last_activation_plus  = activation_values(activations_neural[-2], last_thetas_plus ).to_numpy()
        last_activation_minus = activation_values(activations_neural[-2], last_thetas_minus).to_numpy()

        cost_plus             = cost_function_sigmoid([last_activation_plus] , classification_matrix, [last_thetas_plus] , lambda_value)
        cost_minus            = cost_function_sigmoid([last_activation_minus], classification_matrix, [last_thetas_minus], lambda_value)

        num_grad   = (cost_plus - cost_minus)/(2*step) # it's a column DataFrame
        num_grad_total = pd.concat([num_grad_total, num_grad], axis=1)

    num_grad_total = num_grad_total.T

    dimensional_error(num_grad_total.shape, gradient[-1].shape)

    num_grad_total.index   = gradient[-1].index
    num_grad_total.columns = gradient[-1].columns

    _ = (   np.abs( gradient[-1].to_numpy() - num_grad_total.to_numpy() ) <= tolerance   )

    return _, num_grad_total



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


