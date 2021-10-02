import numpy as np

from Projeto2.neural_network.errors import dimensional_error
from Projeto2.neural_network.auxiliars import print_list, convert_1D_to_column




##################################################
def initiate_constants(shape, limit):
    '''
    shape: 
    limit: 

    returns an array of given shape of random numbers
    '''

    sample = np.random.uniform(low=-1*limit, high=limit, size=shape)

    return(sample)



##################################################
def sigmoid_function(x_data, thetas):
    '''
    x_data: column vector; matrix with entries on the columns
    thetas: column vector

    returns a column vector of the sigmoid function
    '''

    z_value = np.dot(thetas.T, x_data)

    sigmoid = 1 / (   1 + np.e**(-1*z_value)   ) 

    return(sigmoid)



##################################################
def cost_function(x_data, y_data, thetas):
    pass


##################################################
def activation_values(x_data, thetas):
    '''
    '''

    activation = [x_data] # requires x_0 = 1 to be added before

    for i in range(len(thetas)):

        y_data = sigmoid_function(x_data=activation[i], thetas=thetas[i])

        if i < len(thetas) - 1: # if this is not the last, add bias

            y_data = np.concatenate(   ( [1], y_data )   )

        activation.append(y_data)

    return(activation)



##################################################
def error_value_layer(y_data, activation_values, thetas):

    dimensional_error(activation_values[-1].shape, y_data.shape)

    last_error = activation_values[-1] - y_data
    
    errors = [last_error]

    for i in range(   len(thetas)-1, 0, -1   ): # backwards propagation

        sum_theta_deltas = np.dot( thetas[i][1:], errors[-1] ) # do not include bias coefficient

        error = sum_theta_deltas * activation_values[i][1:] * (1 - activation_values[i][1:])

        errors.append(error)

    errors.reverse()

    return(errors)



##################################################
def gradient_value_layer(activation_values, errors):
    '''
    Calculates de gradient on a given layer
    '''

    gradient = 0
    pass


##################################################
def test_gradient():
    pass
    



##################################################
def derivative_value_layer():
    pass





