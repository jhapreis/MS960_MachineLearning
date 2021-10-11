import numpy as np
import pandas as pd

from Projeto2.neural_network.errors import dimensional_error, matrix_multiplication_error, single_dimension_error



# =============================================================================
def initiate_constants(shape, limit=1):
    '''
    shape: 
    limit: 

    returns an array of given shape of random numbers
    '''

    sample = np.random.uniform(low=-1*limit, high=limit, size=shape)

    sample = pd.DataFrame(sample)

    return(sample) 



# =============================================================================
def add_x0_column(x_data):
    '''
    This function is built to...  

    Parameters
    ----------
    x_data:
        .
    '''

    x_data.insert(
        loc=0, column='x_0', value=np.ones(   (x_data.shape[0], 1)   )
        )

    return(x_data)



# =============================================================================
def z_value(x_data, thetas):
    '''
    x_data: matrix with entries on the columns
    thetas: matrix to apply ON EVERY COLUMN of x_data

    returns ...
    '''

    matrix_multiplication_error(thetas.T.shape, x_data.shape)
    

    number_of_elements = x_data.shape[1]

    z_values = np.zeros(   (thetas.T.shape[0], x_data.shape[1])   ).T

    for i in range(number_of_elements):
        x_column    = x_data[ x_data.columns[i] ]
        z_value     = np.dot(thetas.T, x_column)
        z_values[i] = z_value

    z_values         = pd.DataFrame(z_values.T) 
    z_values.index   = thetas.T.index
    z_values.columns = x_data.columns

    return(z_values)



# =============================================================================
def sigmoid_function(x_data, thetas):
    '''
    z_value:

    returns a column vector of the sigmoid function
    '''

    z = z_value(x_data, thetas)

    sigmoid = 1 / (   1 + np.e**(-1*z)   ) # applies element-wise

    return(sigmoid)



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
def delta_value_layer(y_data, activation_values, thetas):
    '''
    '''

    dimensional_error(activation_values[-1].shape, y_data.shape)


    last_error = activation_values[-1] - y_data


    dimensional_error(last_error.shape, y_data.shape)


    errors = [last_error]

    for i in range( len(thetas)-1, 0, -1 ): # backwards propagation

        sum_theta_deltas = np.dot( thetas[i].iloc[1:], errors[-1] ) # exclude theta of bias

        if (sum_theta_deltas.shape != 1) and (sum_theta_deltas.shape != (1, 1)):
            dimensional_error(sum_theta_deltas.shape, activation_values[i].shape)

        error = sum_theta_deltas * activation_values[i].to_numpy() * (1 - activation_values[i].to_numpy())
        
        error = pd.DataFrame(error, index=activation_values[i].index, columns=activation_values[i].columns)
        errors.append(error)

    errors.reverse()

    return(errors)



# =============================================================================
def Delta_layer(y_data, activations, thetas):
    '''
    y_data: 
    activations: list with the avtivation values of all layers
    thetas: 

    returns...
    '''

    deltas = delta_value_layer(y_data, activations, thetas)


    summed_deltas = []

    for i in range( len(deltas)-1, -1, -1 ): # backward, for every layer L, L-1, ..., 1

        delta_layer      = deltas[i]
        activation_layer = add_x0_column(activations[i].T).T # add bias

        single_dimension_error(delta_layer.shape, activation_layer.shape, axis=1)

        a_times_delta = np.zeros((activation_layer.shape[0], delta_layer.shape[0]))

        dimensional_error(thetas[i].shape, a_times_delta.shape)

        for j in range( delta_layer.shape[1] ): # for every data entry; N datas

            delta_column = delta_layer[ delta_layer.columns[j] ]
            activ_column = activation_layer[ activation_layer.columns[j] ]

            matrix_delta = np.full( a_times_delta.shape, delta_column )
            matrix_activ = np.full( a_times_delta.T.shape, activ_column ).T

            dimensional_error(matrix_activ.shape, matrix_delta.shape)

            dimensional_error(matrix_activ.shape, a_times_delta.shape)

            a_times_delta += matrix_activ*matrix_delta

        a_times_delta = pd.DataFrame(a_times_delta, index=thetas[i].index, columns=thetas[i].columns)

        summed_deltas.append(a_times_delta)

    summed_deltas.reverse()

    return(summed_deltas) 




