import numpy as np
import pandas as pd
from progress.bar import IncrementalBar

from Projeto2.neural_network.errors import dimensional_error, matrix_multiplication_error, single_dimension_error, check_equal_values

from Projeto2.neural_network.insiders import *
from Projeto2.neural_network.midders import *




# =============================================================================
def BackPropagation_NeuralNetwork(
    df_images, 
    df_labels,
    init_thetas_range,
    number_of_layers,
    mult_hidden_layers,
    aditional_layers, 
    orig_labels, 
    max_tries,
    max_cost, 
    lambda_value,
    learning_rate, 
    msg
    ):
    '''
    '''

    bar = IncrementalBar('            backprop', max=max_tries, suffix='%(percent).1f%% - ETA: %(eta_td)s - %(avg).1f s/item') 


    class_matrix = classification_matrix( df_labels, orig_labels)
    dimensions   = neural_net_dimension(  df_images, class_matrix, number_of_layers, mult_hidden_layers, aditional_layers ) # without bias
    thetas       = thetas_layers(dimensions, limit=init_thetas_range)

    msg_dim = f"\n\n      Dimensions: {dimensions}... {df_images.shape[1]} images.\n"
    print(msg_dim)
    msg += msg_dim


    '''
    First roll
    '''
    activations = activation_layer(df_images, class_matrix, thetas)
    grad        = gradient_layer(class_matrix, activations, thetas, lambda_value=lambda_value)
    cost        = cost_function_sigmoid(activations, class_matrix, thetas, lambda_value=lambda_value)

    tries = 1 # first try for this lambda_value or value_frac


    '''
    Backpropagation regression
    '''

    while (   tries <= max_tries   ) and (   np.all(cost > max_cost)   ):

        activations = activation_layer(df_images, class_matrix, thetas)
        grad        = gradient_layer(class_matrix, activations, thetas, lambda_value=lambda_value)
        cost        = cost_function_sigmoid(activations, class_matrix, thetas, lambda_value=lambda_value)

        thetas      = update_thetas(thetas, grad, learning_rate)

        if bar:
            bar.next()

        tries += 1


    if (tries > max_tries):
        msg_result = f"\n\n            Number of tries exceeded (> {max_tries}).\n\n"
    elif ( np.all(cost <= max_cost) ):
        msg_result = f"\n\n            Success! After {tries} trie(s) (<= {max_tries}), the costs are now under {max_cost}\n\n"
    else:
        msg_result = f"\n\n            No conditional. After {tries} trie(s) (<= {max_tries}).\n\n"
    print(msg_result)
    msg += msg_result


    return(thetas, cost, msg)


