import numpy as np
import sys
from progress.bar import IncrementalBar

from Projeto2.neural_network.insiders import *
from Projeto2.neural_network.midders import *



# =============================================================================
def BackPropagation_NeuralNetwork(
    df_images, 
    class_matrix,
    dimensions,
    thetas, 
    max_tries,
    max_cost, 
    lambda_value,
    learning_rate, 
    msg,
    cost_metric='any'
    ):
    '''
    This function is built to...
    
    Returns
    -------
        thetas after Gradient Descendent (GD)

        total_costs, which are the costs in every step, on a DataFrame
        
        msg, if msg is requested

    '''

    bar = IncrementalBar('            backprop', max=max_tries, suffix='%(percent).1f%% - ETA: %(eta_td)s - %(avg).1f s/item') 


    '''
    First roll
    '''
    total_costs = pd.DataFrame()

    activations = activation_layer(df_images, class_matrix, thetas)
    cost        = cost_function_sigmoid(activations, class_matrix, thetas, lambda_value=lambda_value)
    grad        = gradient_layer(class_matrix, activations, thetas, lambda_value=lambda_value)
    compare     = compare_cost_categories(cost, max_cost, cost_metric)    


    '''
    Check gradient
    '''
    grad_check, numerical_grad = test_gradient(
        gradient=grad,
        thetas=thetas,
        activations_neural=activations,
        classification_matrix=class_matrix,
        lambda_value=lambda_value,
        )

    if np.all(grad_check) != True: #if failed on the grad_check
        print("            O programa falhou na checagem do gradiente, no primeiro passo ;-;\n")
        return (1, numerical_grad, grad[-1])
    else:
        print("            Teste do gradiente: ok.")
        pass


    tries = 1

    '''
    Backpropagation regression
    '''
    while (   tries <= max_tries   ) and (compare):


        total_costs = pd.concat([total_costs, cost], axis=1)

        activations = activation_layer(df_images, class_matrix, thetas)
        cost        = cost_function_sigmoid(activations, class_matrix, thetas, lambda_value=lambda_value)
        grad        = gradient_layer(class_matrix, activations, thetas, lambda_value=lambda_value)
        compare     = compare_cost_categories(cost, max_cost, cost_metric)
        
        thetas      = update_thetas(thetas, grad, learning_rate)

        if bar:
            bar.next()

        tries += 1


    if (tries > max_tries):
        msg_result = f"\n\n            Number of tries exceeded (> {round(max_tries)}).\n\n"
    elif ( tries <= max_tries ):
        msg_result = f"\n\n            Success! After {round(tries)} trie(s) (<= {max_tries}), the costs are now under {max_cost}\n\n"
    else:
        msg_result = f"\n\n            No conditional. After {round(tries)} trie(s) (<= {max_tries}).\n\n"
    print(msg_result)
    if msg != False:
        msg += msg_result

    total_costs.columns = [str(i+1) for i in range(total_costs.shape[1])]

    return(thetas, total_costs, msg)




# =============================================================================
def BackPropagation_CalculateValidation(
    df_training,
    df_valid,
    orig_labels,
    label_valid, 
    class_matrix_training,
    dimensions,
    thetas, 
    max_tries,
    max_cost, 
    lambda_value,
    learning_rate, 
    msg,
    progress_bar=False,
    cost_metric='any'
    ):
    """
    """

    _ = BackPropagation_NeuralNetwork(
        df_images     = df_training,
        class_matrix  = class_matrix_training,
        dimensions    = dimensions,
        thetas        = thetas,
        max_tries     = max_tries,
        max_cost      = max_cost,
        lambda_value  = lambda_value,
        learning_rate = learning_rate,
        msg           = msg,
        cost_metric   = cost_metric  
        )
    
    if _[0] == 1: # erro no teste do gradiente
        return (1, _[1], _[2], lambda_value) # 1, numerical_grad, gradient[-1], lambda_value
    else:
        (thetas, cost_steps, msg) = _

    class_matrix_validation = classification_matrix(label_valid, orig_labels)
    activation_validation   = activation_layer(df_valid, class_matrix_validation, thetas) 
    cost_validation         = cost_function_sigmoid(activation_validation, class_matrix_validation, thetas, lambda_value=lambda_value)

    if progress_bar != False:
        progress_bar.next()
        print('\n')

    return (thetas, cost_steps, cost_validation, lambda_value, msg)



# =============================================================================
def BackPropagation_CurvaAprendizado(
    value_frac_treino,
    df_training,
    df_label_training,
    df_valid,
    df_label_valid,
    orig_labels,
    thetas_range, 
    max_tries,
    max_cost, 
    lambda_value,
    learning_rate, 
    msg,
    progress_bar=False,
    random_state=None,
    number_of_layers=3,
    mult_hidden_layer=0,
    additional_neurons=25,
    cost_metric='any'
    ):
    """
    """

    images = df_training.sample(frac=value_frac_treino, random_state=random_state, axis=1).T.sort_index().T
    labels = df_label_training.T[images.columns].T.sort_index()

    class_matrix_training = classification_matrix(labels, orig_labels)
    dimensions            = neural_net_dimension(images, class_matrix_training, number_of_layers, mult_hidden_layer, additional_neurons) # without bias
    init_thetas           = thetas_layers(dimensions, limit=thetas_range)


    _ = BackPropagation_NeuralNetwork(
        df_images     = images,
        class_matrix  = class_matrix_training,
        dimensions    = dimensions,
        thetas        = init_thetas,
        max_tries     = max_tries,
        max_cost      = max_cost,
        lambda_value  = lambda_value,
        learning_rate = learning_rate,
        msg           = msg,
        cost_metric   = cost_metric  
        )

    if _[0] == 1: # erro no teste do gradiente
        return (1, _[1], _[2], value_frac_treino) # 1, numerical_grad, gradient[-1], value_frac_treino
    else:
        (thetas, cost_steps, msg) = _

    class_matrix_validation = classification_matrix(df_label_valid, orig_labels)
    activation_validation   = activation_layer(df_valid, class_matrix_validation, thetas) 
    cost_validation         = cost_function_sigmoid(activation_validation, class_matrix_validation, thetas, lambda_value=lambda_value)

    if progress_bar != False:
        progress_bar.next()
        print('\n')

    return (thetas, cost_steps, cost_validation, value_frac_treino, msg)


