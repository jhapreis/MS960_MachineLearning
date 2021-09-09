from matplotlib.pyplot import axis, cla
import pandas as pd
import numpy as np



# =============================================================================
def logistic_function(x_data, coefficients):
    '''
    This function is built to...

    Parameters
    ----------
    x_data:
        Entries are on column-like input. [ [x_1], [x_2], ..., [x_n] ]
    coefficients:
        .
    '''

    theta_times_x = np.dot(coefficients.T, x_data)

    h_theta_x = 1 / (   1 + np.exp(-1*theta_times_x)   )
    
    return(h_theta_x)



# =============================================================================
def residual_function_logistic(x_data, y_data, coefficients):
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

    logistic = logistic_function(x_data=x_data, coefficients=coefficients)

    # sum_zero_term = np.dot(       y_data, np.log(logistic)     )
    # sum_one_term  = np.dot( (1 - y_data), np.log(1 - logistic) )

    zero_term = y_data * np.log(logistic)
    one_term  = (1 - y_data) * np.log(1 - logistic)

    residual_individual = -1*(zero_term + one_term) / data_size

    residual = np.array( residual_individual.sum(axis=1) ).reshape( len(residual_individual), 1 )

    return(residual)



# =============================================================================
def decision_frontier(x_data, coefficients, limit=0.5):
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

    logistic_value = logistic_function(x_data, coefficients)

    y_class = (logistic_value >= limit)[0].astype(int)

    return(y_class)



# =============================================================================
def logistic_gradient_residual_function(x_data, y_data, coefficients):
    '''
    grad = 1/m * np.dot( X.T, g(X*theta) - y )
 
    Parameters
    ----------
    x_data:
        Entries are on column-like input. [ [x_1], [x_2], ..., [x_n] ]
    y_data:
        .
    coefficients:
        .

    returns:
        gradient (thetas X classifications)
    '''

    logistic  = logistic_function(coefficients=coefficients, x_data=x_data)
    data_size = logistic.shape[1]

    residuals = (logistic - y_data)

    gradient  = np.dot(x_data, residuals.T) / data_size

    return(gradient)



# =============================================================================
def Logistic_Gradient_Descendent(x_data_init, y_data_init, initial_guess, learning_rate, min_residual=1E-3, max_tries=100, normalize=1):

    if normalize == 1:
        x_data, y_data = logistic_normalization(x_data_init), y_data_init
    else:
        x_data, y_data = x_data_init, y_data_init


    thetas   = initial_guess
    residual = residual_function_logistic(x_data=x_data, y_data=y_data, coefficients=thetas)
    tries    = 1

    residuals_values = residual

    # print(residuals_values.shape)

    while (tries < max_tries) and (np.all(residual) > min_residual):


        gradient = logistic_gradient_residual_function(x_data, y_data, thetas)

        thetas  -= learning_rate*gradient
        
        residual = residual_function_logistic(x_data=x_data, y_data=y_data, coefficients=thetas)
        tries    += 1
        residuals_values = np.append( residuals_values, residual, axis=1 )


    if np.all(residual) <= min_residual:
        print(f'\n   Success! The residual is under the minimal value.\nAfter {tries} steps.\n\n')
    elif tries >= max_tries:
        print(f'\n   Failed! The number of tries was exceeded.\n\n')


    if normalize == 1:
        coefficients = undo_logistic_normalization(x_data_init, thetas)
    else:
        coefficients = thetas


    # coefficients     = pd.DataFrame(coefficients, columns=['label_'+str(i+1) for i in range(coefficients.shape[1])], index=[('theta_'+str(i)) for i in range(coefficients.shape[0])])
    
    coefficients     = pd.DataFrame(coefficients, columns=['label_'+str(i+1) for i in range(coefficients.shape[1])], index=[('theta_'+str(i)) for i in range(coefficients.shape[0])])

    residuals_values = pd.DataFrame(residuals_values)
    residuals_values.columns = ['try_'+str(i) for i in range(residuals_values.shape[1])]
    residuals_values.index   = coefficients.columns


    return( coefficients, residuals_values  )



# =============================================================================
def logistic_normalization(x_data):
    '''
    This function is built to... y_data is already normalized (y = {0,1})

    Parameters
    ----------
    x_data:
        Entries are on row-like input. [ [x_1], [x_2], ..., [x_n] ]        
    '''

    x_min = x_data.min(axis=0)
    x_max = x_data.max(axis=0)    

    x_min = np.full( x_data.shape, x_min )
    x_max = np.full( x_data.shape, x_max )

    delta_x = x_max - x_min

    x_data_norm = np.divide(     (x_data - x_min) , delta_x     )

    return(x_data_norm)



# =============================================================================
def undo_logistic_normalization(x_data, thetas_norm):
    '''
    This function is built to retrieve the solution of  

        thetas.T*x = thetas_norm.T*x_norm 
    
    Which gives
    
        x.T*thetas = (thetas_norm.T*x_data_norm).T

    Solve for thetas.

    It solves by least-squares method, cause the solution is not unique. 

    Parameters
    ----------
    x_data:
        Entries are on row-like input. [ [x_1], [x_2], ..., [x_n] ]
    thetas:
        Values of the normalized regression
    '''

    x_data_norm = logistic_normalization(x_data)

    thetas_norm_times_x_norm = np.dot(thetas_norm.T, x_data_norm).T

    solution, residual, rank, singular = np.linalg.lstsq(a=x_data.T, b=thetas_norm_times_x_norm, rcond=None)

    thetas = solution   

    return(thetas)



# =============================================================================
def classification_result(x_data, thetas_regression):
    '''
    This function is built to...  

    Parameters
    ----------
    x_data:
        .
    thetas_regression:
        .
    '''


    logistic = logistic_function(x_data, thetas_regression)

    logistic = pd.DataFrame(logistic, columns=x_data.columns)

    logistic.index = [ (i+1) for i in range(logistic.shape[0]) ]

    classification = pd.DataFrame( logistic.idxmax(axis=0), columns=['label'] )

    return(classification)



# =============================================================================
def Correct_Classification(x_data, thetas_regression, original_labels):
    '''
    This function is built to...  

    Parameters
    ----------
    x_data:
        .
    thetas_regression:
        .
    original_labels:
        .
    '''

    classification = classification_result(x_data, thetas_regression)

    if classification.shape != original_labels.shape:
            raise ValueError(f"\n\'classification\' {classification.shape} has not the same shape as \'original_labels\' {original_labels.shape}\n")

    correct = (classification == original_labels).astype(int)

    return(classification, correct)




# # =============================================================================
# def undo_logistic_normalization(x_data, thetas):
#     '''
#     This function is built to... thetas = thetas_norm*x_norm / x

#     Parameters
#     ----------
#     x_data:
#         Entries are on row-like input. [ [x_1], [x_2], ..., [x_n] ]
#     y_data:
#         Row vector; one-dimensional array
#     thetas:
#         Values of the normalized regression
#     '''

#     x_min = x_data.min(axis=0)
#     x_max = x_data.max(axis=0)    

#     x_min = np.full( x_data.shape, x_min )
#     x_max = np.full( x_data.shape, x_max )

#     delta_x = x_max - x_min

#     # x_ratio = np.dot(     thetas.T, x_data - delta_x     ).diagonal().reshape( (thetas.shape[1], 1) )

#     # thetas = np.divide( thetas_times_x, x_data )

#     x_ratio = np.divide( (x_data - x_min), (x_data*delta_x) )

#     # print(x_ratio)
#     # print(delta_x.shape)

#     return(thetas)


