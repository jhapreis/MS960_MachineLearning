import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt



# =============================================================================
def linear_regression_function(x_data, theta_0, theta_1):
    '''
    This function is built to...

    Parameters
    ----------
    x_data: DataFrame
        .
    theta_0:
        .
    theta_1:
        .

    '''

    function_value = theta_0*x_data + theta_1

    return(function_value)



# =============================================================================
def residual_function(x_data, y_data, coefficients):
    '''
    This function is built to...

    Parameters
    ----------
    x_data: DataFrame
        .
    y_data:
        .
    coefficients:
        .

    '''

    if len(coefficients) != 2:
        raise NotImplementedError("I can't handle anything different from two-dimensional yet")

    residuals = linear_regression_function(x_data, coefficients[0], coefficients[1]) - y_data
    data_size = residuals.shape[0]

    cost_function = 1/(2*data_size) * sum(residuals**2)

    return(cost_function)



# =============================================================================
def gradient_residual_function(x_data, y_data, coefficients):
    '''
    This function is built to...

    Parameters
    ----------
    x_data: DataFrame
        .
    y_data:
        .
    coefficients:
        theta_0 (slope) and theta_1 (intercept)

    '''

    residuals = linear_regression_function(x_data, coefficients[0], coefficients[1]) - y_data
    data_size = residuals.shape[0]

    gradient_theta_0 = sum(residuals*x_data)/data_size
    gradient_theta_1 = sum(residuals*1)/data_size

    return(gradient_theta_0, gradient_theta_1)




# =============================================================================
def Gradient_Descendent(x_data, y_data, initial_guess, learning_rate, min_residual=1E-3, max_tries=100):
    '''
    This function is built to...

    Parameters
    ----------
    x_data: DataFrame
        .
    y_data:
        .
    initial_guess:
        theta_0 (slope) and theta_1 (intercept)
    '''

    theta_0  = initial_guess[0]
    theta_1  = initial_guess[1]
    
    residual = residual_function(x_data, y_data, [theta_0, theta_1])
    tries    = 0    

    residuals_values = [residual]

    while (tries <= max_tries) and (residual > min_residual):

        gradient = gradient_residual_function(x_data, y_data, [theta_0, theta_1])
        
        theta_0  -= learning_rate*gradient[0]
        theta_1  -= learning_rate*gradient[1]
        
        residual = residual_function(x_data, y_data, [theta_0, theta_1])
        tries    += 1
        residuals_values.append(residual)

    if residual <= min_residual:
        print('\n   Success! The residual is under the minimal value.\n\n')
    elif tries > max_tries:
        print('\n   Failed! The number of tries was exceded.\n\n')

    coefficients     = pd.DataFrame([theta_0, theta_1], columns=['values'], index=['theta_0', 'theta_1'])
    residuals_values = pd.DataFrame(residuals_values, columns=['residuals'])
    
    return( coefficients, residuals_values  )



