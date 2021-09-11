import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time
from datetime import timedelta

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
def expoential_regression_function(x_data, theta_0, theta_1):
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

    y_values = theta_1*np.e**(theta_0*x_data)

    return(y_values)



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
def normalization(x_data, y_data):
    '''
    This function is built to...

    Parameters
    ----------
    x_data: DataFrame
        .
    y_data:
        .
    '''

    x_data_norm = (x_data - x_data.min()) / (x_data.max() - x_data.min())
    y_data_norm = (y_data - y_data.min()) / (y_data.max() - y_data.min())

    return(x_data_norm, y_data_norm)



# =============================================================================
def undo_normalization_coefficients(x_data, y_data, coefficients):
    '''
    This function is built to...

    Parameters
    ----------
    
    '''

    x_difference = x_data.max() - x_data.min()
    y_difference = y_data.max() - y_data.min()
    x_min = x_data.min()
    y_min = y_data.min()

    theta_0 = y_difference/x_difference*coefficients[0]
    theta_1 = y_difference*( coefficients[1] - coefficients[0]*x_min/x_difference ) + y_min
    
    return(theta_0, theta_1)



# =============================================================================
def Gradient_Descendent(x_data_init, y_data_init, initial_guess, learning_rate, min_residual=1E-3, max_tries=100, normalize=1):
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

    if normalize == 1:
        x_data, y_data = normalization(x_data_init, y_data_init)
    else:
        x_data, y_data = x_data_init, y_data_init

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
        print(f'\n   Success! The residual is under the minimal value.\nAfter {tries} steps.\n\n')
    elif tries > max_tries:
        print(f'\n   Failed! The number of tries was exceded.\nThe residual is {residual}\n\n')
  
    if normalize == 1:
        coefficients = undo_normalization_coefficients(x_data_init, y_data_init, [theta_0, theta_1])

    coefficients     = pd.DataFrame(coefficients, columns=['values'], index=['theta_0', 'theta_1'])
    residuals_values = pd.DataFrame(residuals_values, columns=['residuals'])

    return( coefficients, residuals_values  )



# =============================================================================
def Gradient_Descendent_Exponential(x_data_init, y_data_init,  em , learning_rate, min_residual=1E-3, max_tries=100, normalize=1):
    '''
    This function is built to...

    Parameters
    ----------
    
    '''

    time_start = time()

    y_data_log = np.log( y_data_init )

    coefficients, residuals = Gradient_Descendent(
        x_data_init=x_data_init, 
        y_data_init=y_data_log, 
        initial_guess=initial_guess, 
        learning_rate=learning_rate, 
        min_residual=min_residual, 
        max_tries=max_tries,
        normalize=normalize
    )

    theta_1_exp = np.e**(coefficients['values']['theta_1'])

    '''
    Plots
    '''
    fig, ax = plt.subplots(2,2,figsize=(12,10))

    steps      = [ i for i in range(len(residuals)) ]

    theta_0 = coefficients['values']['theta_0']
    theta_1 = coefficients['values']['theta_1']

    # Original data
    ax[0,0].scatter(x_data_init, y_data_init, color='orange')
    ax[0,0].set_xlabel('number of days', fontsize=10)
    ax[0,0].set_ylabel('amount of cases', fontsize=10)

    # Log cases
    ax[0,1].scatter(x_data_init, y_data_log, color='black')
    x_linear = np.linspace(0,x_data_init.max(),10000)
    ax[0,1].plot( x_linear, linear_regression_function(x_linear, theta_0, theta_1) )
    ax[0,1].set_xlabel('number of days', fontsize=10)
    ax[0,1].set_ylabel('log of amount of cases', fontsize=10)

    # Residuals
    ax[1,0].scatter( steps, residuals, color='green' )
    ax[1,0].set_xlabel("number of iterations", fontsize=10)
    ax[1,0].set_ylabel('residual', fontsize=10)

    # Regression exponential
    ax[1,1].scatter(x_data_init, y_data_init, color='gray')
    x_exp = np.linspace(0, 120, 1000)
    ax[1,1].plot( x_exp, expoential_regression_function(x_exp, theta_0, theta_1_exp) )
    ax[1,1].set_xlabel('number of days', fontsize=10)
    ax[1,1].set_ylabel('amount of cases', fontsize=10)

    # plt.show()

    plt.savefig(f'Projeto1/exercicio_1/results/exponential/exp.png')


    time_final = time()
    print(f'\n\nTime elapsed: {timedelta(seconds=time_final-time_start)}\n\n')

    return(coefficients, residuals)


