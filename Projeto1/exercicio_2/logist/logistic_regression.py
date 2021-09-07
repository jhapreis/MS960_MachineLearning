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
    '''

    h_theta_x = 1 / (   1 + np.e**(-1*np.dot(coefficients.T, x_data))   )

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

    zero_term = y_data*np.log( logistic_function(x_data=x_data, coefficients=coefficients) )
    one_term  = (1 - y_data)*np.log( 1 - logistic_function(x_data=x_data, coefficients=coefficients) )

    residual = sum(zero_term + one_term) / data_size

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

    # if logistic_value >= limit:
    #     y_class = 1
    # elif logistic_value < limit:
    #     y_class = 0
    # else:
    #     raise ValueError("Error on y_class atribute value.")

    y_class = (logistic_value >= limit)[0].astype(int)

    return(y_class)


