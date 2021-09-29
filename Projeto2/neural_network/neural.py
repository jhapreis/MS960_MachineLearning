import numpy as np



##################################################
def sigmoid_function(x_data, thetas):
    '''
    x_data: column vector; matrix with entries on the columns
    thetas: column vector

    returns a column vector of the sigmoid function
    '''

    y_data  = np.dot(thetas.T, x_data)

    sigmoid = 1 / (   1 + np.e**(-1*y_data)   ) 

    return(sigmoid)



##################################################
def test_gradient_value():
    pass



##################################################
def error_value():
    pass



##################################################
def derivative_value():
    pass



