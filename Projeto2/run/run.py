import numpy as np

from Projeto2.neural_network.neural import initiate_constants, activation_values, error_value_layer
from Projeto2.neural_network.auxiliars import print_list, convert_1D_to_column



# print(initiate_constants((2,3), 1E-4))


x_data = np.array([1, 1, 1])

y_data = np.array([1])

thetas = [
    np.array([[0.57, -0.29], [0.47, 0.17],   [-0.52, 0.08]]),
    np.array([[0.13, 0.48],  [-0.69, -0.73], [-0.70, 0.11]]),
    np.array([[0.55], [0.76], [-0.25]])
    ]



number_of_layers = len(thetas) + 1


activation = activation_values(x_data, thetas)



errors = error_value_layer(y_data, activation, thetas)




