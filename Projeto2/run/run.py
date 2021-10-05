import numpy as np
import pandas as pd

from Projeto2.neural_network.neural import *
from Projeto2.neural_network.auxiliars import print_list, convert_1D_to_column
from Projeto2.neural_network.errors import *

import cfg



'''
Beggining
'''
df_images = pd.read_csv("../data/test/sample_images.csv", index_col=0)
df_labels = pd.read_csv("../data/test/sample_labels.csv", index_col=0)

class_matrix = classification_matrix(df_labels, cfg.labels)

dimensions = neural_net_dimension( df_images, class_matrix, cfg.number_of_layers, cfg.mult_hidden_layer ) # without bias

thetas = []
for i in range( len(dimensions)-1 ):
    theta = generate_thetas(dimensions[i]+1, dimensions[i+1], limit=cfg.init_thetas_range) #include bias on the starting layer
    thetas.append(theta)
del(theta)



'''
Activations
'''
activations = [df_images] # calculate for the first layer
for i in range( len(dimensions)-1 ): # calculate for the others layers
    activation = activation_values(activations[-1], thetas[i])
    activations.append(activation)
activations[-1].index = class_matrix.index
del(activation)



'''
Deltas
'''
deltas = Delta_layer(class_matrix, activations, thetas)



'''
Gradient
'''
grad = gradient_layer(class_matrix, activations, thetas, lambda_value=cfg.lambda_value)





# x_data = np.array([1, 1, 1])

# y_data = np.array([1])

# thetas = [
#     np.array([[0.57, -0.29], [0.47, 0.17],   [-0.52, 0.08]]),
#     np.array([[0.13, 0.48],  [-0.69, -0.73], [-0.70, 0.11]]),
#     np.array([[0.55], [0.76], [-0.25]])
#     ]


# activation = activation_values(df_images, thetas)


# errors = error_value_layer(y_data, activation, thetas)




# cost = cost_function_sigmoid(df_images, class_matrix, )



