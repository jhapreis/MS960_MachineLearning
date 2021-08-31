import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time
from datetime import timedelta

from Projeto1.exercicio_1.multiple_gd.gradient_descendent_multiple import Multiple_Gradient_Descendent, multiple_linear_function, multiple_normalization

# thetas = np.array([1, 1, 1])
# x_data = np.array([ [0, 1, 2, 3], [2, 4, 5, 7] ])
# y_data = np.array([-1, -1, 0, 0])

thetas = np.array([1, 1])
x_data = np.array([ [0, 1, 2, 3] ])
y_data = np.array([2, 3, 4, 5])

# y = multiple_linear_function(thetas, x_data)
# print(y)

coefficients, residuals_values = Multiple_Gradient_Descendent(
    x_data_init=x_data, 
    y_data_init=y_data, 
    initial_guess=thetas, 
    learning_rate=1E-2,
    min_residual=1E-5,
    max_tries=1E5, 
    normalize=0
    )
print(f"{coefficients}\n\n")

fig, ax = plt.subplots(1,1,figsize=(10,16))

ax.scatter(x_data, y_data)

x = np.array([ np.linspace(0,3,1000) ])
y = multiple_linear_function(coefficients, x)
ax.plot(x[0], y[0], color='orange')

plt.show()



