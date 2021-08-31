import numpy as np
import matplotlib.pyplot as plt

from Projeto1.exercicio_1.gd.gradient_descendent import Gradient_Descendent_Exponential

x = np.random.rand(100,)
b = 10*np.random.rand(100,)
y = 5*np.e**(3*x) + b

coefficients, residuals = Gradient_Descendent_Exponential(
    x_data_init=x, 
    y_data_init=y, 
    initial_guess=[1,1], 
    learning_rate=1E-3,
    min_residual=1E-6,
    max_tries=1E5,
    normalize=1 
    )

print(coefficients)
