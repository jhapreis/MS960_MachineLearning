import numpy as np
import matplotlib.pyplot as plt

from gradient_descendent import Gradient_Descendent, linear_regression_function

# y = x + 2

# x_data = np.array([1, 2, 3, 4])
# y_data = np.array([3, 4, 5, 6])

x_data = 10*np.random.rand(100,)
b = np.random.rand(100,)
y_data = 3*(1 + np.random.rand(100,))*x_data + b

coefficients, residuals = Gradient_Descendent(
    x_data_init=x_data, 
    y_data_init=y_data, 
    initial_guess=[1,1], 
    learning_rate=1E-2, 
    min_residual=1E-6, 
    max_tries=1E6,
    normalize=1
    )

print(f'\n\n{coefficients}\n\n')

# print(residuals, '\n\n')

fig, axes = plt.subplots( 2,1,figsize=(10,15) )

axes[0].scatter(x_data, y_data)
x_values = np.linspace(0,10,1000)
axes[0].plot( x_values, linear_regression_function(x_values, coefficients.loc['theta_0'][0], coefficients.loc['theta_1'][0]) )

x_values = [i for i in range(len(residuals))]
axes[1].scatter(x_values, residuals, color='orange')

plt.show()


