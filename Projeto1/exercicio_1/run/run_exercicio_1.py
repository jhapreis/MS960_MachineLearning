import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from Projeto1.exercicio_1.gd.gradient_descendent import Gradient_Descendent_Exponential, expoential_regression_function, linear_regression_function
from Projeto1.exercicio_1.multiple_gd.gradient_descendent_multiple import Multiple_Gradient_Descendent, polynomial_matrice, multiple_linear_function
 

'''
Read csv and main parameters
'''
df_casos_covid = pd.read_csv('Projeto1/data/casesBrazil.csv')
learning_rate  = 1E-1
min_residual   = 1E-3
max_tries      = 1E4

'''
Which regression?
'''
exp_reg = 1

pol_reg = 0    
polynomial_degree = 10


'''
Exponential Regression
'''
if exp_reg == 1:

    coefficients, residuals = Gradient_Descendent_Exponential(
        x_data_init=df_casos_covid['day'], 
        y_data_init=df_casos_covid['cases'], 
        initial_guess=[1,1],
        learning_rate=learning_rate,
        min_residual=min_residual,
        max_tries=max_tries 
        )
    print(f'\nExponential Regression:\n\n{coefficients}\n\n')

    '''
    Save results to csv file
    '''
    results = pd.DataFrame([learning_rate, min_residual, max_tries, exp_reg, pol_reg, polynomial_degree])
    results.index = ['learning_rate', 'min_residual', 'max_tries', 'exp_reg', 'pol_reg', 'polynomial_degree']
    results.columns = coefficients.columns
    results = pd.concat([results, coefficients])
    results.to_csv(f'Projeto1/exercicio_1/results/exponential/exp.csv')




'''
Polynomial Regression
'''
if pol_reg == 1:

    # thetas = np.array([1, 1, 1, 1]) #polynomial degree = len(thetas) - 1

    thetas = np.array( (polynomial_degree+1)*[1] ) #initial guess: all starts at 1

    x_data = np.array( df_casos_covid['day'] ).reshape(  (1,df_casos_covid.shape[0])  )
    y_data = np.array( df_casos_covid['cases'] )
    x_data_polynomial = polynomial_matrice(x_data, thetas)

    coefficients, residuals_values = Multiple_Gradient_Descendent(
        x_data_init=x_data_polynomial, 
        y_data_init=y_data, 
        initial_guess=thetas,
        learning_rate=learning_rate,
        min_residual=min_residual,
        max_tries=max_tries 
        )
    print(f'\nPolynomial Regression:\n\n{coefficients}\n\n')

    '''
    Save results to csv file
    '''
    results = pd.DataFrame([learning_rate, min_residual, max_tries, exp_reg, pol_reg, polynomial_degree])
    results.index = ['learning_rate', 'min_residual', 'max_tries', 'exp_reg', 'pol_reg', 'polynomial_degree']
    results.columns = coefficients.columns
    results = pd.concat([results, coefficients])
    results.to_csv(f'Projeto1/exercicio_1/results/polynomial/pol_{polynomial_degree}.csv')

    '''
    Plot
    '''
    fig, ax = plt.subplots(1,1,figsize=(20,16))
    plt.title(f'Regressão por polinômio de grau {polynomial_degree}')
    ax.scatter(x_data, y_data, color='black')

    x = np.array([ np.linspace(0,134,100000) ])
    x_polynomial = polynomial_matrice(x, coefficients)
    y = multiple_linear_function(coefficients, x_polynomial)
    ax.plot(x_polynomial[0], y[0], color='orange')

    plt.savefig(f'Projeto1/exercicio_1/results/polynomial/pol_{polynomial_degree}.png')


