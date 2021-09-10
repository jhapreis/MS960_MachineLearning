import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from Projeto1.exercicio_2.logist.logistic_regression import logistic_function, residual_function_logistic, decision_frontier, logistic_gradient_residual_function, Logistic_Gradient_Descendent, logistic_normalization



print('\n\nRegression with test sample...')

classifications = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

df = pd.read_csv('../../../Projeto1/data/exercicio_2/data_sample.csv', index_col=0)
df_sample = df.iloc[:, :-1]
df_label = pd.DataFrame(df['label'])



'''
Classification matrix
'''

classification_matrix = pd.DataFrame()

for i in classifications:
    column = (df_label == i).astype(int)
    classification_matrix = pd.concat([classification_matrix, column], axis=1)

classification_matrix = classification_matrix.T
classification_matrix.index = ['label_'+str(i+1) for i in range(classification_matrix.shape[0])]

# print(classification_matrix.shape)




'''
Random thetas
'''

thetas = np.random.rand(df_sample.shape[1], len(classifications))/2
# # thetas = np.ones((df_sample.shape[0], 1))
# print(thetas.shape)
# print(pd.DataFrame(thetas), '\n')




'''
Gradient Descendent
'''

coefficients, residuals_values = Logistic_Gradient_Descendent(
    x_data_init=df_sample.T,
    y_data_init=classification_matrix,
    initial_guess=thetas,
    learning_rate=1E-3,
    min_residual=1E-1,
    max_tries=1E4,
    normalize=0
)

# print(coefficients, '\n\n')
# print(residuals_values, '\n\n')

thetas = pd.DataFrame(thetas, columns=['label_'+str(i+1) for i in range(coefficients.shape[1])], index=[('theta_'+str(i)) for i in range(coefficients.shape[0])])
thetas.to_csv('../../../Projeto1/exercicio_2/results/regression/thetas_initial.csv')
coefficients.to_csv('../../../Projeto1/exercicio_2/results/regression/thetas_regression.csv')
residuals_values.to_csv('../../../Projeto1/exercicio_2/results/regression/residuals.csv')



print('   \ndone\n\n')


# =============================================================================


'''
Perguntar: "a imagem é um número _i_?"
'''

# x_data, y_data = logistic_normalization(df_sample.T), classification_matrix


# logistic = logistic_function(x_data, thetas)
# print(pd.DataFrame(logistic))


# residual = residual_function_logistic(x_data=x_data, y_data=y_data, coefficients=thetas)
# # print(pd.DataFrame(residual))
# print(residual.shape)


# gradient = logistic_gradient_residual_function(x_data=x_data, y_data=y_data, coefficients=thetas)
# # print(gradient.shape)
# print(pd.DataFrame(gradient))

# learning = 1E-3
# thetas_1 = thetas - learning*gradient #1E-1 vs 1E-3
# # print(pd.DataFrame(thetas_1))
# print(f'\nthetas_eq: {np.array_equal(thetas, thetas_1)}\n')


# residual_1 = residual_function_logistic(x_data=x_data, y_data=y_data, coefficients=thetas_1)
# # print(residual.shape)
# print(f'\nresidual_eq: {np.array_equal(residual, residual_1)}\n')