import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from Projeto1.exercicio_2.logist.logistic_regression import logistic_function, residual_function_logistic, decision_frontier

df_image = pd.read_csv('Projeto1/data/imageMNIST.csv', header=None).T
df_label = pd.read_csv('Projeto1/data/labelMNIST.csv')

# print(df_image)

thetas = np.random.rand(df_image.shape[0], 1)/1000
# thetas = np.ones((df_image.shape[0], 1))
# print(thetas)

logistic = logistic_function(df_image, thetas)
# print(logistic.shape)

y_decision = decision_frontier(df_image, thetas)
print(y_decision)

# plt.plot(logistic.T, y_decision)
# plt.show()