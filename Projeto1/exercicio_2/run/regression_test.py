import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from Projeto1.exercicio_2.logist.logistic_regression import Correct_Classification



print('\n\nUsing thetas on the data...\n\n')

df = pd.read_csv('../../../Projeto1/data/exercicio_2/data_image.csv', index_col=0)
df_image = df.iloc[:, :-1].T
df_label = pd.DataFrame(df['label'])
# print(df_label, '\n\n')



thetas_regression = pd.read_csv('../../../Projeto1/exercicio_2/results/regression/thetas_regression.csv', index_col=0)
residuals_values  = pd.read_csv('../../../Projeto1/exercicio_2/results/regression/residuals.csv', index_col=0).T
# print('\n\n', thetas_regression)
# print('\n\n', residuals_values)



'''
Classifications
'''

classification, corrects = Correct_Classification(df_image, thetas_regression, df_label)

# print('\n\nClassifications:')
# print(classification, '\n\n')

# print('\n\nCorrects:')
# print(corrects, '\n\n')

results = pd.DataFrame()
results['result']   = corrects.value_counts().sort_index()
results['ratio(%)'] = 100*corrects.value_counts(normalize=True).sort_index()
results.index = ['0', '1']

total = pd.DataFrame(    [ classification.shape[0], 100 ], index=results.columns, columns=['total']).T

results = results.append(total)



'''
Save to csv
'''

results.to_csv('../../../Projeto1/exercicio_2/results/analysis/results.csv')
classification.to_csv('../../../Projeto1/exercicio_2/results/analysis/classification.csv')
corrects.to_csv('../../../Projeto1/exercicio_2/results/analysis/corrects.csv')



'''
Print result
'''

print(f'\n\nResults:\n{results}.\n\n')







'''
Plots residuals
'''

# fig, ax = plt.subplots(1, 2, figsize=(16,8)) #residual; 

# steps    = [i for i in range(residuals_values.shape[0])]
# residual = residuals_values.sample(1, axis=1)
# # for image in residuals_values.columns:
# #     plt.scatter(steps, residuals_values[f'{image}'])
# ax[0].scatter(steps, residual, color='orange')
# ax[0].set_xlabel('steps', fontsize=14)
# ax[0].set_ylabel('residual', fontsize=14)
# ax[0].set_title(f'Residual {residual.columns[0]}', fontsize=16)

# plt.show()



