import pandas as pd

from gradient_descendent import Gradient_Descendent_Exponential
 


df_casos_covid = pd.read_csv('data/casesBrazil.csv')

coefficients, residuals_values = Gradient_Descendent_Exponential(
    x_data_init=df_casos_covid['day'], 
    y_data_init=df_casos_covid['cases'], 
    initial_guess=[1,1],
    learning_rate=1E-1,
    min_residual=1E-3,
    max_tries=1E4 
    )

print(coefficients, '\n\n')


