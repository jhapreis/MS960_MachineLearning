import pandas as pd

df_image = pd.read_csv('Projeto1/data/imageMNIST.csv', header=None).T
df_image.index = ['px_'+str(i) for i in range(df_image.shape[0])]
df_image.columns = ['image_'+str(i) for i in range(df_image.shape[1])]

df_label = pd.read_csv('Projeto1/data/labelMNIST.csv', header=None)

df_image = df_image.T
df_image['label'] = df_label[0].values

df_sample = df_image.sample(frac=0.25).sort_index()

df_data = df_image.drop(index=df_sample.index)



df_sample.to_csv('Projeto1/data/exercicio_2/data_sample.csv')
df_data.to_csv('Projeto1/data/exercicio_2/data_image.csv')


