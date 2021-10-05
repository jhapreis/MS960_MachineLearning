import pandas as pd

import cfg


print('\n\nSplitting data and test csv files...\n')


'''
Read CSVs
'''
df_data  = pd.read_csv("../data/original/imageMNIST.csv", header=None).T
df_label = pd.read_csv("../data/original/labelMNIST.csv", header=None)

df_data.index   = ['pixel_'+str(i+1) for i in range(df_data.shape[0])]
df_data.columns = ['image_'+str(i+1) for i in range(df_data.shape[1])]

df_label.index   = df_data.columns
df_label.columns = ['value']



'''
Take a sample
'''
df_data_sample  = df_data.sample(frac=cfg.sample_fraction, random_state=cfg.random_state, axis=1).T.sort_index().T
df_label_sample = df_label.T[df_data_sample.columns].T.sort_index()



'''
Separate (or not)
'''
if cfg.separate_data == True:
    df_data  = df_data.drop(columns=df_data_sample.columns)
    df_label = df_label.T.drop(columns=df_data_sample.columns).T



'''
Save new CSV files
'''
df_data.to_csv("../data/test/images.csv")

df_data_sample.to_csv("../data/test/sample_images.csv")

df_label.to_csv("../data/test/labels.csv")

df_label_sample.to_csv("../data/test/sample_labels.csv")



print('   done!\n')
