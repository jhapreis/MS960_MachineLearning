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
Take a sample for test and another for validation
    sample for training
    remove the training sample from original
    sample for validation
    revome validation sample from original (already removed training sample)
    the rest is the test sample
'''

df_data_treino  = df_data.sample(frac=cfg.SAMPLE_TREINO, random_state=cfg.RANDOM_STATE, axis=1).T.sort_index().T
df_label_treino = df_label.T[df_data_treino.columns].T.sort_index()

df_data  = df_data.drop(   columns=df_data_treino.columns)
df_label = df_label.T.drop(columns=df_data_treino.columns).T

df_data_valid  = df_data.sample(frac=cfg.SAMPLE_VALID, random_state=cfg.RANDOM_STATE, axis=1).T.sort_index().T
df_label_valid = df_label.T[df_data_valid.columns].T.sort_index()

df_data  = df_data.drop(   columns=df_data_valid.columns)
df_label = df_label.T.drop(columns=df_data_valid.columns).T

df_data_test  = df_data
df_label_test = df_label



'''
Save new CSV files
'''

training_folder   = "../data/training"
validation_folder = "../data/validation"
test_folder       = "../data/test"


df_data_treino.to_csv(f"{training_folder}/images.csv")
df_label_treino.to_csv(f"{training_folder}/labels.csv")

df_data_valid.to_csv(f"{validation_folder}/images.csv")
df_label_valid.to_csv(f"{validation_folder}/labels.csv")

df_data_test.to_csv(f"{test_folder}/images.csv")
df_label_test.to_csv(f"{test_folder}/labels.csv")



print('   done!\n')
