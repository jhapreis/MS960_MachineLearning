import sys
from pathlib import Path
import numpy as np
import pandas as pd
from time import time
from datetime import timedelta
from progress.bar import IncrementalBar



_ = Path().resolve().parent.parent # Add [...]\MS960_MachineLearning\Projeto2 to PYTHONPATH
sys.path.insert(0, str(_))

from Projeto2.neural_network.plots import plot_curva_aprendizado, plot_optimize_lambda
from Projeto2.neural_network.neural import BackPropagation_NeuralNetwork
from Projeto2.neural_network.midders import *
from Projeto2.neural_network.auxiliars import *
from Projeto2.neural_network.errors import *
from Projeto2.neural_network.SendEmail_AWS import SendEmailAWS, CONFIG_FILE_EMAILS

import cfg



'''
Type of run
'''

msg = '' # message to send email, if requested

if cfg.OPTIMIZE_LAMBDA == True:
    msg_run = f"This is a run for OPTIMIZE_LAMBDA.\n"
    print(msg_run)
    msg += msg_run
    
elif cfg.CURVA_APRENDIZADO == True:
    msg_run = f"This is a run for CURVA_APRENDIZADO.\n"
    print(msg_run)
    msg += msg_run

elif cfg.SAVE_THETAS == True:
    msg_run = f"This is a run for SAVE_THETAS.\n"
    print(msg_run)
    msg += msg_run

else:
    msg_run = f"Not defined type of run.\n"
    print(msg_run)
    msg += msg_run



'''
Beggining
'''
time_start = time()

training_folder  = "../data/training"
valid_folder     = "../data/validation"

df_images_training = pd.read_csv(f"{training_folder}/images.csv", index_col=0)
df_labels_training = pd.read_csv(f"{training_folder}/labels.csv", index_col=0)

df_images_valid = pd.read_csv(f"{valid_folder}/images.csv", index_col=0)
df_labels_valid = pd.read_csv(f"{valid_folder}/labels.csv", index_col=0)

msg_start = f"\n\nRunning neural network... Training with {df_images_training.shape[1]} images.\n\n"
print(msg_start)
msg += msg_start




'''
While loop for the Backpropagation Execution
'''
# total_costs = pd.DataFrame()
cost_treino   = pd.DataFrame()
cost_valid    = pd.DataFrame()


bar_lambdas     = False
bar_frac_treino = False
if cfg.TRACKING == True:
    bar_lambdas     = IncrementalBar('      lambdas', max = len(cfg.LAMBDA_VALUE)     , suffix='%(percent).1f%%')
    bar_frac_treino = IncrementalBar('   fracs'     , max = len(cfg.VALUE_FRAC_TREINO), suffix='%(percent).1f%%')



Check_ValuesFrac_Lambdas(cfg.VALUE_FRAC_TREINO, cfg.LAMBDA_VALUE) # check if both loops are going to be executed; intertravamento



for frac_values in cfg.VALUE_FRAC_TREINO:


    images = df_images_training.sample(frac=frac_values, random_state=cfg.RANDOM_STATE, axis=1).T.sort_index().T
    labels = df_labels_training.T[images.columns].T.sort_index()


    for lambda_value in cfg.LAMBDA_VALUE:


        thetas, cost, msg = BackPropagation_NeuralNetwork(
            df_images          = images,
            df_labels          = labels,
            init_thetas_range  = cfg.INIT_THETAS_RANGE,
            number_of_layers   = cfg.NUMBER_OF_LAYERS,
            mult_hidden_layers = cfg.MULT_HIDDEN_LAYER,
            aditional_layers   = cfg.ADDITIONAL_LAYERS,
            orig_labels        = cfg.LABELS,
            max_tries          = cfg.MAX_TRIES,
            max_cost           = cfg.MAX_COST,
            lambda_value       = lambda_value,
            learning_rate      = cfg.LEARNING_RATE,
            msg                = msg
        )
        
        
        '''
        use the thetas from the training result to calculate the cost of the validation
        '''
        class_matrix_validation = classification_matrix(df_labels_valid, cfg.LABELS)
        activation_validation   = activation_layer(df_images_valid, class_matrix_validation, thetas) 
        _ = cost_function_sigmoid(activation_validation, class_matrix_validation, thetas, lambda_value=lambda_value)

        cost_valid  = pd.concat([cost_valid, _], axis=1)
        cost_treino = pd.concat([cost_treino, cost], axis=1)


        if bar_lambdas:
            # print('\n')
            bar_lambdas.next()
            print('\n')
    

    if bar_frac_treino:
        print('\n\n')
        bar_frac_treino.next()
        print('\n')


cost_treino.columns = [str(i+1) for i in range(cost_treino.shape[1])]
cost_valid.columns  = [str(i+1) for i in range(cost_valid.shape[1]) ]



'''
Time elapsed
'''
time_end = time()
msg_time = f"\n\nDone. Finished after {timedelta(seconds = time_end - time_start)}.\n"
print(msg_time)
msg += msg_time



'''
Final
'''
msg_cost = f"   \n\n\nValor de custo final:\n\n{cost}\n"
print(msg_cost)
msg += msg_cost

if (cfg.CURVA_APRENDIZADO == True) or (cfg.OPTIMIZE_LAMBDA == True):

    cost_treino.to_csv(f"{training_folder}/results/cost.csv")
    cost_valid.to_csv(f"{valid_folder}/results/cost.csv")

    if cfg.CURVA_APRENDIZADO == True:
        plot_curva_aprendizado(cost_treino, cost_valid, df_images_training.shape[1], cfg.VALUE_FRAC_TREINO, cfg.MAX_TRIES)
    elif cfg.OPTIMIZE_LAMBDA == True:
        plot_optimize_lambda(cfg.LAMBDA_VALUE, cost_valid, cfg.MAX_TRIES)

elif cfg.SAVE_THETAS == True:
    for i in range(len(thetas)):
        thetas[i].to_csv(f"{training_folder}/results/thetas_{i+1}{i+2}.csv")



'''
Send Email
'''
if (cfg.SEND_EMAIL == True) and (CONFIG_FILE_EMAILS == True):
    subject = "[MS_960] Projeto 2"
    # msg     = f"{msg_dim}{msg_result}{msg_time}{msg_cost}"
    SendEmailAWS(subject, msg)


