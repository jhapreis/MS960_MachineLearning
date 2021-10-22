import sys
from pathlib import Path
import numpy as np
import pandas as pd
from time import time
from datetime import timedelta
from progress.bar import IncrementalBar

import concurrent.futures

_ = Path().resolve().parent.parent # Add [...]\MS960_MachineLearning\Projeto2 to PYTHONPATH
sys.path.insert(0, str(_))

from Projeto2.neural_network.plots         import plot_optimize_lambda, plot_curva_aprendizado
from Projeto2.neural_network.neural        import BackPropagation_NeuralNetwork, BackPropagation_CalculateValidation, BackPropagation_CurvaAprendizado
from Projeto2.neural_network.midders       import *
from Projeto2.neural_network.errors        import *
from Projeto2.neural_network.SendEmail_AWS import SendEmailAWS, CONFIG_FILE_EMAILS

import cfg



"""
PLEASE, PAY ATTENTION: THIS SCRIPT WILL ASK REQUEST, FROM YOUR OPERATIONAL SYSTEM, MULTIPLE INSTANCES OF PYTHON (MULTIPROCESSING)
BE AWARE OF THAT

General structure for the script:

    1) Type of run

    2) Beggining

    3) Progress bar and DataFrames to store data

    4) Backpropagation:

        - for OPTMIZE_LAMBDA
            Once the unique change is lambda_value, it's worth it to calculate all preliminars before
            Backpropagation
            After that, use the thetas from the training result to calculate the cost of the validation
        
        - for CURVA_APRENIZADO
            It runs a backpropagation for every percentage of sample on the df_training
            Backpropagation
            After that, use the thetas from the training result to calculate the cost of the validation
    
    5) Time elapsed

    6) Final

    7) Save CSVs and plot graphics

    8) Send Email
"""





if __name__ == '__main__':

    with concurrent.futures.ProcessPoolExecutor() as executor:

        '''
        1. Type of run and configs
        '''
        if cfg.SEND_EMAIL == True:
            msg = '' # message to send email, if requested
        else:
            msg = False  
        if cfg.OPTIMIZE_LAMBDA == True:

            res_folder = cfg.OPTLAMBDA_FOLDER 
            msg_run = f"This is a run for OPTIMIZE_LAMBDA.\n"
            print(msg_run)

            lambda_values = []

            if cfg.TRACKING == True:
                bar_lambdas = IncrementalBar('      lambdas', max = len(cfg.LAMBDA_VALUE), suffix='%(percent).1f%%')
            else:
                bar_lambdas = False

            if msg != False:
                msg += msg_run      
        elif cfg.CURVA_APRENDIZADO == True:

            res_folder = cfg.CURVA_APRENDIZADO_FOLDER 
            msg_run = f"This is a run for CURVA_APRENDIZADO.\n"
            print(msg_run)

            value_fracs   = []

            if cfg.TRACKING == True:
                bar_aprend  = IncrementalBar('      aprendizado', max = len(cfg.CURVA_APRENDIZADO), suffix='%(percent).1f%%')
            else:
                bar_aprend = False
            if msg != False:
                msg += msg_run      
        elif cfg.SAVE_THETAS == True:

            res_folder = cfg.SAVETHETAS_FOLDER

            msg_run = f"This is a run for SAVE_THETAS.\n"
            print(msg_run)

            lambda_values = []   
            
            if cfg.TRACKING == True:
                bar_lambdas = IncrementalBar('      lambdas', max = len(cfg.LAMBDA_VALUE), suffix='%(percent).1f%%')
            else:
                bar_lambdas = False

            if msg != False:
                msg += msg_run  
        else:
            res_folder = cfg.OTHERS_FOLDER




        '''
        2. Beggining: read data and normalize
        '''
        time_start = time()

        df_images_training = pd.read_csv(f"{cfg.TRAINING_FOLDER}/images.csv"  , index_col=0)
        # df_images_training = normalize_data(df_images_training)
        df_labels_training = pd.read_csv(f"{cfg.TRAINING_FOLDER}/labels.csv"  , index_col=0)

        df_images_valid    = pd.read_csv(f"{cfg.VALIDATION_FOLDER}/images.csv", index_col=0)
        # df_images_valid    = normalize_data(df_images_valid)
        df_labels_valid    = pd.read_csv(f"{cfg.VALIDATION_FOLDER}/labels.csv", index_col=0)

        msg_start = f"\n\nRunning neural network... Training with {df_images_training.shape[1]} images.\n\n"
        print(msg_start)
        if msg != False:
            msg += msg_start




        '''
        3. DataFrames to store the results data
        '''
        cost_treino         = pd.DataFrame()
        cost_valid          = pd.DataFrame()
        thetas_all          = [] # stores all thetas that results from the Neural Network
        cost_steps_all      = [] # stores all training cost and every step
        num_grad_error      = []
        backprop_grad_error = []
        flag_grad_error     = []
        

        
            


        '''
        Multiple executor: PLEASE, NOTICE THAT THE ORDER OF EXECUTION IS NOT NECESSARILY THE SAME AS IT APPEARS
                           SO, YOU SHOULD KEEP TRACK OF LAMBDA_VALUES AND/OR FRAC_VALUES, IN ORDER TO MATCH THE RESULTS
        '''

        if (cfg.OPTIMIZE_LAMBDA == True) or (cfg.SAVE_THETAS == True):

            images = df_images_training.sample(frac=cfg.VALUE_FRAC_TREINO[0], random_state=cfg.RANDOM_STATE, axis=1).T.sort_index().T
            labels = df_labels_training.T[images.columns].T.sort_index()

            class_matrix_training = classification_matrix(labels, cfg.LABELS)
            dimensions            = neural_net_dimension(images, class_matrix_training, cfg.NUMBER_OF_LAYERS, cfg.MULT_HIDDEN_LAYER, cfg.ADDITIONAL_LAYERS) # without bias
            init_thetas           = thetas_layers(dimensions, limit=cfg.INIT_THETAS_RANGE)


            msg_dim = f"      Dimensions: {dimensions}... {images.shape[1]} images.\n"
            print(msg_dim)
            if msg != False:
                msg += msg_dim


            results = [ 
                executor.submit(
                    BackPropagation_CalculateValidation,
                    images,                # df_training
                    df_images_valid,       # df_valid
                    cfg.LABELS,            # orig_labels
                    df_labels_valid,       # label_valid
                    class_matrix_training, # class_matrix_training
                    dimensions,            # dimensions
                    init_thetas,           # thetas
                    cfg.MAX_TRIES,         # max_tries
                    cfg.MAX_COST,          # max_cost
                    _,                     # lambda_value
                    cfg.LEARNING_RATE,     # learning_rate
                    msg,                   # msg
                    bar_lambdas,           # progress_bar=False
                    cfg.COST_MEASUREMENT   # cost_metric='any'
                )   

                for _ in cfg.LAMBDA_VALUE
            ]



        elif cfg.CURVA_APRENDIZADO == True:
            
            dimensions = [
                df_images_training.shape[0], 
                cfg.MULT_HIDDEN_LAYER*df_images_training.shape[0]+cfg.ADDITIONAL_LAYERS,
                len(cfg.LABELS)
                ]
            msg_dim = f"      Dimensions: {dimensions}... max of {df_images_training.shape[1]} images.\n"
            print(msg_dim)
            if msg != False:
                msg += msg_dim
            
            results = [ 
                executor.submit(
                    BackPropagation_CurvaAprendizado,
                    _,                     # value_frac_treino
                    df_images_training,    # df_training
                    df_labels_training,    # df_label_training
                    df_images_valid,       # df_valid
                    df_labels_valid,       # df_label_valid
                    cfg.LABELS,            # orig_labels
                    cfg.INIT_THETAS_RANGE, # thetas_range, 
                    cfg.MAX_TRIES,         # max_tries,
                    cfg.MAX_COST,          # max_cost, 
                    cfg.LAMBDA_VALUE[0],   # lambda_value,
                    cfg.LEARNING_RATE,     # learning_rate,
                    msg,                   # msg,
                    bar_aprend,            # progress_bar=False,
                    cfg.RANDOM_STATE,      # random_state=None,
                    cfg.NUMBER_OF_LAYERS,  # number_of_layers=3,
                    cfg.MULT_HIDDEN_LAYER, # mult_hidden_layer=0,
                    cfg.ADDITIONAL_LAYERS, # additional_neurons=25
                    cfg.COST_MEASUREMENT   # cost_metric='any'  
                )

                for _ in cfg.VALUE_FRAC_TREINO
            ]



        for f in concurrent.futures.as_completed(results): 
            """
            Erro no teste do gradiente:
                f.results():
                    for OPTIMIZE_LAMBDA:
                        - 0 = status (0,1,-1)
                        - 1 = numerical_grad
                        - 2 = gradient[-1]
                        - 3 = lambda_value
                    for CURVA_APRENDIZADO:
                        - 0 = status (0,1,-1)
                        - 1 = numerical_grad
                        - 2 = gradient[-1]
                        - 3 = value_frac
                    for SAVE_THETAS:
                        - 0 = 
                        - 1 = 
                        - 2 = 
                        - 3 = 

            Aprovado no teste do gradiente:
                f.results():
                    for OPTIMIZE_LAMBDA:
                        - 0 = thetas
                        - 1 = cost_steps
                        - 2 = cost_validation
                        - 3 = lambda_value
                        - 4 = msg
                    for CURVA_APRENDIZADO:
                        - 0 = thetas
                        - 1 = cost_steps
                        - 2 = cost_validation
                        - 3 = value_frac
                        - 4 = msg
                    for SAVE_THETAS:
                        - 0 = 
                        - 1 = 
                        - 2 = 
                        - 3 = 
                        - 4 = 
            """

            if f.result()[0] == 1: # erro no teste do gradiente
                num_grad_error.append(f.result()[1])
                backprop_grad_error.append(f.result()[2])
                flag_grad_error.append(f.result()[3])

            else:
                thetas_all.append(    f.result()[0])
                cost_steps_all.append(f.result()[1])

                cost_treino = pd.concat([cost_treino, f.result()[1][f.result()[1].columns[-1]]], axis=1)
                cost_valid  = pd.concat([cost_valid , f.result()[2]], axis=1)

                if cfg.OPTIMIZE_LAMBDA == True:
                    lambda_values.append(f.result()[3])
                elif cfg.CURVA_APRENDIZADO == True:
                    value_fracs.append(  f.result()[3])

                msg += f.result()[4]

        columns_labels      = [str(i+1) for i in range(cost_treino.shape[1])]
        cost_treino.columns = columns_labels
        cost_valid.columns  = columns_labels




        if cfg.OPTIMIZE_LAMBDA: # add DataFrame for lambdas on costs

            lambda_values = pd.DataFrame(lambda_values, columns=['lambdas'], index=columns_labels)

            cost_treino   = pd.concat([cost_treino, lambda_values.T], axis=0)
            cost_treino.sort_values(by='lambdas', axis='columns', inplace=True)

            cost_valid    = pd.concat([cost_valid, lambda_values.T] , axis=0)
            cost_valid.sort_values(by='lambdas', axis='columns', inplace=True)

            cost_treino.to_csv(f"{res_folder}/cost_treino.csv")
            cost_valid.to_csv(f"{res_folder}/cost_valid.csv")

            plot_optimize_lambda(
                cost_valid=cost_valid,
                title=f'{len(cost_valid.loc["lambdas"])} lambdas; {round(cfg.MAX_TRIES)} execuções por tamanho; {images.shape[1]} imagens', 
                file=f"{cfg.OPTLAMBDA_FOLDER}/lambdas.png"
                )


        elif cfg.SAVE_THETAS:

            if ( len(thetas_all) > 1) or (len(cost_steps_all) > 1):
                raise ValueError("It's supposed to only have a single thetas among thetas_all when running SAVE_THETAS.\n")

            thetas_all     = thetas_all[0]
            cost_steps_all = cost_steps_all[0]

            cost_treino.to_csv(f"{res_folder}/cost_treino.csv")
            cost_valid.to_csv(f"{res_folder}/cost_valid.csv")
            cost_steps_all.to_csv(f"{res_folder}/cost_steps.csv")


            for i in range(   len(thetas_all)   ): 
                thetas_all[i].to_csv(f"{res_folder}/thetas_{i+1}_{i+2}.csv")



        elif cfg.CURVA_APRENDIZADO: # add DataFrame for value_fracs on costs

            value_fracs   = df_images_training.shape[1]*np.array(value_fracs)
            value_samples = pd.DataFrame(value_fracs, columns=['samples'], index=columns_labels)

            cost_treino = pd.concat([cost_treino, value_samples.T], axis=0)
            cost_treino.sort_values(by='samples', axis='columns', inplace=True)

            cost_valid  = pd.concat([cost_valid, value_samples.T] , axis=0)
            cost_valid.sort_values(by='samples', axis='columns', inplace=True)

            cost_treino.to_csv(f"{res_folder}/cost_treino.csv")
            cost_valid.to_csv(f"{res_folder}/cost_valid.csv")

            plot_curva_aprendizado(
                cost_treino=cost_treino,
                cost_valid=cost_valid,
                max_tries=cfg.MAX_TRIES,
                file=f"{cfg.CURVA_APRENDIZADO_FOLDER}/aprendizado.png"
                )        
            





        '''
        5. Time elapsed
        '''
        time_end = time()
        msg_time = f"\n\nDone. Finished after {timedelta(seconds = time_end - time_start)}.\n"
        print(msg_time)
        if msg != False:
            msg += msg_time


        '''
        8. Send Email
        '''
        if (cfg.SEND_EMAIL == True) and (CONFIG_FILE_EMAILS == True):
            subject = "[MS_960] Projeto 2"
            # msg     = f"{msg_dim}{msg_result}{msg_time}{msg_cost}"
            SendEmailAWS(subject, msg)


