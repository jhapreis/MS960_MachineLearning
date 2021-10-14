import sys
from pathlib import Path
import numpy as np
import pandas as pd
from time import time
from datetime import timedelta

_ = Path().resolve().parent.parent # Add [...]\MS960_MachineLearning\Projeto2\run to PYTHONPATH
sys.path.insert(0, str(_))

from Projeto2.neural_network.neural import *
from Projeto2.neural_network.auxiliars import *
from Projeto2.neural_network.errors import *
from Projeto2.neural_network.SendEmail_AWS import SendEmailAWS, CONFIG_FILE_EMAILS

import cfg






'''
Beggining
'''
print("\n\n      Running neural network...\n")

time_start = time()

df_images = pd.read_csv("../data/test/sample_images.csv", index_col=0)
df_labels = pd.read_csv("../data/test/sample_labels.csv", index_col=0)

class_matrix = classification_matrix(df_labels, cfg.LABELS)
dimensions   = neural_net_dimension( df_images, class_matrix, cfg.NUMBER_OF_LAYERS, cfg.MULT_HIDDEN_LAYER, cfg.ADDITIONAL_LAYERS ) # without bias
thetas       = thetas_layers(dimensions, limit=cfg.INIT_THETAS_RANGE)

msg_dim = f"\n   Dimensions: {dimensions}\n"
print(msg_dim)



'''
First roll
'''
activations = activation_layer(df_images, class_matrix, thetas)
grad        = gradient_layer(class_matrix, activations, thetas, lambda_value=cfg.LAMBDA_VALUE)
cost        = cost_function_sigmoid(activations, class_matrix)



'''
While loop
'''
tries       = 1
total_costs = pd.DataFrame()

while (   tries <= cfg.MAX_TRIES   ) and (   np.all(cost > cfg.MAX_COST)   ):

    total_costs[f"{tries}"] = cost

    thetas      = update_thetas(thetas, grad, learning_rate=cfg.LEARNING_RATE)

    activations = activation_layer(df_images, class_matrix, thetas)
    grad        = gradient_layer(class_matrix, activations, thetas, lambda_value=cfg.LAMBDA_VALUE)
    cost        = cost_function_sigmoid(activations, class_matrix)

    if (cfg.TRACKING == True) and (tries % cfg.PRINT_FLAG == 0):
        print(   f"{tries}/{cfg.MAX_TRIES}\n"   )   

    tries += 1


if (tries > cfg.MAX_TRIES):
    msg_result = f"\n   Number of tries exceeded (> {cfg.MAX_TRIES}).\n"
elif ( np.all(cost <= cfg.MAX_COST) ):
    msg_result = f"\n   Success! After {tries} trie(s) (<= {cfg.MAX_TRIES}), the costs are now under {cfg.MAX_COST} \n"
else:
    msg_result = "\n   No conditional\n"
print(msg_result)


'''
Time elapsed
'''
time_end = time()
msg_time = f"\n      Done. Finished after {timedelta(seconds = time_end - time_start)}. \n"
print(msg_time)



'''
Final
'''
msg_cost = f"   Valor de custo final:\n\n{cost}\n"
print(msg_cost)

total_costs.to_csv("../data/results/costs.csv")

for i in range(len(thetas)):
    thetas[i].to_csv(f"../data/results/thetas_{i+1}{i+2}.csv")


if (cfg.SEND_EMAIL == True) and (CONFIG_FILE_EMAILS == True):
    subject = "[MS_960] Projeto 2"
    msg     = f"{msg_dim}{msg_result}{msg_time}{msg_cost}"
    SendEmailAWS(subject, msg)


