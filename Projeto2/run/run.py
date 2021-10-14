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
from Projeto2.neural_network.send_email import SendEmail

import cfg






'''
Beggining
'''
print("\n\n      Running neural network...\n")

time_start = time()

df_images = pd.read_csv("../data/test/sample_images.csv", index_col=0)
df_labels = pd.read_csv("../data/test/sample_labels.csv", index_col=0)

class_matrix = classification_matrix(df_labels, cfg.labels)
dimensions   = neural_net_dimension( df_images, class_matrix, cfg.number_of_layers, cfg.mult_hidden_layer, cfg.additional_layers ) # without bias
thetas       = thetas_layers(dimensions, limit=cfg.init_thetas_range)

msg_dim = f"\n   Dimensions: {dimensions}\n"
print(msg_dim)



'''
First roll
'''
activations = activation_layer(df_images, class_matrix, thetas)
grad        = gradient_layer(class_matrix, activations, thetas, lambda_value=cfg.lambda_value)
cost        = cost_function_sigmoid(activations, class_matrix)



'''
While loop
'''
tries       = 1
total_costs = pd.DataFrame()

while (   tries <= cfg.max_tries   ) and (   np.all(cost > cfg.max_cost)   ):

    total_costs[f"{tries}"] = cost

    thetas      = update_thetas(thetas, grad, learning_rate=cfg.learning_rate)

    activations = activation_layer(df_images, class_matrix, thetas)
    grad        = gradient_layer(class_matrix, activations, thetas, lambda_value=cfg.lambda_value)
    cost        = cost_function_sigmoid(activations, class_matrix)

    if (cfg.tracking == True) and (tries % cfg.flag == 0):
        print(   f"{tries}/{cfg.max_tries}\n"   )   

    tries += 1


if (tries > cfg.max_tries):
    msg_result = f"\n\n   Number of tries exceeded (> {cfg.max_tries}).\n\n"
elif ( np.all(cost <= cfg.max_cost) ):
    msg_result = f"\n\n   Success! After {tries} trie(s) (<= {cfg.max_tries}), the costs are now under {cfg.max_cost} \n\n"
else:
    msg_result = "\n\n   No conditional\n\n"
print(msg_result)


'''
Time elapsed
'''
time_end = time()
msg_time = f"\n\n      Done. Finished after {timedelta(seconds = time_end - time_start)}. \n\n"
print(msg_time)



'''
Final
'''
msg_cost = f"\n\n   Valor de custo final:\n\n{cost}\n\n"
print(msg_cost)

total_costs.to_csv("../data/results/costs.csv")

for i in range(len(thetas)):
    thetas[i].to_csv(f"../data/results/thetas_{i+1}{i+2}.csv")


if cfg.send_email == True:
    subject = "[MS_960] Projeto 2"
    msg     = f"{msg_dim}{msg_result}{msg_time}{msg_cost}"
    SendEmail(subject, msg)


