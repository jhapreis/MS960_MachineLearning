# NEURAL NETWORK
# =============================================================================

labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] # official labels

number_of_layers  = 3 # start, middle(hidden), end
mult_hidden_layer = 0 # n times the start layer 
additional_layers = 25 #   

init_thetas_range = 1E-1 # range to initiate the thetas constants

lambda_value      = 1E-1 # lambda for the regularization

max_cost          = 1E-2      
max_tries         = int(5E0)  
learning_rate     = 1E-1 

tracking          = True
flag              = 0.1*max_tries # every 10%, print
send_email        = True



# DATA SAMPLE
# =============================================================================

sample_fraction = 0.10 #
random_state    = None #
separate_data   = True # separate the data from test to further applications


