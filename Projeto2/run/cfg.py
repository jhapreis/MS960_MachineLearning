# NEURAL NETWORK
# =============================================================================

LABELS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] # official labels

NUMBER_OF_LAYERS  = 3 # start, middle(hidden), end
MULT_HIDDEN_LAYER = 0 # n times the start layer 
ADDITIONAL_LAYERS = 25#   

INIT_THETAS_RANGE = 1E-1 # range to initiate the thetas constants

LAMBDA_VALUE      = 1E-3 # lambda for the regularization

MAX_COST          = 1E-2      
MAX_TRIES         = int(1E4)  
LEARNING_RATE     = 1E-1 

TRACKING          = True
PRINT_FLAG        = 0.1*MAX_TRIES # every 10%, print
SEND_EMAIL        = False



# DATA SAMPLE
# =============================================================================

SAMPLE_FRACTION = 0.10 #
RANDOM_STATE    = None #
SEPARATE_DATA   = True # separate the data from test to further applications


