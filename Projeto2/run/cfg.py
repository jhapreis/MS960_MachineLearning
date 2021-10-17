import numpy as np
import sys



# =============================================================================
# NEURAL NETWORK
# =============================================================================

LABELS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] # official labels

NUMBER_OF_LAYERS  = 3 # start, middle(hidden), end
MULT_HIDDEN_LAYER = 0 # n times the start layer 
ADDITIONAL_LAYERS = 25 #   

INIT_THETAS_RANGE = 1E-4 # range to initiate the thetas constants

MAX_COST          = 5E-2      
MAX_TRIES         = 1E3  
LEARNING_RATE     = 5E-1 

# LAMBDA_VALUE      = [1E-1] # lambda for the regularization
LAMBDA_VALUE      = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]

NUMBER_VALUE_FRAC = 10
# VALUE_FRAC_TREINO = np.arange(0.1, 1.1, 1/NUMBER_VALUE_FRAC) # from 10% to 100% of the sample_treino, with given step
VALUE_FRAC_TREINO = [1]


SAVE_THETAS       = False
CURVA_APRENDIZADO = False
OPTIMIZE_LAMBDA   = True
SEND_EMAIL        = True
TRACKING          = True






# =============================================================================
# ERROR HANDLING ON PARAMETERS COMBINATION
# =============================================================================

if (CURVA_APRENDIZADO == True) and (OPTIMIZE_LAMBDA == True):
    print(f"    You cannot run both CURVA_APRENDIZADO e OPTIMIZE_LAMBDA as True at the same time. Choose one.\n")
    sys.exit()

elif (CURVA_APRENDIZADO == True) and (len(LAMBDA_VALUE) > 1):
    print(f"    You are trying to run CURVA_APRENDIZADO, but we found that there is more than one LAMBDA_VALUE (actually, {len(LAMBDA_VALUE)}). Choose only one kind.\n")
    sys.exit()

elif (len(VALUE_FRAC_TREINO) > 1) and (OPTIMIZE_LAMBDA == True):
    print(f"    You are trying to run OPTIMIZE_LAMBDA, but we found that there is more than one VALUE_FRAC_TREINO (actually, {len(VALUE_FRAC_TREINO)}). Choose only one kind.\n")
    sys.exit()

if (OPTIMIZE_LAMBDA == True) or (CURVA_APRENDIZADO == True):
    if SAVE_THETAS == True:
        print("Doesn't allow SAVE_THETAS if OPTIMIZE_LAMBDA or if trying to generate CURVA_APRENDIZADO\n")
        SAVE_THETAS = False









# =============================================================================
# DATA SAMPLE
# =============================================================================

SAMPLE_TREINO = 0.60 #
SAMPLE_VALID  = 0.20 #
SAMPLE_TESTE  = 0.20 #
RANDOM_STATE  = None #
SEPARATE_DATA = True # separate the data from test to further applications; not implemented



SAMPLE_VALID = SAMPLE_VALID / (1 - SAMPLE_TREINO) # remove training, we need to update the percentage


