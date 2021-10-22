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
MAX_TRIES         = 1E2 
LEARNING_RATE     = 1E-1 
COST_MEASUREMENT  = 'any'

SEND_EMAIL        = False
TRACKING          = False

"""SAVE_THETAS"""
SAVE_THETAS       = True
CURVA_APRENDIZADO = False
OPTIMIZE_LAMBDA   = False
LAMBDA_VALUE      = [0] 
VALUE_FRAC_TREINO = [1.] 

"""CURVA_APRENDIZADO"""
# SAVE_THETAS       = False
# CURVA_APRENDIZADO = True
# OPTIMIZE_LAMBDA   = False
# NUMBER_VALUE_FRAC = 10
# VALUE_FRAC_TREINO = np.arange(0.1, 1.1, 1/NUMBER_VALUE_FRAC) # from 10% to 100% of the sample_treino, with given step
# LAMBDA_VALUE      = [1.] # lambda for the regularization

"""OPTMIZE_LAMBDA"""
# SAVE_THETAS       = False
# CURVA_APRENDIZADO = False
# OPTIMIZE_LAMBDA   = True
# VALUE_FRAC_TREINO = [1.]
# LAMBDA_VALUE      = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1., 1.2, 1.5, 1.7, 1.9, 2.5, 3., 10.]






# =============================================================================
# NEURAL NETWORK
# =============================================================================

TRAINING_FOLDER          = "../data/training"
VALIDATION_FOLDER        = "../data/validation"
OPTLAMBDA_FOLDER         = "../data/results/lambda_opt"
CURVA_APRENDIZADO_FOLDER = "../data/results/curva_aprendizado"
SAVETHETAS_FOLDER        = "../data/results/save_thetas"
OTHERS_FOLDER            = "../data/results/others"




# =============================================================================
# ERROR HANDLING ON PARAMETERS COMBINATION
# =============================================================================

if (CURVA_APRENDIZADO == True) and (OPTIMIZE_LAMBDA == True):
    print(f"    You are trying to run CURVA_APRENDIZADO and LAMBDA_VALUE. Choose only one kind.\n")
    sys.exit(-1)

elif (CURVA_APRENDIZADO == True) and (len(LAMBDA_VALUE) > 1):
    print(f"    You are trying to run CURVA_APRENDIZADO, but we found that there is more than one LAMBDA_VALUE (actually, {len(LAMBDA_VALUE)}). Choose only one kind.\n")
    sys.exit(-1)

elif (len(VALUE_FRAC_TREINO) > 1) and (OPTIMIZE_LAMBDA == True):
    print(f"    You are trying to run OPTIMIZE_LAMBDA, but we found that there is more than one VALUE_FRAC_TREINO (actually, {len(VALUE_FRAC_TREINO)}). Choose only one kind.\n")
    sys.exit(-1)


if (OPTIMIZE_LAMBDA + CURVA_APRENDIZADO + SAVE_THETAS > 1):
    print(f"    There's was attempt to run more then one kind at the same time. Please, change the cfg configs.")
    sys.exit(-1)

if (SAVE_THETAS == True and len(VALUE_FRAC_TREINO) > 1) or (SAVE_THETAS == True and len(LAMBDA_VALUE) > 1):
    print(f"    There's was attempt to run more then one kind at the same time, cause the lists have len > 1. Please, change the cfg configs.")
    sys.exit(-1)








# =============================================================================
# DATA SAMPLE
# =============================================================================

SAMPLE_TREINO = 0.60 #
SAMPLE_VALID  = 0.20 #
SAMPLE_TESTE  = 0.20 #
RANDOM_STATE  = None #
SEPARATE_DATA = True # separate the data from test to further applications; not implemented



SAMPLE_VALID = SAMPLE_VALID / (1 - SAMPLE_TREINO) # remove training, we need to update the percentage


