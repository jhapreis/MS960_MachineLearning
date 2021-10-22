import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



# =============================================================================
def gen_image(image, classifications='false'):
    
    data = np.array( image ) 
    
    two_d = (np.reshape(data, (20, 20)) * (2**8 - 1)).T
    title = image.name
    if np.all(classifications) != 'false':
        title = f"{title}; orig={classifications['original']}, given={classifications['atribuido']}" 

    fig, ax = plt.subplots(1,1)

    ax.imshow(two_d, interpolation='nearest', cmap='gray')
    
    ax.set_title(title)
    ax.axis('off')
    
    return fig



# =============================================================================
def distribution_figs_per_page(number_of_figures, height=4, width=4):
    '''
    '''

    figures_per_page = height*width
    number_of_pages  = number_of_figures // figures_per_page
    remaining        = number_of_figures % figures_per_page

    _ = number_of_pages*[figures_per_page]
    _.append(remaining)

    return _



# =============================================================================
def multiple_gen_image(   images, classifications, nrows=4, ncols=4, figsize=(12,12)   ):

    figs = []

    images_per_pg = nrows*ncols
    quantidade_pg = (images.shape[1] // images_per_pg) + 1

    image_number = 0
    
    for n in range(quantidade_pg):

        fig, axarr = plt.subplots(nrows, ncols, figsize=figsize) # generate page of images

        for i in range(nrows): # fill that page with the figures

            for j in range(ncols):

                ax = axarr[i,j]

                if images.shape[1] > image_number: # if there is another image to plot

                    image = images[ images.columns[image_number] ]
                    classification = classifications.iloc[image_number]
                    data  = np.array( image ) 

                    title = image.name
                    title = f"{title}; orig={classification['original']}, given={classification['atribuido']}" 


                else: # else, plot a black figure with no title

                    data = np.zeros((20,20))
                    title = ''
            
                two_d = (np.reshape(data, (20, 20)) * (2**8 - 1)).T


                ax.imshow(two_d, interpolation='nearest', cmap='gray')
                
                plt.rcParams['text.color'] = 'white'
                ax.set_title(title)
                ax.axis('off')

                image_number += 1

        figs.append(fig)
    
    return figs



# =============================================================================
def plot_curva_aprendizado(cost_treino, cost_valid, max_tries, file="../data/results/curva_aprendizado.png"):


    sample_steps = cost_treino.loc['samples']
    cost_treino  = cost_treino.drop(labels='samples', axis=0)
    cost_valid   = cost_valid.drop(labels='samples', axis=0)

    if len(sample_steps) == 1:
        print("Only one value_frac. Cannot plot graph.\n")
        return -1

    fig, ax = plt.subplots(figsize=(10,6))

    ax.scatter(sample_steps, cost_treino.mean(axis=0), color='orange')
    ax.scatter(sample_steps, cost_valid.mean(axis=0) , color='blue'  )
    ax.plot(sample_steps, cost_treino.mean(axis=0), color='orange', label='Treino')
    ax.plot(sample_steps, cost_valid.mean(axis=0) , color='blue'  , label='Validação')

    ax.set_xlabel('Tamanho da amostra de treinamento', fontsize=14)
    ax.set_ylabel('Valor médio da função de custo', fontsize=14)
    ax.set_title(f'{cost_treino.shape[1]} tamanhos de amostra; {round(max_tries)} execuções por tamanho')
    plt.suptitle('Curva de aprendizado médio', fontsize=18)
    plt.legend()

    plt.savefig(file)

    return 0



# =============================================================================
def plot_optimize_lambda(cost_valid, title='', parameter='mean', file="../data/results/curva_lambdas.png"):

    if len(cost_valid.loc['lambdas']) == 1:
        print("Only one lambda_value. Cannot plot graph.\n")
        return -1

    '''
    Check +/- np.inf or negative cost and remove
    '''
    _ = np.any(cost_valid == np.inf, axis=0)
    if np.any(_) == True:
        indx = [_.index[i] for i in range(_.shape[0]) if _[i] == True]
        print(f"Deleting {indx} from cost_valid due to inf values.")
        cost_valid = cost_valid.drop(indx, axis=1)

    _ = np.any(cost_valid < 0      , axis=0)
    if np.any(_) == True:
        indx = [_.index[i] for i in range(_.shape[0]) if _[i] == True]
        print(f"Deleting {indx} from cost_valid due to negative values.")
        cost_valid = cost_valid.drop(indx, axis=1)

    lambdas     = cost_valid.loc['lambdas']
    cost_valid  = cost_valid.drop(labels='lambdas', axis=0)

    fig, ax = plt.subplots(figsize=(10,6))

    if parameter == 'max':
        ax.scatter(lambdas, cost_valid.max(axis=0), color='orange')
        ax.plot(   lambdas, cost_valid.max(axis=0), color='orange', label='Validação')
        type_metric = 'máximo'
    elif parameter == 'min':
        ax.scatter(lambdas, cost_valid.min(axis=0), color='orange')
        ax.plot(   lambdas, cost_valid.min(axis=0), color='orange', label='Validação')
        type_metric = 'mínimo'
    else:
        ax.scatter(lambdas, cost_valid.mean(axis=0), color='orange')
        ax.plot(   lambdas, cost_valid.mean(axis=0), color='orange', label='Validação')
        type_metric = 'médio'
    
    ax.set_xlabel(r'Valores do hiperparâmetro de regularização $\lambda$', fontsize=14)
    ax.set_ylabel(f'Valor {type_metric} da função de custo da validação', fontsize=14)
    ax.set_title(title)
    plt.suptitle('Otimização do parâmetro de regularização', fontsize=18)
    plt.legend()

    plt.savefig(file)

    return 0


