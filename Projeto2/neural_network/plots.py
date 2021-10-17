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

    # fig, axarr = plt.subplots(nrows, ncols, figsize=figsize)
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
def plot_curva_aprendizado(cost_treino, cost_valid, sample_size, value_frac_treino, max_tries, folder="../data/results/curva_aprendizado.png"):

    x = sample_size*value_frac_treino

    fig, ax = plt.subplots(figsize=(10,6))

    ax.plot(x, cost_treino.mean(axis=0), color='orange', label='Treino')
    ax.plot(x, cost_valid.mean(axis=0) , color='blue'  , label='Validação')

    ax.set_xlabel('Tamanho da amostra de treinamento', fontsize=14)
    ax.set_ylabel('Valor médio da função de custo', fontsize=14)
    ax.set_title(f'{len(value_frac_treino)} tamanhos de amostra; {round(max_tries)} execuções por tamanho')
    plt.suptitle('Curva de aprendizado médio', fontsize=18)
    # plt.xticks(np.arange(0,3500,500))
    plt.legend()

    plt.savefig(folder)

    # plt.show()



# =============================================================================
def plot_optimize_lambda(lambdas, cost_valid, max_tries, folder="../data/results/curva_lambdas.png"):
    
    fig, ax = plt.subplots(figsize=(10,6))

    ax.plot(lambdas, cost_valid.mean(axis=0) , color='grey', label='Validação')

    ax.set_xlabel(r'Valores do hiperparâmetro de regularização $\lambda$', fontsize=14)
    ax.set_ylabel('Valor médio da função de custo da validação', fontsize=14)
    ax.set_title(f'{len(lambdas)} lambdas; {round(max_tries)} execuções por tamanho')
    plt.suptitle('Otimização do parâmetro de regularização', fontsize=18)
    # plt.xticks(np.arange(0,3500,500))
    plt.legend()

    plt.savefig(folder)


