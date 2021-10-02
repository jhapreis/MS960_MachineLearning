import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



def gen_image(data, label, bit=8):
    
    image_number = data.columns[0]
    name = label.loc[image_number][0]
    # print(name)
    
    data = np.array( data ) 
    
    two_d = (np.reshape(data, (20, 20)) * (2**bit - 1)).T
    plt.imshow(two_d, interpolation='nearest', cmap='gray')
    plt.title(name)
    
    return plt


