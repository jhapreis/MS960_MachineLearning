import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def deg_1(x, thetas):

    y = thetas[0] + thetas[1]*x[1]
    return(y)

def deg_2(x, thetas):
    
    y = thetas[0] + thetas[1]*x[1] + thetas[2]*(x[2]**2)

    return(y)


def function(func, x, thetas):

    y = func(x, thetas)

    return(y)


x = [1, 2, 3]
thetas = [2, 3, 4]

print(function(x=x, thetas=thetas, func=deg_1))