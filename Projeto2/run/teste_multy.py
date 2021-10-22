import numpy as np
import pandas as pd
import time

import concurrent.futures



def function(x, y, z):
    time.sleep(2)
    return x + y + z

def function_2(x, y, z):
    time.sleep(2)
    return (x, y, z)


start = time.perf_counter()

x = np.arange(0,  10,    1)
y = np.arange(0,   1,  0.1)
z = np.arange(0, 0.1, 0.01)




if __name__ == '__main__': # main program

    a = 0
    a += 1
    print("teste")
    
    with concurrent.futures.ProcessPoolExecutor() as executor:

        results = [ executor.submit(function_2, x[i], y[i], z[i]) for i in range(len(x)) ]

        for f in concurrent.futures.as_completed(results):
            print(f.result()[0])



    finish = time.perf_counter()

    print(f'Finished in {round(finish-start, 2)} second(s)')
