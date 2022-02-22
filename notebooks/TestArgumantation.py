from numba import jit
import random
import numpy as np
import training.DataArgumentation as da

X = []
Y = []
for n in range(1000):
 signal = np.random.rand(3000)
 X.append(signal)
 Y.append(n%10)
print('X:', len(X))
print('Y:', len(Y))
X,Y = da.augment_data(X,Y, 3000,3)
print('X:', len(X))
print('Y:', len(Y))