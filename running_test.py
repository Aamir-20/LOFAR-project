# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 19:36:05 2023

@author: Aamir
"""
from pandas import read_csv
from ast import literal_eval
from time import time
import numpy as np
t1 = time()
df = read_csv("test_train_data.csv")
y = df.values[:,-2:]
print(y)
print((type(y)))
print(len(y))
print(type(y[0]))
print(type(y[0][0]))

for i, el in enumerate(y):
    for j in range(2):
        y[i][j] = np.array(literal_eval(y[i][j]))

print(y)
print((type(y)))
print(len(y))
print(type(y[0]))
print(type(y[0][0]))
print(type(y[0][1]))
print("Run time: ", time()-t1)

# X = df.values[:,:3]
# print(X)
# print((type(X)))
# print(len(X))
# print(type(X[0]))
# print(type(X[0][0]))
# print(type(X[0][1]))
# print(type(X[0][2]))
