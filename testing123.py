# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from scipy.fft import fft, ifft
import numpy as np
x = np.array([1.0, 2.0, 1.0, -1.0, 1.5])
print(x)
y = fft(x)
print(y)
yinv = ifft(y)
print(yinv)
calculated = np.array([])
for i in range(len(y)):
    val = 0
    for n in range(len(x)):
        val+= np.exp(-2*np.pi*n*i/5*1j)*x[n]
 
    calculated = np.append(calculated, val)
 
print(calculated)
print(len(calculated))