# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 11:14:41 2023

@author: Aamir
"""

import autograd.numpy as np
import pylab as pl
import os,sys


def main():
    const_c = 3e8
    
    nu_min = 0.58e9
    nu_max = 2.50e9
    
    lambda2_min = (const_c/nu_max)**2
    lambda2_max = (const_c/nu_min)**2
    
    print(lambda2_min, lambda2_max)
    
    # make data regularly spaced in frequency:
    nu = np.linspace(nu_min, nu_max, 512)
    lambda2 = (const_c/nu)**2
    lambda2 = lambda2[::-1] # reversing list so it's in ascending order.
    
    print("This is lambda squared: ", lambda2)

    Q, U = simulate_QU(0.1, 0.8, 0.9, lambda2)
    print("This is Q simulated: ", Q)
    print("This is U simulated: ", U)


def simulate_QU(phi_0, chi_0, P_0,l2):
    Q = P_0*np.cos(2*(phi_0*l2+chi_0))
    U = P_0*np.sin(2*(phi_0*l2+chi_0))
    return Q, U    
    
if __name__ == "__main__":
    main()