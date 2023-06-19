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

    
    # make data regularly space in lambda^2:
    #t1 = np.linspace(lambda2_min, lambda2_max, 512)
         
    #print("This is lambda squared: ", lambda2)

    Q, U = simulate_QU(50, 1.5, 1, lambda2)
    #print("This is Q simulated: ", Q)
    #print("This is U simulated: ", U)
    print("The maximum of Q is: ", max(Q))
    print("The minimum of Q is: ", min(Q))
    print("The maximum of U is: ", max(U))
    print("The minimum of U is: ", min(U))
    print("The maximum of nu is: ", max(nu))
    print("The minimum of nu is: ", min(nu))    
    plot_sim(Q, U, nu, "Frequency")
    
    
    Q, U = simulate_QU(50, 1.5, 1, lambda2)
    #print("This is Q simulated: ", Q)
    #print("This is U simulated: ", U)
    print("The maximum of Q is: ", max(Q))
    print("The minimum of Q is: ", min(Q))
    print("The maximum of U is: ", max(U))
    print("The minimum of U is: ", min(U))
    print("The maximum of nu is: ", max(nu))
    print("The minimum of nu is: ", min(nu))    
    plot_sim(Q, U, lambda2, "$\lambda^2$")    


def simulate_QU(phi_0, chi_0, P_0, l2):
    Q = P_0*np.cos(2*(phi_0*l2+chi_0))
    U = P_0*np.sin(2*(phi_0*l2+chi_0))
    return Q, U


def plot_sim(Q, U, nu, title="$\lambda^2$"):
    ax2 = pl.subplot(111)
    
    ax2.plot(nu, Q, linestyle='-', color='c', lw=1.0, label="Q")
    ax2.plot(nu, U, linestyle='-', color='b', lw=1.0, label="U")
    ax2.set_ylabel("Q,U")
    ax2.set_xlabel(title)
    
    ax2.legend()
    pl.show()
    
    
if __name__ == "__main__":
    main()