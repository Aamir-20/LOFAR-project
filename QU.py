# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 11:14:41 2023

@author: Aamir
"""

from scipy.fft import fft, ifft, rfft, irfft, fft2, ifft2, fftshift, fftfreq
import autograd.numpy as np
import pylab as pl


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
    #1 t1 = np.linspace(lambda2_min, lambda2_max, 512)
      
    
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
    plot_sim(Q, U, lambda2, "$\lambda^2$")  
    
    faraday_depth_recovery(Q, U)

    
    

def simulate_QU(phi_0, chi_0, P_0, l2):
    Q = P_0*np.cos(2*(phi_0*l2+chi_0))
    U = P_0*np.sin(2*(phi_0*l2+chi_0))
    return Q, U


# def faraday_depth_recovery(Q, U):
#     ft = fft(np.array(Q+1j*U)) # produces 512 complex numbers to be mapped with phi  
#     # pl.plot(np.abs(ft))
#     # pl.plot(np.real(ft))
#     # pl.plot(np.imag(ft))
#     phi = np.linspace(-500,500,512)
#     pl.plot(phi,np.real(ft),ls='--',c='c',label="Real")
#     pl.plot(phi,np.imag(ft),ls=':',c='c',label="Imag")
#     pl.plot(phi,np.abs(ft),ls='-',c='grey',label="Abs")
#     pl.xlim(-200,200)
#     pl.ylim(-1,1)
#     pl.legend(fontsize=14)
#     pl.xlabel(r"Faraday Depth [rad m$^{-2}$]", fontsize=14)
#     return ft


def faraday_depth_recovery(Q, U):
    ft = fft(np.array(Q+1j*U)) # produces 512 complex numbers to be mapped with phi  
    # pl.plot(np.abs(ft))
    # pl.plot(np.real(ft))
    # pl.plot(np.imag(ft))
    phi = np.linspace(-500,500,512)
    # time_step = 1000/512
    # phi = fftfreq(ft.size, d=time_step)
    pl.plot(phi,np.real(ft),ls='--',c='c',label="Real")
    pl.plot(phi,np.imag(ft),ls=':',c='c',label="Imag")
    pl.plot(phi,np.abs(ft),ls='-',c='grey',label="Abs")
    pl.xlim(-200,200)
    pl.ylim(-1,1)
    pl.legend(fontsize=14)
    pl.xlabel(r"Faraday Depth [rad m$^{-2}$]", fontsize=14)
    return ft




def plot_sim(Q, U, nu, title):
    ax2 = pl.subplot(111)
    
    ax2.plot(nu, Q, linestyle='-', color='c', lw=1.0, label="Q")
    ax2.plot(nu, U, linestyle='-', color='b', lw=1.0, label="U")
    ax2.set_ylabel("Q,U")
    ax2.set_xlabel(title)
    
    ax2.legend()
    pl.show()
    
    
if __name__ == "__main__":
    main()