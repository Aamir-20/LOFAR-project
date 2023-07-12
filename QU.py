# -*- coding: utf-8 -*-
import autograd.numpy as np
import pylab as pl


def main():
    const_c = 3e8
    
    nu_min, nu_max = 0.58e9, 2.50e9
    
    lambda2_min = (const_c/nu_max)**2
    lambda2_max = (const_c/nu_min)**2
    
    # Adjust parameters here:
    phi_0 = 50
    chi_0 = 1.5
    P_0 = 1
    N = 512
    
    # make data regularly spaced in frequency:
    nu = np.linspace(nu_min, nu_max, N)
    lambda2 = (const_c/nu)**2
    
    # make data regularly spaced in lambda^2: 
    lambda_spaced = np.linspace(lambda2_min, lambda2_max, N)    
    
    Q, U = simulate_QU(phi_0, chi_0, P_0, lambda_spaced)  
    #plot_sim(Q, U, nu, "Frequency")  
    plot_sim(Q, U, lambda2, "$\lambda^2$")  
    
    faraday_depth_recovery(Q, U, phi_0, chi_0, P_0, np.ones(len(lambda2)))
    

def simulate_QU(phi_0, chi_0, P_0, lambda2):
    
    Q = P_0*np.cos(2*(phi_0*lambda2+chi_0))
    U = P_0*np.sin(2*(phi_0*lambda2+chi_0))
    
    return Q, U


def plot_sim(Q, U, nu, title):
    
    ax2 = pl.subplot(111)
    
    ax2.plot(nu, Q, linestyle='-', color='c', lw=1.0, label="Q")
    ax2.plot(nu, U, linestyle='-', color='b', lw=1.0, label="U")
    ax2.set_ylabel("Q,U")
    ax2.set_xlabel(title)
    
    ax2.legend()
    pl.show()
    
    
def calc_k(W):
    
    K = np.sum(W)
    
    return K


def calc_l0(W,lambda2):
    
    K = calc_k(W)
    l0 = (1./K)*np.sum(W*lambda2)
    
    return l0

    

def faraday_depth_recovery(Q, U, phi_0, chi_0, P_0, W=512):
    """Aim is to recover Faraday depth spectrum.
    """
    const_c = 3e8
    N = 512
    
    nu_min, nu_max = 0.58e9, 2.50e9
    
    lambda2_min = (const_c/nu_max)**2
    lambda2_max = (const_c/nu_min)**2
    
    # make data regularly spaced in lambda^2: 
    lambda_spaced = np.linspace(lambda2_min, lambda2_max, N)  

  
    phi_min, phi_max = -200, 200
    phi = np.linspace(phi_min, phi_max, N)

    P = Q + 1j * U # Produces N complex numbers to be mapped with phi  
    K = calc_k(W)
    l0 = calc_l0(W, lambda_spaced)

    yplot = []
    for _ in range(len(phi)):    
        f = 1/K*(np.sum(P*W*np.exp(-2*1j*phi[_]*(lambda_spaced-l0))))
        yplot.append(f)


    yplot = np.array(yplot)


    pl.plot(phi, np.abs(yplot), ls='-', c='grey', label="Abs")
    pl.plot(phi, np.real(yplot), ls='--', c='c', label="Real")
    pl.plot(phi, np.imag(yplot), ls=':', c='c', label="Imag")
    pl.xlim(-200, 200)
    pl.xlabel(r"Faraday Depth [rad m$^{-2}$]")
    pl.legend()
    pl.grid()
    pl.show()  
    

    
if __name__ == "__main__":
    main()