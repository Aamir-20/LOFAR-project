# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 11:53:55 2023

@author: Aamir
"""

from scipy.fft import fft, fftfreq, ifft, ifftn
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


def main():
    tut3()
    #

def tut3():
    N = 30
    f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, sharex='col', sharey='row')
    xf = np.zeros((N,N))
    xf[0, 5] = 1
    xf[0, N-5] = 1
    Z = ifftn(xf)
    ax1.imshow(xf, cmap=cm.Reds)
    ax4.imshow(np.real(Z), cmap=cm.gray)
    xf = np.zeros((N, N))
    xf[5, 0] = 1
    xf[N-5, 0] = 1
    Z = ifftn(xf)
    ax2.imshow(xf, cmap=cm.Reds)
    ax5.imshow(np.real(Z), cmap=cm.gray)
    xf = np.zeros((N, N))
    xf[5, 10] = 1
    xf[N-5, N-10] = 1
    Z = ifftn(xf)
    ax3.imshow(xf, cmap=cm.Reds)
    ax6.imshow(np.real(Z), cmap=cm.gray)
    plt.show()
    

def tut2():
    # Number of sample points
    N = 600
    # sample spacing
    T = 1.0 / 800.0
    x = np.linspace(0.0, N*T, N, endpoint=False)
    y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)
    yf = fft(y)
    xf = fftfreq(N, T)[:N//2]
    plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
    plt.grid()
    plt.show()


def tut1():
    time_step = 0.05
    time_vec = np.arange(0, 10, time_step) # np array from 0 to 10 didvided by time step
    period = 5
    sig = (np.sin(2*np.pi*time_vec/period) + 0.25*np.random.randn(time_vec.size))
    
    sig_fft = fft(sig)
    
    amplitude = np.abs(sig_fft)
    power = amplitude**2
    angle = np.angle(sig_fft)
    
    sample_freq = fftfreq(sig.size, d=time_step)
    
    print(amplitude)
    print(sample_freq)
    
    amp_freq = np.array([amplitude, sample_freq])
    amp_pos = amplitude.argmax() # the position of the amplitude, i.e maximum value
    peak_freq = amp_freq[1, amp_pos]
    
    print(amp_freq)
    print(amp_pos)
    print(peak_freq)
    
    high_freq_fft = sig_fft.copy()
    high_freq_fft[np.abs(sample_freq) > peak_freq] = 0
    filtered_sig = ifft(high_freq_fft)
    
    print(high_freq_fft)
    print(filtered_sig)


if __name__ == "__main__":
    main()