# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 11:53:55 2023

@author: Aamir
"""

from scipy.fft import fft, fftfreq, ifft
import numpy as np

def main():
    tut2()

def tut2():
    ...
##




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