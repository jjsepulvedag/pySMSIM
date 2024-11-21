import numpy as np
from utils import fft
import matplotlib.pyplot as plt

def windowB():
    '''Box window function'''
    return None

def windowE(times, duration): # after Saragoni and Hart (1973)
    '''Exponential window function'''

    epsilon = 0.3 # Saragoni and Hart (1973), page 655 Boore (2003)
    eta = 0.05 # Saragoni and Hart (1973), page 655 Boore (2003)

    # Eqtns 25, 26, and 27, Boore (2003)
    b = -(epsilon*np.log(eta))/(1+epsilon*(np.log(epsilon) - 1)) 
    c = b/epsilon
    a = (np.exp(1)/epsilon)**b

    f_Tgm = 1/1.3 # page 655, Boore (2003) 
    t_eta = duration*f_Tgm # eqtn 28, Boore (2003)
    
    w = a*((times/t_eta)**b)*np.exp(-c*(times/t_eta)) # eqn 24, Boore (2003)

    return w, t_eta

def get_noiseB():
    '''white noise with a box window function'''
    return None

def get_noiseE(gmParam):
    '''white noise with an Exponential window function'''

    gmDuration = gmParam['gmDuration']
    sampleRate = gmParam['sampleRate']

    times = np.linspace(0, gmDuration, sampleRate*gmDuration)

    noise0 = np.random.normal(0, 1, sampleRate*gmDuration) # gaussian noise
    window, t_eta = windowE(times, gmDuration)

    windowed_noise = noise0*window
    fas_wn, ff_wn = fft(windowed_noise, 1/sampleRate, windowed_noise.shape[0])
    fas_phase = np.angle(fas_wn)
    fas_wn = np.abs(fas_wn)
    root_mean_square = np.sqrt(np.mean(fas_wn**2))
    normalized_spectral_amplitudes = fas_wn/root_mean_square

    return normalized_spectral_amplitudes, fas_phase, ff_wn

def plot_noise(gmParam):
    '''plots noise given input paramters'''

    gmDuration = gmParam['gmDuration']
    sampleRate = gmParam['sampleRate']

    times = np.linspace(0, gmDuration, sampleRate*gmDuration)

    noise0 = np.random.normal(0, 1, sampleRate*gmDuration) # gaussian noise
    window, t_eta = windowE(times, gmDuration)

    windowed_noise = noise0*window
    fas_wn, ff_wn = fft(windowed_noise, 1/sampleRate, windowed_noise.shape[0])
    fas_phase = np.angle(fas_wn)
    fas_wn = np.abs(fas_wn)
    root_mean_square = np.sqrt(np.mean(fas_wn**2))
    normalized_spectral_amplitudes = fas_wn/root_mean_square

    fig, axs = plt.subplots(2,2, figsize=(10, 8))
    fig.suptitle('White Noise for SMSIM')
    axs[0, 0].plot(times, noise0, linewidth=0.35, 
                   label='Gaussian WN')
    axs[0, 1].plot(times, windowed_noise, linewidth=0.35, 
                   label='Windowed WN')
    axs[0, 1].plot(times, window, linewidth=1.25, 
                   label='Window function')
    axs[1, 0].plot(ff_wn, fas_wn, linewidth=0.35, 
                   label='Windowed WN')
    axs[1, 1].plot(ff_wn, normalized_spectral_amplitudes, linewidth=0.35, 
                   label='Normalized windowed WN')

    for i in [0, 1]:
        axs[0, i].set_xlabel('Time (s)')
        axs[0, i].set_ylabel('Acceleration (cm/s2)')
        axs[0, i].set_xlim(0, gmDuration)
        axs[0, i].legend()
        
    for i in [0, 1]:
        axs[1, i].set_xlabel('Frequency (HZ)')
        axs[1, i].set_ylabel('Fourier Amplitude')
        axs[1, i].set_xscale('log')
        axs[1, i].set_yscale('log')
        axs[1, i].legend()



    plt.show()

    return None

if __name__=='__main__':

    gmParam = {'gmDuration': 25, # sec
               'sampleRate':200 # Hz
              }
    
    # noise_spectra, phase, ffreqs = noise(gmParam)
    plot_noise(gmParam)