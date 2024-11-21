import numpy as np
import target_fas
import white_noise
import matplotlib.pyplot as plt

def get_gm(gmParam, sourceParam, pathParam, siteParam):
    '''returns ground motion acceleration time series'''

    noise_spectra, phase, ffreqs = white_noise.get_noiseE(gmParam)
    freqs = ffreqs + 0.001
    fas = target_fas.get_targetFAS(freqs, sourceParam, pathParam, siteParam)

    final_fas = fas*noise_spectra

    gm_duration = gmParam['gmDuration']
    gm_sampling = gmParam['sampleRate']
    accs = np.fft.irfft(final_fas*np.exp(1j*phase))
    times = np.linspace(0, gm_duration, gm_duration*gm_sampling)

    return accs, times

def plot_gm(gmParam, sourceParam, pathParam, siteParam):
    '''Plots a single ground motion time series'''
    accs, times = get_gm(gmParam, sourceParam, pathParam, siteParam)
    accs = accs[:5000]

    fig, axs = plt.subplots(1,1)
    fig.suptitle('Ground motion record')
    axs.plot(times, accs, linewidth=0.75)
    axs.set_xlabel('Time (s)')
    axs.set_ylabel('Acceleration (cm/s3)')
    axs.set_xlim(0, gmParam['gmDuration'])
    axs.grid(which='both', alpha=0.5)
    plt.show()

    return None 

def plot_spectrum(gmParam, sourceParam, pathParam, siteParam, n_fas=100):
    '''plots target spectrum, last spectrum of a gm realization, and averaged 
    spectrum after n realizations'''

    noise_0, phase_0, ffreqs = white_noise.get_noiseE(gmParam)
    freqs = ffreqs+0.01
    fas = target_fas.get_targetFAS(freqs, sourceParam, pathParam, siteParam)
    all_fas = np.zeros((n_fas, ffreqs.shape[0]))
    all_fas[0,:] = fas*noise_0

    for i in range(n_fas-1):
        noise_i, phase_i, ffreqs_i = white_noise.get_noiseE(gmParam)
        all_fas[i+1,:] = fas*noise_i

    fig, axs = plt.subplots(1,1)
    fig.suptitle('Fourier Amplitude Spectrum')
    axs.plot(freqs, fas, label='Target FAS', linewidth=2.0)
    axs.plot(ffreqs, all_fas[99,:], label='Last realization', linewidth=0.5)
    axs.plot(ffreqs, np.mean(all_fas, axis=0), label='Average FAS')
    axs.set_xlabel('Frequency (Hz)')
    axs.set_ylabel('Fourier Amplitude')
    axs.set_xlim(0.01, 100)
    axs.set_ylim(0.00001, 100)
    axs.set_xscale('log')
    axs.set_yscale('log')
    axs.legend()
    axs.grid(which='both')
    plt.show()

    return None

if __name__=='__main__':

    # ----------------------- Defining input parameters ---------------------- #
    # Source
    sourceParam = {'M_w': 4.0, 
                   'Vs_src': 3.6, # km/s
                   'rho_src': 2800, # gm/cc
                   'dSigma':100 # bars
                   }
    # Path
    Z_r = lambda r: 1/r if r<40 else ((1/40)*((40/r)**0.5) if r>=40 else None)
    Q_f = lambda f: 680*(f**0.38) # page 643, Boore (2003)
    pathParam = {'R':10, # km
                 'Z_r': Z_r, # picewise funct
                 'Q_f': Q_f, # table 4 Boore (2003)
                 'c_Q': 3.5 # km/s, table 4 Boore (2003)
                 }
    # Site
    siteParam = {'k0':0.045 # Lee et al. (2022)
                 }
    # Ground motion
    gmParam = {'gmDuration': 25, # sec
               'sampleRate':200 # Hz
               }

    # ----------------- Plotting target FAS with realizations ---------------- #
    # plot_spectrum(gmParam, sourceParam, pathParam, siteParam, n_fas=1000)
    # ------------------- Plotting one realization of SMSIM ------------------ #
    plot_gm(gmParam, sourceParam, pathParam, siteParam)


