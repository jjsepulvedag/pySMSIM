import numpy as np

def fft(y, dt, n):
    nfft = 2*n - 1
    fa_spectrum = np.fft.rfft(y, n=nfft, axis=0) * dt
    fa_freq = np.fft.rfftfreq(nfft, dt)

    return fa_spectrum, fa_freq

if __name__ == '__main__':

    print('Nothing to show')