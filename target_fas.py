import numpy as np
import matplotlib.pyplot as plt

def source_spectrum(freqs, M_w, Vs_src, rho_src, dSigma, R0):
    '''returns source spectrum'''

    def magConv(M_w):
        M_0 = 10**((3/2)*(M_w + 10.7)) # eqtn 2.5, Baker et al. (2021)
        return M_0

    def cornerFreq(M_0, Vs_src, dSigma):
            f0 = 4.9e6*Vs_src*(dSigma/M_0)**(1/3) # eqtn 4
            return f0
    
    def constantC(Vs_src, rho_src, R0):
        R_TP= 0.55 # radiation pattern (usually 0.55) 
        V = 1/np.sqrt(2) # page 641, Boore (2003)
        F = 2 # effect of the free surface, page 641 Boore (2003)

        C = R_TP*V*F/(4*np.pi*rho_src*(Vs_src**3)*R0)
        C = C*(1/1000**3)*(1/100**3) # To cancel cm3 with km3, final units= s^3/(g*km)
        return C

    M_0 = magConv(M_w)
    f0 = cornerFreq(M_0, Vs_src, dSigma) # eqtn 4, Boore (2003)
    C = constantC(Vs_src, rho_src, R0) # eqtn 7, Boore (2003)
    
    # Sa = 1/((1 + (freqs/f0)**2)) # table 2, Boore (2003)
    Sa = (4*np.pi*freqs**2)/((1 + (freqs/f0)**2)) # table 2, Boore (2003)
    Sb = 1 # table 2, Boore (2003)
    S = Sa*Sb # eqtn 6, Boore (2003)

    E = C*M_0*S # eqtn 5, Boore (2003)
    
    return E

def path_spectrum(freqs, R, Z_r, Q_f, c_Q):
    '''returns path spectrum'''

    Q_fs = np.array(list(map(Q_f, freqs)))
    P = Z_r(R)*np.exp(-np.pi*freqs*R/(Q_fs*c_Q)) # eqtn 8, Boore (2003)
    
    return P

def site_spectrum(freqs, k0):
    '''return site spectrm'''
    def A_f():
        return 1

    def D_f(freqs, k0):
        d_f = np.exp(-np.pi*k0*freqs) # eqtn 20, Boore (2003)
        # d_f = 1/np.sqrt(1+(freqs/15)**8)
        return d_f

    d_f = D_f(freqs, k0)
    a_f = A_f()

    G = a_f*d_f
    
    return G
       

def get_targetFAS(freqs, sourceParam, pathParam, siteParam):
    '''returns target Fourier Amplitude Spectrum'''

    # Source parameters
    M_w = sourceParam['M_w']
    Vs_src = sourceParam['Vs_src']
    rho_src = sourceParam['rho_src']
    dSigma = sourceParam['dSigma']
    R0 = 1 # reference distance, usually taken as 1, page 642 Boore (2003)

    # Path parameters
    R = pathParam['R']
    Z_r = pathParam['Z_r']
    Q_f = pathParam['Q_f'] 
    c_Q = pathParam['c_Q']

    # Site parameters
    k0 = siteParam['k0']


    # Computation of spectrums
    E = source_spectrum(freqs, M_w, Vs_src, rho_src, dSigma, R0) # Source spectrum 
    P = path_spectrum(freqs, R, Z_r, Q_f, c_Q)
    G = site_spectrum(freqs, k0)

    target_FAS = E*P*G

    return target_FAS


def plot_targetFAS(freqs, sourceParam, pathParam, siteParam):
    '''plots target Fourier Amplitude Spectrum'''

    fas = get_targetFAS(freqs, sourceParam, pathParam, siteParam)

    plt.plot(freqs, fas, label='Target FAS')
    plt.title('Target Fourier Amplitude Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Fourier Amplitude')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.grid(which='both', alpha=0.5)
    plt.show()


    return None

if __name__=='__main__':

    freqs = np.linspace(0.1, 100, 5000)

    sourceParam = {'M_w': 4.0, 
                   'Vs_src': 3.6, # km/s
                   'rho_src': 2800, # gm/cc
                   'dSigma':100 # bars
                   }
    
    Z_r = lambda r: 1/r if r<40 else ((1/40)*((40/r)**0.5) if r>=40 else None)
    Q_f = lambda f: 680*(f**0.38) # page 643, Boore (2003)
    pathParam = {'R':10, # km
                 'Z_r': Z_r, # picewise funct
                 'Q_f': Q_f, # table 4 Boore (2003)
                 'c_Q': 3.5 # km/s, table 4 Boore (2003)
                 }

    siteParam = {'k0':0.045 # Lee et al. (2022)
                 }


    # fas = get_targetFAS(freqs, sourceParam, pathParam, siteParam)
    plot_targetFAS(freqs, sourceParam, pathParam, siteParam)



