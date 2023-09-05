
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

params = {'text.usetex' : True,
          'font.size' : 11,
          'font.family' : 'lmodern'
          }
plt.rcParams.update(params)

def mfreqz(b, a, Fs):
    wz, hz = signal.freqz(b, a)
    Mag = 20*np.log10(abs(hz))
    Phase = np.unwrap(np.arctan2(np.imag(hz),np.real(hz)))*(180/np.pi)
    Freq = wz*Fs/(2*np.pi)
    fig = plt.figure(figsize=(10, 6))
    sub1 = plt.subplot(2, 1, 1)
    sub1.plot(Freq, Mag, 'r', linewidth=2)
    sub1.axis([1, Fs/2, -100, 5])
    sub1.set_title('Magnitute Response', fontsize=20)
    sub1.set_xlabel('Frequency [Hz]', fontsize=20)
    sub1.set_ylabel('Magnitude [dB]', fontsize=20)
    sub1.grid() 

    sub2 = plt.subplot(2, 1, 2)
    sub2.plot(Freq, Phase, 'g', linewidth=2)
    sub2.set_ylabel('Phase (degree)', fontsize=20)
    sub2.set_xlabel(r'Frequency (Hz)', fontsize=20)
    sub2.set_title(r'Phase response', fontsize=20)
    sub2.grid() 

    plt.subplots_adjust(hspace=0.5)
    fig.tight_layout()
    plt.show() 

def impz(b, a):
    impulse = np.repeat(0., 60)
    impulse[0] = 1.
    x = np.arange(0, 60)
    response = signal.lfilter(b, a, impulse)
    fig = plt.figure(figsize=(10, 6))

    plt.subplot(211)
    plt.stem(x, response, 'm', use_line_collection=True)
    plt.ylabel('Amplitude', fontsize=15)
    plt.xlabel(r'n (samples)', fontsize=15)
    plt.title(r'Impulse response', fontsize=15)

    plt.subplot(212)
    step = np.cumsum(response)
    plt.stem(x, step, 'g', use_line_collection=True)
    plt.ylabel('Amplitude', fontsize=15)
    plt.xlabel(r'n (samples)', fontsize=15)
    plt.title(r'Step response', fontsize=15)
    plt.subplots_adjust(hspace=0.5)
    fig.tight_layout()

    plt.show()




Fs = 4*775
fp = np.array([1510/2, 1570/2])
fs = np.array([1394/2, 1704/2])
Ap = 1
As = 40
wp = fp/(Fs/2)
ws = fs/(Fs/2)
N, wc = signal.ellipord(wp, ws, Ap, As)
print('Order of the filter=', N)
print('Cut-off frequency=', wc)
z, p = signal.ellip(N, Ap, As, wc, 'bandpass')
print('Numerator Coefficients:', z)
print('Denominator Coefficients:', p)
mfreqz(z, p, Fs)