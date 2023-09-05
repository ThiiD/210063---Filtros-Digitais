import matplotlib.pyplot as plt
from control.matlab import *
import scipy.signal as signal
import numpy as np

params = {'text.usetex' : True,
          'font.size' : 11,
          'font.family' : 'lmodern'
          }
plt.rcParams.update(params)

fig_width_cm = 18/2.4
fig_height_cm = 7/2.4

def mfreqz(b, a, Fs):
    wz, hz = signal.freqz(b, a, worN=10000)
    Mag = 20*np.log10(abs(hz))
    Phase = np.unwrap(np.arctan2(np.imag(hz),np.real(hz)))*(180/np.pi)
    Freq = wz*Fs/(2*np.pi)
    fig, axs = plt.subplots(figsize=(fig_width_cm, 1.4*fig_height_cm), nrows = 2, ncols = 1, sharex= True)
    axs[0].plot(Freq, Mag, linewidth=2, color = "tab:blue", label = "Filtro - Código")
    axs[0].set_xlim([600, 1000])
    axs[0].set_ylim([-90,10])
    axs[0].legend(loc='upper right')
    axs[0].set_ylabel('Módulo [dB]')
    axs[0].grid() 

    axs[1] = plt.subplot(2, 1, 2)
    axs[1].plot(Freq, Phase, linewidth=2, color = "tab:blue", label = "Filtro - Código")
    axs[1].set_ylabel('Fase [Graus]')
    axs[1].set_xlabel('Frequência [Hz]')
    axs[1].set_xlim([600, 1000])
    axs[1].legend(loc='upper right')
    axs[1].grid() 
    fig.tight_layout()
    plt.savefig("resposta_codigo.pdf", bbox_inches = 'tight')
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

Ho = 2.7113
b0 = 0
b1 = 5.4746*10**14
b2 = 0
b3 = 4.8027*10**7
b4 = 0
b5 = 1
b6 = 0

a0 = 1.2809*10**22
a1 = 1.0175*10**17
a2 = 1.6434*10**15
a3 = 8.7008*10**9
a4 = 7.0238*10**7
a5 = 1.8586*10**2
a6 = 1

num = np.array([b6,b5,b4,b3,b2,b1,b0])
den = np.array([a6,a5,a4,a3,a2,a1,a0])

w = np.linspace(0,10000,100000)

G = tf(Ho*num, den)



mag, phase, w = bode(G,w)

mag = 20*np.log10(mag)
phase = np.rad2deg(phase)

fig, axs = plt.subplots(figsize=(fig_width_cm, 1.4*fig_height_cm), nrows = 2, ncols = 1, sharex=True)
axs[0].plot(w/(2*np.pi),mag, linewidth = 2, color = 'tab:blue', label = "Filtro - Livro")
axs[0].set_ylim([-90, 10])
axs[0].set_ylabel("Modulo [dB]")
axs[0].legend(loc='upper right')
axs[0].grid()


axs[1].plot(w/(2*np.pi),phase, linewidth = 2, color = 'tab:blue', label = "Filtro - Livro")
axs[1].set_xlim([600,1000])
axs[1].set_ylabel("Fase [Graus]")
axs[1].set_xlabel("Frequência [Hz]")
axs[1].legend(loc='upper right')
axs[1].grid()
plt.tight_layout()
plt.savefig("resposta_livro.pdf", bbox_inches='tight')
plt.show(block = False)


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
print(f'ZEROS: {np.roots(z)}')
print(f'POLOS: {np.roots(p)}')
G_python = tf(z,p)
plt.figure()
pzmap(G_python, plot = True)
plt.figure()
G_livro = c2d(G, 1/Fs)
p_livro, z_livro = pzmap(G_livro, plot = True)
print(f'ZEROS LIVRO: {z_livro}')
print(f'POLOS LIVRO: {p_livro}')
mfreqz(z, p, Fs)



g = 2**27
numG = g*num
denG = g*den

print(f'numG: {numG}')
print(f'denG: {denG}')