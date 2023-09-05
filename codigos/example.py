import matplotlib.pyplot as plt
from control.matlab import *
import scipy.signal as signal
import numpy as np

Fs = 4*775
fp = np.array([1510/2, 1570/2])
fs = np.array([1394/2, 1704/2])
Ap = 1
As = 40
wp = fp/(Fs/2)
ws = fs/(Fs/2)

f = 770
tf = 6*1/f
t = np.linspace(0,tf,1000)
sig = 0.3*np.sin(2*np.pi*f/2*t) + 1*np.sin(2*np.pi*f*t) + 0.1*np.sin(2*np.pi*2*f*t)


fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.plot(t, sig)
ax1.set_title('10 Hz and 20 Hz sinusoids')
ax1.axis([0, 1, -2, 2])
N, wc = signal.ellipord(wp, ws, Ap, As)
sos = signal.ellip(N, Ap, As, wc, 'bandpass', output='sos')
filtered = signal.sosfilt(sos, sig)
ax2.plot(t, filtered)
ax2.set_title('After 17 Hz high-pass filter')
ax2.axis([0, 0.008, -2, 2])
ax2.set_xlabel('Time [seconds]')
plt.tight_layout()
plt.show()