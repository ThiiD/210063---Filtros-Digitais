import scipy.signal as signal
import numpy as np
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