import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import cmath

fig_width_cm = 18/ 2.4
fig_height_cm = 18/2.4

# 2.05580199e-03 -8.27899010e-05  1.99917959e-03  0.00000000e+00  -1.99917959e-03  8.27899010e-05 -2.05580199e-03
a0 = 2.05580199e-03
a1 = -8.27899010e-05
a2 = 1.99917959e-03
a3 = 0
a4 = -1.99917959e-03
a5 = 8.27899010e-05
a6 = -2.05580199e-03

# --------------------
# 1.         -0.06020081  2.93894423 -0.11797858  2.88129071 -0.05786016   0.94225528
d0 = 1
d1 = -0.06020081
d2 = 2.93894423
d3 = -0.11797858
d4 = 2.88129071
d5 = -0.05786016
d6 = 0.94225528


num = np.array([a0, a1, a2, a3, a4, a5, a6])
den = np.array([d0, d1, d2, d3, d4, d5, d6])

g = 2**27
numG = np.around(num*g, decimals=0)
denG = np.around(den*g, decimals=0)

print(f'NUM: {numG}')
print(f'DEN: {denG}')

# denG = np.flipud(denG)
# numG = np.flipud(numG)
roots = np.roots(denG)
nRoots = np.roots(numG)
print(roots)

for i,_ in enumerate(roots):
    print(f'polo {i}, norma: {abs(roots[i])}')
    

circle1 = plt.Circle((0, 0), 1, color='black', fill = False, linewidth = 2, zorder = 5)
fig, axs = plt.subplots(figsize = (fig_width_cm, fig_height_cm), nrows = 1, ncols = 1)
axs.add_patch(circle1)
axs.scatter(roots.real,roots.imag,s=100,marker='x',color='steelblue',linewidths=2, zorder = 5)
axs.scatter(nRoots.real, nRoots.imag, s=100, marker="o", color="firebrick", linewidths=2, zorder=5)
plt.grid()
plt.savefig("pzmap_gain.png", bbox_inches = "tight")
plt.show(block = False)


f = 770
tf = 6*1/f
t = np.linspace(0,tf,1000)
x = 0.3*np.sin(2*np.pi*f/2*t) + 1*np.sin(2*np.pi*f*t) + 0.1*np.sin(2*np.pi*2*f*t)

u = [0,0,0,0,0,0,0]
y = [0,0,0,0,0,0,0]
filter_signal = []
for _,i in enumerate(x):
    print(i)
    u[6] = u[5]
    u[5] = u[4]
    u[4] = u[3]
    u[3] = u[2]
    u[2] = u[1]
    u[1] = u[0]
    u[0] = i

    y[6] = y[5]
    y[5] = y[4]
    y[4] = y[3]
    y[3] = y[2]
    y[2] = y[1]
    y[1] = y[0]
    y[0] = a6*u[6] + a5*u[5] + a4*u[4] + a3*u[3] + a2*u[2] + a1*u[1] + a0*u[0] - d6*y[6] - d5*y[5] - d4*y[4] - d3*y[3] - d2*y[2] - d1*y[1]

    filter_signal.append(y[0])

fig, axs = plt.subplots(figsize=(fig_width_cm, fig_height_cm), nrows = 2, ncols = 1, sharex=True)
axs[0].plot(t,x)
axs[1].plot(t,filter_signal)
plt.show()
