import numpy as np
import matplotlib.pyplot as plt

g = 9.81
nx, L = 400, 10.0
dx = L/nx
x = (np.arange(nx) +0.5)*dx

h = np.where(x < L/2, 2.0, 1.0)
hu = np.zeros(nx)

def flux(h, hu):
    u = hu / h
    return hu, hu*u + 0.5*g*h**2

t, t_end = 0.0, 0.7
while t < t_end:
    c = np.abs(hu/h) + np.sqrt(g*h)
    dt = min(0.4*dx / c.max(), t_end - t)

    hE = np.pad(h, 1, mode='edge')
    huE = np.pad(hu, 1, mode='edge')
    F1, F2 = flux(hE, huE)
    cE = np.abs(huE/hE)+np.sqrt(g*hE)
    a = np.maximum(cE[:-1], cE[1:])

    f1 = 0.5*(F1[:-1]+F1[1:]) - a*0.5*(hE[1:] - hE[:-1])
    f2 = 0.5*(F2[:-1]+F2[1:]) - a*0.5*(huE[1:] - huE[:-1])

    h = h - dt/dx*(f1[1:]-f1[:-1])
    hu = hu - dt/dx*(f2[1:]-f2[:-1])
    t += dt # t = t + dt
plt.plot(x,h); plt.xlabel("x"); plt.ylabel("h"); plt.show()
