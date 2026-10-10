import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

#valores constantes
g = 9.81
nx = ny = 150
l = 10.0
dx = dy = l/nx
x = (np.arange(nx)+0.5)*dx
y = (np.arange(ny)+(0.5))*dy
X, Y = np.meshgrid(x, y, indexing="ij") #eixos x e y

h = np.where((X - l/2)**2 + (Y - l/2)**2 < 1.5**2, 2.0, 1.0)
hu = np.zeros_like(h)
hv = np.zeros_like(h)

def fx(h, hu, hv):
    u = hu/h
    return hu, hu*u + 0.5*g*h**2, hv*u

def fy(h, hu, hv):
    v = hv/h
    return hv, hu*v, hv*v + 0.5*g*h**2

def com_paredes(q, sx, sy):
    qe = np.pad(q, 1, mode="edge")
    qe[0, :] *= sx; qe[-1, :] *= sx
    qe[: , 0] *= sy; qe[: , -1] *= sy
    return qe

def passo():
    global h, hu, hv #variáveis globais
    c = np.sqrt(g*h) + np.maximum(np.abs(hu/h), np.abs(hv/h))
    dt = 0.25*dx/c.max()

    hE = com_paredes(h, 1, 1)
    huE = com_paredes(hu, -1, 1)
    hvE = com_paredes(hv, 1, -1)

    Q = (hE[:, 1:-1,], huE[:, 1:-1], hvE[:, 1:-1])
    F = fx(*Q)
    cx = np.abs(Q[1]/Q[0]) + np.sqrt(g*Q[0])
    ax = np.maximum(cx[:-1], cx[1:])
    Fx = [0.5*(F[k][:-1] + F[k][1:]) - 0.5*ax*(Q[k][1:] - Q[k][:-1]) for k in range(3)]

    P = (hE[1:-1, :], huE[1:-1, :], hvE[1:-1, :])
    G = fy(*P)
    cy = np.abs(P[2]/P[0]) + np.sqrt(g*P[0])
    ay = np.maximum(cy[:, :-1], cy[:, 1:])
    Gy = [0.5*(G[k][:, :-1] + G[k][:, 1:]) - 0.5*ay*(P[k][:, 1:] - P[k][:, :-1]) for k in range(3)]

    h = h - dt/dx*(Fx[0][1:] - Fx[0][:-1]) - dt/dy*(Gy[0][:, 1:] - Gy[0][:, :-1])
    hu = hu - dt/dx*(Fx[1][1:]-Fx[1][:-1]) - dt/dy*(Gy[1][:, 1:] - Gy[1][:, :-1])
    hv = hv - dt/dx*(Fx[2][1:]-Fx[2][:-1]) - dt/dy*(Gy[2][:, 1:] - Gy[2][:, :-1])
    return dt

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(h.T, origin="lower", extent=(0.0, l, 0.0, l), cmap="viridis", vmin=0.7, vmax=1.3)
fig.colorbar(im, label="h")
ax.set_xlabel("x"); ax.set_ylabel("y")
titulo = ax.set_title("t=0.00")
t = 0.0

PASSO_POR_QUADRO = 3

def atualizar(quadro): #frames
    global t
    for _ in range(PASSO_POR_QUADRO):
        t += passo()
    im.set_data(h.T)
    titulo.set_text(f"t = {t:.2f}")
    return im, titulo

anim = FuncAnimation(fig, atualizar, frames=200, interval=30, blit=False)
plt.show()
