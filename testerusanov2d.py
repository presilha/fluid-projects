import numpy as np
import matplotlib.pyplot as plt

g = 9.81
nx = ny = 200
L = 10.0
dx = dy = L / nx
x = (np.arange(nx) + 0.5) * dx
y = (np.arange(ny) + 0.5) * dy
X, Y = np.meshgrid(x, y, indexing="ij")          # eixo 0 = x, eixo 1 = y

# coluna circular de água no centro
h = np.where((X - L/2)**2 + (Y - L/2)**2 < 1.5**2, 2.0, 1.0)
hu = np.zeros_like(h)
hv = np.zeros_like(h)

def fx(h, hu, hv):
    u = hu / h
    return hu, hu*u + 0.5*g*h**2, hv*u

def fy(h, hu, hv):
    v = hv / h
    return hv, hu*v, hv*v + 0.5*g*h**2

def com_paredes(q, sinal_x, sinal_y):
    """Uma camada de células fantasma refletindo a parede."""
    qe = np.pad(q, 1, mode="edge")
    qe[0, :] *= sinal_x;  qe[-1, :] *= sinal_x
    qe[:, 0] *= sinal_y;  qe[:, -1] *= sinal_y
    return qe

t, t_end = 0.0, 0.6
while t < t_end:
    c = np.sqrt(g*h) + np.maximum(np.abs(hu/h), np.abs(hv/h))
    dt = min(0.25 * dx / c.max(), t_end - t)

    hE  = com_paredes(h,  1,  1)
    huE = com_paredes(hu, -1, 1)     # u inverte nas paredes em x
    hvE = com_paredes(hv, 1, -1)     # v inverte nas paredes em y

    # fluxos nas faces em x
    Q = (hE[:, 1:-1], huE[:, 1:-1], hvE[:, 1:-1])
    F = fx(*Q)
    cx = np.abs(Q[1]/Q[0]) + np.sqrt(g*Q[0])
    ax = np.maximum(cx[:-1], cx[1:])
    Fx = [0.5*(F[k][:-1] + F[k][1:]) - 0.5*ax*(Q[k][1:] - Q[k][:-1]) for k in range(3)]

    # fluxos nas faces em y
    P = (hE[1:-1, :], huE[1:-1, :], hvE[1:-1, :])
    G = fy(*P)
    cy = np.abs(P[2]/P[0]) + np.sqrt(g*P[0])
    ay = np.maximum(cy[:, :-1], cy[:, 1:])
    Gy = [0.5*(G[k][:, :-1] + G[k][:, 1:]) - 0.5*ay*(P[k][:, 1:] - P[k][:, :-1]) for k in range(3)]

    h  -= dt/dx*(Fx[0][1:] - Fx[0][:-1]) + dt/dy*(Gy[0][:, 1:] - Gy[0][:, :-1])
    hu -= dt/dx*(Fx[1][1:] - Fx[1][:-1]) + dt/dy*(Gy[1][:, 1:] - Gy[1][:, :-1])
    hv -= dt/dx*(Fx[2][1:] - Fx[2][:-1]) + dt/dy*(Gy[2][:, 1:] - Gy[2][:, :-1])
    t += dt

plt.imshow(h.T, origin="lower", extent=(0, L, 0, L), cmap="viridis")
plt.colorbar(label="h")
plt.xlabel("x"); plt.ylabel("y")
plt.show()
