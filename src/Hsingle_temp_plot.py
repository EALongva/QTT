# Testing that the trajectories are consistent with solutions of the Lindblad equation

import matplotlib.pyplot as plt
import numpy as np
import qutip as qp
import random as rnd
import time as time
import math as math
from scipy.linalg import expm
from mpl_toolkits.mplot3d import Axes3D

""" utility """

def dag(x):
    # hermitian conjugate of 3 dim array with (2,1) matrixes along first axis
    return np.transpose(np.conj(x), (0,2,1))

# importing trajectories 

### SINGLE TEMPERATURE
location    = "../dat/RUD/"
loadname    = location + "Hsim_S_4000_N_2000"
QT          = np.load(loadname + ".npy")
times       = np.load(loadname + "_times.npy")
info        = np.load(loadname + "_info.npy")
psi0        = np.load(loadname + "_psi0.npy")
H           = np.load(loadname + "_ham.npy")

### printing out the info for the datafile
print("number of trajectories: ", info[0])
print("number of timesteps: ", info[1])
print("temperature: ", info[2])
print("interaction strength parameter theta: ", info[3])
print("Hamiltonian: ", H)

temperature = info[2] #temps[2]
dt          = times[-1] / info[1]
theta       = info[3] # the value for theta must be consistent globally (you have to check the program simulating the trajectories)

### Trajectories solution ###

rhos = np.zeros((QT[:,0,0,0].size,QT[0,:,0,0].size, 2, 2), dtype=QT.dtype) # array of all density matrices

for s in range(QT[:,0,0,0].size):
    rhos[s] = QT[s] @ dag(QT[s])

rho = np.mean(rhos, axis=0)

#rho = rhos[0]

rx = rho[:,1,0] + rho[:,0,1]
ry = 1j*(rho[:,1,0] - rho[:,0,1])
rz = rho[:,0,0] - rho[:,1,1]

R = [rx.real, ry.real, rz.real]

xyz = ['x', 'y', 'z']
alpha = [0.9, 0.7, 0.5]
lines = ['-', '--', '-.']


### Lindblad solution ###

#H = 0.0*qp.sigmax() #0.5*(omega0 - omega)*qp.sigmaz() + 0.5*omega1*qp.sigmax()

A_SWAPm = qp.basis(2,0)*qp.basis(2,1).dag()
A_SWAPp = qp.basis(2,1)*qp.basis(2,0).dag()

a = np.sqrt( (theta**2) / (2.0*dt)) # with theta 0.01 and N = 10000 and T = 1.0 this is 1.0, but with T not equal to 1.0 dt changes and this is no longer 1.0

n       = 1/(np.exp(1/temperature)-1)

gammap  = np.sqrt(n)
gammam  = np.sqrt(n+1)

#A = A_SWAPm.copy()

c_ops = [gammam*a*A_SWAPm, gammap*a*A_SWAPp]

p0 = 1/(1 + np.exp(-1/temperature))
p1 = 1 - p0

#psi0 = np.sqrt(p0) * qp.basis(2,0) + np.sqrt(p1) * qp.basis(2,1)

# init state is loaded at the beginning

result = qp.mesolve(qp.Qobj(H), qp.Qobj(psi0), times, c_ops, [])

rhoLB = np.array(result.states)

lx = rhoLB[:,1,0] + rhoLB[:,0,1]
ly = 1j*(rhoLB[:,1,0] - rhoLB[:,0,1])
lz = rhoLB[:,0,0] - rhoLB[:,1,1]

L = [lx, ly, lz]


"""
### compares all bloch vector components on individual plots and the relative error for each

plt.style.use('ggplot')
fig = plt.figure()
ax1 = fig.add_subplot(131)

for r, u, a, l in zip(R, xyz, alpha, lines):
    ax1.plot(times, r, l, label='LB' + u, alpha=a)

ax1.set_xlabel('t')
ax1.set_title('Quantum trajectories average')
ax1.legend(['x', 'y', 'z'])

ax2 = fig.add_subplot(132)

for r, u, a, l in zip(L, xyz, alpha, lines):        # the notation for l L and r R gets a bit confusing here
    ax2.plot(times, r, l, label='LB' + u, alpha=a)

ax2.set_xlabel('t')
ax2.set_title('Lindblad solution')
ax2.legend(['x', 'y', 'z'])

### relative error ###

ex = np.abs( (rx - lx)/lx )
ey = np.abs( (ry - ly)/ly )
ez = np.abs( (rz - lz)/lz )

E = [ex, ey, ez]

ax3 = fig.add_subplot(133)

for r, u, a, l in zip(E, xyz, alpha, lines):        # the notation for l L and r R, e E gets a bit confusing here but meh
    ax3.plot(times, r, l, label='LB' + u, alpha=a)

ax3.set_xlabel('t')
ax3.set_title('Relative error')
ax3.legend(['x', 'y', 'z'])

fig.suptitle("Comparison of trajectories vs Lindblad solution, bloch vector components\nRandom Unitary Diffusion with SWAP+ and SWAP- interaction (p+ = 0.5)")
fig.set_size_inches(16,9)
figname = "../../fig/RUD/QT_test_SWAPpm_test.png"
fig.savefig(figname, dpi=400)
"""

"""
plt.style.use('ggplot')
fig = plt.figure()
ax = fig.add_subplot()
ax.plot(times, rz.real, label='QTT')
ax.plot(times, lz.real, label='LB')
ax.legend()
ax.set_xlabel('t')
ax.set_ylabel(r'$\langle \sigma_z \rangle$')
title = r'comparison of $\langle \sigma_z \rangle$ for QTT and LB solutions, ' + r'$\theta$ = ' + str(theta) + ', temperature: ' + str(temperature) + ', trajectories: ' + str(info[0]) + ', timesteps: ' + str(info[1])
ax.set_title(title)
"""

plt.style.use('ggplot')
fig = plt.figure()
ax1 = fig.add_subplot(131)
ax1.plot(times, rx.real, label='QTT')
ax1.plot(times, lx.real, label='LB')
ax1.legend()
ax1.set_xlabel('t')
ax1.set_ylabel(r'$\langle \sigma_x \rangle$')

ax2 = fig.add_subplot(132)
ax2.plot(times, ry.real, label='QTT')
ax2.plot(times, ly.real, label='LB')
ax2.legend()
ax2.set_xlabel('t')
ax2.set_ylabel(r'$\langle \sigma_y \rangle$')

ax3 = fig.add_subplot(133)
ax3.plot(times, rz.real, label='QTT')
ax3.plot(times, lz.real, label='LB')
ax3.legend()
ax3.set_xlabel('t')
ax3.set_ylabel(r'$\langle \sigma_z \rangle$')

rx = rho[:,1,0] + rho[:,0,1]
ry = 1j*(rho[:,1,0] - rho[:,0,1])
rz = rho[:,0,0] - rho[:,1,1]

delta = (H[0,0] - H[1,1])
epsilon = np.abs(1j*(H[1,0] - H[0,1]))

bigtitle = (r'comparison of $\langle \sigma \rangle$ for QTT and LB solutions, ' + r'$\theta$ = ' + str(theta) + \
 ', temperature: ' + str(temperature) + ', trajectories: ' + str(info[0]) + ', timesteps: ' + str(info[1]) + \
 '\n' + r'$H = \frac{1}{2}\Delta \sigma_z + \frac{1}{2}\epsilon \sigma_y$' + ' with ' + r'$\Delta = $' + str(delta.real) + r' and $\epsilon = $' + str(epsilon.real))
 

fig.suptitle(bigtitle)
fig.set_size_inches(16,9)

figname = "../fig/RUD/Hsingletemp_" + "traj_" + str(info[0]) + ".png"
fig.savefig(figname, dpi=400)

plt.show()