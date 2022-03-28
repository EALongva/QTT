# comparing trajectories and Lindblad - <sigma_z> for varying temperatures

import matplotlib.pyplot as plt
import numpy as np
import qutip as qp
import random as rnd
import time as time
import math as math
from scipy.linalg import expm
from mpl_toolkits.mplot3d import Axes3D

def dag(x):
    # hermitian conjugate of 3 dim array with (2,1) matrixes along first axis
    return np.transpose(np.conj(x), (0,2,1))

location    = "../dat/RUD/"
loadname    = location + "Hsim_S_2000_N_1500_NTemps_5" # change
QT          = np.load(loadname + ".npy")
times       = np.load(loadname + "_times.npy")
temps       = np.load(loadname + "_temps.npy")
info        = np.load(loadname + "_info.npy")
psi0        = np.load(loadname + "_psi0.npy")
H           = np.load(loadname + "_ham.npy")

#print(times.size)
#print(QT[0,0,:,0,0].size)


### Lindblad solution ###

A_SWAPm = qp.basis(2,0)*qp.basis(2,1).dag()
A_SWAPp = qp.basis(2,1)*qp.basis(2,0).dag()

# to get the correct scaling we need to have theta (must know from the dataset) and dt = times[-1]/times.size

N = info[1]
finaltime = times[-1]
dt = finaltime/N
theta = info[3]

a = np.sqrt( 0.5*(theta**2) / dt)

#psi0  = (qp.basis(2,0) + qp.basis(2,1))/np.sqrt(2) # init state

### QTT solution ###

rhos = np.zeros((QT[:,0,0,0,0].size, QT[0,:,0,0,0].size,QT[0,0,:,0,0].size, 2, 2), dtype=QT.dtype) # array of all density matrices

plt.style.use('ggplot')
fig = plt.figure()
plt.gca().invert_yaxis()
ax1 = fig.add_subplot()

clr = ['b', 'g', 'r', 'c', 'y']

for t in range(temps.size):
    for s in range(QT[0,:,0,0,0].size):
        rhos[t,s] = QT[t,s] @ dag(QT[t,s])

    rho = np.mean(rhos[t], axis=0)

    rz = rho[:,0,0] - rho[:,1,1]

    ax1.plot(times, rz.real, color=clr[t])

    gammap = np.sqrt(1/(np.exp(1/temps[t]) - 1))
    gammam = np.sqrt(1/(1 - np.exp(-1/temps[t])))

    c_ops = [gammam*a*A_SWAPm, gammap*a*A_SWAPp]
    result = qp.mesolve(qp.Qobj(H), qp.Qobj(psi0), times, c_ops, [])

    rhoLB = np.array(result.states)
    lz = (rhoLB[:,0,0] - rhoLB[:,1,1])

    ax1.plot(times, lz.real, color=clr[t], linestyle='--', label=str(round(temps[t],1)))
    
    #Tz = (1 - np.exp(-1/temps[t]))/(1 + np.exp(-1/temps[t]))
    #ax1.axhline(y=Tz, color=clr[t], linestyle="-", label=str(temps[t]))


ax1.set_xlabel('t')
ax1.set_ylabel(r'$\langle \sigma_z \rangle$')

delta = (H[0,0] - H[1,1])
epsilon = np.abs(1j*(H[1,0] - H[0,1]))

title = (r'comparison of $\langle \sigma_z \rangle$ for QTT and LB solutions, ' + r'$\theta$ = ' + str(theta) + \
', trajectories: ' + str(info[0]) + ', timesteps: ' + str(info[1]) + \
 '\n' + r'$H = \frac{1}{2}\Delta \sigma_z + \frac{1}{2}\epsilon \sigma_y$' + ' with ' + r'$\Delta = $' + str(delta.real) + r' and $\epsilon = $' + str(epsilon.real))


ax1.set_title(title)
ax1.legend()

"""
plt.style.use('ggplot')
fig = plt.figure()
ax = fig.add_subplot()
ax.plot(times, rz.real, label='QTT')
ax.plot(times, lz.real, label='LB')
ax.legend()
ax.set_xlabel('t')
ax.set_ylabel(r'$\langle \sigma_x \rangle$')

title = r'comparison of $\langle \sigma_x \rangle$ for QTT and LB solutions, ' + r'$\theta$ = ' + str(theta) + ', temperature: ' + str(temperature) + ', trajectories: ' + str(info[0]) + ', timesteps: ' + str(info[1])

ax.set_title(title)
"""

fig.set_size_inches(16,9)

figname = "../fig/RUD/Hmultitemp_" + "traj_" + str(info[0]) + ".png"
fig.savefig(figname, dpi=400)

plt.show()