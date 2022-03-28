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
loadname    = location + "Hsim_S_1_N_20000_tet_0p01"
QT          = np.load(loadname + ".npy")
times       = np.load(loadname + "_times.npy")
info        = np.load(loadname + "_info.npy")
psi0        = np.load(loadname + "_psi0.npy")
H           = np.load(loadname + "_ham.npy")

print(QT.shape)


### printing out the info for the datafile
print("number of trajectories: ", info[0])
S = info[0]
print("number of timesteps: ", info[1])
N = info[1]
print("temperature: ", info[2])
T = info[2]
print("interaction strength parameter theta: ", info[3])

temperature = info[2] #temps[2]
dt          = times[-1] / info[1] # info[-1] is info[3] so theta hm
theta       = info[3] # the value for theta must be consistent globally (you have to check the program simulating the trajectories)

### Trajectories solution ###

rho = QT[0] @ dag(QT[0])

#print(rhos[:,0,0,0].size)

#rho = np.mean(rhos, axis=0) #dont take the averages, plot the trajectories

#rho = rhos[0]

#rx = rho[:,1,0] + rho[:,0,1]
#ry = 1j*(rho[:,1,0] - rho[:,0,1])
#rz = rho[:,0,0] - rho[:,1,1]

#R = [rx.real, ry.real, rz.real]

#xyz = ['x', 'y', 'z']
#alpha = [0.9, 0.7, 0.5]
#lines = ['-', '--', '-.']

### Bloch sphere plot of trajectories


#print(rhos[:5,0,0,0].size)


plt.style.use('ggplot')
fontsize=12

fig = plt.figure()
ax1 = fig.add_subplot(121, projection='3d')

ax1.set_title('Trajectory on Bloch sphere',fontsize=fontsize)
ax1.view_init(-30,60)
sphere = qp.Bloch(axes=ax1)
sphere.point_size = [0.3,0.3,0.3,0.3]

rx = rho[:,1,0] + rho[:,0,1]
ry = 1j*(rho[:,1,0] - rho[:,0,1])
rz = rho[:,0,0] - rho[:,1,1]

R = [rx.real, ry.real, rz.real]

sphere.add_points(R)

sphere.make_sphere()


ax2 = fig.add_subplot(122)
ax2.plot(rx.real, ry.real)
ax2.set_xlabel(r'$\langle \sigma_x \rangle$')
ax2.set_ylabel(r'$\langle \sigma_y \rangle$')



fig.suptitle('Single trajectory simulation on the Bloch sphere\n' \
+ 'with theta= ' + str(theta) + ' N= ' + str(N) + ' T= ' + str(T))
fig.set_size_inches(16,9)
fig.tight_layout()

"""
filename = "../fig/RUD/bloch_singletraj_H_sigz.png"
fig.savefig(filename, dpi=400)
"""


plt.show()
