# Testing that the trajectories are consistent with solutions of the Lindblad equation

import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
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
loadname    = location + "Hsim_S_4000_N_6000_tet_0p01"
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
print("total sim time: ", times[-1])

temperature = info[2] #temps[2]
dt          = times[-1] / info[1]
theta       = info[3] # the value for theta must be consistent globally (you have to check the program simulating the trajectories)

rhos = np.zeros((QT[:,0,0,0].size,QT[0,:,0,0].size, 2, 2), dtype=QT.dtype) # array of all density matrices


# master loop - computing freq for each individual traj and storing it in 'frequencies'-array

frequencies = np.zeros(QT[:,0,0,0].size)

ping0 = time.perf_counter() #timing for loop

for s in range(QT[:,0,0,0].size):
    
    rho = QT[s] @ dag(QT[s])

    rx = rho[:,1,0] + rho[:,0,1]
    ry = 1j*(rho[:,1,0] - rho[:,0,1])

    eps = 1e-1 # sensitivity for detecting minima

    angles = np.arctan2(ry.real, rx.real)
    abs_angles = np.abs(angles)

    minima = ma.masked_less(abs_angles, eps).mask

    
    # for loop to determine suitable sensitivity in cut off
    
    count = 0; long_count = 0

    for i in range(minima.size - 1):

        if minima[i] or minima[i + 1]:
            
            if count > long_count:

                long_count = count
            
            count = 0

        else:

            count += 1

    sens = int(np.floor(long_count/2)) # computing sensitivity

    
    # for loop to cut off extra maxima around peaks due to diffusion "noise"

    cut_minima = minima[sens:]
    cutoff_minima = np.copy(minima)
    cutoff_minima[:sens] = 0

    for i in range(cutoff_minima.size):

        if cutoff_minima[i]:

            cutoff_minima[ i + 1 : (i + sens) ] = 0


    # computing frequency

    rotcount = np.sum(cutoff_minima)

    total_angle = 2*np.pi * rotcount + abs_angles[-1]

    rotations = total_angle/(2*np.pi)

    frequencies[s] = rotations / times[-1]



ping1 = time.perf_counter()
forlooptime = ping1 - ping0
print("time for loop: ", forlooptime)

avgFreq = np.mean(frequencies)
print("estimated average frequency: ", avgFreq)

plt.style.use('ggplot')
fontsize=12

fig = plt.figure()
ax1 = fig.add_subplot()

ax1.plot(frequencies)
ax1.axhline(y=avgFreq, color='blue', linestyle="-")
ax1.set_xlabel('trajectory')
ax1.set_ylabel('frequency')


fig.suptitle('Single trajectory simulation on the Bloch sphere\n' \
+ 'with theta= ' + str(theta) + ' N= ' + str(N) + ' T= ' + str(T))
fig.set_size_inches(16,9)
fig.tight_layout()

plt.show()








"""
rx = rho[:,1,0] + rho[:,0,1]
ry = 1j*(rho[:,1,0] - rho[:,0,1])
rz = rho[:,0,0] - rho[:,1,1]

plt.style.use('ggplot')
fontsize=12

fig = plt.figure()
ax1 = fig.add_subplot(121)

ax1.plot(rx.real, ry.real)
ax1.set_xlabel(r'$\langle \sigma_x \rangle$')
ax1.set_ylabel(r'$\langle \sigma_y \rangle$')


# computing angles of sigx sigy - plane using np arctan2 and plotting it
# note arctan2( y-coordinates, x-coordinates )



angles = np.arctan2(ry.real, rx.real)
abs_angles = np.abs(angles)

eps = 1e-1

minima = ma.masked_less(abs_angles, eps).mask

# for loop to determine suitable sensitivity in cut off
count = 0; long_count = 0

for i in range(minima.size - 1):

    if minima[i] or minima[i + 1]:
        
        if count > long_count:

            long_count = count
        
        count = 0

    else:

        count += 1



print("counting false chain: ", long_count)

# computing sensitivity

sens = int(np.floor(long_count/2))

print("true: ", sumTrue, "false: ", sumFalse, "sensitivity: ", sens)


# for loop to cut off extra maxima around peaks due to diffusion "noise"

cut_minima = minima[sens:]
cutoff_minima = np.copy(minima)
cutoff_minima[:sens] = 0

for i in range(cutoff_minima.size):

    if cutoff_minima[i]:

        cutoff_minima[ i + 1 : (i + sens) ] = 0

ping1 = time.perf_counter()
forlooptime = ping1 - ping0
print("time for loop: ", forlooptime)


# computing frequency

rotcount = np.sum(cutoff_minima)

total_angle = 2*np.pi * rotcount + abs_angles[-1]

rotations = total_angle/(2*np.pi)

freq = rotations / times[-1]

print("total angle rotated: ", total_angle, "number of rotations: ", rotations, "estimated freq: ", freq)
"""
