# test environment

import matplotlib.pyplot as plt
import numpy as np
import qutip as qp
import random as rnd
import time as time
from datetime import timedelta
import math as math
from scipy.linalg import expm
from mpl_toolkits.mplot3d import Axes3D

"""
bas0 = np.array(([1.0], [0.0]), dtype='complex128')
bas1 = np.array(([0.0], [1.0]), dtype='complex128')

psi0 = np.sqrt(0.5)*(bas0 + bas1)

temperature = 0.5

theta = 0.01

# interaction hamiltonians
Usp = np.array(([0,0,0,1],[0,-1,0,0],[0,0,0,0],[1,0,0,1]), dtype='complex128')
Usm = np.array(([0,0,0,0],[0,-1,1,0],[0,1,0,0],[0,0,0,1]), dtype='complex128')

# environment for RUD
env = np.copy(bas0)

# full state system x environment
Psi = np.kron(psi0, env)

# interaction strength parameters
pdensity        = 1/(np.exp(1/temperature)-1)
gammap          = pdensity
gammam          = pdensity + 1

# temperature dependent interaction strength
thetap          = np.sqrt(gammap) * theta
thetam          = np.sqrt(gammam) * theta

# second order expansion of time evolution operator
U_p = np.eye(4) -1j*thetap*Usp - (thetap**2/2) * Usp @ Usp
U_m = np.eye(4) -1j*thetam*Usm - (thetam**2/2) * Usm @ Usm


# test out evolving one timestep

newPsi = U_m @ Psi

print(U_m, Psi)
print(newPsi)

psi_p  =  (newPsi[0] + newPsi[1])*bas0 + (newPsi[2] + newPsi[3])*bas1
                
prob_p   = 0.5* ( np.conj(psi_p).T @ psi_p )

psi_m = (newPsi[0] - newPsi[1])*bas0 + (newPsi[2] - newPsi[3])*bas1

prob_m = 0.5 * ( np.conj(psi_m).T @ psi_m ) 

#print(prob_p + prob_m)
"""

"""
fnum = 0.01
fnumstr = str(fnum).replace('.', 'p')
print(fnumstr)
"""

"""
import matplotlib.pyplot as plt
import numpy as np

ax = plt.figure().add_subplot(projection='3d')

# Make the grid
x, y, z = np.meshgrid(np.arange(-0.8, 1, 0.2),
                      np.arange(-0.8, 1, 0.2),
                      np.arange(-0.8, 1, 0.8))

# Make the direction data for the arrows
u = np.sin(np.pi * x) * np.cos(np.pi * y) * np.cos(np.pi * z)
v = -np.cos(np.pi * x) * np.sin(np.pi * y) * np.cos(np.pi * z)
w = (np.sqrt(2.0 / 3.0) * np.cos(np.pi * x) * np.cos(np.pi * y) *
     np.sin(np.pi * z))

ax.quiver(x, y, z, u, v, w, length=0.1, normalize=True)

plt.show()
"""