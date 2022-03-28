# simulating single temperature system varying the initial states

from methods import *

# some simulation variables
timesteps = 1000
finaltime = 10.0
dt = finaltime / timesteps
traj = 200
theta = 0.1
nTemp = 1
nStates = 10 
resolution = 200
temperature = 0.5


p0 = 1/(1 + np.exp(-1/temperature))
p1 = 1 - p0
psi0_final = np.sqrt(p0) * bas0 + np.sqrt(p1) * bas1 #final state for some temperature

psi0_yminus = np.sqrt(0.5) * (bas0 - 1j*bas1)
psi0_yplus = np.sqrt(0.5) * (bas0 - 1j*bas1)
psi0_xplus = np.sqrt(0.5) * (bas0 + bas1)
psi0_xminus = np.sqrt(0.5) * (bas0 - bas1)

print(np.pi)


phi = np.linspace(0, 2*np.pi - (2*np.pi/nStates), nStates)

alpha = np.cos(phi)
beta = 1j*np.sin(phi)

psi0 = np.zeros((nStates,2,1), dtype='complex128')

for i in range(nStates):

    psi0[i,:] = alpha[i]*bas0 + beta[i]*bas1

# defining the Hamiltonian

delta = 1.0
eps = 0.0

H = 1.0*(0.5*delta*sigmaz + 0.5*eps*sigmay)

#(S, N, finaltime, psi0, H, temperature, theta=0.1, seed=1337, res=1000, test=1.0)

simulationMultiInit(traj, timesteps, finaltime, psi0, H, temperature, theta, seed=1337, res=resolution, test=1.0)


### end