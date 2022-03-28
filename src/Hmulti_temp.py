# simulating single temperature system varying the initial states

from methods import *

# some simulation variables
timesteps = 1500
finaltime = 15.0
dt = finaltime / timesteps
traj = 2000
theta = 0.1
nTemp = 5
resolution = 150
temperatures = np.linspace(0.1,0.9,nTemp)

psi0_yminus = np.sqrt(0.5) * (bas0 - 1j*bas1)
psi0_yplus = np.sqrt(0.5) * (bas0 - 1j*bas1)
psi0_xplus = np.sqrt(0.5) * (bas0 + bas1)
psi0_xminus = np.sqrt(0.5) * (bas0 - bas1)

# defining the Hamiltonian

delta = 0.5
eps = 0.2

H = 1.0*(0.5*delta*sigmaz + 0.5*eps*sigmay)

#(S, N, finaltime, psi0, H, temperature, theta=0.1, seed=1337, res=1000, test=1.0)

simulationHamTemps(traj, timesteps, finaltime, psi0_xplus, H, temperatures, theta, seed=1337, res=resolution, test=1.0)


### end