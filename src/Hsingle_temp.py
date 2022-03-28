# simulating single temperature system varying the initial states

from methods import *

# some simulation variables
timesteps = 20000
finaltime = 20.0
dt = finaltime / timesteps
traj = 1
theta = 0.01
nTemp = 1
resolution = 1000
temperature = 0.5


p0 = 1/(1 + np.exp(-1/temperature))
p1 = 1 - p0
psi0_final = np.sqrt(p0) * bas0 + np.sqrt(p1) * bas1 #final state for some temperature

psi0_yminus = np.sqrt(0.5) * (bas0 - 1j*bas1)
psi0_yplus = np.sqrt(0.5) * (bas0 - 1j*bas1)
psi0_xplus = np.sqrt(0.5) * (bas0 + bas1)
psi0_xminus = np.sqrt(0.5) * (bas0 - bas1)

# defining the Hamiltonian

eps = 1.0
omega0 = 1.0
omega = 1.1
delta = omega0 - omega

#H = 1.0*(0.5*delta*sigmaz + 0.5*eps*sigmay)
#H = np.array( 0.5*omega0*sigmaz + 1j * 0.25 * eps * (np.exp(1j*omega) * sigmam - np.exp(-1j*omega) * sigmap ) , dtype='complex128' )
H = np.array( (0.5 * delta * sigmaz + 0.5 * eps * sigmay ), dtype='complex128' )


#(S, N, finaltime, psi0, H, temperature, theta=0.1, seed=1337, res=1000, test=1.0)

simulationHam(traj, timesteps, finaltime, psi0_final, H, temperature, theta, seed=281740, res=resolution, test=1.0)


### end