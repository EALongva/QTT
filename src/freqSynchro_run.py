# running the freqSynchro method

from methods import *


M = 24
S = 1000
N = 1000
res = 1000

omega0 = 1.0
dOmega = 0.02

finaltime = 20.0
U = [Up_z, Um_z]
temperature = 0.5

### likely wrong with current hamiltonian? testing it anyways
p0 = 1/(1 + np.exp(-1/temperature))
p1 = 1 - p0
psi0_final = np.sqrt(p0) * bas0 + np.sqrt(p1) * bas1

psi0 = psi0_final



freqSynchro(M, S, N, omega0, dOmega, finaltime, psi0, U, temperature, theta=0.1, sig_strength=0.1, seed=1337, res=res, test=1.0)