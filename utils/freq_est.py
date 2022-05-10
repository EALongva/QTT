# using QTT class to calculate frequency for a range of signal frequencies

from QTT import *

def main(M, S, N, theta, simtime, psi0, ncpu, burnin, delta, ddelta, epsilon):

    # input variables

    env = 'z'
    meas_basis = 'x'
    dt = simtime/N
    
    class_instance = QTT(env, meas_basis, theta=theta, seed=seed)

    class_instance.freqSynchro(M, S, N, burnin, psi0, simtime, ncpu, delta, ddelta, epsilon)

    result = class_instance.freqResult

    path = '../data/freq/'
    filename = path + 'freq' + '_M_' + str(M) + '_S_' + str(S) + '_N_' + str(N) + \
        '_burnin_N_' + str(burnin) + '_dt_' + str(dt).replace('.', 'p') + \
        '_theta_' + str(theta).replace('.', 'p') + '_delta_' + str(delta).replace('.', 'p') + \
        '_ddelta_' + str(ddelta).replace('.', 'p') + '_eps_' + str(epsilon).replace('.', 'p')

    np.save(filename, result)

    return 0


def plot(M, S, N, theta, simtime, psi0, ncpu, burnin, delta, ddelta, epsilon):

    dt = simtime/N

    path = '../data/freq/'
    loadname = path + 'freq' + '_M_' + str(M) + '_S_' + str(S) + '_N_' + str(N) + \
        '_burnin_N_' + str(burnin) + '_dt_' + str(dt).replace('.', 'p') + \
        '_theta_' + str(theta).replace('.', 'p') + '_delta_' + str(delta).replace('.', 'p') + \
        '_ddelta_' + str(ddelta).replace('.', 'p') + '_eps_' + str(epsilon).replace('.', 'p')
    
    result = np.load(loadname + '.npy')


    class_instance = QTT(env='z', meas_basis='x', theta=theta, seed=seed)
    temperature = class_instance.temperature

    omega = np.linspace(delta-ddelta, delta+ddelta, M)

    freqdiff = result - omega

    plt.style.use('ggplot')
    fig = plt.figure()

    ax = fig.add_subplot()
    ax.plot(omega, freqdiff)
    ax.set_xlabel(r'$\omega$')
    ax.set_ylabel(r'$\Omega - \omega$')

    title = r'Frequency difference in system $\Omega$ and external signal $\omega$, ' + \
            r'$\theta$ = ' + str(theta) + ', temperature: ' + str(temperature) + \
            ', \n trajectories: ' + str(int(S)) + ', timesteps: ' + str(int(N)) + \
            ', samples: ' + str(int(M)) + '\n' + r'$H = \Delta \sigma_z + \epsilon \sigma_y$'

    ax.set_title(title)
    fig.set_size_inches(16,9)

    path = '../figure/freq/'
    figname = path + 'eps' + str(epsilon).replace('.', '') +  '_delta' + str(delta).replace('.', '') + \
         '_freqEstimate' + '_S_' + str(S) + '_N_' + str(N) + \
        '_burnin_N_' + str(burnin) + '_dt_' + str(dt).replace('.', 'p') + \
        '_theta_' + str(theta).replace('.', 'p') + '_delta_' + str(delta).replace('.', 'p') + \
        '.png'
    fig.savefig(figname, dpi=400)

    plt.show()

    return 0


#def plot_bugged()



### variables

M = 16
S = 400
N = 10000
burnin = 200000

delta = 0.0025
ddelta = 0.0015
epsilon = 0.02
theta = 0.05

simtime = 100.0
seed = 1947571
ncpu = 4
psi0 = xplus

#main(M, S, N, theta, simtime, psi0, ncpu, burnin, delta, ddelta, epsilon)
#plot(M, S, N, theta, simtime, psi0, ncpu, burnin, delta, ddelta, epsilon)



### spamming runs:
# (to be fair i am still not sure if the way the frequency is calculated 
# is accurate for all epsilons)

#main(M, S, N, theta, simtime, psi0, ncpu, burnin, delta, ddelta, epsilon=0.1)
#plot(M, S, N, theta, simtime, psi0, ncpu, burnin, delta, ddelta, epsilon=0.1)
"""
main(M, S, N, theta, simtime, psi0, ncpu, burnin, delta, ddelta, epsilon=0.01)
plot(M, S, N, theta, simtime, psi0, ncpu, burnin, delta, ddelta, epsilon=0.01)

main(M, S, N, theta, simtime, psi0, ncpu, burnin, delta, ddelta, epsilon=0.001)
plot(M, S, N, theta, simtime, psi0, ncpu, burnin, delta, ddelta, epsilon=0.001)
"""


main(M, S, N, theta, simtime, psi0, ncpu, burnin, delta, ddelta, epsilon=0.01)
plot(M, S, N, theta, simtime, psi0, ncpu, burnin, delta, ddelta, epsilon=0.01)