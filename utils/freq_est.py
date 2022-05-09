# using QTT class to calculate frequency for a range of signal frequencies

from QTT import *

def main(M, S, N, theta, simtime, psi0, ncpu, burnin, delta, ddelta, epsilon):

    # input variables

    env = 'z'
    meas_basis = 'x'
    
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

    path = '../data/freq/'
    loadname = path + 'freq' + '_M_' + str(M) + '_S_' + str(S) + '_N_' + str(N) + \
        '_burnin_N_' + str(burnin) + '_dt_' + str(dt).replace('.', 'p') + \
        '_theta_' + str(theta).replace('.', 'p') + '_delta_' + str(delta).replace('.', 'p') + \
        '_ddelta_' + str(ddelta).replace('.', 'p') + '_eps_' + str(epsilon).replace('.', 'p')
    
    result = np.load(loadname + '.npy')


    class_instance = QTT(env, meas_basis, theta=theta, seed=seed)
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

    figname = "../figure/burnin/" + 'burninest' + '_S_' + str(S) + '_N_' + str(N) + \
        '_burnin_N_' + str(burnin) + '_dt_' + str(dt).replace('.', 'p') + \
        '_theta_' + str(theta_value).replace('.', 'p') + '_delta_' + str(delta).replace('.', 'p') + \
        '_eps_' + str(epsilon).replace('.', 'p') + '.png'
    fig.savefig(figname, dpi=400)

    plt.show()

    return 0




### variables

M = 5
S = 40
N = 100
burnin = 2000

delta = 0.001
ddelta = 0.003
epsilon = 0.001
theta = 0.01

simtime = 1.0
seed = 1948571
ncpu = 4
psi0 = xplus

main(M, S, N, theta, simtime, psi0, ncpu, burnin, delta, ddelta, epsilon)
plot(M, S, N, theta, simtime, psi0, ncpu, burnin, delta, ddelta, epsilon)