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


def main4(S, N, theta, simtime, psi0, ncpu, burnin, epsilon):

    # input variables

    env = 'z'
    meas_basis = 'x'
    dt = simtime/N
    
    class_instance = QTT(env, meas_basis, theta=theta, seed=seed)

    result = class_instance.freqSynchro4(S, N, burnin, psi0, simtime, ncpu, epsilon)

    #result = class_instance.freqResult

    path = '../data/freq/'
    filename = path + 'theta' + str(theta).replace('.', '') + \
        '_eps' + str(epsilon).replace('.', '') + '_dt' + str(dt).replace('.', '') + \
        '_S_' + str(S) + '_N_' + str(N) + '_burnin_N_' + str(burnin) 

    np.save(filename, result)

    return 0

def plot4(S, N, theta, simtime, psi0, ncpu, burnin, epsilon):

    dt = simtime/N

    path = '../data/freq/'
    loadname = path + 'theta' + str(theta).replace('.', '') + \
        '_eps' + str(epsilon).replace('.', '') + '_dt' + str(dt).replace('.', '') + \
        '_S_' + str(S) + '_N_' + str(N) + '_burnin_N_' + str(burnin) 
    
    result = np.load(loadname + '.npy')

    #print(result.shape)

    measured_freq = []

    for MC in result:

        class_instance = QTT(env='z', meas_basis='x', theta=theta, seed=seed)
        tmp = class_instance.fft_freq(MC, dt)
        measured_freq.append(tmp)

    Omega = np.asarray(measured_freq)
    delta = np.array([-1.5*epsilon, -0.5*epsilon, 0.5*epsilon, 1.5*epsilon])

    print(Omega.shape)


    #plt.style.use('ggplot')
    fig = plt.figure()
    ax = fig.add_subplot()

    #ax.scatter(delta, Omega-delta)
    ax.scatter(delta, Omega-delta, s=4.0)

    print(Omega, delta)


    """
    ax1 = fig.add_subplot(221)
    ax1.scatter(rx, ry, color='r', s=1.5)
    ax1.set_xlabel(r'$\langle \sigma_x \rangle$')
    ax1.set_ylabel(r'$\langle \sigma_y \rangle$')
    ax1.set_title('z-axis')
    """


    ### making a polyfit to the freq points
    """
    polyfit = np.polyfit(delta, Omega-delta, 3)
    xarray = np.linspace(-2.5*epsilon, 2.5*epsilon, 200)
    poly = np.polyval(polyfit, xarray)

    ax.plot(xarray, poly, ls='--', color='black', alpha=0.7)
    ax.plot([0, 1], [1, 0], transform=ax.transAxes, ls='--', color='black', alpha=0.7)
    """
    
    fig.set_size_inches(16,9)

    path = '../figure/freq/freq4/'
    figname = path + 'testplot_' + 'eps' + str(epsilon).replace('.', '') + '.png'

    fig.savefig(figname, dpi=400)

    return 0


def mainDeltaArray(S, N, theta, simtime, psi0, ncpu, burnin, delta, epsilon):

    # input variables

    env = 'z'
    meas_basis = 'x'
    dt = simtime/N
    
    class_instance = QTT(env, meas_basis, theta=theta, seed=seed)

    result = class_instance.freqSimulationResult(S, N, burnin, psi0, simtime, ncpu, delta, epsilon)


    path = '../data/freq/'
    filename = path + 'theta' + str(theta).replace('.', '') + \
        '_eps' + str(epsilon).replace('.', '') + '_dt' + str(dt).replace('.', '') + \
        '_S_' + str(S) + '_N_' + str(N) + '_freqN_' + str(delta.size) + '_burninN_' + str(burnin) 

    np.save(filename, result)

    return 0

#"{0:.5f}".format(gamma_d)

#def plot_bugged()



### variables
"""
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

main(M, S, N, theta, simtime, psi0, ncpu, burnin, delta, ddelta, epsilon=0.01)
plot(M, S, N, theta, simtime, psi0, ncpu, burnin, delta, ddelta, epsilon=0.01)

main(M, S, N, theta, simtime, psi0, ncpu, burnin, delta, ddelta, epsilon=0.001)
plot(M, S, N, theta, simtime, psi0, ncpu, burnin, delta, ddelta, epsilon=0.001)
"""

#main(M, S, N, theta, simtime, psi0, ncpu, burnin, delta, ddelta, epsilon=0.01)
#plot(M, S, N, theta, simtime, psi0, ncpu, burnin, delta, ddelta, epsilon=0.01)



### for freqSynchro4

"""
# run 1
S = 80
N = 50000
burnin = 100000

epsilon = 0.02
theta = 0.01

dt = 0.01
simtime = N*dt
seed = 1947571
ncpu = 4
psi0 = xplus

#main4(S, N, theta, simtime, psi0, ncpu, burnin, epsilon)
plot4(S, N, theta, simtime, psi0, ncpu, burnin, epsilon)
"""
"""
# run 2
S = 80
N = 100000
burnin = 100000

epsilon = 0.04
theta = 0.01

dt = 0.01
simtime = N*dt
seed = 1947571
ncpu = 4
psi0 = xplus

#main4(S, N, theta, simtime, psi0, ncpu, burnin, epsilon) # results ready
plot4(S, N, theta, simtime, psi0, ncpu, burnin, epsilon)
"""


def delta_spacing(start, stop, M):

    M_ = int((M+1)/2)

    logstart = np.logspace(-abs(start), 1, M_, base=abs(start))
    logstop = np.logspace(-abs(stop), 1, M_, base=abs(stop))
    result = np.concatenate( [ -1 * np.flip(logstart[1:]) , logstop[1:] ])

    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
    print('delta values : \n', result, '\n')

    return result



S = 52
N = 100000
burnin = 100000

epsilon = 0.04
delta = epsilon * delta_spacing(-4.0, 4.0, 24)
theta = 0.01

dt = 0.01
simtime = N*dt
seed = 1947571
ncpu = 4
psi0 = xplus


mainDeltaArray(S, N, theta, simtime, psi0, ncpu, burnin, delta, epsilon)