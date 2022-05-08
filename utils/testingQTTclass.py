# testingQTTclass

from QTT import *
import multiprocessing as mp
from tqdm import tqdm #love this

def f(S, N, seed):

    # example func

    print(seed)
    rnd.seed(seed)

    MC_traj = np.zeros((S, N, 2, 1), dtype='complex128')

    for s in range(S):

        traj_result = np.zeros((N, 2, 1), dtype='complex128')

        for i in range(N):

            p = rnd.random()
            time.sleep(0.001)

            if p >= 0.5:

                traj_result[i] = bas0

            else:

                traj_result[i] = bas1

        MC_traj[s] = traj_result

    return MC_traj

def mpTest(S, N, seed, ncpu):

    #ncpu = number of CPUs to use

    args = []

    for i in range(ncpu):
        dseed = seed + i
        args.append([S, N, dseed])

    pool = mp.Pool(ncpu)
    result = pool.starmap(f, args)
    pool.close()

    avgres = np.mean(result, axis=(0,1))

    print(avgres)

    return 0

# para test run
"""
ncpu = 4
S = int(ncpu * 200)
N = 3
seed = 200

#test = f(S, N, seed)
#print(test)

start = time.perf_counter()
mpTest(S, N, seed, ncpu)
fin = time.perf_counter()

print('time: ', fin - start)

"""


def qttMC(S, N, seed):

    env = 'z'
    meas_basis = 'x'
    simtime = 30.0
    delta = 0
    epsilon = 0
    psi = bas0

    #dt = 0.01
    #total_simtime = N * dt

    class_instance = QTT(env, meas_basis, seed=seed)
    class_instance.system_hamiltonian(delta, epsilon)

    class_instance.MC(S, psi, N, simtime)

    #rho_result = class_instance.rho() #compute rho after taking the mean of all trajectories IMPORTANT

    return class_instance.mcResult
    

def pool_qttMC(S, N, seed, ncpu):

    args = []
    for cpu in range(ncpu):
        dseed = seed + cpu
        args.append([S, N, dseed])

    pool = mp.Pool(ncpu)
    result = pool.starmap(qttMC, args)
    pool.close()



    return result


def pool_qttMC_alt(S, N, seed, ncpu):

    args = []
    for d in tqdm(range(S)):
        dseed = seed + d
        args.append([S, N, dseed])

    pool = mp.Pool(ncpu)
    result = pool.starmap(qttMC, args)
    pool.close()

    

    return result


"""
# initial test

# input variables
S = 800
N = 10000
seed = 1234
ncpu = 4

#pool_qttMC(S, N, seed, ncpu)
#pool_qttMC_alt(S, N, seed, ncpu)

env = 'z'
meas_basis = 'x'
simtime = 30.0
delta = 0
epsilon = 0
psi = bas0

#dt = 0.01
#total_simtime = N * dt

class_instance = QTT(env, meas_basis, seed=seed)
class_instance.system_hamiltonian(delta, epsilon)

# testing the parallel version of the monte carlo simulation

start_time = time.perf_counter()

class_instance.paraMC(S, bas0, N, simtime, ncpu)

finish_time = time.perf_counter()

print('MC simulation time: ', (finish_time-start_time)) # approx 2.5 minutes for S=800 N=10000

"""

def results(S, N, simtime, psi0, theta, delta, epsilon, seed):

    class_instance = QTT(env, meas_basis, theta=theta, seed=seed)
    class_instance.system_hamiltonian(delta, epsilon)

    start_time = time.perf_counter()

    class_instance.paraMC(S, psi0, N, simtime, ncpu)
    class_instance.rho()

    print('timesteps ', class_instance.timesteps)

    finish_time = time.perf_counter()
    print('MC trajectories simulation time: ', (finish_time-start_time))

    path = '../data/QTT_class_test/'
    filename_result_qtt = path + 'resultQTT_' + 'S_' + str(S) + '_N_' + str(N) + '_delta_' + str(delta).replace('.', 'p') + '_eps_' + str(epsilon).replace('.', 'p')

    np.save(filename_result_qtt, class_instance.rhoResult)

    class_instance.lindblad()

    path = '../data/QTT_class_test/'
    filename_result_LB = path + 'resultLB_' + 'S_' + str(S) + '_N_' + str(N) + '_delta_' + str(delta).replace('.', 'p') + '_eps_' + str(epsilon).replace('.', 'p')

    np.save(filename_result_LB, class_instance.rhoLB)

    return 0


def plot_results(S, N, simtime, psi0, theta, delta, epsilon, seed):

    class_instance = QTT(env, meas_basis, theta=theta, seed=seed)
    class_instance.system_hamiltonian(delta, epsilon)

    path = '../data/QTT_class_test/'
    filename_result_qtt = path + 'resultQTT_' + 'S_' + str(S) + '_N_' + str(N) + '_delta_' + str(delta).replace('.', 'p') + '_eps_' + str(epsilon).replace('.', 'p')

    resultQTT = np.load(filename_result_qtt + '.npy')

    class_instance.lindblad()

    path = '../data/QTT_class_test/'
    filename_result_LB = path + 'resultLB_' + 'S_' + str(S) + '_N_' + str(N) + '_delta_' + str(delta).replace('.', 'p') + '_eps_' + str(epsilon).replace('.', 'p')

    resultLB = np.load(filename_result_LB + '.npy')

    print(resultLB.shape)
    print(resultQTT.shape)

    # get the bloch vectors and time array

    class_instance.rhoResult = resultQTT
    class_instance.rhoLB = resultLB
    class_instance.finaltime = simtime
    class_instance.timesteps = N

    blochvecs_QTT = class_instance.blochvec()
    blochvecs_LB = class_instance.blochvec(LB=True)
    time_array = class_instance.times()

    ### plotting

    plt.style.use('ggplot')
    fig = plt.figure()
    ax1 = fig.add_subplot(131)
    ax1.plot(time_array, blochvecs_QTT[0].real, label='QTT')
    ax1.plot(time_array, blochvecs_LB[0].real, label='LB')
    ax1.legend()
    ax1.set_xlabel('t')
    ax1.set_ylabel(r'$\langle \sigma_x \rangle$')

    ax2 = fig.add_subplot(132)
    ax2.plot(time_array, blochvecs_QTT[1].real, label='QTT')
    ax2.plot(time_array, blochvecs_LB[1].real, label='LB')
    ax2.legend()
    ax2.set_xlabel('t')
    ax2.set_ylabel(r'$\langle \sigma_y \rangle$')

    ax3 = fig.add_subplot(133)
    ax3.plot(time_array, blochvecs_QTT[2].real, label='QTT')
    ax3.plot(time_array, blochvecs_LB[2].real, label='LB')
    ax3.legend()
    ax3.set_xlabel('t')
    ax3.set_ylabel(r'$\langle \sigma_z \rangle$')

    bigtitle = (r'comparison of $\langle \sigma \rangle$ for QTT and LB solutions, ' + r'$\theta$ = ' + str(class_instance.theta) + \
    ', temperature: ' + str(class_instance.temperature) + ', trajectories: ' + str(S) + ', timesteps: ' + str(N) + \
    '\n' + r'$H = \frac{1}{2}\Delta \sigma_z + \frac{1}{2}\epsilon \sigma_y$' + ' with ' + r'$\Delta = $' + str(delta) + r' and $\epsilon = $' + str(epsilon))

    fig.suptitle(bigtitle)
    fig.set_size_inches(16,9)

    figname = '../figure/QTT_class_test/Hsingletemp_' + 'traj_' + str(S) + '_timesteps_' + str(N) + '.png'
    fig.savefig(figname, dpi=400)

    return 0

"""
# input variables
S = 400
N = 100000
seed = 1234
ncpu = 4

#pool_qttMC(S, N, seed, ncpu)
#pool_qttMC_alt(S, N, seed, ncpu)

env = 'z'
meas_basis = 'x'
simtime = 1000.0
delta = 0
epsilon = 0
psi0 = xplus

#results(S, N, simtime, psi0, delta, epsilon, seed)
plot_results(S, N, simtime, psi0, delta, epsilon, seed)
"""

"""
# input variables
S = 800
N = 10000
seed = 1234
ncpu = 4

#pool_qttMC(S, N, seed, ncpu)
#pool_qttMC_alt(S, N, seed, ncpu)

env = 'z'
meas_basis = 'x'
simtime = 100.0
theta = 0.05
delta = 0
epsilon = 0
psi0 = xplus

#results(S, N, simtime, psi0, theta, delta, epsilon, seed)
#plot_results(S, N, simtime, psi0, theta, delta, epsilon, seed)
# """

