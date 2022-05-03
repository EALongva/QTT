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



# will have to calculate rho in some other way than this

# input variables
S = 5
N = 3
seed = 1234
ncpu = 4

#pool_qttMC(S, N, seed, ncpu)
#pool_qttMC_alt(S, N, seed, ncpu)