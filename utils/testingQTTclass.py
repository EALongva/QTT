# testingQTTclass

from QTT import *
import multiprocessing as mp
from tqdm import tqdm #love this

def f(S, N, seed):

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

# run

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

