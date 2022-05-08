# burnin calculation
# single trajectory 1000 N save final state for each N = 1000 repeat starting at previous final state

from QTT import *

def main(S, N, delta, epsilon):

    result = np.zeros((S, 2, 1), dtype='complex128')

    env = 'z'
    meas_basis = 'x'
    seed = 381693

    #delta = 0.1
    #epsilon = 2.0

    dt = 0.01
    total_simtime = N * dt

    class_instance = QTT(env, meas_basis, seed=seed)
    class_instance.system_hamiltonian(delta, epsilon)

    psi = xplus

    for s in tqdm(range(S)):

        trajSim = class_instance.Traj(psi, N, total_simtime)
        result[s] = trajSim[-1]
        psi = np.copy(trajSim[-1])


    path = '../data/burnin/'
    filename = path + 'burnin' + '_S_' + str(S) + '_N_' + str(N) + '_delta_' + str(delta).replace('.', 'p') + '_eps_' + str(epsilon).replace('.', 'p')

    np.save(filename, result)

def aux(N, simtime, delta, epsilon):

    #result = np.zeros((S, 2, 1), dtype='complex128')

    env = 'z'
    meas_basis = 'x'
    seed = 381693

    #delta = -0.05
    #epsilon = 0.5
    psi0 = xplus

    #dt = 0.01
    #total_simtime = N * dt

    class_instance = QTT(env, meas_basis, seed=seed)
    class_instance.system_hamiltonian(delta, epsilon)
    print(class_instance.H)

    start = time.perf_counter()
    result = class_instance.Traj(psi0, N, simtime)
    stop = time.perf_counter()

    print('trajectory simulation time: ', (stop - start))

    path = '../data/single_traj/'
    filename = path + 'test_traj' + '_N_' + str(N) + '_delta_' + str(delta).replace('.', 'p') + '_eps_' + str(epsilon).replace('.', 'p')

    np.save(filename, result)

def plot(S, N, delta, epsilon):
    
    path = '../data/burnin/'
    loadname = path + 'burnin' + '_S_' + str(S) + '_N_' + str(N) + '_delta_' + str(delta).replace('.', 'p') + '_eps_' + str(epsilon).replace('.', 'p')

    states = np.load(loadname + '.npy')

    theta = 2 * np.arccos( states[:,0] )
    phi = np.imag( np.log( states[:,1] / np.sin(theta/2) ) )

    
    plt.style.use('ggplot')
    fig = plt.figure()
    ax1 = fig.add_subplot(121)

    ax1.scatter(theta, phi, s=1.5)


    ax2 = fig.add_subplot(122, projection='3d')

    ax2.set_title('Trajectory on Bloch sphere')
    ax2.view_init(-30,60)
    sphere = qp.Bloch(axes=ax2)
    psize = 10.0
    sphere.point_size = [psize,psize,psize,psize]

    rho = states @ dag(states)

    rx = rho[:,1,0] + rho[:,0,1]
    ry = 1j*(rho[:,1,0] - rho[:,0,1])
    rz = rho[:,0,0] - rho[:,1,1]

    R = [rx, ry, rz]

    sphere.add_points(R)

    sphere.make_sphere()


    fig.set_size_inches(16,9)

    figname = "../figure/burnin/" + 'burnin' + '_S_' + str(S) + '_N_' + str(N) + '_delta_' + str(delta).replace('.', 'p') + '_eps_' + str(epsilon).replace('.', 'p') + '_last500' + '.png'
    #fig.savefig(figname, dpi=400)

    plt.show()
    
def auxplot(N, delta, epsilon):

    path = '../data/single_traj/'
    loadname = path + 'test_traj' + '_N_' + str(N) + '_delta_' + str(delta).replace('.', 'p') + '_eps_' + str(epsilon).replace('.', 'p')

    states = np.load(loadname + '.npy')

    theta = 2 * np.arccos( states[:,0] )
    phi = np.imag( np.log( states[:,1] / np.sin(theta/2) ) )

    
    plt.style.use('ggplot')
    fig = plt.figure()
    ax1 = fig.add_subplot(121)

    ax1.scatter(theta, phi, s=1.5)

    ax2 = fig.add_subplot(122, projection='3d')

    ax2.set_title('Trajectory on Bloch sphere')
    ax2.view_init(-30,60)
    sphere = qp.Bloch(axes=ax2)
    psize = 10.0
    sphere.point_size = [psize,psize,psize,psize]

    rho = states @ dag(states)

    rx = rho[:,1,0] + rho[:,0,1]
    ry = 1j*(rho[:,1,0] - rho[:,0,1])
    rz = rho[:,0,0] - rho[:,1,1]

    R = [rx.real, ry.real, rz.real]

    sphere.add_points(R)

    sphere.make_sphere()


    fig.set_size_inches(16,9)
    """fig.suptitle('Single trajectory simulation on the Bloch sphere\n' \
                + 'with theta= ' + str(theta) + ' N= ' + str(N) + ' T= ' + str(T))"""

    figname = "../figure/single_traj/" + 'test_traj' + '_N_' + str(N) + '_delta_' + str(delta).replace('.', 'p') + '_eps_' + str(epsilon).replace('.', 'p') + '_last500' + '.png'
    fig.savefig(figname, dpi=400)


# main 

S = 100
N = 2000
delta = 0.5
epsilon = -0.25
simtime = 20.0

#main(S, N, delta, epsilon)
#aux(N, simtime, delta, epsilon)

# plot
plot(S, N, delta, epsilon)
#auxplot(N, delta, epsilon)