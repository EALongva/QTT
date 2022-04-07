# burnin calculation
# single trajectory 1000 N save final state for each N = 1000 repeat starting at previous final state

from QTT import *

def main(S, N, delta, epsilon):

    result = np.zeros((S, 2, 1), dtype='complex128')

    env = 'z'
    meas_basis = 'x'
    seed = 381693

    delta = 0.1
    epsilon = 2.0

    qtt = QTT(env, meas_basis, seed=seed)
    qtt.system_hamiltonian(delta, epsilon)

    psi = bas0

    for s in range(S):

        temp = qtt.Traj(psi, N)
        result[s] = temp[-1]
        psi = np.copy(temp[-1])


    path = '../data/burnin/'
    filename = path + 'burnin' + '_S_' + str(S) + '_N_' + str(N) + '_delta_' + str(delta).replace('.', 'p') + '_eps_' + str(epsilon).replace('.', 'p')

    np.save(filename, result)

def plot(S, N, delta, epsilon):
    
    path = '../data/burnin/'
    loadname = path + 'burnin' + '_S_' + str(S) + '_N_' + str(N) + '_delta_' + str(delta).replace('.', 'p') + '_eps_' + str(epsilon).replace('.', 'p')

    states = np.load(loadname + '.npy')[9500:-1]

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

    figname = "../figure/burnin/" + 'burnin' + '_S_' + str(S) + '_N_' + str(N) + '_delta_' + str(delta).replace('.', 'p') + '_eps_' + str(epsilon).replace('.', 'p') + '_last500' + '.png'
    fig.savefig(figname, dpi=400)

    plt.show()
    




# main 

S = 1000
N = 1000
delta = 0.1
epsilon = 0.2

main(S, N, delta, epsilon)

# plot

#plot(S, N, delta, epsilon)