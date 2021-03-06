# burnin calculation
# single trajectory 1000 N save final state for each N = 1000 repeat starting at previous final state

from QTT import *

def main(S, N, simtime, delta, epsilon):

    result = np.zeros((S, 2, 1), dtype='complex128')

    env = 'z'
    meas_basis = 'x'
    seed = 381693

    #delta = 0.1
    #epsilon = 2.0

    #dt = 0.01
    total_simtime = simtime

    class_instance = QTT(env, meas_basis, seed=seed)
    class_instance.system_hamiltonian(delta, epsilon)

    psi = xplus

    for s in tqdm(range(S)):

        trajSim = class_instance.Traj(psi, N, total_simtime)
        result[s] = trajSim[-1]
        psi = trajSim[-1]


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

def burninest(S, N, burnin, simtime, seed, theta, delta, epsilon):

    result = np.zeros((S, 2, 1), dtype='complex128')

    env = 'z'
    meas_basis = 'x'
    #seed = 38165193

    #delta = 0.1
    #epsilon = 2.0

    dt = simtime/N

    class_instance = QTT(env, meas_basis, theta=theta, seed=seed)
    class_instance.system_hamiltonian(delta, epsilon)

    print('gamma plus = ', class_instance.gammap, 'gamma minus = ', class_instance.gammam )

    #theta = class_instance.theta

    ### burnin

    burnintime = burnin*dt
    psi0 = xplus
    psi = class_instance.Burnin(psi0, burnin, burnintime)

    for s in tqdm(range(S)):

        trajSim = class_instance.Traj(psi, N, simtime)
        result[s] = trajSim[-1]
        psi = trajSim[-1]


    path = '../data/burnin/'
    filename = path + 'burninest' + '_S_' + str(S) + '_N_' + str(N) + \
        '_burnin_N_' + str(burnin) + '_dt_' + str(dt).replace('.', 'p') + \
        '_theta_' + str(theta).replace('.', 'p') + \
        '_delta_' + str(delta).replace('.', 'p') + '_eps_' + str(epsilon).replace('.', 'p')

    np.save(filename, result)

    return 0

def plot(S, N, delta, epsilon):
    
    path = '../data/burnin/'
    loadname = path + 'burnin' + '_S_' + str(S) + '_N_' + str(N) + '_delta_' + str(delta).replace('.', 'p') + '_eps_' + str(epsilon).replace('.', 'p')

    states = np.load(loadname + '.npy')#[100:1100]

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

    figname = "../figure/burnin/" + 'burnin' + '_S_' + str(S) + '_N_' + str(N) + \
        '_delta_' + str(delta).replace('.', 'p') + '_eps_' + \
        str(epsilon).replace('.', 'p') + '.png'

    fig.savefig(figname, dpi=400)

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

def burninestplot2(S, N, burnin, simtime, theta, delta, epsilon):

    ### initialize QTT class

    env = 'z'
    meas_basis = 'x'
    seed = 38165193

    #delta = 0.1
    #epsilon = 2.0

    dt = simtime/N
    finaltime = simtime*S + burnin*dt

    class_instance = QTT(env, meas_basis, theta=theta, seed=seed)
    class_instance.system_hamiltonian(delta, epsilon)
    theta_value = class_instance.theta
    temperature = class_instance.temperature

    ### loading data

    path = '../data/burnin/'
    loadname = path + 'burninest' + '_S_' + str(S) + '_N_' + str(N) + \
        '_burnin_N_' + str(burnin) + '_dt_' + str(dt).replace('.', 'p') + \
        '_theta_' + str(theta_value).replace('.', 'p') + \
        '_delta_' + str(delta).replace('.', 'p') + '_eps_' + str(epsilon).replace('.', 'p')

    states = np.load(loadname + '.npy')

    rho = states @ dag(states)

    rx = rho[:,1,0] + rho[:,0,1]
    ry = 1j*(rho[:,1,0] - rho[:,0,1])
    rz = rho[:,0,0] - rho[:,1,1]

    R = [rx, ry, rz]

    theta       = np.arccos( rz )
    phi         = np.arctan2(ry.real, rx.real)


    ### compute lindblad solution

    lindblad_timesteps = 1000
    class_instance.finaltime = finaltime
    #class_instance.inittime = burnin*dt
    class_instance.state0 = xplus #states[0]
    class_instance.timesteps = lindblad_timesteps
    class_instance.lindblad()
    L = class_instance.blochvecLB
    times = class_instance.times_result

    total_timesteps = int(burnin + S*N)
    burnin_ratio = burnin/total_timesteps
    #burnin_index = np.ceil(lindblad_timesteps*burnin_ratio)
    burnin_index = int(lindblad_timesteps*burnin_ratio)

    print(total_timesteps, burnin, burnin_ratio, burnin_index)

    ###


    plt.style.use('ggplot')
    fig = plt.figure()
    

    ax1 = fig.add_subplot(131)
    ax1.scatter(theta, phi, s=1.5)
    ax1.set_xlabel(r'$\theta$')
    ax1.set_ylabel(r'$\phi$')
    ax1.set_title('2d projection')


    ax2 = fig.add_subplot(132, projection='3d')

    ax2.view_init(-30,60)
    sphere = qp.Bloch(axes=ax2)
    psize = 10.0
    sphere.point_size = [psize,psize,psize,psize]

    sphere.add_points(R)
    sphere.make_sphere()
    ax2.set_title('Trajectory on Bloch sphere')


    ax3 = fig.add_subplot(133)

    xyz = ['x', 'y', 'z']

    for component, comp in zip( L, xyz ):

        ax3.plot(times, component, label=comp)

    ax3.axvline(x=times[burnin_index], color='y', linestyle="-", label='burnin end')
    ax3.set_xlabel('time')
    ax3.legend()
    ax3.set_title('Lindblad solution, bloch vector components')


    bigtitle = (r'Single trajectory simulation with Lindblad solution for comparison, ' + \
            'using random unitary diffusion, \n' + r'$\theta$ = ' + str(theta_value) + \
            r', $\Delta t =$ ' + str(dt) + ', temperature: ' + str(temperature) + ', timesteps: ' + str(int(N*S)) + \
            ', burnin timesteps: ' + str(burnin) + '\n' + r'$H = \Delta \sigma_z + \epsilon \sigma_y$' + \
            ' with ' + r'$\Delta = $' + str(delta) + r' and $\epsilon = $' + str(epsilon))

    fig.suptitle(bigtitle)
    fig.set_size_inches(16,9)

    figname = "../figure/burnin/improved/" + 'theta' + str(theta_value).replace('.', '')  \
        + '_eps' + str(epsilon).replace('.', '') + '_delta' + str(delta).replace('.', '') + \
        '_S_' + str(S) + '_N_' + str(N) + '_burnin_N_' + str(burnin) + '_dt' + \
        str(dt).replace('.', '') + '.png'
    
    fig.savefig(figname, dpi=400)

    plt.show()

    return 0

def burninestplot3(S, N, burnin, simtime, theta, delta, epsilon):

    ### loading data

    dt = simtime/N

    path = '../data/burnin/'
    loadname = path + 'burninest' + '_S_' + str(S) + '_N_' + str(N) + \
        '_burnin_N_' + str(burnin) + '_dt_' + str(dt).replace('.', 'p') + \
        '_theta_' + str(theta).replace('.', 'p') + \
        '_delta_' + str(delta).replace('.', 'p') + '_eps_' + str(epsilon).replace('.', 'p')

    states = np.load(loadname + '.npy')

    rho = states @ dag(states)

    rx = rho[:,1,0] + rho[:,0,1]
    ry = 1j*(rho[:,1,0] - rho[:,0,1])
    rz = rho[:,0,0] - rho[:,1,1]

    R = [rx, ry, rz]


    plt.style.use('ggplot')
    fig = plt.figure()

    ax1 = fig.add_subplot(131)
    ax1.scatter(rx, ry, color='r', s=1.5)
    ax1.set_xlabel(r'$\langle \sigma_x \rangle$')
    ax1.set_ylabel(r'$\langle \sigma_y \rangle$')

    ax2 = fig.add_subplot(132)
    ax2.scatter(rz, ry, color='b', s=1.5)
    ax2.set_xlabel(r'$\langle \sigma_z \rangle$')
    ax2.set_ylabel(r'$\langle \sigma_y \rangle$')

    ax3 = fig.add_subplot(133)
    ax3.scatter(rx, rz, color='g', s=1.5)
    ax3.set_xlabel(r'$\langle \sigma_x \rangle$')
    ax3.set_ylabel(r'$\langle \sigma_z \rangle$')

    bigtitle = ('Single trajectory simulation using random unitary diffusion, \n' + r'$\theta$ = ' + str(theta) + \
            r', $\Delta t =$ ' + str(dt) + ', temperature: ' + str(0.5) + ', timesteps: ' + str(int(N*S)) + \
            ', burnin timesteps: ' + str(burnin) + '\n' + r'$H = \Delta \sigma_z + \epsilon \sigma_y$' + \
            ' with ' + r'$\Delta = $' + str(delta) + r' and $\epsilon = $' + str(epsilon))

    fig.suptitle(bigtitle)
    fig.set_size_inches(16,9)

    figname = "../figure/burnin/improved/bvecs" + '_theta' + str(theta).replace('.', '')  \
        + '_eps' + str(epsilon).replace('.', '') + '_delta' + str(delta).replace('.', '') + \
        '_S_' + str(S) + '_N_' + str(N) + '_burnin_N_' + str(burnin) + '_dt' + \
        str(dt).replace('.', '') + '.png'
    
    fig.savefig(figname, dpi=400)

    plt.show()

    return 0



### function below does not give the correct lindblad solution

def burninestplot(S, N, burnin, simtime, delta, epsilon):

    ### initialize QTT class

    env = 'z'
    meas_basis = 'x'
    seed = 381693

    #delta = 0.1
    #epsilon = 2.0

    dt = simtime/N
    finaltime = simtime + burnin*dt

    class_instance = QTT(env, meas_basis, seed=seed)
    class_instance.system_hamiltonian(delta, epsilon)
    theta_value = class_instance.theta
    temperature = class_instance.temperature

    ### loading data

    path = '../data/burnin/'
    loadname = path + 'burninest' + '_S_' + str(S) + '_N_' + str(N) + \
        '_burnin_N_' + str(burnin) + '_dt_' + str(dt).replace('.', 'p') + \
        '_theta_' + str(theta_value).replace('.', 'p') + \
        '_delta_' + str(delta).replace('.', 'p') + '_eps_' + str(epsilon).replace('.', 'p')

    states = np.load(loadname + '.npy')#[100:1100]

    rho = states @ dag(states)

    rx = rho[:,1,0] + rho[:,0,1]
    ry = 1j*(rho[:,1,0] - rho[:,0,1])
    rz = rho[:,0,0] - rho[:,1,1]

    R = [rx, ry, rz]

    theta       = np.arccos( rz )
    phi         = np.arctan2(ry.real, rx.real)


    ### compute lindblad solution

    class_instance.finaltime = finaltime
    class_instance.inittime = burnin*dt
    class_instance.state0 = states[0]
    class_instance.timesteps = int(N*S)
    class_instance.lindblad()
    L = class_instance.blochvecLB

    ###


    plt.style.use('ggplot')
    fig = plt.figure()
    

    ax1 = fig.add_subplot(131)
    ax1.scatter(theta, phi, s=1.5)
    ax1.set_xlabel(r'$\theta$')
    ax1.set_ylabel(r'$\phi$')
    ax1.set_title('2d projection')


    ax2 = fig.add_subplot(132, projection='3d')

    ax2.view_init(-30,60)
    sphere = qp.Bloch(axes=ax2)
    psize = 10.0
    sphere.point_size = [psize,psize,psize,psize]

    sphere.add_points(R)
    sphere.make_sphere()
    ax2.set_title('Trajectory on Bloch sphere')


    ax3 = fig.add_subplot(133, projection='3d')

    ax3.view_init(-30,60)
    sphere = qp.Bloch(axes=ax3)
    psize = 10.0
    sphere.point_size = [psize,psize,psize,psize]
    sphere.add_points(L)
    sphere.make_sphere()
    ax3.set_title('Lindblad solution on Bloch sphere')

    bigtitle = (r'Single trajectory simulation with Lindblad solution for comparison, ' + \
            'using random unitary diffusion, \n' + r'$\theta$ = ' + str(theta_value) + \
            r', $\Delta t =$ ' + str(dt) + ', temperature: ' + str(temperature) + ', timesteps: ' + str(int(N*S)) + \
            ', burnin timesteps: ' + str(burnin) + '\n' + r'$H = \Delta \sigma_z + \epsilon \sigma_y$' + \
            ' with ' + r'$\Delta = $' + str(delta) + r' and $\epsilon = $' + str(epsilon))

    fig.suptitle(bigtitle)
    fig.set_size_inches(16,9)

    figname = "../figure/burnin/" + 'burninest' + '_S_' + str(S) + '_N_' + str(N) + \
        '_burnin_N_' + str(burnin) + '_dt_' + str(dt).replace('.', 'p') + \
        '_theta_' + str(theta_value).replace('.', 'p') + '_delta_' + str(delta).replace('.', 'p') + \
        '_eps_' + str(epsilon).replace('.', 'p') + '.png'
    fig.savefig(figname, dpi=400)

    plt.show()

    return 0



### main 
"""
S = 10000
N = 200
delta = 0.0
epsilon = 0.0
simtime = 2.0

#main(S, N, simtime, delta, epsilon)
plot(S, N, delta, epsilon)
"""

### aux
"""
S = 100
N = 2000
delta = 0.5
epsilon = -0.25
simtime = 20.0

#aux(N, simtime, delta, epsilon)
#auxplot(N, delta, epsilon)
"""

"""
### burnin

S = 2000
N = 200
burnin = 200000
delta = 1.5
epsilon = 1.5
simtime = 2.0

#burninest(S, N, burnin, simtime, delta, epsilon)
#burninestplot(S, N, burnin, simtime, delta, epsilon)


### runs
# leaving theta and dt both at 0.01
# will have to run these with a proper title for each

#burninest(S, N, burnin, simtime, 0.1, 0.1)
burninestplot(S, N, burnin, simtime, 0.1, 0.1)

#burninest(S, N, burnin, simtime, 0.01, 0.1)
burninestplot(S, N, burnin, simtime, 0.01, 0.1)

#burninest(S, N, burnin, simtime, 0.01, 0.01)
burninestplot(S, N, burnin, simtime, 0.01, 0.01)

#burninest(S, N, burnin, simtime, 0.1, 0.01)
burninestplot(S, N, burnin, simtime, 0.1, 0.01)




#burninest(S, N, burnin, simtime, 0.01, 0.02)
burninestplot(S, N, burnin, simtime, 0.01, 0.02)

#burninest(S, N, burnin, simtime, 0.01, 0.03)
burninestplot(S, N, burnin, simtime, 0.01, 0.03)




#burninest(S, N, burnin, simtime, 0.015, 0.02)
burninestplot(S, N, burnin, simtime, 0.015, 0.02)

#burninest(S, N, burnin, simtime, 0.02, 0.02)
burninestplot(S, N, burnin, simtime, 0.02, 0.02)

#burninest(S, N, burnin, simtime, 0.025, 0.02)
burninestplot(S, N, burnin, simtime, 0.025, 0.02)

#burninest(S, N, burnin, simtime, 0.05, 0.02)
burninestplot(S, N, burnin, simtime, 0.05, 0.02)




#burninest(S, N, burnin, simtime, 0.005, 0.01)
burninestplot(S, N, burnin, simtime, 0.005, 0.01)

#burninest(S, N, burnin, simtime, 0.015, 0.01)
burninestplot(S, N, burnin, simtime, 0.015, 0.01)

#burninest(S, N, burnin, simtime, 0.03, 0.01)
burninestplot(S, N, burnin, simtime, 0.03, 0.01)

#burninest(S, N, burnin, simtime, 0.06, 0.01)
burninestplot(S, N, burnin, simtime, 0.06, 0.01)



#burninest(S, N, burnin, simtime, 0.0002, 0.001)
burninestplot(S, N, burnin, simtime, 0.0002, 0.001)

#burninest(S, N, burnin, simtime, 0.0005, 0.001)
burninestplot(S, N, burnin, simtime, 0.0005, 0.001)

#burninest(S, N, burnin, simtime, 0.0010, 0.001)
burninestplot(S, N, burnin, simtime, 0.0010, 0.001)

#burninest(S, N, burnin, simtime, 0.0020, 0.001)
burninestplot(S, N, burnin, simtime, 0.0020, 0.001)


### giving it a shot with some more timesteps


S = 5000
N = 300
burnin = 800000
simtime = 3.0

#burninest(S, N, burnin, simtime, 0.02, 0.001)
#burninestplot(S, N, burnin, simtime, 0.02, 0.001)

burninest(S, N, burnin, simtime, 0.0, 0.0)
burninestplot(S, N, burnin, simtime, 0.0, 0.0)


"""

S = 5000
N = 10
burnin = 100000
simtime = 0.1
seed = 194357126


"""
burninest(S, N, burnin, simtime, seed, 0.01, delta=0.01, epsilon=0.02)
burninestplot2(S, N, burnin, simtime, 0.01, delta=0.01, epsilon=0.02)

burninest(S, N, burnin, simtime, seed, 0.01, delta=0.015, epsilon=0.03)
burninestplot2(S, N, burnin, simtime, 0.01, delta=0.015, epsilon=0.03)

### this one was kind of interesting
burninest(S, N, burnin, simtime, seed, 0.01, delta=0.02, epsilon=0.04)
burninestplot2(S, N, burnin, simtime, 0.01, delta=0.02, epsilon=0.04)

burninest(S, N, burnin, simtime, seed, 0.01, delta=0.025, epsilon=0.05)
burninestplot2(S, N, burnin, simtime, 0.01, delta=0.025, epsilon=0.05)

burninest(S, N, burnin, simtime, seed, 0.01, delta=0.03, epsilon=0.06)
burninestplot2(S, N, burnin, simtime, 0.01, delta=0.03, epsilon=0.06)

burninest(S, N, burnin, simtime, seed, 0.01, delta=0.021, epsilon=0.042)
burninestplot2(S, N, burnin, simtime, 0.01, delta=0.021, epsilon=0.042)

burninest(S, N, burnin, simtime, seed, 0.01, delta=0.019, epsilon=0.038)
burninestplot2(S, N, burnin, simtime, 0.01, delta=0.019, epsilon=0.038)
"""

burninest(S, N, burnin, simtime, seed, 0.01, delta=0.02, epsilon=0.04)
burninestplot2(S, N, burnin, simtime, 0.01, delta=0.02, epsilon=0.04)
burninestplot3(S, N, burnin, simtime, 0.01, delta=0.02, epsilon=0.04)
