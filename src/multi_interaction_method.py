from methods import *




def simulationHam_v2(S, N, finaltime, psi0, H, temperature, theta, Usm, Usp, env, infostring, seed=1337, res=1000, test=1.0):

    info            = np.array([S, N, temperature, theta])

    T               = temperature # single temperature

    result          = np.zeros(( S, res, 2, 1), dtype='complex128')
    result[:,0]     = psi0
    psi             = np.copy(psi0)
    newpsi          = np.zeros((2,1), dtype='complex128')
    times           = np.linspace(0,finaltime,res)
    dt              = finaltime/N

    skip            = int(np.floor(N/res)) # resolution must be less than timesteps (N)

    # full state system x environment
    Psi = np.kron(psi0, env)

    pdensity        = 1/(np.exp(1/temperature)-1)
    gammap          = pdensity
    gammam          = pdensity + 1

    # temperature dependent interaction strength
    thetap          = np.sqrt(gammap) * theta
    thetam          = np.sqrt(gammam) * theta
    print(gammap, gammam)
    print(thetap, thetam)

    # second order expansion of time evolution operator
    U_p = np.eye(4) -1j*thetap*Usp - (thetap**2/2) * Usp @ Usp
    U_m = np.eye(4) -1j*thetam*Usm - (thetam**2/2) * Usm @ Usm

    # second order expansion of Hamiltonian evolution
    H_expansion = np.eye(2) -1j*dt*H - (dt**2/2) * H @ H

    for s in range(S):

        # initial state for each traj
        Psi = np.kron(psi0, env)

        if s<1:

            print('debug on, timing first trajectory simulation ... ')
            ping0 = time.perf_counter()

        elif s==2:

            ping0 = time.perf_counter()

        seed += 1

        rnd.seed(seed)

        r = 1

        for n in range(N-1):

            p = rnd.random()

            # alternating evenly, temperature dependence in interaction strength parameter              
            if (n+1)%2 == 1:

                newPsi = U_p @ Psi

            else:

                newPsi = U_m @ Psi

            ### measuring in x-basis
            """
            psi_p  = np.sqrt(0.5) * ( (newPsi[0] + newPsi[1])*bas0 + (newPsi[2] + newPsi[3])*bas1 )
            
            prob   = np.conj(psi_p).T @ psi_p

            #print("random:", p)
            #print("calc prob:", prob)

            if p <= prob: # measure |x_+>

                #print("xplus")

                temp        = (newPsi[0] + newPsi[1])*bas0 + (newPsi[2] + newPsi[3])*bas1
                norm        = 1/np.sqrt(np.conj(temp).T @ temp)
                newpsi_s    = norm*temp

            else:   # measure |x_->

                #print("xminus")

                temp        = (newPsi[0] - newPsi[1])*bas0 + (newPsi[2] - newPsi[3])*bas1
                norm        = 1/np.sqrt(np.conj(temp).T @ temp)
                newpsi_s    = norm*temp
            """

            ### measuring in y-basis
            

            psi_p  = np.sqrt(0.5) * ( (newPsi[0] - 1j*newPsi[1] ) * bas0 + (newPsi[2] - 1j*newPsi[3] ) * bas1 )
            
            prob   = np.conj(psi_p).T @ psi_p
            
            if p <= prob: # measure |y_+>

                temp        = (newPsi[0] - 1j*newPsi[1] ) * bas0 + (newPsi[2] - 1j*newPsi[3] ) * bas1
                norm        = 1/np.sqrt(np.conj(temp).T @ temp)
                newpsi_s    = norm*temp

            else:   # measure |y_->

                temp        = (newPsi[0] + 1j*newPsi[1] ) * bas0 + (newPsi[2] + 1j*newPsi[3] ) * bas1
                norm        = 1/np.sqrt(np.conj(temp).T @ temp)
                newpsi_s    = norm*temp
            

            ### measuring in z-basis
            """

            psi_p  = newPsi[0] * bas0 + newPsi[2]*bas1
            
            prob   = np.conj(psi_p).T @ psi_p

            if p <= prob: # measure |z_+> |0>

                temp        = newPsi[0] * bas0 + newPsi[2] * bas1
                norm        = 1/np.sqrt(np.conj(temp).T @ temp)
                newpsi_s    = norm*temp

            else:   # measure |z_-> |1>

                temp        = newPsi[1] * bas0 + newPsi[3] * bas1
                norm        = 1/np.sqrt(np.conj(temp).T @ temp)
                newpsi_s    = norm*temp

            """


            # Hamiltonian time evolution

            H_newpsi_s = H_expansion @ newpsi_s

            if (n+1+skip)%skip == 0:


                result[s,r] = H_newpsi_s

                r += 1

            Psi     = np.kron(H_newpsi_s, env) # entangling updated system state with a new environment state

        if s<1:

            ping1 = time.perf_counter()
            simtime = ping1 - ping0
            print('time of first trajectory sim: ', simtime )
            print('estimated finish time: ', time.ctime(time.time()+1.3*simtime*(S-1)))

        elif s==2:
            ping1 = time.perf_counter()
            simtime = ping1 - ping0
            print('estimated finish time after 2nd traj: ', time.ctime(time.time()+simtime*(S-s)))
            print('estimated total simulation time: ', str(timedelta(seconds=simtime*(S-s))))


    direc = "../dat/XDiffusion/"
    # should find a better filename system
    filename = direc + "TESTsim" + "_S_" + str(S) + "_N_" + str(N) + "_tet_" + str(theta).replace('.', 'p') + "_env_" + infostring
    np.save(filename, result)

    timesname = filename + "_times"
    np.save(timesname, times)

    psi0name = filename + "_psi0"
    np.save(psi0name, psi0)

    hamname = filename + "_ham"
    np.save(hamname, H)

    infoname = filename + "_info"
    np.save(infoname, info)

    return 0



# simulating single temperature system varying the initial states

#p0 = 1/(1 + np.exp(-1/temperature))
#p1 = 1 - p0
#psi0_final = np.sqrt(p0) * bas0 + np.sqrt(p1) * bas1 #final state for some temperature

yminus = np.sqrt(0.5) * (bas0 - 1j*bas1)
yplus = np.sqrt(0.5) * (bas0 - 1j*bas1)
xplus = np.sqrt(0.5) * (bas0 + bas1)
xminus = np.sqrt(0.5) * (bas0 - bas1)

### possible interactions with (z,x,y)-environment bits

# z-basis
Usm = np.array(([0,0,0,0],[0,-1,1,0],[0,1,0,0],[0,0,0,1]), dtype='complex128')
Usp = np.array(([0,0,0,1],[0,-1,0,0],[0,0,0,0],[1,0,0,1]), dtype='complex128')

Usmt = 0.5*( np.kron( sigmax, sigmax ) + np.kron( sigmay, sigmay ) + np.kron( sigmaz, sigmaz - np.eye(2) ) )
Uspt = 0.5*( np.kron( sigmax, sigmax ) - np.kron( sigmay, sigmay ) + np.kron( sigmaz, sigmaz - np.eye(2) ) )

Um_z = 0.5*( np.kron( sigmax, sigmax ) + np.kron( sigmay, sigmay ) )
Up_z = 0.5*( np.kron( sigmax, sigmax ) - np.kron( sigmay, sigmay ) )

# x-basis

Um_x = 0.5*( np.kron( sigmax, sigmay ) + np.kron( sigmay, sigmaz ) )
Up_x = 0.5*( np.kron( sigmax, sigmay ) - np.kron( sigmay, sigmaz ) )

# y-basis

Um_y = 0.5*( np.kron( sigmax, sigmax ) + np.kron( sigmay, sigmaz ) )
Up_y = 0.5*( np.kron( sigmax, sigmax ) - np.kron( sigmay, sigmaz ) )

# defining the Hamiltonian

delta = 1.0
eps = 0.0

H = 1.0*(0.5*delta*sigmaz + 0.5*eps*sigmay)

#(S, N, finaltime, psi0, H, temperature, theta=0.1, seed=1337, res=1000, test=1.0)

# some simulation variables
timesteps = 2000
finaltime = 20.0
dt = finaltime / timesteps
traj = 1000
theta = 0.1
nTemp = 1
resolution = 1000
temperature = 0.5
environment = np.copy(xplus)
infostring = "xp" # string to be passed to filename, specifies the basis of environment bits

simulationHam_v2(traj, timesteps, finaltime, xplus, H, temperature, theta, Um_x, Up_x, environment, infostring, seed=1339184, res=resolution, test=1.0)


### end