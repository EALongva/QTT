# methods for QTT
# this file contains functions used in the other programs

import sys as sys
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import qutip as qp
import random as rnd
import time as time
from datetime import timedelta
import math as math
from scipy.linalg import expm
from mpl_toolkits.mplot3d import Axes3D

# A list of methods to solve Random Unitary Diffusion in Quantum Trajectory Theory

def dag(x, test=False):

    if test==True:

        print("testing the dag(x) function")

        bas0 = np.array(([1.0], [0.0]))
        bas1 = np.array(([0.0], [1.0]))

        dagtest = np.array([bas0, bas1, 1j*bas0, 1j*bas1])

        result1 = dag(dagtest) @ dagtest
        result2 = dagtest @ dag(dagtest)

        print("input array:")
        print(dagtest)
        print("output with dag(array) @ array:")
        print(result1)
        print("output with array @ dag(array):")
        print(result2)
    
    else:
        # hermitian conjugate of 3 dim array with (2,1) matrices along first axis
        return np.transpose(np.conj(x), (0,2,1))

### interactions
# legacy interactions, we now use the exponatiated interaction matrices Usm and Usp expanded to second order in theta :)

def SWAPp(p, psi, theta):

    # theta interaction strength
    # might add alpha = psi[0], beta = psi[1] for clarity
    # maybe include a test in these that shows normalization of states

    temp_xplus = psi[0] * np.cos(theta) * bas0 + (psi[1] - 1j * psi[0] * np.sin(theta)) * bas1
    prod_xplus = np.conj(temp_xplus.T) @ temp_xplus
    prob_xplus = 0.5 * (prod_xplus)

    if p <= prob_xplus:

        newpsi      = temp_xplus/np.sqrt(prod_xplus)

    else:

        temp        = psi[0] * np.cos(theta) * bas0 + (psi[1] + 1j * psi[0] * np.sin(theta)) * bas1
        normalize   = np.sqrt(np.conj(temp.T) @ temp)
        newpsi      = temp/normalize

    return newpsi

def SWAPm(p, psi, theta):

    temp_xplus = (psi[0] - 1j * psi[1] * np.sin(theta)) * bas0 + psi[1] * np.cos(theta) * bas1
    prod_xplus = np.conj(temp_xplus.T) @ temp_xplus
    prob_xplus = 0.5 * (prod_xplus)

    if p <= prob_xplus:

        newpsi      = temp_xplus/np.sqrt(prod_xplus)

    else:

        temp        = (psi[0] + 1j * psi[1] * np.sin(theta)) * bas0 + psi[1] * np.cos(theta) * bas1
        normalize   = np.sqrt(np.conj(temp.T) @ temp)
        newpsi      = temp/normalize

    return newpsi

### simulation methods with no Hamiltonian

def simulationLagacy(S, N, finaltime, psi0, temperature, theta=0.1, seed=1337, res=1000, test=1.0):

    info            = np.array([S, N, temperature, theta])

    T               = temperature # single temperature

    result          = np.zeros(( S, res, 2, 1), dtype='complex128')
    result[:,0]     = psi0
    psi             = np.copy(psi0)
    newpsi          = np.zeros((2,1), dtype='complex128')
    times           = np.linspace(0,finaltime,res)

    skip            = int(np.floor(N/res)) # resolution must be less than timesteps (N)

    pdensity        = 1/(np.exp(1/temperature)-1)
    gammap          = pdensity
    gammam          = pdensity + 1

    # temperature dependent interaction strength
    thetap          = np.sqrt(gammap) * theta
    thetam          = np.sqrt(gammam) * theta

    for s in range(S):

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
            
            # alternating evenly
            
            if (n+1)%2 == 1:

                newpsi = SWAPm(p, psi, thetam)
            
            else:

                newpsi = SWAPp(p, psi, thetap)
            

            if (n+1+skip)%skip == 0:

                result[s,r] = newpsi

                r += 1

            psi     = newpsi
            newpsi  = np.zeros((2,1), dtype='complex128')

        psi = np.copy(psi0)

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

    direc = "../../../dat/RUD/"
    # should find a better filename system
    filename = direc + "simulation" + "_S_" + str(S) + "_N_" + str(N)
    np.save(filename, result)

    timesname = filename + "_times"
    np.save(timesname, times)

    psi0name = filename + "_psi0"
    np.save(psi0name, psi0)

    infoname = filename + "_info"
    np.save(infoname, info)

    return 0

def simulation(S, N, finaltime, psi0, temperature, theta=0.1, seed=1337, res=1000, test=1.0):

    # could also implement standard first and second order expansions replacing the scipy expm expansion (which is very slow)

    info            = np.array([S, N, temperature, theta])

    T               = temperature # single temperature only
    Tsize           = 1

    result          = np.zeros((Tsize, S, res, 2, 1), dtype='complex128')
    result[:,:,0]   = psi0
    newpsi_s        = np.zeros((2,1), dtype='complex128')
    times           = np.linspace(0,finaltime,res)

    skip            = int(np.floor(N/res)) # resolution must be less than timesteps (N)

    # interaction hamiltonians
    Usp = np.array(([0,0,0,1],[0,-1,0,0],[0,0,0,0],[1,0,0,1]), dtype='complex128')
    Usm = np.array(([0,0,0,0],[0,-1,1,0],[0,1,0,0],[0,0,0,1]), dtype='complex128')

    # environment for RUD
    env = np.copy(bas0)

    # full state system x environment
    Psi = np.kron(psi0, env)

    # interaction strength parameters
    pdensity        = 1/(np.exp(1/temperature)-1)
    gammap          = pdensity
    gammam          = pdensity + 1

    # temperature dependent interaction strength
    thetap          = np.sqrt(gammap) * theta
    thetam          = np.sqrt(gammam) * theta

    # second order expansion of time evolution operator
    U_p = np.eye(4) -1j*thetap*Usp - (thetap**2/2) * Usp @ Usp
    U_m = np.eye(4) -1j*thetam*Usm - (thetam**2/2) * Usm @ Usm

    # third order expansion of time evolution operator
    #U_p3 = np.eye(4) -1j*thetap*Usp - (thetap**2/2.0) * (Usp @ Usp) + 1j*(thetap**3/6.0) * (Usp @ Usp @ Usp)
    #U_m3 = np.eye(4) -1j*thetam*Usm - (thetam**2/2.0) * (Usm @ Usm) + 1j*(thetam**3/6.0) * (Usm @ Usm @ Usm)

    for t in range(Tsize):

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


                if (n+1+skip)%skip == 0:


                    result[t,s,r] = newpsi_s

                    r += 1

                Psi     = np.kron(newpsi_s, env) # entangling updated system state with a new environment state

            if s<1:

                ping1 = time.perf_counter()
                simtime = ping1 - ping0
                print('time of first trajectory sim: ', simtime )
                print('estimated finish time: ', time.ctime(time.time()+1.3*simtime*(S-1)))

            elif s==2:
                ping1 = time.perf_counter()
                simtime = ping1 - ping0
                print('estimated finish time after 2nd traj: ', time.ctime(time.time()+simtime*(S-s)))
                print('estimated total simulation time: ', str(timedelta(seconds=simtime*(S-s)*Tsize)))

    direc = "../dat/RUD/"
    filename = direc + "sim" + "_S_" + str(S) + "_N_" + str(N) + "_NTemps_" + str(Tsize)
    np.save(filename, result)

    timesname = filename + "_times"
    np.save(timesname, times)

    psi0name = filename + "_psi0"
    np.save(psi0name, psi0)

    infoname = filename + "_info"
    np.save(infoname, info)

    return 0

def simulationTemps(S, N, finaltime, psi0, temperatures, theta=0.01, seed=1337, res=1000, debug=True, intStrength=True):

    # needs functions: RUD_SWAPp, RUD_SWAPm, dag(x)
    # timesteps = N, finaltime = T, traj = S

    info            = np.array([S, N, 0, theta])

    result          = np.zeros((temperatures.size, S, res, 2, 1), dtype='complex128')
    result[:,:,0]   = psi0
    psi             = np.copy(psi0)
    newpsi          = np.zeros((2, 1), dtype='complex128')
    times           = np.linspace(0, finaltime, res)

    skip = int(np.ceil(N/res))

    ### if using intStrength=True
    pdensity        = 1/(np.exp(1/temperatures)-1)
    gammap          = pdensity
    gammam          = pdensity + 1

    # temperature dependent interaction strength
    thetap          = np.sqrt(gammap) * theta
    thetam          = np.sqrt(gammam) * theta


    ### if intStrength=False
    p0 = 1/(1 + np.exp(-1/temperatures)) # where 1/temperature is actually beta * dE, where beta is 1/kb*T and dE is the energy gap (dE/kb = 1)

    for t in range(temperatures.size):

        for s in range(S):

            if debug and s<1:
                print('debug on, timing first trajectory simulation ... ')
                ping0 = time.perf_counter()

            seed += 1
            rnd.seed(seed)

            r = 1

            for n in range(N-1):

                if intStrength:

                    p = rnd.random()

                    if (n+1)%2 == 1:

                        newpsi = SWAPm(p, psi, thetam[t])
                    
                    else:

                        newpsi = SWAPp(p, psi, thetap[t])


                else:

                    pm = rnd.random()

                    if pm <= p0[t]: #probability of swap minus interaction

                        p = rnd.random()

                        newpsi = SWAPm(p, psi, theta)

                    else:

                        p = rnd.random()

                        newpsi = SWAPp(p, psi, theta)

                if (n+1+skip)%skip == 0:

                        result[t,s,r] = newpsi

                        r += 1

                psi     = newpsi
                newpsi  = np.zeros((2,1), dtype='complex128')

            psi = np.copy(psi0)

            if debug and s<1:
                ping1 = time.perf_counter()
                simtime = ping1 - ping0
                print('time of first trajectory sim: ', simtime )
                print('projected total sim time: ', simtime*S*temperatures.size)
                print('estimated finish time: ', time.ctime(time.time()+1.2*simtime*(S-1)*temperatures.size))
    
    # only save the amount of points given in resolution (this only applies to the N points in each trajectory)

    if intStrength:

        interaction = '_intStrength'

    else:
        
        interaction = '_intProbability'


    direc = "../../../dat/RUD/"
    filename = direc + "simulationTemps" + "_S_" + str(S) + "_N_" + str(N) + "_NTemps_" + str(temperatures.size) + interaction
    np.save(filename, result)

    timesname = filename + "_times"
    np.save(timesname, times)

    tempsname = filename + "_temps"
    np.save(tempsname, temperatures)

    psi0name = filename + "_psi0"
    np.save(psi0name, psi0)

    infoname = filename + "_info"
    np.save(infoname, info)

    return 0

### simulations with Hamiltonian

def simulationHam(S, N, finaltime, psi0, H, temperature, theta=0.1, seed=1337, res=1000, test=1.0):

    info            = np.array([S, N, temperature, theta])

    T               = temperature # single temperature

    result          = np.zeros(( S, res, 2, 1), dtype='complex128')
    result[:,0]     = psi0
    psi             = np.copy(psi0)
    newpsi          = np.zeros((2,1), dtype='complex128')
    times           = np.linspace(0,finaltime,res)
    dt              = finaltime/N

    skip            = int(np.floor(N/res)) # resolution must be less than timesteps (N)

    # interaction hamiltonians
    Usp = np.array(([0,0,0,1],[0,-1,0,0],[0,0,0,0],[1,0,0,1]), dtype='complex128')
    Usm = np.array(([0,0,0,0],[0,-1,1,0],[0,1,0,0],[0,0,0,1]), dtype='complex128')

    # environment for RUD
    env = np.copy(bas0)

    # full state system x environment
    Psi = np.kron(psi0, env)

    pdensity        = 1/(np.exp(1/temperature)-1)
    gammap          = pdensity
    gammam          = pdensity + 1

    # temperature dependent interaction strength
    thetap          = (np.sqrt(gammap) * theta)
    thetam          = (np.sqrt(gammam) * theta)
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


    direc = "../dat/RUD/"
    # should find a better filename system
    filename = direc + "Hsim" + "_S_" + str(S) + "_N_" + str(N) + "_tet_" + str(theta).replace('.', 'p')
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

### full simulation with multiple temperatures and hamiltonian

def simulationHamTemps(S, N, finaltime, psi0, H, temperatures, theta=0.1, seed=1337, res=1000, test=1.0):

    # could also implement standard first and second order expansions replacing the scipy expm expansion (which is very slow)

    info            = np.array([S, N, 0, theta])

    Tsize           = temperatures.size

    result          = np.zeros((Tsize, S, res, 2, 1), dtype='complex128')
    result[:,:,0]   = psi0
    newpsi_s        = np.zeros((2,1), dtype='complex128')
    times           = np.linspace(0,finaltime,res)
    dt              = finaltime/N

    skip            = int(np.floor(N/res)) # resolution must be less than timesteps (N)

    # interaction hamiltonians
    Usp = np.array(([0,0,0,1],[0,-1,0,0],[0,0,0,0],[1,0,0,1]), dtype='complex128')
    Usm = np.array(([0,0,0,0],[0,-1,1,0],[0,1,0,0],[0,0,0,1]), dtype='complex128')

    # environment for RUD
    env = np.copy(bas0)

    # full state system x environment
    Psi = np.kron(psi0, env)

    # interaction strength parameters
    pdensity        = 1/(np.exp(1/temperatures)-1)
    gammap          = pdensity
    gammam          = pdensity + 1

    # temperature dependent interaction strength
    thetap          = np.sqrt(gammap) * theta
    thetam          = np.sqrt(gammam) * theta

    # second order expansion of time evolution operator -> array with operator corr to each temperature
    U_p = np.zeros((Tsize, 4, 4), dtype='complex128')
    U_m = np.zeros((Tsize, 4, 4), dtype='complex128')
    
    for t in range(Tsize):
        U_p[t] = np.eye(4) -1j*thetap[t]*Usp - (thetap[t]**2/2) * Usp @ Usp
        U_m[t] = np.eye(4) -1j*thetam[t]*Usm - (thetam[t]**2/2) * Usm @ Usm


    # second order expansion of Hamiltonian
    H_expansion = np.eye(2) -1j*dt*H - (dt**2/2) * H @ H

    # third order expansion of time evolution operator
    #U_p3 = np.eye(4) -1j*thetap*Usp - (thetap**2/2.0) * (Usp @ Usp) + 1j*(thetap**3/6.0) * (Usp @ Usp @ Usp)
    #U_m3 = np.eye(4) -1j*thetam*Usm - (thetam**2/2.0) * (Usm @ Usm) + 1j*(thetam**3/6.0) * (Usm @ Usm @ Usm)

    for t in range(Tsize):

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

                    newPsi = U_p[t] @ Psi

                else:

                    newPsi = U_m[t] @ Psi
                
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

                H_newpsi_s = H_expansion @ newpsi_s

                if (n+1+skip)%skip == 0:


                    result[t,s,r] = H_newpsi_s

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
                print('estimated total simulation time: ', str(timedelta(seconds=simtime*(S-s)*Tsize)))

    direc = "../dat/RUD/"
    filename = direc + "Hsim" + "_S_" + str(S) + "_N_" + str(N) + "_NTemps_" + str(Tsize)
    np.save(filename, result)

    timesname = filename + "_times"
    np.save(timesname, times)

    tempsname = filename + "_temps"
    np.save(tempsname, temperatures)

    psi0name = filename + "_psi0"
    np.save(psi0name, psi0)

    hamname = filename + "_ham"
    np.save(hamname, H)

    infoname = filename + "_info"
    np.save(infoname, info)

    return 0


### simulation multiple initial states, single temperature

def simulationMultiInit(S, N, finaltime, psi0, H, temperature, theta=0.1, seed=1337, res=1000, test=1.0):

    # could also implement standard first and second order expansions replacing the scipy expm expansion (which is very slow)

    info            = np.array([S, N, temperature, theta])

    Psize           = psi0[:,0].size

    result          = np.zeros((Psize, S, res, 2, 1), dtype='complex128')
    
    print(Psize)

    for p in range(Psize):

        result[p,:,:,:]   = psi0[p]
    
    newpsi_s        = np.zeros((2,1), dtype='complex128')
    times           = np.linspace(0,finaltime,res)
    dt              = finaltime/N

    skip            = int(np.floor(N/res)) # resolution must be less than timesteps (N)

    # interaction hamiltonians
    Usp = np.array(([0,0,0,1],[0,-1,0,0],[0,0,0,0],[1,0,0,1]), dtype='complex128')
    Usm = np.array(([0,0,0,0],[0,-1,1,0],[0,1,0,0],[0,0,0,1]), dtype='complex128')

    # environment for RUD
    env = np.copy(bas0)

    # full state system x environment
    #Psi = np.kron(psi0, env)

    # interaction strength parameters
    pdensity        = 1/(np.exp(1/temperature)-1)
    gammap          = pdensity
    gammam          = pdensity + 1

    # temperature dependent interaction strength
    thetap          = np.sqrt(gammap) * theta
    thetam          = np.sqrt(gammam) * theta

    # second order expansion of time evolution operator
    U_p = np.eye(4) -1j*thetap*Usp - (thetap**2/2) * Usp @ Usp
    U_m = np.eye(4) -1j*thetam*Usm - (thetam**2/2) * Usm @ Usm


    # second order expansion of Hamiltonian
    H_expansion = np.eye(2) -1j*dt*H - (dt**2/2) * H @ H

    # third order expansion of time evolution operator
    #U_p3 = np.eye(4) -1j*thetap*Usp - (thetap**2/2.0) * (Usp @ Usp) + 1j*(thetap**3/6.0) * (Usp @ Usp @ Usp)
    #U_m3 = np.eye(4) -1j*thetam*Usm - (thetam**2/2.0) * (Usm @ Usm) + 1j*(thetam**3/6.0) * (Usm @ Usm @ Usm)

    for t in range(Psize):

        for s in range(S):

            # initial state for each traj
            Psi = np.kron(psi0[t], env)

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

                H_newpsi_s = H_expansion @ newpsi_s

                if (n+1+skip)%skip == 0:


                    result[t,s,r] = H_newpsi_s

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
                print('estimated total simulation time: ', str(timedelta(seconds=simtime*(S-s)*Psize)))

    direc = "../dat/RUD/"
    filename = direc + "HsimMultiInit" + "_S_" + str(S) + "_N_" + str(N) + "_NStates_" + str(Psize)
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

### frequency calculation

def freq(S, N, finaltime, psi0, H, U, temperature, theta=0.1, seed=1337, res=1000, test=1.0):

    #info            = np.array([S, N, temperature, theta])

    T               = temperature # single temperature

    result          = np.zeros(( S, res, 2, 1), dtype='complex128')
    result[:,0]     = psi0
    psi             = np.copy(psi0)
    newpsi          = np.zeros((2,1), dtype='complex128')
    times           = np.linspace(0,finaltime,res)
    dt              = finaltime/N

    skip            = int(np.floor(N/res)) # resolution must be less than timesteps (N)

    # interaction hamiltonians
    #Usp = np.array(([0,0,0,1],[0,-1,0,0],[0,0,0,0],[1,0,0,1]), dtype='complex128')
    #Usm = np.array(([0,0,0,0],[0,-1,1,0],[0,1,0,0],[0,0,0,1]), dtype='complex128')

    # environment for RUD
    env = np.copy(bas0)

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
    U_p = np.eye(4) -1j*thetap*U[0] - (thetap**2/2) * U[0] @ U[0]
    U_m = np.eye(4) -1j*thetam*U[1] - (thetam**2/2) * U[1] @ U[1]

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


    # calculating frequency of the run

    QT = result

    rhos = np.zeros((QT[:,0,0,0].size,QT[0,:,0,0].size, 2, 2), dtype=QT.dtype) # array of all density matrices

    # master loop - computing freq for each individual traj and storing it in 'frequencies'-array

    frequencies = np.zeros(QT[:,0,0,0].size)

    for s in range(QT[:,0,0,0].size):
        
        rho = QT[s] @ dag(QT[s])

        rx = rho[:,1,0] + rho[:,0,1]
        ry = 1j*(rho[:,1,0] - rho[:,0,1])

        eps = 1e-1 # sensitivity for detecting minima

        angles = np.arctan2(ry.real, rx.real)
        abs_angles = np.abs(angles)

        minima = ma.masked_less(abs_angles, eps).mask

        
        # for loop to determine suitable sensitivity in cut off
        
        count = 0; long_count = 0

        for i in range(minima.size - 1):

            if minima[i] or minima[i + 1]:
                
                if count > long_count:

                    long_count = count
                
                count = 0

            else:

                count += 1

        sens = int(np.floor(long_count/2)) # computing sensitivity

        
        # for-loop to cut off extra maxima around peaks due to diffusion "noise"

        cut_minima = minima[sens:]
        cutoff_minima = np.copy(minima)
        cutoff_minima[:sens] = 0

        for i in range(cutoff_minima.size):

            if cutoff_minima[i]:

                cutoff_minima[ i + 1 : (i + sens) ] = 0


        # computing frequency

        rotcount = np.sum(cutoff_minima)

        total_angle = 2*np.pi * rotcount + abs_angles[-1]

        rotations = total_angle/(2*np.pi)

        frequencies[s] = rotations / times[-1]

    measured_frequency = np.mean(frequencies)

    """
    direc = "../dat/RUD/"
    # should find a better filename system
    filename = direc + "Hsim" + "_S_" + str(S) + "_N_" + str(N) + "_tet_" + str(theta).replace('.', 'p')
    np.save(filename, result)

    timesname = filename + "_times"
    np.save(timesname, times)

    psi0name = filename + "_psi0"
    np.save(psi0name, psi0)

    hamname = filename + "_ham"
    np.save(hamname, H)

    infoname = filename + "_info"
    np.save(infoname, info)
    """

    return measured_frequency

def freqSynchro(M, S, N, omega0, dOmega, finaltime, psi0, U, temperature, theta=0.1, sig_strength=1.0, seed=1337, res=1000, test=1.0):

    # M: number of quantum trajectory simulation to perform
    # S: number of monte carlo simulations (trajectories per run)
    # N: number of timesteps per trajectory
    # dOmega: frequency difference (system frequency, ie H_0 = omega/2 * sigmaz)

    # U = [U_p, U_m]

    info = np.array([M, S, N, omega0, dOmega, finaltime, temperature, theta, sig_strength] )

    traj_freq = np.zeros(M)

    """ test """

    omega = np.linspace(omega0-dOmega, omega0+dOmega, M)
    H = np.zeros((M, 2, 2), dtype='complex128')

    """
    for i in range(M):
        H[i] = np.array( 0.5*omega0*sigmaz + 1j * 0.25 * sig_strength * (np.exp(1j*omega[i]) * sigmam - np.exp(-1j*omega[i]) * sigmap ) , dtype='complex128' )
    """

    for i in range(M):
        H[i] = np.array( (0.5 * (omega0 - omega[i]) * sigmaz + 0.5 * sig_strength * sigmay ), dtype='complex128' )


    for m in range(M):

        print(m)

        if m==0:
            ping_freq_0 = time.perf_counter()

        traj_freq[m] = freq(S, N, finaltime, psi0, H[m], U, temperature, theta, seed, res, test)

        # updating seed
        seed += int(S*N)

        if m==0:
            ping_freq_1 = time.perf_counter()
            print("estimated finish time: ", (ping_freq_1-ping_freq_0)*M)


    

    direc = "../dat/freq/"
    # should find a better filename system
    filename = direc + "FREQ" + "_M_" + str(M) + "_S_" + str(S) + "_N_" + str(N) + "_tet_" + str(theta).replace('.', 'p') + "_dOmega_" + str(dOmega).replace('.', 'p')
    np.save(filename, traj_freq)

    timesname = filename + "_omega"
    np.save(timesname, omega)

    infoname = filename + "_info"
    np.save(infoname, info)

    return 0






### useful objects

bas0 = np.array(([1.0], [0.0]), dtype='complex128')
bas1 = np.array(([0.0], [1.0]), dtype='complex128')

yplus = np.sqrt(0.5) * (bas0 + 1j*bas1)
yminus = np.sqrt(0.5) * (bas0 - 1j*bas1)

xplus = np.sqrt(0.5) * (bas0 + bas1)
xminus = np.sqrt(0.5) * (bas0 - bas1)

sigmax = np.array(([0.0, 1.0], [1.0, 0.0]), dtype='complex128')
sigmay = np.array(([0.0, -1.0j], [1.0j, 0.0]), dtype='complex128')
sigmaz = np.array(([1.0, 0.0], [0.0, -1.0]), dtype='complex128')

sigmap = np.array(([0.0, 1.0], [0.0, 0.0]), dtype='complex128')
sigmam = np.array(([0.0, 0.0], [1.0, 0.0]), dtype='complex128')

# z-basis

Um_z = 0.5*( np.kron( sigmax, sigmax ) + np.kron( sigmay, sigmay ) )
Am_z = 0.5*np.array([sigmax, sigmay])
Bm_z = 0.5*np.array([sigmax, sigmay])

Up_z = 0.5*( np.kron( sigmax, sigmax ) - np.kron( sigmay, sigmay ) )
Ap_z = 0.5*np.array([sigmax, -1*sigmay])
Bp_z = 0.5*np.array([sigmax, sigmay])

# x-basis

Um_x = 0.5*( np.kron( sigmax, sigmay ) + np.kron( sigmay, sigmaz ) )
Am_x = 0.5*np.array([sigmax, sigmay])
Bm_x = 0.5*np.array([sigmay, sigmaz])

Up_x = 0.5*( np.kron( sigmax, sigmay ) - np.kron( sigmay, sigmaz ) )
Ap_x = 0.5*np.array([sigmax, -1*sigmay])
Bp_x = 0.5*np.array([sigmay, sigmaz])

# y-basis

Um_y = 0.5*( np.kron( sigmax, sigmax ) + np.kron( sigmay, sigmaz ) )
Am_y = 0.5*np.array([sigmax, sigmay])
Bm_y = 0.5*np.array([sigmax, sigmaz])

Up_y = 0.5*( np.kron( sigmax, sigmax ) - np.kron( sigmay, sigmaz ) )
Ap_y = 0.5*np.array([sigmax, -1*sigmay])
Bp_y = 0.5*np.array([sigmax, sigmaz])

### end of methods

### improvement list

# add line at the start of the saved files with value of theta and other information so that this can be read off easily when plotting 
## fixed by saving independent "info"-array with each file