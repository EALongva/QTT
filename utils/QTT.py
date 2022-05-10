# QTT class

from os import times_result
import sys as sys
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import qutip as qp
import random as rnd
import time as time
from datetime import timedelta
import math as math
import multiprocessing as mp
from tqdm import tqdm
from scipy.linalg import expm
from mpl_toolkits.mplot3d import Axes3D

""" utilities """

def dag(x):

    # hermitian conjugate of 3 dim array with (2,1) matrices along first axis
    return np.transpose(np.conj(x), (0,2,1))

### useful objects

bas0 = np.array(([1.0], [0.0]), dtype='complex128')
bas1 = np.array(([0.0], [1.0]), dtype='complex128')

xplus = np.sqrt(0.5) * (bas0 + bas1)
xminus = np.sqrt(0.5) * (bas0 - bas1)

yplus = np.sqrt(0.5) * (bas0 + 1j*bas1)
yminus = np.sqrt(0.5) * (bas0 - 1j*bas1)

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



""" class struct """

class QTT:

    """ Master class for all quantum trajectory theory methods. """

    def __init__(self, env, meas_basis, hamiltonian=0, dt=0.01, theta=0.01, temperature=0.5, seed=1337 ):

        self.env            = env
        self.meas           = meas_basis
        self.seed           = seed

        ### for debugging:

        self.debug = False


        rnd.seed(seed) # seeding the rng

        self.temperature    = temperature
        self.theta          = theta
        self.dt             = dt
        self.inittime       = 0.0
        self.finaltime      = 0.0
        self.timesteps      = 1

        # storing results from simulations
        self.trajectory = np.zeros((1,2,1), dtype='complex128')
        self.mcResult   = np.zeros((1,1,2,1), dtype='complex128')
        self.state      = np.zeros((2,1), dtype='complex128')
        self.state0     = np.zeros((2,1), dtype='complex128')
        #self.rhoResult  = np.zeros((1,2,2), dtype='complex128')
        self.rhoResult  = 0
        self.blochvecResult = []
        self.freqResult = 0

        # Lindblad solutions using QuTiP
        self.rhoLB      = 0
        self.blochvecLB = []
        
        # defining the system Hamiltonian
        if hamiltonian == 0:
            print('Defaulting to empty hamiltonian')
            self.H          = np.zeros((2, 2), dtype='complex128')
        else:
            self.H          = hamiltonian

        
        # variables not defined by init args
        self.resolution     = 0


        # assign interaction hamiltonian corresponding to chosen environment

        if env == 'x':

            self.Up = Up_x
            self.Ap = Ap_x
            self.Bp = Bp_x
            
            self.Um = Um_x
            self.Am = Am_x
            self.Bm = Bm_x

            self.env_state = xplus

        elif env == 'y':

            self.Up = Up_y
            self.Ap = Ap_y
            self.Bp = Bp_y
            
            self.Um = Um_y
            self.Am = Am_y
            self.Bm = Bm_y

            self.env_state = yminus

        elif env == 'z':

            self.Up = Up_z
            self.Ap = Ap_z
            self.Bp = Bp_z
            
            self.Um = Um_z
            self.Am = Am_z
            self.Bm = Bm_z

            self.env_state = bas0

        else:

            print("invalid environment chosen: expected type string with arguments 'x', 'y' or 'z', but got: ", self.env)


        ### lindblad operators

        self.Lp = qp.basis(2,1)*qp.basis(2,0).dag()
        self.Lm = qp.basis(2,0)*qp.basis(2,1).dag()

        # second order expansions of system hamiltonian and interaction hamiltonian for single temperature, for multiple 
        # temperatures I will have to implement a different method could potentially just make a MC method for multiple 
        # temperatures where the expansions are updated for each temp

        pdensity        = 1 / ( np.exp( 1/self.temperature ) - 1 )
        self.gammap          = pdensity
        self.gammam          = pdensity + 1

        # temperature dependent interaction strength
        self.thetap     = np.sqrt(self.gammap) * self.theta
        self.thetam     = np.sqrt(self.gammam) * self.theta

        self.H_expansion    = np.eye(2) - 1j*self.dt * self.H - (self.dt**2/2) * self.H @ self.H
        self.Up_expansion   = np.eye(4) - 1j*self.thetap * self.Up - (self.thetap**2/2) * self.Up @ self.Up
        self.Um_expansion   = np.eye(4) - 1j*self.thetam * self.Um - (self.thetam**2/2) * self.Um @ self.Um


    def evolution_ops(self):

        self.H_expansion    = np.eye(2) - 1j*self.dt * self.H - (self.dt**2/2) * self.H @ self.H
        self.Up_expansion   = np.eye(4) - 1j*self.thetap * self.Up - (self.thetap**2/2) * self.Up @ self.Up
        self.Um_expansion   = np.eye(4) - 1j*self.thetam * self.Um - (self.thetam**2/2) * self.Um @ self.Um

        return 0


    def system_hamiltonian(self, delta, epsilon):
        # Hamiltonian in the rotating reference system

        self.H = delta * sigmaz + epsilon * sigmay

        return 0


    def rho(self):

        rhos = np.zeros((self.mcResult[:,0,0,0].size,self.mcResult[0,:,0,0].size, 2, 2), dtype='complex128') # array of all density matrices

        for s in range(self.mcResult[:,0,0,0].size):
            rhos[s] = self.mcResult[s] @ dag(self.mcResult[s])

        self.rhoResult = np.mean(rhos, axis=0)

        return self.rhoResult


    def blochvec(self, LB=False):
        # calculating bloch vector

        print('compute rho self.rho() before computing blochvecs')

        if LB is False:
            x = self.rhoResult[:,1,0] + self.rhoResult[:,0,1]
            y = 1j*(self.rhoResult[:,1,0] - self.rhoResult[:,0,1])
            z = self.rhoResult[:,0,0] - self.rhoResult[:,1,1]
        else:
            x = self.rhoLB[:,1,0] + self.rhoLB[:,0,1]
            y = 1j*(self.rhoLB[:,1,0] - self.rhoLB[:,0,1])
            z = self.rhoLB[:,0,0] - self.rhoLB[:,1,1]

        self.blochvecResult = [x,y,z]

        return [x, y, z]


    def times(self):

        times_result = np.linspace(self.inittime, self.finaltime, self.timesteps)

        return times_result


    def MC(self, S, psi_sys_0, timesteps, finaltime, dseed=0, traj_resolution=0):
        # Monte Carlo simulation over S trajectories

        self.state0 = psi_sys_0
        self.timesteps = timesteps

        if dseed == 0:
            dseed = self.seed

        print('seed: ', dseed)

        rnd.seed(dseed)

        MC_traj = np.zeros((S, timesteps, 2, 1), dtype='complex128')

        for s in range(S):

            MC_traj[s] = self.Traj(psi_sys_0, timesteps, finaltime, traj_resolution)
            #self.seed += 1

        self.mcResult = MC_traj

        return MC_traj


    def paraMC(self, S, psi_sys_0, timesteps, finaltime, ncpu, traj_resolution=0):

        self.finaltime  = finaltime
        self.dt         = finaltime / timesteps
        self.state0     = psi_sys_0
        self.timesteps  = timesteps

        if S%ncpu == 0:
            S_ = int(S/ncpu)
        else:
            print('please enter a valid combination of S and ncpu')

        args = []
        for d in range(ncpu):
            dseed = self.seed + d
            args.append([S_, psi_sys_0, timesteps, finaltime, dseed])

        pool = mp.Pool(ncpu)
        result = pool.starmap(self.MC, args)
        pool.close()

        self.mcResult = np.concatenate(np.asarray(result)) #tiling trajectory results from all cpus

        print('successfully simulated ', self.mcResult[:,0,0,0].size, ' trajectories')

        return 0


    def burnin_estimate(self):
        return 0


    def Traj(self, psi_sys_0, timesteps, finaltime, traj_resolution=0):

        if traj_resolution != 0:
            self.resolution = traj_resolution
        else:
            self.resolution = timesteps

        # Single trajectory simulation over N timesteps
        # set up state: system tensordot environment
        # evolve with interaction hamiltonian
        # measure composite system -> collapse entanglement
        # evolve system state
        # return the new system state

        self.finaltime  = finaltime
        self.dt         = finaltime / timesteps
        self.timesteps  = timesteps
        self.state0     = psi_sys_0

        self.evolution_ops()

        Psi = np.array(np.kron(psi_sys_0, self.env_state), dtype='complex128')

        skip = int(np.floor( timesteps / self.resolution )) # resolution must be less than or equal to timesteps (N)

        traj_result = np.zeros((self.resolution, 2, 1), dtype='complex128')
        traj_result[0] = psi_sys_0

        r = 1

        for n in range( timesteps - 1 ):

            p = rnd.random()

            # alternating evenly, temperature dependence in interaction strength parameter
            if (n+1)%2 == 1:

                newPsi = self.Up_expansion @ Psi

            else:

                newPsi = self.Um_expansion @ Psi
            
            newpsi_s = self.measure(newPsi, p)

            # Hamiltonian time evolution

            H_newpsi_s = self.H_expansion @ newpsi_s

            if (n+1+skip)%skip == 0:

                traj_result[r] = H_newpsi_s

                r += 1

            Psi     = np.kron(H_newpsi_s, self.env_state) # entangling updated system state with a new environment state

        self.state = H_newpsi_s
        self.trajectory = traj_result

        return traj_result


    def Burnin(self, psi_sys_0, timesteps, finaltime, traj_resolution=0):

        if traj_resolution != 0:
            self.resolution = traj_resolution
        else:
            self.resolution = timesteps

        # Single trajectory simulation over N timesteps
        # set up state: system tensordot environment
        # evolve with interaction hamiltonian
        # measure composite system -> collapse entanglement
        # evolve system state
        # return the new system state

        self.finaltime  = finaltime
        self.dt         = finaltime / timesteps
        self.timesteps  = timesteps
        self.state0     = psi_sys_0

        self.evolution_ops()

        Psi = np.array(np.kron(psi_sys_0, self.env_state), dtype='complex128')

        skip = int(np.floor( timesteps / self.resolution )) # resolution must be less than or equal to timesteps (N)

        traj_result = np.zeros((self.resolution, 2, 1), dtype='complex128')
        traj_result[0] = psi_sys_0

        r = 1

        for n in tqdm(range( timesteps - 1 )):

            p = rnd.random()

            # alternating evenly, temperature dependence in interaction strength parameter
            if (n+1)%2 == 1:

                newPsi = self.Up_expansion @ Psi

            else:

                newPsi = self.Um_expansion @ Psi
            
            newpsi_s = self.measure(newPsi, p)

            # Hamiltonian time evolution

            H_newpsi_s = self.H_expansion @ newpsi_s

            if (n+1+skip)%skip == 0:

                traj_result[r] = H_newpsi_s

                r += 1

            Psi     = np.kron(H_newpsi_s, self.env_state) # entangling updated system state with a new environment state

        self.state = H_newpsi_s
        self.trajectory = traj_result

        return traj_result


    def lindblad(self):

        timearray   = np.linspace(self.inittime, self.finaltime, self.timesteps)

        print(timearray.shape)

        a           = np.sqrt( (self.theta**2) / (2.0*self.dt))

        c_ops       = [ np.sqrt(self.gammam) * a * self.Lm, np.sqrt(self.gammap) * a * self.Lp ]

        print('collapse operators: ', c_ops)
        print('init state: ', self.state0)
        print('hamiltonian: ', self.H)

        lb_result   = qp.mesolve(qp.Qobj(self.H), qp.Qobj(self.state0), timearray, c_ops, [])

        self.rhoLB  = np.array(lb_result.states)

        lx = self.rhoLB[:,1,0] + self.rhoLB[:,0,1]
        ly = 1j*(self.rhoLB[:,1,0] - self.rhoLB[:,0,1])
        lz = self.rhoLB[:,0,0] - self.rhoLB[:,1,1]

        self.blochvecLB = [lx, ly, lz]

        return lb_result


    def interaction_evolve(self):

        # most likely not needed but will exist inside of the trajectory simulation

        return 0

    def system_evolve(self):

        return 0



    def measure(self, Psi, p):
        # measuring environment after sys@env interaction

        if self.meas == 'x':

            xp_meas = ( Psi[0] + Psi[1] ) * bas0 + ( Psi[2] + Psi[3] ) * bas1

            prob_xp_meas = ( np.sqrt(0.5) * np.conj( xp_meas ).T ) @ ( np.sqrt(0.5) * xp_meas )

            if p <= prob_xp_meas:
                # corresponds to measuring the environment to be in |x+>
                norm = 1 / np.sqrt( np.conj( xp_meas ).T @ xp_meas )
                psi_s = norm * xp_meas

            else:
                # corresponds to measuring the environment to be in |x->
                xm_meas = ( Psi[0] - Psi[1] ) * bas0 + ( Psi[2] - Psi[3] ) * bas1
                norm = 1 / np.sqrt( np.conj( xm_meas ).T @ xm_meas )
                psi_s = norm * xm_meas

        elif self.meas == 'y':

            yp_meas = ( Psi[0] - 1j*Psi[1] ) * bas0 + ( Psi[2] - 1j*Psi[3] ) * bas1

            prob_yp_meas = ( np.sqrt(0.5) * np.conj( yp_meas ).T ) @ ( np.sqrt(0.5) * yp_meas )

            if p <= prob_yp_meas:
                # corresponds to measuring the environment to be in |y+>
                norm = 1 / np.sqrt( np.conj( yp_meas ).T @ yp_meas )
                psi_s = norm * yp_meas

            else:
                # corresponds to measuring the environment to be in |y->
                ym_meas = ( Psi[0] + 1j*Psi[1] ) * bas0 + ( Psi[2] + 1j*Psi[3] ) * bas1
                norm = 1 / np.sqrt( np.conj( ym_meas ).T @ ym_meas )
                psi_s = norm * ym_meas

        elif self.meas == 'z':

            zp_meas = ( Psi[0] + Psi[2] ) * bas0

            prob_zp_meas = ( np.sqrt(0.5) * np.conj( zp_meas ).T ) @ ( np.sqrt(0.5) * zp_meas )

            if p <= prob_zp_meas:
                # corresponds to measuring the environment to be in |z+>
                norm = 1 / np.sqrt( np.conj( zp_meas ).T @ zp_meas )
                psi_s = norm * zp_meas

            else:
                # corresponds to measuring the environment to be in |z->
                zm_meas = ( Psi[1] + Psi[3] ) * bas1
                norm = 1 / np.sqrt( np.conj( zm_meas ).T @ zm_meas )
                psi_s = norm * zm_meas

        return psi_s


    def freq(self):

        # master loop - computing freq for each individual traj and storing it in 'frequencies'-array

        QT = self.mcResult

        S = QT[:,0,0,0].size

        times = self.times()

        frequencies = np.zeros(S)

        for s in range(S):

            rho = QT[s] @ dag(QT[s])

            rx = rho[:,1,0] + rho[:,0,1]
            ry = 1j*(rho[:,1,0] - rho[:,0,1])

            #print('bloch vec x: ', rx.shape)

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

            #cut_minima = minima[sens:]
            cutoff_minima = np.copy(minima)

            ### debugging
            if sens <= 1:

                """
                print('angles shape ', angles.shape, 'minima shape: ', minima.shape)
                print('sens: ', sens)
                print('cutoff_minima shape: ', cutoff_minima.shape)

                # could even make it so you can init the class with a designated debug folder
                debugpath = '../data/debug/freq/trajectory_debug_freq_minima'
                np.save(debugpath, QT[s])

                bugresult = np.asarray([rx, ry])
                debugpath2 = '../data/debug/freq/bvecXY_debug_freq_minima'
                np.save(debugpath2, bugresult)

                self.debug = True 
                """
                break
            
            else:

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

        return measured_frequency

    def freqSynchro(self, M, S, N, burnin, psi0, simtime, ncpu, delta0, dDelta, epsilon):

        # M: number of quantum trajectory simulation to perform
        # S: number of monte carlo simulations (trajectories per run)
        # N: number of timesteps per trajectory
        # dDelta: frequency difference (system frequency, ie H_0 = omega/2 * sigmaz)

        ### finding burnin state:

        burnin_finaltime = burnin * (simtime/N)

        traj_freq = np.zeros(M)

        """ test """

        omega = np.linspace(delta0-dDelta, delta0+dDelta, M)

        """
        for i in range(M):
            Harray[i] = np.array( 0.5*delta0*sigmaz + 1j * 0.25 * epsilon * (np.exp(1j*omega[i]) * sigmam - np.exp(-1j*omega[i]) * sigmap ) , dtype='complex128' )
        """
        """
        for i in range(M):
            Harray[i] = self.system_hamiltonian(omega[i], epsilon)
        """

        for m in range(M):

            self.system_hamiltonian(omega[m], epsilon)
 
            self.Burnin(psi0, burnin, burnin_finaltime)
            burnin_state = self.state

            self.paraMC(S, burnin_state, N, simtime, ncpu)

            self.seed += ncpu + 1 # keeping fresh seeding for all MC simulations

            # computing bloch vectors
            #self.rho()
            #self.blochvec()

            traj_freq[m] = self.freq()
        

        self.freqResult = traj_freq

        return traj_freq





### What should the class structure be like?

# initialize class object with interaction hamiltonian and measurement basis,
# this can preferably be changed or chosen simply by initializing some attribute 

# Needs the basic simulation method taking where N, dt or T must be defined

# method for doing the monte carlo simulation and saving results to file, exactly
# how the savefile should be handled is still not entirely clear

# additional methods should include: converting system states to density matrix,
# converting to spherical coordinates (pure states on surface), calculating 
# frequency, plotting, comparing QTT with master equation ...



""" testing QTT """

if __name__ == '__main__':

    env = 'z'
    meas_basis = 'x'

    seed = 381693

    #hamiltonian = 

    test = QTT(env, meas_basis, seed=seed)
    print(test.Up)

    delta = 0.1
    epsilon = 2.0
    test.system_hamiltonian(delta, epsilon)
    print(test.H)