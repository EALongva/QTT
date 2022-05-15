# frequency analysis, attempt to work out new method for frequency calculation

from QTT import *

### util

def oldFreq(angles, simtime):

    eps = 1e-1
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

    cutoff_minima[:sens] = 0

    for i in range(cutoff_minima.size):

        if cutoff_minima[i]:

            cutoff_minima[ i + 1 : (i + sens) ] = 0


    rotcount = np.sum(cutoff_minima)

    total_angle = 2*np.pi * rotcount + abs_angles[-1]

    rotations = total_angle/(2*np.pi)

    measured_frequency = rotations / simtime

    return measured_frequency

def oldFreqAvg(trajectories, simtime):

    S = trajectories[:,0,0,0].size

    freq_result = np.zeros(S)

    for s in tqdm(range(S)):

        rho = trajectories[s] @ dag(trajectories[s])

        x = ( rho[:,1,0] + rho[:,0,1] ).real
        rx = x - np.mean(x)

        z = ( rho[:,0,0] - rho[:,1,1] ).real
        rz = z - np.mean(z)

        angles = np.arctan2(rx, rz)

        freq_result[s] = oldFreq(angles, simtime)

    avgfreq = np.mean(freq_result)

    return avgfreq

def fft_angles2freq(angles, dt):

    fft_result  = np.fft.fft(angles)
    freq_array  = np.fft.fftfreq(angles.size, d=dt)
    fftmeasfreq = np.abs( freq_array[ np.argmax( fft_result.imag ) ] )

    return fftmeasfreq

def fft_freq(trajectories, dt):

    S = trajectories[:,0,0,0].size

    freq_result = np.zeros(S)

    for s in tqdm(range(S)):

        rho = trajectories[s] @ dag(trajectories[s])

        x = ( rho[:,1,0] + rho[:,0,1] ).real
        rx = x - np.mean(x)

        z = ( rho[:,0,0] - rho[:,1,1] ).real
        rz = z - np.mean(z)

        angles = np.arctan2(rx, rz)

        freq_result[s] = fft_angles2freq(angles, dt)

    avgfreq = np.mean(freq_result)

    return avgfreq


### main script

def main(S, N, theta, simtime, psi0, ncpu, burnin, epsilon):

    dt = simtime/N
    times = np.linspace(0, simtime, N)

    path = '../data/freq/'
    loadname = path + 'theta' + str(theta).replace('.', '') + \
        '_eps' + str(epsilon).replace('.', '') + '_dt' + str(dt).replace('.', '') + \
        '_S_' + str(S) + '_N_' + str(N) + '_burnin_N_' + str(burnin) 
    
    result = np.load(loadname + '.npy')

    print('shape of input file ', result.shape)

    ### test average frequency methods

    sample_result = result[3,0:(int(S/1))]
    S_ = sample_result[:,0,0,0].size

    old_method = oldFreqAvg(sample_result, simtime)
    print('[OLD METHOD] average frequency for ', S_, ' trajectories : ', old_method)

    fft_method = fft_freq(sample_result, dt)
    print('[FFT METHOD] average frequency for ', S_, ' trajectories : ', fft_method)


    ### lets take a closer look at the trajectories

    result2 = result[2]

    print('shape of trajectories for single frequency ', result2.shape)
    
    plt.style.use('ggplot')
    """
    for s in tqdm( range( int(S/20) ) ):

        states = result2[s]
        rho = states @ dag(states)

        x = ( rho[:,1,0] + rho[:,0,1] ).real
        rx = x - np.mean(x)

        y = ( 1j*(rho[:,1,0] - rho[:,0,1]) ).real
        ry = y - np.mean(y)
        
        z = ( rho[:,0,0] - rho[:,1,1] ).real
        rz = z - np.mean(z)
        
        
        fig = plt.figure()

        ax1 = fig.add_subplot(221)
        ax1.scatter(rx, ry, color='r', s=1.5)
        ax1.set_xlabel(r'$\langle \sigma_x \rangle$')
        ax1.set_ylabel(r'$\langle \sigma_y \rangle$')
        ax1.set_title('z-axis')

        ax2 = fig.add_subplot(222)
        ax2.scatter(rz, ry, color='b', s=1.5)
        ax2.set_xlabel(r'$\langle \sigma_z \rangle$')
        ax2.set_ylabel(r'$\langle \sigma_y \rangle$')
        ax2.set_title('x-axis')

        ax3 = fig.add_subplot(223)
        ax3.scatter(rx, rz, color='g', s=1.5)
        #ax3.scatter(np.mean(rx), np.mean(rz), color='y', s=4.0)
        ax3.set_xlabel(r'$\langle \sigma_x \rangle$')
        ax3.set_ylabel(r'$\langle \sigma_z \rangle$')
        ax3.set_title('y-axis')

        ax4 = fig.add_subplot(224, projection='3d')
        ax4.view_init(-30,60)
        sphere = qp.Bloch(axes=ax4)
        psize = 10.0
        sphere.point_size = [psize,psize,psize,psize]
        sphere.add_points( [rx, ry, rz] )
        sphere.make_sphere()
        ax4.set_title('Trajectory on Bloch sphere')

        fig.set_size_inches(16,9)

        figname = "../figure/freqcalc/test1/" + 'eps' + str(epsilon).replace('.', '')  \
            + '_sample' + str(s) + '.png'
        
        fig.savefig(figname, dpi=400)

    print('fin')

    """    
    ### just taking the mean of the points along the "best" axis atleast gives us the center.
    ### developing this futher:
    ### subtract the new "center" to align coordinate system -> then try to use arctan2 to find angles
    
    """
    test_traj = result2[0]
    rho = test_traj @ dag(test_traj)

    x = ( rho[:,1,0] + rho[:,0,1] ).real
    rx = x - np.mean(x)

    y = ( 1j*(rho[:,1,0] - rho[:,0,1]) ).real
    ry = y - np.mean(y)
    
    z = ( rho[:,0,0] - rho[:,1,1] ).real
    rz = z - np.mean(z)

    angles = np.arctan2(rx, rz)

    fig = plt.figure()
    ax1 = fig.add_subplot(131)
    ax1.plot(times, angles)

    ### fft test
    fft_result = np.fft.fft(angles)
    test = np.fft.fftfreq(N, d=dt)
    fftmeasfreq = test[ np.argmax( fft_result.imag ) ]
    print('fft measured frequency: ', fftmeasfreq)

    ### test it with old method
    oldfreq_result = oldFreq(angles, simtime)
    print('fft measured frequency (old method): ', oldfreq_result)

    ax2 = fig.add_subplot(132)
    ax2.plot(test, fft_result.real)
    ax2.plot(test, fft_result.imag)
    ax2.set_xlim([-0.05,0.05])

    ax3 = fig.add_subplot(133)
    ax3.plot(test, fft_result.real)
    ax3.plot(test, fft_result.imag)
    ax3.set_xlim([-1.0, 1.0])

    fig.set_size_inches(16,9)

    figname = "../figure/freqcalc/test1/" + 'eps' + str(epsilon).replace('.', '')  \
        + '_zxplaneARCTAN2' + '.png'
    
    fig.savefig(figname, dpi=400)
    """


    """
    ### trying to find the rotational axis

    states = result2[0] # single trajectory

    #print(centerstate)

    rho = states @ dag(states)

    rx = np.mean( rho[:,1,0] + rho[:,0,1] )
    ry = np.mean( 1j*(rho[:,1,0] - rho[:,0,1]) )
    rz = np.mean( rho[:,0,0] - rho[:,1,1] )
    
    
    
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.view_init(30,60)
    sphere = qp.Bloch(axes=ax)
    psize = 15.0
    sphere.point_size = [psize,psize,psize,psize]
    sphere.add_points( [rx, ry, rz] )
    sphere.make_sphere()
    ax.set_title('Trajectory on Bloch sphere')

    fig.set_size_inches(16,9)

    figname = "../figure/freqcalc/test1/" + 'singletrajectory_rotaxis' + '.png'
    
    fig.savefig(figname, dpi=400)"""

    return 0




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

main(S, N, theta, simtime, psi0, ncpu, burnin, epsilon)