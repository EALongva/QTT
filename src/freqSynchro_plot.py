# plotting results from freqSynchro_run

from methods import *

location    = "../dat/freq/"
loadname    = location + "FREQ_M_24_S_1000_N_1000_tet_0p1_dOmega_0p02"
freq = np.load(loadname + ".npy")
omega = np.load(loadname + "_omega.npy")
info = np.load(loadname + "_info.npy")


M = info[0]
S = info[1]
N = info[2]
omega0 = info[3]
dOmega = info[4]
finaltime = info[5]
temperature = info[6]
theta = info[7]
sig_strength = info[8]

freqdiff = freq -omega

plt.style.use('ggplot')
fig = plt.figure()

ax = fig.add_subplot()
ax.plot(omega, freqdiff)
ax.set_xlabel(r'$\omega$')
ax.set_ylabel(r'$\Omega - \omega$')

title = r'Frequency difference in system $\Omega$ and external signal $\omega$, ' + r'$\theta$ = ' + str(theta) + ', temperature: ' \
+ str(temperature) + ', \n trajectories: ' + str(int(S)) + ', timesteps: ' + str(int(N)) + ', samples: ' + str(int(M)) + '\n' \
+ r'with $H = \frac{\omega_0}{2}\sigma_z + \frac{i\epsilon}{4} ( \exp(i\omega t)\sigma_- - \exp(-i\omega t)\sigma_+  )$'

ax.set_title(title)
fig.set_size_inches(16,9)


figname = "../fig/freq/system_signal_frequency_diff_" + "M_" + str(M) + "_N_" + str(N) + "_S_" + str(S) + "_tet_" + str(theta).replace('.', 'p') + "_dOmega_" + str(dOmega).replace('.', 'p') + ".png"
fig.savefig(figname, dpi=400)


plt.show()