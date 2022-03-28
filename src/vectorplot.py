# vector plot showing the effect of some interaction on states (bloch sphere)

import numpy as np
import matplotlib.pyplot as plt
import qutip as qp

def dag(X):
    return np.transpose( np.conj(X), (0,2,1) )

bas0 = np.array(([1.0], [0.0]), dtype='complex128')
bas1 = np.array(([0.0], [1.0]), dtype='complex128')

sigmax = np.array(([0.0, 1.0], [1.0, 0.0]), dtype='complex128')
sigmay = np.array(([0.0, -1.0j], [1.0j, 0.0]), dtype='complex128')
sigmaz = np.array(([1.0, 0.0], [0.0, -1.0]), dtype='complex128')

yplus = np.sqrt(0.5) * (bas0 + 1j*bas1)
yminus = np.sqrt(0.5) * (bas0 - 1j*bas1)
xplus = np.sqrt(0.5) * (bas0 + bas1)
xminus = np.sqrt(0.5) * (bas0 - bas1)

### possible interactions with (z,x,y)-environment bits

# z-basis
Usm = np.array(([0,0,0,0],[0,-1,1,0],[0,1,0,0],[0,0,0,1]), dtype='complex128')
Usp = np.array(([0,0,0,1],[0,-1,0,0],[0,0,0,0],[1,0,0,1]), dtype='complex128')

#Usmt = 0.5*( np.kron( sigmax, sigmax ) + np.kron( sigmay, sigmay ) + np.kron( sigmaz, sigmaz - np.eye(2) ) )
#Uspt = 0.5*( np.kron( sigmax, sigmax ) - np.kron( sigmay, sigmay ) + np.kron( sigmaz, sigmaz - np.eye(2) ) )

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


### testing

# sett opp et array med tilstander på overflaten av bloch-kula

# velg vekselvirkning og omgivelser

# to mulige utfall etter måling, velg "målebasis"

# Har et uttrykk fra Joakims notater

# U = sum_j A_j tensor B_j

### for z veksekvirkning og omgivelser MINUS

# theta, phi angles in the bloch sphere


def vecplot2d(RES, Am, Ap, Bm, Bp, env, meas):

    theta = np.linspace(np.pi/RES, np.pi - np.pi/RES, RES)
    phi = np.linspace(2*np.pi/RES, 2*np.pi - 2*np.pi/RES, RES) # maybe change to 2*np.pi - (2*np.pi)/RES

    states = np.zeros((RES*RES, 2, 1), dtype='complex128')

    ind = 0
    for i in range(RES):
        for j in range(RES):

            states[ind] = np.cos(theta[i]/2) * bas0 + (np.cos(phi[j]) + 1j*np.sin(phi[j])) * np.sin(theta[i]/2) * bas1
            ind += 1


    #states = np.cos(theta/2) * bas0 + (np.cos(phi) + 1j*np.sin(phi)) * np.sin(theta/2) * bas1

    X = (dag(states) @ sigmax @ states)[:,0,0]
    Y = (dag(states) @ sigmay @ states)[:,0,0]
    Z = (dag(states) @ sigmaz @ states)[:,0,0]

    P = [X.real, Y.real, Z.real]

    # measuring U- meas+

    C = ( np.abs( np.conj(meas[0]).T @ env )**2 )[0,0]

    X1 = C*X
    Y1 = C*Y
    Z1 = C*Z

    X2_ = ( 1j * ( dag(states) @ Am[0] @ sigmax @ states ) * ( np.conj(env).T @ Bm[0] @ meas[0] ) * ( np.conj(meas[0]).T @ env ) \
            + 1j * ( dag(states) @ Am[1] @ sigmax @ states ) * ( np.conj(env).T @ Bm[1] @ meas[0] ) * ( np.conj(meas[0]).T @ env ) )[:,0,0]
    X2  = X2_ + np.conj(X2_)

    Y2_ = ( 1j * ( dag(states) @ Am[0] @ sigmay @ states ) * ( np.conj(env).T @ Bm[0] @ meas[0] ) * ( np.conj(meas[0]).T @ env ) \
            + 1j * ( dag(states) @ Am[1] @ sigmay @ states ) * ( np.conj(env).T @ Bm[1] @ meas[0] ) * ( np.conj(meas[0]).T @ env ) )[:,0,0]
    Y2  = Y2_ + np.conj(Y2_)

    Z2_ = ( 1j * ( dag(states) @ Am[0] @ sigmaz @ states ) * ( np.conj(env).T @ Bm[0] @ meas[0] ) * ( np.conj(meas[0]).T @ env ) \
            + 1j * ( dag(states) @ Am[1] @ sigmaz @ states ) * ( np.conj(env).T @ Bm[1] @ meas[0] ) * ( np.conj(meas[0]).T @ env ) )[:,0,0]
    Z2  = Z2_ + np.conj(Z2_)

    absX1 = np.conj(X1)*X1
    nsx = X1/absX1; dnx = 1/absX1 * ( X2 - nsx * (nsx * X2))

    absY1 = np.conj(Y1)*Y1
    nsy = Y1/absY1; dny = 1/absY1 * ( Y2 - nsy * (nsy * Y2))

    absZ1 = np.conj(Z1)*Z1
    nsz = Z1/absZ1; dnz = 1/absZ1 * ( Z2 - nsz * (nsz * Z2))

    R_mp = [dnx, dny, dnz]

    # measuring U- meas-

    X2_ = ( 1j * ( dag(states) @ Am[0] @ sigmax @ states ) * ( np.conj(env).T @ Bm[0] @ meas[1] ) * ( np.conj(meas[1]).T @ env ) \
            + 1j * ( dag(states) @ Am[1] @ sigmax @ states ) * ( np.conj(env).T @ Bm[1] @ meas[1] ) * ( np.conj(meas[1]).T @ env ) )[:,0,0]
    X2  = X2_ + np.conj(X2_)

    Y2_ = ( 1j * ( dag(states) @ Am[0] @ sigmay @ states ) * ( np.conj(env).T @ Bm[0] @ meas[1] ) * ( np.conj(meas[1]).T @ env ) \
            + 1j * ( dag(states) @ Am[1] @ sigmay @ states ) * ( np.conj(env).T @ Bm[1] @ meas[1] ) * ( np.conj(meas[1]).T @ env ) )[:,0,0]
    Y2  = Y2_ + np.conj(Y2_)

    Z2_ = ( 1j * ( dag(states) @ Am[0] @ sigmaz @ states ) * ( np.conj(env).T @ Bm[0] @ meas[1] ) * ( np.conj(meas[1]).T @ env ) \
            + 1j * ( dag(states) @ Am[1] @ sigmaz @ states ) * ( np.conj(env).T @ Bm[1] @ meas[1] ) * ( np.conj(meas[1]).T @ env ) )[:,0,0]
    Z2  = Z2_ + np.conj(Z2_)

    absX1 = np.conj(X1)*X1
    nsx = X1/absX1; dnx = 1/absX1 * ( X2 - nsx * (nsx * X2))

    absY1 = np.conj(Y1)*Y1
    nsy = Y1/absY1; dny = 1/absY1 * ( Y2 - nsy * (nsy * Y2))

    absZ1 = np.conj(Z1)*Z1
    nsz = Z1/absZ1; dnz = 1/absZ1 * ( Z2 - nsz * (nsz * Z2))

    R_mm = [dnx, dny, dnz]

    # measuring U+ meas+

    X2_ = ( 1j * ( dag(states) @ Ap[0] @ sigmax @ states ) * ( np.conj(env).T @ Bp[0] @ meas[0] ) * ( np.conj(meas[0]).T @ env ) \
            + 1j * ( dag(states) @ Ap[1] @ sigmax @ states ) * ( np.conj(env).T @ Bp[1] @ meas[0] ) * ( np.conj(meas[0]).T @ env ) )[:,0,0]
    X2  = X2_ + np.conj(X2_)

    Y2_ = ( 1j * ( dag(states) @ Ap[0] @ sigmay @ states ) * ( np.conj(env).T @ Bp[0] @ meas[0] ) * ( np.conj(meas[0]).T @ env ) \
            + 1j * ( dag(states) @ Ap[1] @ sigmay @ states ) * ( np.conj(env).T @ Bp[1] @ meas[0] ) * ( np.conj(meas[0]).T @ env ) )[:,0,0]
    Y2  = Y2_ + np.conj(Y2_)

    Z2_ = ( 1j * ( dag(states) @ Ap[0] @ sigmaz @ states ) * ( np.conj(env).T @ Bp[0] @ meas[0] ) * ( np.conj(meas[0]).T @ env ) \
            + 1j * ( dag(states) @ Ap[1] @ sigmaz @ states ) * ( np.conj(env).T @ Bp[1] @ meas[0] ) * ( np.conj(meas[0]).T @ env ) )[:,0,0]
    Z2  = Z2_ + np.conj(Z2_)

    absX1 = np.conj(X1)*X1
    nsx = X1/absX1; dnx = 1/absX1 * ( X2 - nsx * (nsx * X2))

    absY1 = np.conj(Y1)*Y1
    nsy = Y1/absY1; dny = 1/absY1 * ( Y2 - nsy * (nsy * Y2))

    absZ1 = np.conj(Z1)*Z1
    nsz = Z1/absZ1; dnz = 1/absZ1 * ( Z2 - nsz * (nsz * Z2))

    R_pp = [dnx, dny, dnz]

    # measuring U+ meas-

    X2_ = ( 1j * ( dag(states) @ Ap[0] @ sigmax @ states ) * ( np.conj(env).T @ Bp[0] @ meas[1] ) * ( np.conj(meas[1]).T @ env ) \
            + 1j * ( dag(states) @ Ap[1] @ sigmax @ states ) * ( np.conj(env).T @ Bp[1] @ meas[1] ) * ( np.conj(meas[1]).T @ env ) )[:,0,0]
    X2  = X2_ + np.conj(X2_)

    Y2_ = ( 1j * ( dag(states) @ Ap[0] @ sigmay @ states ) * ( np.conj(env).T @ Bp[0] @ meas[1] ) * ( np.conj(meas[1]).T @ env ) \
            + 1j * ( dag(states) @ Ap[1] @ sigmay @ states ) * ( np.conj(env).T @ Bp[1] @ meas[1] ) * ( np.conj(meas[1]).T @ env ) )[:,0,0]
    Y2  = Y2_ + np.conj(Y2_)

    Z2_ = ( 1j * ( dag(states) @ Ap[0] @ sigmaz @ states ) * ( np.conj(env).T @ Bp[0] @ meas[1] ) * ( np.conj(meas[1]).T @ env ) \
            + 1j * ( dag(states) @ Ap[1] @ sigmaz @ states ) * ( np.conj(env).T @ Bp[1] @ meas[1] ) * ( np.conj(meas[1]).T @ env ) )[:,0,0]
    Z2  = Z2_ + np.conj(Z2_)

    absX1 = np.conj(X1)*X1
    nsx = X1/absX1; dnx = 1/absX1 * ( X2 - nsx * (nsx * X2))

    absY1 = np.conj(Y1)*Y1
    nsy = Y1/absY1; dny = 1/absY1 * ( Y2 - nsy * (nsy * Y2))

    absZ1 = np.conj(Z1)*Z1
    nsz = Z1/absZ1; dnz = 1/absZ1 * ( Z2 - nsz * (nsz * Z2))

    R_pm = [dnx, dny, dnz]

    return P, R_mp, R_mm, R_pp, R_pm





"""
RES = 20
eps = 1.0 # interaction strength parameter, normally theta but used as an angle here

theta = np.linspace(np.pi/RES, np.pi - np.pi/RES, RES)
phi = np.linspace(2*np.pi/RES, 2*np.pi - 2*np.pi/RES, RES) # maybe change to 2*np.pi - (2*np.pi)/RES

states = np.zeros((RES*RES, 2, 1), dtype='complex128')

ind = 0
for i in range(RES):
    for j in range(RES):

        states[ind] = np.cos(theta[i]/2) * bas0 + (np.cos(phi[j]) + 1j*np.sin(phi[j])) * np.sin(theta[i]/2) * bas1
        ind += 1


#states = np.cos(theta/2) * bas0 + (np.cos(phi) + 1j*np.sin(phi)) * np.sin(theta/2) * bas1

X = (dag(states) @ sigmax @ states)[:,0,0]
Y = (dag(states) @ sigmay @ states)[:,0,0]
Z = (dag(states) @ sigmaz @ states)[:,0,0]


### RUD environment in |0>, measuring |x> +/-

env = np.copy(bas0)
meas = np.copy(xplus)

# measuring x+

C = ( np.abs( np.conj(meas).T @ env )**2 )[0,0]

X1 = C*X
Y1 = C*Y
Z1 = C*Z

X2_ = ( 1j * ( dag(states) @ Am_z[0] @ sigmax @ states ) * ( np.conj(env).T @ Bm_z[0] @ meas ) * ( np.conj(meas).T @ env ) \
        + 1j * ( dag(states) @ Am_z[1] @ sigmax @ states ) * ( np.conj(env).T @ Bm_z[1] @ meas ) * ( np.conj(meas).T @ env ) )[:,0,0]
X2  = X2_ + np.conj(X2_)

Y2_ = ( 1j * ( dag(states) @ Am_z[0] @ sigmay @ states ) * ( np.conj(env).T @ Bm_z[0] @ meas ) * ( np.conj(meas).T @ env ) \
        + 1j * ( dag(states) @ Am_z[1] @ sigmay @ states ) * ( np.conj(env).T @ Bm_z[1] @ meas ) * ( np.conj(meas).T @ env ) )[:,0,0]
Y2  = Y2_ + np.conj(Y2_)

Z2_ = ( 1j * ( dag(states) @ Am_z[0] @ sigmaz @ states ) * ( np.conj(env).T @ Bm_z[0] @ meas ) * ( np.conj(meas).T @ env ) \
        + 1j * ( dag(states) @ Am_z[1] @ sigmaz @ states ) * ( np.conj(env).T @ Bm_z[1] @ meas ) * ( np.conj(meas).T @ env ) )[:,0,0]
Z2  = Z2_ + np.conj(Z2_)

#print(X2, Y2, Z2)


# now we have the bloch vectors on the form: n = R1 * eps*R2 (R = X, Y, Z)

nx = X1 + eps*X2 + np.conj(X1 + eps*X2)
ny = Y1 + eps*Y2 + np.conj(Y1 + eps*Y2)
nz = Z1 + eps*Z2 + np.conj(Z1 + eps*Z2)
#print(nx, ny, nz)

"""

"""
nsx = X1/(np.abs(X1)); dnx = 1/(np.abs(X1)) * ( X2 - nsx * (nsx * X2))
nsy = Y1/(np.abs(Y1)); dny = 1/(np.abs(Y1)) * ( Y2 - nsy * (nsy * Y2))
nsz = Z1/(np.abs(Z1)); dnz = 1/(np.abs(Z1)) * ( Z2 - nsz * (nsz * Z2))
"""


"""
dnx = ( X2 * np.abs(X1) - (X1/np.abs(X1)) * (X1 * X2) ) / ( np.abs(X1)**2 )
dny = ( Y2 * np.abs(Y1) - (Y1/np.abs(Y1)) * (Y1 * Y2) ) / ( np.abs(Y1)**2 )
dnz = ( Z2 * np.abs(Z1) - (Z1/np.abs(Z1)) * (Z1 * Z2) ) / ( np.abs(Z1)**2 )
"""

"""
absX1 = np.conj(X1)*X1
nsx = X1/absX1; dnx = 1/absX1 * ( X2 - nsx * (nsx * X2))

absY1 = np.conj(Y1)*Y1
nsy = Y1/absY1; dny = 1/absY1 * ( Y2 - nsy * (nsy * Y2))

absZ1 = np.conj(Z1)*Z1
nsz = Z1/absZ1; dnz = 1/absZ1 * ( Z2 - nsz * (nsz * Z2))
"""

RES = 24 # real resolution is RES*RES :)
env = np.copy(xplus)
meas = [np.copy(bas0), np.copy(bas1)]

# same environment basis and measurement basis does not compute

P, R_mp, R_mm, R_pp, R_pm = vecplot2d(RES, Am_x, Ap_x, Bm_x, Bp_x, env, meas)


#print(X2, Y2, Z2)

plt.style.use('ggplot')
fig = plt.figure()

ax1 = fig.add_subplot(321)
ax1.quiver(P[1], P[2], R_mp[1], R_mp[2], color='r', label=r'$| 0 \rangle$')
ax1.quiver(P[1], P[2], R_mm[1], R_mm[2], color='b', label=r'$| 1 \rangle$')
ax1.set_xlim([-1, 1])
ax1.set_ylim([-1, 1])
ax1.set_xlabel(r'$\langle\sigma_y\rangle$')
ax1.set_ylabel(r'$\langle\sigma_z\rangle$')
ax1.set_title('Emission interaction')

plt.legend()

ax2 = fig.add_subplot(322)
ax2.quiver(P[1], P[2], R_pp[1], R_pp[2], color='r')#, label=r'$| x_+ \rangle$')
ax2.quiver(P[1], P[2], R_pm[1], R_pm[2], color='b')#, label=r'$| x_- \rangle$')
ax2.set_xlim([-1, 1])
ax2.set_ylim([-1, 1])
ax2.set_xlabel(r'$\langle\sigma_y\rangle$')
ax2.set_ylabel(r'$\langle\sigma_z\rangle$')
ax2.set_title('Absorption interaction')

ax3 = fig.add_subplot(323)
ax3.quiver(P[0], P[1], R_mp[0], R_mp[1], color='r')#, label=r'$| x_+ \rangle$')
ax3.quiver(P[0], P[1], R_mm[0], R_mm[1], color='b')#, label=r'$| x_- \rangle$')
ax3.set_xlim([-1, 1])
ax3.set_ylim([-1, 1])
ax3.set_xlabel(r'$\langle\sigma_x\rangle$')
ax3.set_ylabel(r'$\langle\sigma_y\rangle$')
#ax3.set_title('Emission interaction')

ax4 = fig.add_subplot(324)
ax4.quiver(P[0], P[1], R_pp[0], R_pp[1], color='r')#, label=r'$| x_+ \rangle$')
ax4.quiver(P[0], P[1], R_pm[0], R_pm[1], color='b')#, label=r'$| x_- \rangle$')
ax4.set_xlim([-1, 1])
ax4.set_ylim([-1, 1])
ax4.set_xlabel(r'$\langle\sigma_x\rangle$')
ax4.set_ylabel(r'$\langle\sigma_y\rangle$')
#ax4.set_title('Absorption interaction')

ax5 = fig.add_subplot(325)
ax5.quiver(P[2], P[0], R_mp[0], R_mp[1], color='r')#, label=r'$| x_+ \rangle$')
ax5.quiver(P[2], P[0], R_mm[0], R_mm[1], color='b')#, label=r'$| x_- \rangle$')
ax5.set_xlim([-1, 1])
ax5.set_ylim([-1, 1])
ax5.set_xlabel(r'$\langle\sigma_z\rangle$')
ax5.set_ylabel(r'$\langle\sigma_x\rangle$')
#ax3.set_title('Emission interaction')

ax6 = fig.add_subplot(326)
ax6.quiver(P[2], P[0], R_pp[0], R_pp[1], color='r')#, label=r'$| x_+ \rangle$')
ax6.quiver(P[2], P[0], R_pm[0], R_pm[1], color='b')#, label=r'$| x_- \rangle$')
ax6.set_xlim([-1, 1])
ax6.set_ylim([-1, 1])
ax6.set_xlabel(r'$\langle\sigma_z\rangle$')
ax6.set_ylabel(r'$\langle\sigma_x\rangle$')
#ax4.set_title('Absorption interaction')





bigtitle = (r'Changes to the system state using $H_{int} = \frac{1}{2}( \sigma_x \otimes \sigma_y \pm \sigma_y \otimes \sigma_z )$ with $|x_+\rangle$ environment and measuring in the $z$-basis')


fig.suptitle(bigtitle)
fig.set_size_inches(16,21)

### REMEMBER TO CHANGE FILENAME TO NOT OVERWRITE PREVIOUS PLOTS
"""
figname = "../fig/vectorplots/vectorplot_blochsphere2d_env_xp_meas_zbas" + ".png"
fig.savefig(figname, dpi=400)
"""

plt.show()

#print(dnx, dny, dnz)

"""
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.quiver(X.real, Y.real, Z.real, dnx.real, dny.real, dnz.real, length=0.001) 
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
plt.show()
"""


"""
fig = plt.figure()
ax = fig.add_subplot(121,projection='3d')

# Create the mesh in polar coordinates and compute corresponding Z.

X = (dag(states) @ sigmax @ states)[:,0,0].real
Y = (dag(states) @ sigmay @ states)[:,0,0].real
Z = (dag(states) @ sigmaz @ states)[:,0,0].real

ax.set_title('Vectors on Bloch sphere',fontsize=12)
ax.view_init(-30,60)
sphere = qp.Bloch(axes=ax)
psize = 10.0
sphere.point_size = [psize,psize,psize,psize]

R = [X,Y,Z]

sphere.add_points(R)

sphere.make_sphere()


ax2 = fig.add_subplot(122)
ax2.plot(X, Y)
ax2.set_xlabel(r'$\langle \sigma_x \rangle$')
ax2.set_ylabel(r'$\langle \sigma_y \rangle$')

plt.show()
"""
