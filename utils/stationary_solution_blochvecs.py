### plotting stationary solution bloch vectors of TLS with a classical drive hamiltonian
### varying the damping and gain rates (gamma factors), strength of signal (epsilon) 
### and frequency (omega)

import numpy as np
import matplotlib.pyplot as plt

def mx(gamma_g, gamma_d, eps, delta):

    nom     = 4 * eps * ( gamma_g - gamma_d )

    denom   = ( gamma_d + gamma_g )**2 + 8 * ( eps**2 + 2 * delta**2 )

    return nom/denom


def my(gamma_g, gamma_d, eps, delta):

    nom     = 16 * eps * delta * ( gamma_g - gamma_d )

    denom   = ( gamma_d + gamma_g ) * ( ( gamma_d + gamma_g )**2 + 8 * ( eps**2 + 2 * delta**2 ) )

    return nom/denom


def mz(gamma_g, gamma_d, eps, delta):

    nom     = ( gamma_d - gamma_g ) * ( ( gamma_d + gamma_g )**2 + 16 * delta**2 )

    denom   = ( gamma_d + gamma_g ) * ( ( gamma_d + gamma_g )**2 + 8 * ( eps**2 + 2 * delta**2 ) )

    return nom/denom


def plot(temperature, constant, width):

    ### calculating gamma_g and gamma_d 

    pdensity         = 1 / ( np.exp( 1/temperature ) - 1 )
    gamma_g          = pdensity
    gamma_d          = pdensity + 1

    values = np.linspace( constant - width/2 , constant + width/2, 1000 )

    epsconstant = [ mx(gamma_g, gamma_d, constant, values), \
        my(gamma_g, gamma_d, constant, values), mz(gamma_g, gamma_d, constant, values) ]

    delconstant = [ mx(gamma_g, gamma_d, values, constant), \
        my(gamma_g, gamma_d, values, constant), mz(gamma_g, gamma_d, values, constant) ]

    vec = ['x', 'y', 'z']

    plt.style.use('ggplot')
    fig = plt.figure()

    ax1 = fig.add_subplot(121)

    for m in range(3):

        ax1.plot(values, epsconstant[m], label=vec[m])

    ax1.legend()
    ax1.set_title(r'$\epsilon$ constant')
    ax1.set_xlabel(r'$\Delta$')

    ax2 = fig.add_subplot(122)

    for m in range(3):

        ax2.plot(values, delconstant[m], label=vec[m])

    ax2.legend()
    ax2.set_title(r'$\Delta$ constant')
    ax2.set_xlabel(r'$\epsilon$')


    fig.suptitle('Stationary solution of a TLS with an external signal in the '\
        +'rotating-wave approximation.\n \
        With temperature ' + str(temperature) + ' with corresponding rates ' + \
            r'$\Gamma_g = $ ' + "{0:.5f}".format(gamma_g) + r' $\Gamma_d = $ ' + \
                "{0:.5f}".format(gamma_d)  \
                + ', and constant value ' + '{0:.2f}'.format(constant)     )
    fig.supylabel('bloch vector components')

    fig.set_size_inches(16,9)

    path = '../figure/stationary/'
    figname = path + 'width' + str(width).replace('.', '')+ '_const' + str(constant).replace('.', '')  \
         + '_T' + str(temperature).replace('.', '') \
         + '.png'
    fig.savefig(figname, dpi=400)

    plt.show()

    return 0


### runs

plot(2.0, 1.0, 10.0)
plot(1.0, 1.0, 10.0)
plot(0.8, 1.0, 10.0)
plot(0.5, 1.0, 10.0)
plot(0.3, 1.0, 10.0)
plot(0.1, 1.0, 10.0)




plot(0.5, 0.2, 60.0)
plot(0.5, 0.5, 60.0)
plot(0.5, 0.8, 60.0)
plot(0.5, 1.0, 60.0)
plot(0.5, 1.2, 60.0)
plot(0.5, 1.6, 60.0)

