# testing the class structure because its been ages :)
# documentation: https://docs.python.org/3/tutorial/classes.html

import numpy as np

r = 2.0

class class_test:

    def __init__(self, x, y):
        
        self.x = x
        self.y = y

    def method_defaultcheck(self, N, newx=0):

        if newx != 0:
            self.x = newx

        print(x)

    def method1(self, N):

        testarray = np.linspace(self.x, self.y, N)

        return testarray

    def method2(self, N):

        print(self.method1(N))

    def method3(self, N):

        print(r * self.method1(N))

class test2:

    def __init__(self):

        self.null = 0

    def method1(self, name):

        if name == 'x':

            self.name = 0

        elif name == 'y':

            self.name = 1

        elif name == 'z':

            self.name = 2

        else:

            print('no valid environment chosen, must be a string: x, y or z, ', name, ' not a valid argument')

            self.name = 'invalid arg'


"""
x = 2.0; y = 5.0
instance = class_test(x, y)

N = 10
Narray = instance.method1(N)

instance.method_defaultcheck(N, 3.0)



instance.method3(N)

# basically all I need to clean up in my code

lol = test2()
lol.method1(2)
print(lol.name)
"""

bas0 = np.array(([1.0], [0.0]), dtype='complex128')
bas1 = np.array(([0.0], [1.0]), dtype='complex128')

xplus = np.sqrt(0.5) * (bas0 + bas1)

Psi = np.array(np.kron(xplus, bas0), dtype='complex256')
Psi_ = np.kron(xplus, bas0)
print(Psi, Psi_)