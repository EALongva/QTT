# testing multiprocessing

from lib2to3.pgen2 import grammar
import os
import numpy as np
import multiprocessing as mp


"""

def info(title):
    print(title)
    print('module name:', __name__)
    print('parent process:', os.getppid())
    print('process id:', os.getpid())

def f(name):
    info('function f')
    print('hello', name)

def g(var):
    gArray = np.zeros(var)
    for i in range(var):
        gArray[i] = i
    print('hello')

if __name__ == '__main__':
    info('main line')
    var = 10
    p = mp.Process(target=g, args=(var,))
    p.start()
    p.join()

"""





def foo(q):
    q.put('hello')

if __name__ == '__main__':
    ctx = mp.get_context('spawn')
    q = ctx.Queue()
    p = ctx.Process(target=foo, args=(q,))
    p.start()
    print(q.get())
    p.join()


#print(os.getppid())