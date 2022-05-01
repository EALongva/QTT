# testing multiprocessing

from lib2to3.pgen2 import grammar
import os as os
import numpy as np
import multiprocessing as mp
import time as time


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

"""




def power(x, n):

    time.sleep(1)

    return x ** n


def main():

    start = time.perf_counter()

    print(f'starting computations on {os.cpu_count()} cores')

    values = ((2, 2), (4, 3), (5, 5), (6, 6))

    with mp.Pool(3) as pool:
        res = pool.starmap(power, values)
        print(res)

    end = time.perf_counter()
    print(f'elapsed time: {end - start}')


if __name__ == '__main__':
    main()    



"""
def power(S, N):

    time.sleep(0.1)

    print('S: ', S, 'N: ', N)


def main():

    start = time.perf_counter()

    print(f'starting computations on {os.cpu_count()} cores')

    S = 100
    N = 1000

    values = [(S, N)]

    with mp.Pool() as pool:
        res = pool.starmap(power, values)
        print(res)

    end = time.perf_counter()
    print(f'elapsed time: {end - start}')


if __name__ == '__main__':
    main()    

"""

#print(os.getppid())