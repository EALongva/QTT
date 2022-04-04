# testing multiprocessing

import os
import numpy as np
import multiprocessing as mp

def info(title):
    print(title)
    print('module name:', __name__)
    print('parent process:', os.getppid())
    print('process id:', os.getpid())

def f(name):
    info('function f')
    print('hello', name)


if __name__ == '__main__':
    info('main line')
    p = mp.Process(target=f, args=('bob',))
    p.start()
    p.join()



#print(os.getppid())