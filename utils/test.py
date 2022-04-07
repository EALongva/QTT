# general testing

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

"""
for key in {'one':'balls', 'two':2, 'three':3}:
    print(key)
"""

# dictionary test
dic = {12:-4, 8:-2, 4:0}

for key, ind in enumerate(dic):
    print(key, ind)


print( dic[12] )

string_array = np.array(['one', 'two', 'three'])