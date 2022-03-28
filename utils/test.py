# general testing

import numpy as np

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
