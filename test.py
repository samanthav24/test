# import pyscal.core as pc
import numpy as np
import time
from init import *
from util import *




start = time.time()
a = 0
for i in range(1000):
    for j in range(i):
        calc_distance(i, j)
print(time.time() - start)
start = time.time()
for i in range(1000):
    for j in range(i+1, 1000):
        
        L = 28.9
        posA, posB = i, j
        # calculate the distance vector
        r1 = abs(posA - posB)

        # periodic boundary condition
        r2 = abs(r1 - L)
        if r1 < r2:
            r = linalg.norm(r1)
        else:
            r = linalg.norm(r2)

print(time.time() - start)