from __future__ import print_function

import numpy as np

import time

x = np.random.randn(1000, 50)

t = np.random.randn(1, 50)

y = np.random.randn(100, 50)

start = time.clock()
for i in range(1000):
	np.sqrt(np.sum((x[i] - t)**2))
print(time.clock() - start)


start = time.clock()
for i in range(100):
	np.sqrt(np.sum((y[i] - t)**2))
print(time.clock() - start)