"""
Dispersive Flies Optimisation

Copyright (C) 2014 Mohammad Majid al-Rifaie

This is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License.

For any query contact:
m.alrifaie@gre.ac.uk

School of Computing & Mathematical Sciences
University of Greenwich, Old Royal Naval College,
Park Row, London SE10 9LS, U.K.

Reference to origianl paper:
Mohammad Majid al-Rifaie (2014), Dispersive Flies Optimisation, Proceedings of the 2014 Federated Conference on Computer Science and Information Systems, 535--544. IEEE.

    @inproceedings{FedCSIS_2014,
		author={Mohammad Majid al-Rifaie},
		pages={535--544},
		title={Dispersive Flies Optimisation},
		booktitle={Proceedings of the 2014 Federated Conference on Computer Science and Information Systems},
		year={2014},
		editor={M. Ganzha, L. Maciaszek, M. Paprzycki},
		publisher={IEEE}
	}
"""

import numpy as np
# import matplotlib as plt
import pandas as pd
from time import perf_counter
import multiprocessing
import concurrent.futures

# FITNESS FUNCTION (SPHERE FUNCTION)
def f(x):  # x IS A VECTOR REPRESENTING ONE FLY
    sum = 0.0
    for i in range(len(x)):
        sum = sum + np.power(x[i], 2)
    return sum


N = 100  # POPULATION SIZE
D = 30  # DIMENSIONALITY
delta = 0.001  # DISTURBANCE THRESHOLD
maxIterations = 3100  # ITERATIONS ALLOWED
lowerB = [-100] * D  # LOWER BOUND (IN ALL DIMENSIONS)
upperB = [100] * D  # UPPER BOUND (IN ALL DIMENSIONS)

t1_start = perf_counter()
# INITIALISATION PHASE
X = np.empty([N, D])  # EMPTY FLIES ARRAY OF SIZE: (N,D)
fitness = [None] * N  # EMPTY FITNESS ARRAY OF SIZE N

# INITIALISE FLIES WITHIN BOUNDS
for i in range(N):
    for d in range(D):
        X[i, d] = np.random.uniform(lowerB[d], upperB[d])

# MAIN DFO LOOP
for itr in range(maxIterations):
    for i in range(N):  # EVALUATION
        fitness[i] = f(X[i,])
    s = np.argmin(fitness)  # FIND BEST FLY

    if (itr % 100 == 0):  # PRINT BEST FLY EVERY 100 ITERATIONS
        print("Iteration:", itr, "\tBest fly index:", s,
              "\tFitness value:", fitness[s])

    # TAKE EACH FLY INDIVIDUALLY
    for i in range(N):
        if i == s: continue  # ELITIST STRATEGY

        # FIND BEST NEIGHBOUR
        left = (i - 1) % N
        right = (i + 1) % N
        bNeighbour = right if fitness[right] < fitness[left] else left

        for d in range(D):  # UPDATE EACH DIMENSION SEPARATELY
            if (np.random.rand() < delta):
                X[i, d] = np.random.uniform(lowerB[d], upperB[d])
                continue;

            u = np.random.rand()
            X[i, d] = X[bNeighbour, d] + u * (X[s, d] - X[i, d])

            # OUT OF BOUND CONTROL
            if X[i, d] < lowerB[d] or X[i, d] > upperB[d]:
                X[i, d] = np.random.uniform(lowerB[d], upperB[d])

for i in range(N): fitness[i] = f(X[i,])  # EVALUATION
s = np.argmin(fitness)  # FIND BEST FLY
t1_stop = perf_counter()
print("\nFinal best fitness:\t", fitness[s])
print("\nBest fly position:\n", X[s,])
print("Time elapsed:", t1_stop - t1_start)
minimum = X[s,].min()
maximum = X[s,].max()
Fmean = X[s,].mean()
standardDev = X[s,].std()
print("The min is: ", minimum, "\nThe Max is: ", maximum, "\nThe Mean is: ", Fmean, "\nThe Standard Deviation is: ",
      standardDev)
