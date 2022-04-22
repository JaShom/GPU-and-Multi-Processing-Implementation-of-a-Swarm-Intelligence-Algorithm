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
import matplotlib as plt
import pandas as pd
from time import perf_counter
import numba
from numba import jit, cuda
import multiprocessing as mp
from multiprocessing import pool
import concurrent.futures
import os
import sys


def scientific(x, n):
    # """Represent a float in scientific notation.
    # This function is merely a wrapper around the 'e' type flag in the
    # formatting specification.
    # """
    n = int(n)
    x = float(x)

    if n < 1: raise ValueError("1+ significant digits required.")

    return ''.join(('{:.', str(n - 1), 'e}')).format(x)


# FITNESS FUNCTION (SPHERE FUNCTION)
def Sphere(x):  # x IS A VECTOR REPRESENTING ONE FLY
    sum = 0.0
    for i in range(len(x)):
        sum = sum + np.power(x[i], 2)
    return sum


t0 = perf_counter()
# file = open("MultiFitness.txt", "r+")

N = 100  # POPULATION SIZE
D = 30  # DIMENSIONALITY
delta = 0.001  # DISTURBANCE THRESHOLD
maxIterations = 3100  # ITERATIONS ALLOWED
lowerB = [-100] * D  # LOWER BOUND (IN ALL DIMENSIONS)
upperB = [100] * D  # UPPER BOUND (IN ALL DIMENSIONS)


def multi():
    lis = []
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
            fitness[i] = Sphere(X[i,])
        s = np.argmin(fitness)  # FIND BEST FLY

        if itr % 100 == 0:  # PRINT BEST FLY EVERY 100 ITERATIONS
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
                if np.random.rand() < delta:
                    X[i, d] = np.random.uniform(lowerB[d], upperB[d])
                    continue

                u = np.random.rand()
                X[i, d] = X[bNeighbour, d] + u * (X[s, d] - X[i, d])

                # OUT OF BOUND CONTROL
                if X[i, d] < lowerB[d] or X[i, d] > upperB[d]:
                    X[i, d] = np.random.uniform(lowerB[d], upperB[d])

    for i in range(N): fitness[i] = Sphere(X[i,])  # EVALUATION
    s = np.argmin(fitness)  # FIND BEST FLY
    lis.append(fitness[s])
    # finalResults = mp.Queue.get(lis)
    # for line in file:
    #     file.write(str(lis))
    #     file.close()
    t1_stop = perf_counter()

    # file1 = fitness[s]
    with open("MultiFitness.csv", "a+") as f:
        f.write(lis, "\n")
        f.close()

    print("\nFinal best fitness:\t", fitness[s])
    print("\nBest fly position:\n", X[s,])
    print("Time elapsed:", t1_stop - t1_start)
    minimum = X[s,].min()
    maximum = X[s,].max()
    Fmean = X[s,].mean()
    standardDev = X[s,].std()
    print("The min is: ", minimum, "\nThe Max is: ", maximum, "\nThe Mean is: ", Fmean, "\nThe Standard Deviation is: ",
          standardDev)
    return lis


def getVal(lis):
    w = lis
    return w


if __name__ == "__main__":
    processes = []
    for _ in range(30):
        p1 = mp.Process(target=multi,)
        p1.start()
        processes.append(p1)
    for process in processes:
        process.join()
    t1 = perf_counter()
    print("Time elapsed:", t1 - t0)
    # print(lis)
    # print("This is the best fly after 30 trials:", lis, f"\nMin = ", min(lis), "\nMax = ", max(lis), "\nMedian = ",
    #       np.median(lis),
    #       "\nMean = ", np.mean(lis), "\nStandard deviation = ", np.std(lis), "\nCounter says ")  # , count)
    # fmean = np.mean(lis)
    # standardDev = np.std(lis)
    # print("Mean: ", scientific(fmean, 3), "Standard deviation: ", scientific(standardDev, 3))
