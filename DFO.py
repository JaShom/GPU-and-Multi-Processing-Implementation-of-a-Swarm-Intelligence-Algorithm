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
from math import floor, log10

count = 0


def scientific(x, n):
    """Represent a float in scientific notation.
	This function is merely a wrapper around the 'e' type flag in the
	formatting specification.
	"""
    n = int(n)
    x = float(x)

    if n < 1: raise ValueError("1+ significant digits required.")

    return ''.join(('{:.', str(n - 1), 'e}')).format(x)


# with tf.device('/GPU:0'):

def Sphere(x):  # x IS A VECTOR REPRESENTING ONE FLY, SPHEREFUNCTION
    sums = 0.0
    for i in range(len(x)):
        sums = sums + np.power(x[i], 2)
    return sums


def rastrigin(x):  # RastriginFunction ∈ [-5.12, 5.12]
    sums = 0.0
    for i in range(len(x)):
        sums += x[i] ** 2 - (10 * np.cos(2 * np.pi * x[i])) + 10
    return sums


def schwefel_1_2(x):  # ∈ [−100,100]
    # x = np.array(x)
    sums = 0.0
    for i in range(len(x)):
        sums = sums + (sums + x[i] ** 2)
    return sums


def rosenbrock(x):  # ∈ [-5, 10] but may be restricted to [-2.048, 2.048]
    sums = 0.0
    for i in range(len(x) - 1):
        xn = x[i + 1]
        new = 100 * np.power(xn - np.power(x[i], 2), 2) + np.power(x[i] - 1, 2)
        sums = sums + new
    return sums


# Check if value are correct later. Mean was a bit large at least 45 min wait time tho.


def ackley(x):  # ∈ [-32.768, 32.768]
    sum = 0.0
    sum2 = 0.0
    for c in x:
        sum += c ** 2.0
        sum2 += np.cos(2.0 * np.pi * c)
    n = float(len(x))
    return -20.0 * np.exp(-0.2 * np.sqrt(sum / n)) - np.exp(sum2 / n) + 20 + np.exp(1)


def griewank(x):  # ∈ [-600, 600]
    p1 = 0
    for i in range(len(x)):
        p1 += x[i] ** 2
        p2 = 1
    for i in range(len(x)):
        p2 *= np.cos(float(x[i] / np.sqrt(i + 1)))
    return 1 + (float(p1) / 4000.0) - float(p2)


def goldstein(x):  # ∈ [-2, 2]
    if len(x) <= 2:
        return goldsteinAid(0, x)
    else:
        total = 0
        for i in range(len(x)):
            total += x[i] * \
                     goldsteinAid(i, x)
    return total


def goldsteinAid(i, x):
    x1 = x[i]
    x2 = x[(i + 1) % len(x)]
    eq1a = np.power(x1 + x2 + 1, 2)
    eq1b = 19 - 14 * x1 + 3 * np.power(x1, 2) - 14 * x2 + 6 * x1 * x2 + 3 * np.power(x2, 2)
    eq1 = 1 + eq1a * eq1b
    eq2a = np.power(2 * x1 - 3 * x2, 2)
    eq2b = 18 - 32 * x1 + 12 * np.power(x1, 2) + 48 * x2 - 36 * x1 * x2 + 27 * np.power(x2, 2)
    eq2 = 30 + eq2a * eq2b
    return eq1 * eq2


def camel6(x):  # x1 ∈ [-3, 3], x2 ∈ [-2, 2]
    if len(x) <= 2:
        return camel6Aid(0, x)
    else:
        total = 0
        for i in range(len(x)):
            total += x[i] * \
                     camel6Aid(i, x)
        return total


def camel6Aid(i, x):
    x1 = x[i]
    x2 = x[(i + 1) % len(x)]
    part1 = (4 - 2.1 * np.power(x1, 2) + (np.power(x1, 4) / 3)) * np.power(x1, 2)
    part2 = x1 * x2
    part3 = (- 4 + 4 * np.power(x2, 2))
    return part1 + part2 + part3


def lunaceksBiRastrigin(x):  # prep for death # ∈ [-5.12,5.12]
    sum = 0.0
    sum1 = 0.0
    sum2 = 0.0
    s = 1 - (1 / (2 * np.sqrt(len(x) + 20) - 8.2))
    d = 1
    mu = 2.5
    mu1 = - np.sqrt(abs((mu ** 2 - d) / s))
    for i in range(len(x)):
        sum += (x[i] - mu) ** 2
        sum1 += (x[i] - mu1) ** 2
        sum2 += 1 - np.cos(2 * np.pi * (x[i] - mu))
    return min(sum, d * len(x) + s * sum1) + 10 * sum2


def schafferN06(x):  # prep for death
    if len(x) <= 2:
        return schafferAid(0, x)
    else:
        total = 0
        for i in range(len(x)):
            total += x[i] * \
                     schafferAid(i, x)
        return total


def schafferAid(i, x):
    x1 = x[i]
    y1 = x[(i + 1) % len(x)]
    xysqrd = x1 ** 2 + y1 ** 2
    return 0.5 + (np.sin(np.sqrt(xysqrd)) - 0.5) / (1 + 0.001 * xysqrd) ** 2


def shiftedRastrigin(x):
    sum = 0.0
    for i in range(len(x)):
        sum = sum + (np.power(x[i], 2) - 10 * np.cos(2 * np.pi * x[i]) + 10)
    return sum


lis = []
t1_start = perf_counter()
N = 100  # POPULATION SIZE
D = 30  # DIMENSIONALITY
delta = 0.001  # DISTURBANCE THRESHOLD
maxIterations = 3100  # ITERATIONS ALLOWED
lowerB = [-100] * D  # LOWER BOUND (IN ALL DIMENSIONS)
upperB = [100] * D  # UPPER BOUND (IN ALL DIMENSIONS)

for i in range(1):
    count += 1
    print(f'This is trial {count}')
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
    lis.append(fitness[s])  # Apply 30 best flies into list

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

print("\n Time elapsed: ", (t1_stop - t1_start))
print("This is the best fly after 30 trials:", lis, "\nMin = ", min(lis), "\nMax = ", max(lis), "\nMedian = ",
      np.median(lis),
      "\nMean = ", np.mean(lis), "\nStandard deviation = ", np.std(lis), "\nCounter says ", count)
fimean = np.mean(lis)
standardDev = np.std(lis)
print("Mean: ", scientific(fimean, 3), "Standard deviation: ", scientific(standardDev, 3))
