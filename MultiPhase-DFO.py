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

# If you do decide to run this to check the time ran, I'd advise you watch a 10-20 min video of something to pass time

import numpy as np
import matplotlib as plt
import pandas as pd
from time import perf_counter
from math import floor, log10
import numba
from numba import jit
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda import gpuarray

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


mod = SourceModule(
    """
        float f(float x[], int D) { // x IS ONE FLY AND D IS DIMENSION
            float sum=0;
            for(int i=0; i<D; i++)
                sum=sum+x[i]*x[i];
            return sum;
        }
    """)
updateFly = SourceModule(
    """
    // UPDATE EACH FLY INDIVIDUALLY
    for(int i=0; i<N; i++) {
        // ELITIST STRATEGY (i.e. DON'T UPDATE BEST FLY)
        if (i==s) continue;

            // FIND BEST NEIGHBOUR FOR EACH FLY
            int left; int right; int bNeighbour;
            left=(i-1)%N; right=(i+1)%N; // INDICES: LEFT & RIGHT FLIES
            if (device_flies[right]<device_flies[left]) bNeighbour = right;
            else bNeighbour = left;

            // UPDATE EACH DIMENSION SEPARATELY
            for (int d=0; d<D; d++) {
                if (r() < delta) { // DISTURBANCE MECHANISM
                    X[i][d] = lowerB[d] + r()*(upperB[d]-lowerB[d]); continue;
                }

                // UPDATE EQUATION
                X[i][d] = X[bNeighbour][d] + r()*( X[s][d]- X[i][d] );

                // OUT OF BOUND CONTROL
                if (X[i][d] < lowerB[d] or X[i][d] > upperB[d])
                X[i][d] = lowerB[d] + r()*(upperB[d]-lowerB[d]);
            }
        }
    """
)
sphereFunc = mod.get_function("f")



def sphere(x):  # x IS A VECTOR REPRESENTING ONE FLY. SPHERE FUNCTION with bounds ∈ [-100, 100]
    sums = 0.0
    for i in range(len(x)):
        sums = sums + np.power(x[i], 2)
    return sums


def rastrigin(x):  # Rastrigin Function, ∈ [-5.12, 5.12]
    sums = 0.0
    for i in range(len(x)):
        sums += x[i] ** 2 - (10 * np.cos(2 * np.pi * x[i])) + 10
    return sums


def schwefel_1_2(x):  # Schwefel 1.2 Function, ∈ [−100,100]
    # x = np.array(x)
    sums = 0.0
    for i in range(len(x)):
        sums = sums + (sums + x[i] ** 2)
    return sums


def rosenbrock(x):  # Rosenbrock Function, ∈ [-5, 10] but may be restricted to [-2.048, 2.048]
    sums = 0.0
    for i in range(len(x) - 1):
        xn = x[i + 1]
        new = 100 * np.power(xn - np.power(x[i], 2), 2) + np.power(x[i] - 1, 2)
        sums = sums + new
    return sums


def ackley(x):  # Ackley Function, ∈ [-32.768, 32.768]
    sum = 0.0
    sum2 = 0.0
    for c in x:
        sum += c ** 2.0
        sum2 += np.cos(2.0 * np.pi * c)
    n = float(len(x))
    return -20.0 * np.exp(-0.2 * np.sqrt(sum / n)) - np.exp(sum2 / n) + 20 + np.exp(1)


def griewank(x):  # Griewank Function, ∈ [-600, 600]
    p1 = 0
    for i in range(len(x)):
        p1 += x[i] ** 2
        p2 = 1
    for i in range(len(x)):
        p2 *= np.cos(float(x[i] / np.sqrt(i + 1)))
    return 1 + (float(p1) / 4000.0) - float(p2)


def goldstein(x):  # Goldstein Function, ∈ [-2, 2]
    if len(x) <= 2:
        return goldsteinAid(0, x)
    else:
        total = 0
        for i in range(len(x)):
            total += x[i] * \
                     goldsteinAid(i, x)
    return total


def goldsteinAid(i, x):  # Main component of the Goldstein Function
    x1 = x[i]
    x2 = x[(i + 1) % len(x)]  # Two flies are taken to be used
    eq1a = np.power(x1 + x2 + 1, 2)
    eq1b = 19 - 14 * x1 + 3 * np.power(x1, 2) - 14 * x2 + 6 * x1 * x2 + 3 * np.power(x2, 2)
    eq1 = 1 + eq1a * eq1b
    eq2a = np.power(2 * x1 - 3 * x2, 2)
    eq2b = 18 - 32 * x1 + 12 * np.power(x1, 2) + 48 * x2 - 36 * x1 * x2 + 27 * np.power(x2, 2)
    eq2 = 30 + eq2a * eq2b
    return eq1 * eq2


def camel6(x):  # Six-Hump Camel-Back Function, x1 ∈ [-3, 3] and x2 ∈ [-2, 2] so decided to use ∈ [-5, 5]
    if len(x) <= 2:
        return camel6Aid(0, x)
    else:
        total = 0
        for i in range(len(x)):
            total += x[i] * \
                     camel6Aid(i, x)
        return total


def camel6Aid(i, x):  # Main component of the Camel-Back Function
    x1 = x[i]
    x2 = x[(i + 1) % len(x)]  # Two flies are also needed
    part1 = (4 - 2.1 * np.power(x1, 2) + (np.power(x1, 4) / 3)) * np.power(x1, 2)
    part2 = x1 * x2
    part3 = (- 4 + 4 * np.power(x2, 2))
    return part1 + part2 + part3


def lunaceksBiRastrigin(x):  # Lunaceks Bi-Rastrigin Function ∈ [-5.12,5.12]
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


def schafferN06(x):  # Schaffer N0 6 Function ∈ [-100,100]
    if len(x) <= 2:
        return schafferAid(0, x)
    else:
        total = 0
        for i in range(len(x)):
            total += x[i] * \
                     schafferAid(i, x)
        return total


def schafferAid(i, x):  # Main component of Schaffer N06 Function
    x1 = x[i]
    y1 = x[(i + 1) % len(x)]  # Uses two flies
    xysqrd = x1 ** 2 + y1 ** 2
    return 0.5 + (np.sin(np.sqrt(xysqrd)) - 0.5) / (1 + 0.001 * xysqrd) ** 2


def shiftedRastrigin(x):  # Shifted Rastrigin Function, ∈ [-5, 5]
    summ = 0.0
    sum1 = 0.0
    fopt = 100
    for i in range(len(x)):
        summ = summ + (np.power(x[i], 2) - 10 * np.cos(2 * np.pi * x[i]) + 10)  # The original Rastrgin function is -
        # used to get the optimal fly, so it can be used for shifting the function

        xopt = 0.0512 * (summ - x[i])  # The optimal fly is stored and ready to be used again
    return sum1 + (np.power(xopt, 2) - 10 * np.cos(2 * np.pi * xopt) + 10 + fopt)  # Shifted Rastrign function is -
    # executed on return here


def shiftedRosenbrock(x):  # Shifted Rosenbrock Function, ∈ [-2, 2]
    sums = 0.0
    sums1 = 0.0
    fopt = 100
    for i in range(len(x) - 1):
        xn = x[i + 1]
        new = 100 * np.power(np.power(x[i], 2) - xn, 2) + np.power(x[i] - 1, 2)  # The original Rosenbrock function is
        # used to get the optimal fly, so it can later be used for shifting

        sums = sums + new
        xopt = 0.02048 * (sums - x[i]) + 1
    return sums1 + 100 * np.power(np.power(xopt, 2) - xopt, 2) + np.power(xopt - 1, 2)  # Shifted Rosenbrock function is
    # executed on return here


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
    X = np.empty([N, D], dtype=np.float32)  # EMPTY FLIES ARRAY OF SIZE: (N,D)
    device_flies = [None] * N  # EMPTY device_flies ARRAY OF SIZE N
    device_flies = gpuarray.to_gpu(device_flies)
    # INITIALISE FLIES WITHIN BOUNDS
    for i in range(N):
        for d in range(D):
            X[i, d] = np.random.uniform(lowerB[d], upperB[d])

    # MAIN DFO LOOP
    for itr in range(maxIterations):
        for i in range(N):  # EVALUATION
         #   gpuSphere = Sphere.to_gpu()
            device_flies = Sphere(X[i,])
        s = np.argmin(device_flies)  # FIND BEST FLY

        if itr % 100 == 0:  # PRINT BEST FLY EVERY 100 ITERATIONS
            print("Iteration:", itr, "\tBest fly index:", s,
                  "\tdevice_flies value:", device_flies[s])

        # TAKE EACH FLY INDIVIDUALLY
        for i in range(N):
            if i == s: continue  # ELITIST STRATEGY

            # FIND BEST NEIGHBOUR
            left = (i - 1) % N
            right = (i + 1) % N
            bNeighbour = right if device_flies[right] < device_flies[left] else left

            for d in range(D):  # UPDATE EACH DIMENSION SEPARATELY
                if np.random.rand() < delta:
                    X[i, d] = np.random.uniform(lowerB[d], upperB[d])
                    continue

                u = np.random.rand()
                X[i, d] = X[bNeighbour, d] + u * (X[s, d] - X[i, d])

                # OUT OF BOUND CONTROL
                if X[i, d] < lowerB[d] or X[i, d] > upperB[d]:
                    X[i, d] = np.random.uniform(lowerB[d], upperB[d])

    for i in range(N): device_flies[i] = Sphere(X[i,])  # EVALUATION
    s = np.argmin(device_flies)  # FIND BEST FLY
    hostFlies = device_flies.get()
    lis.append(hostFlies[s])  # Apply 30 best flies into list

t1_stop = perf_counter()
print("\nFinal best device_flies:\t", hostFlies[s])
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
