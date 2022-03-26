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
from numba import jit, cuda
from time import perf_counter

import warnings

# suppress warnings
# warnings.filterwarnings('ignore')

# from numpy import abs, cos, exp, mean, pi, prod, sin, sqrt, sum
# FITNESS FUNCTION (SPHERE FUNCTION)
# @jit(nopython=True)
counter1 = 0


@jit(target_backend="cuda")
def f(x):  # x IS A VECTOR REPRESENTING ONE FLY, SPHEREFUNCTION
    sums = 0.0
    for i in range(len(x)):
        sums = sums + np.power(x[i], 2)
    return sums


def rastrigin(x):  # RastriginFunction
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


#    return np.sum([np.sum(x[:i]) ** 2
#                   for i in range(len(x))])


def rosenbrock(x):  # ∈ [-5, 10] but may be restricted to [-2.048, 2.048]
    sums = 0.0
    for i in range(len(x) - 1):
        xn = x[i + 1]
        new = 100 * np.power(xn - np.power(x[i], 2), 2) + np.power(x[i] - 1, 2)
        sums = sums + new
    return sums


def ackley(x, a=20, b=0.2, c=2 * np.pi):  # ∈ [-32.768, 32.768], may be restricted to a smaller domain
    sum1 = 0.0
    sum2 = 0.0
    for i in range(len(x)):
        sum1 = sum1 + np.power(x[i], 2)
        sum2 = sum2 + np.cos(c * x[i])
    part1 = -a * np.exp(-b * (np.sqrt(1 / sum1)))  # / x[i])))
    part2 = -np.exp(1 / sum2)  # / x[i])
    return part1 + part2 + a + np.exp(1)


def ackleyRedo(x):
    for i in range(x):
        part1 = - 20 * np.exp(- 0.2 * np.sqrt(np.power(x[i], 2) / x[i]))
        part2 = - np.exp(np.cos(2 * np.pi * x[i]))
    return part1 + part2 + 20 + np.exp(1)


def ackley_fun(x):
    """Ackley function
    Domain: -32 < xi < 32
    Global minimum: f_min(0,..,0)=0
    """
    return -20 * np.exp(-.2 * np.sqrt(.5 * (x[0] ** 2 + x[1] ** 2))) - np.exp(
        .5 * (np.cos(np.pi * 2 * x[0]) + np.cos(np.pi * 2 * x[1]))) + np.exp(1) + 20

def finalAckley(x):
    sum = 0.0
    sum2 = 0.0
    for c in x:
        sum += c**2.0
        sum2 += np.cos(2.0 * np.pi * c)
    n = float(len(x))
    return -20.0 * np.exp(-0.2 * np.sqrt(sum/n)) - np.exp(sum2/n) + 20 + np.exp(1)

def griewank(x):  # ∈ [-600, 600]
    # sums = 0.0
    # prod = 1
    # for i in range(len(x)):
    #     sums = sums + np.power(x[i], 2) / 4000
    #     prod = prod * np.cos(x[i]/np.sqrt(i))
    # return sums - prod + 1
    p1 = 0
    for i in range(len(x)):
        p1 += x[i] ** 2
        p2 = 1
    for i in range(len(x)):
        p2 *= np.cos(float(x[i] / np.sqrt(i + 1)))
    return 1 + (float(p1) / 4000.0) - float(p2)


def goldstein(x):  # ∈ [-2, 2]
    x1 = x[0]
    x2 = x[1]
    for i in range(len(x)):
        eq1 = 1 + (np.power(x1 + x2 + 1, 2))(19 - 14 * x2 + 6 * x1 * x2 + 3 * np.power(x2, 2))
        eq2 = 30 + np.power((2 * x1 - 3 * x2, 2))(
            18 - 32 * x1 + 12 * np.power(x1, 2) + 48 * x2 - 36 * x1 * x2 + 27 * np.power(x2, 2))
        return eq1 * eq2

    return


def camel6(x):  # x1 ∈ [-3, 3], x2 ∈ [-2, 2]
    x1 = x[i, 0]
    x2 = x[i, 1]
    part1 = (4 - 2.1 * np.power(x1, 2) + (np.power(x1, 4) / 3)) * np.power(x1, 2)
    part2 = x1 * x2
    part3 = (- 4 + 4 * np.power(x2, 2))
    return


def shiftedRastrigin(x):
    sum = 0.0
    for i in range(len(x)):
        sum = sum + (np.power(x[i], 2) - 10 * np.cos(2 * np.pi * x[i]) + 10)
    return sum


def shiftedRotatedRastrigin(x):
    return


def lunaceksBiRastrigin(x):  # prep for death
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


lis = []
t0 = perf_counter()

t2 = perf_counter()
N = 100  # POPULATION SIZE
D = 30  # DIMENSIONALITY
delta = 0.001  # DISTURBANCE THRESHOLD
maxIterations = 3100  # ITERATIONS ALLOWED
lowerB = [-32] * D  # LOWER BOUND (IN ALL DIMENSIONS)
upperB = [32] * D  # UPPER BOUND (IN ALL DIMENSIONS)

for i in range(30):
    counter1 += 1
    print("This is trial ", counter1)
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
            fitness[i] = finalAckley(X[i,])
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
    lis.append(fitness[s])
t3 = perf_counter()
print(lis)
print("\nFinal best fitness:\t", fitness[s])
print("\nBest fly position:\n", X[s,])
print("\n1% Time elapsed: ", t3 - t2)

t1 = perf_counter()
print("\n Time elapsed: ", (t1 - t0) / 60, "mins")
print("This is the best fly after 30 trials:", lis)
print("Min = ", min(lis))
print("Max = ", max(lis))
print("Median = ", np.median(lis))
print("Mean = ", np.mean(lis))
print("Standard deviation = ", np.std(lis))
print("Counter says ", counter1)
