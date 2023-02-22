#!/usr/bin/env python

""" Python code submission file.

IMPORTANT:
- Do not include any additional python packages.
- Do not change the existing interface and return values of the task functions.
"""

import numpy as np
from scipy.optimize import linprog


def task1():
    """ Start of your code
    """

    def simplex_method(c, A, x, beta):
        # x...basic feasible solution
        cols, rows = A.shape
        iteration = 0
        while True:
            print("%" + 120 * "-")
            print(f"%\t\tIteration {iteration}:")
            print(f"%\t\tCost: {c.dot(x)}")
            print(f"%\t\tIntermediate solution: {x}")
            print(f"%\t\tBasic: {set(beta.values())}, Non basic: {set(range(rows)) - set(beta.values())}")
            iteration += 1
            B = np.vstack(A[:, beta[i]] for i in range(cols)).T  # make b an m x m matrix
            B_1 = np.linalg.inv(B)
            cBT_B1 = c[[beta[i] for i in range(cols)]].T.dot(B_1)
            c_ = c - cBT_B1.dot(A)
            c_[[beta[i] for i in range(cols)]] = np.infty  # remove the indexes in the basis
            j = np.argmin(c_)  # Pivoting rule (a)
            print(f"%\t\t\\bar{{c}} = {c_}")
            if c_[j] >= 0:
                print("%\t\tWe have found a solution!")
                print(f"%\t\tSolution: {x}")
                print("%" + 120 * "-")
                return x
            u = B_1.dot(A[:, j])
            if np.all(u <= 0):
                theta_star = np.infty
                x[j] = theta_star
                return x
            theta = [x[beta[i]] / u_i if u_i > 0 else np.infty for i, u_i in enumerate(u)]
            l = np.argmin(theta)
            theta_star = theta[l]
            y = np.zeros(shape=x.shape)
            y[j] = theta_star
            for i in range(cols):
                if i == l:
                    continue
                y[beta[i]] = x[beta[i]] - theta_star * u[i]
            beta[l] = j  # replacing l with j for new basis
            x = y

    A = np.array([
        [100, 120, 70, 80, 1, 0, 0],
        [7, 10, 8, 8, 0, 1, 0],
        [1, 1, 1, 1, 0, 0, 1]
    ], dtype=np.float_)
    x = np.array([0, 0, 0, 0, 100_000, 8000, 1000], dtype=np.float_)  # first basic f. solution
    beta = {0: 4, 1: 5, 2: 6}  #
    c = np.array([-35, -40, -20, -30, 0, 0, 0], dtype=np.float_)  # - since we are minimizing
    assert np.all(A.dot(x) == np.array([100_000, 8000, 1000])) and np.all(x >= 0)  # feasibility check
    print(simplex_method(c=c, A=A, x=x, beta=beta))
    print(linprog(c=c[:-3], A_ub=A[:, :-3], b_ub=[100_000, 8000, 1000]))
    """ End of your code
    """


if __name__ == '__main__':
    # execute simplex method
    task1()
