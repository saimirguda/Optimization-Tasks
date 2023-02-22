#!/usr/bin/env python

""" Python code submission file.

IMPORTANT:
- Do not include any additional python packages.
- Do not change the existing interface and return values of the task functions.
- Prior to your submission, check that the pdf showing your plots is generated.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import approx_fprime
from typing import Callable

# Modify the following global variables to be used in your functions
""" Start of your code
"""

""" End of your code
"""


def task1():
    """ Characterization of Functions

        Requirements for the plots:
            - ax[0] Contour plot for a)
            - ax[1] Contour plot for b)
            - ax[2] Contour plot for c)
        Choose the number of contour lines such that the stationary points and the function can be well characterized. 
    """
    print('\nTask 1')

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle('Task 1 - Contour plots of functions', fontsize=16)

    ax[0].set_title('a)')
    ax[0].set_xlabel('$x_1$')
    ax[0].set_ylabel('$x_2$')

    ax[1].set_title('b)')
    ax[1].set_xlabel('$x_1$')
    ax[1].set_ylabel('$x_2$')

    ax[2].set_title('c)')
    ax[2].set_xlabel('$x_1$')
    ax[2].set_ylabel('$x_2$')

    """ Start of your code
    """
    """ - ax[0] Contour plot for a) """
    # Example for plot usage
    x1, x2 = np.meshgrid(np.linspace(-5, 5), np.linspace(-0.85, 5))
    # x1 = 0; x2 <= -0.874 => log(<=0) => numerical issue
    ax[0].contour(x1, x2, func_1a(np.array([x1, x2])), 50)
    stat_points = [(0, 0)]
    ax[0].scatter([sp[0] for sp in stat_points], [sp[1] for sp in stat_points])
    # TODO: ADD MARKERS TO STATIONARY points

    """ - ax[1] Contour plot for b) """
    x1, x2 = np.meshgrid(np.linspace(-2, 2), np.linspace(-2, 2))  # TODO MAYBE EDIT THIS TOO
    ax[1].contour(x1, x2, func_1b(np.array([x1, x2])), 50)
    stat_points = [(0, 0), (1, -1 / 2), (-1, 1 / 2)]
    ax[1].scatter([sp[0] for sp in stat_points], [sp[1] for sp in stat_points])
    # TODO: ADD MARKERS TO STATIONARY points

    """ - ax[2] Contour plot for c) """
    x1, x2 = np.meshgrid(np.linspace(-2, 2), np.linspace(-2, 2))  # TODO MAYBE EDIT THIS TOO
    ax[2].contour(x1, x2, func_1c(np.array([x1, x2])), 50)
    stat_points = [(0, 0), (-4 / 3, 0), (-1, -1), (-1, 1)]
    ax[2].scatter([sp[0] for sp in stat_points], [sp[1] for sp in stat_points])
    # TODO: ADD MARKERS TO STATIONARY points
    """ End of your code
    """
    return fig


# Modify the function bodies below to be used for function value and gradient computation
def approx_grad_task1(func: Callable, x: np.ndarray) -> np.ndarray:
    """ Numerical Gradient Computation
        @param x Vector of size (2,)
        This function shall compute the gradient approximation for a given point 'x' and a function 'func'
        using the given central differences formulation for 2D functions. (Task1 functions)
        @return The gradient approximation
    """
    assert (len(x) == 2)
    x1, x2, e = x[0], x[1], 1e-6
    d_x1 = func(np.array([x1 + e, x2])) - func(np.array([x1 - e, x2]))
    d_x2 = func(np.array([x1, x2 + e])) - func(np.array([x1, x2 - e]))
    return np.array([d_x1, d_x2]) / (2 * e)


def approx_grad_task2(func: Callable, x: np.ndarray, *args) -> np.ndarray:
    """ Numerical Gradient Computation
        @param x Vector of size (n,)
        This function shall compute the gradient approximation for a given point 'x' and a function 'func'
        using scipy.optimize.approx_fprime(). (Task2 functions)
        @return The gradient approximation
    """
    # Imo this method should also take kwargs.
    return approx_fprime(x, func, 1e-6, *args)


def func_1a(x: np.ndarray) -> float:
    """ Computes and returns the function value for function 1a) at a given point x
        @param x Vector of size (2,)
    """
    x1, x2 = x[0], x[1]  # I like to match the code to the formula, hence the conversion
    return np.log(1 + 1 / 2 * (x1 ** 2 + 3 * x2 ** 3))


def grad_1a(x: np.ndarray) -> np.ndarray:
    """ Computes and returns the analytical gradient result for function 1a) at a given point x
        @param x Vector of size (2,)
    """
    x1, x2 = x[0], x[1]
    denom = (2 + x1 ** 2 + 3 * x2 ** 3)
    dx1 = 2 * x1 / denom
    dx2 = 9 * x2 ** 2 / denom
    return np.array([dx1, dx2])


def func_1b(x: np.ndarray) -> float:
    """ Computes and returns the function value for function 1b) at a given point x
        @param x Vector of size (2,)
    """
    x1, x2 = x[0], x[1]  # I like to match the code to the formula, hence the conversion
    return (x1 - 2 * x2) ** 4 + 64 * x1 * x2


def grad_1b(x: np.ndarray) -> np.ndarray:
    """ Computes and returns the analytical gradient result for function 1b) at a given point x
        @param x Vector of size (2,)
    """
    x1, x2 = x[0], x[1]
    dx1 = 64 * x2 + 4 * (x1 - 2 * x2) ** 3
    dx2 = 64 * x1 - 8 * (x1 - 2 * x2) ** 3
    return np.array([dx1, dx2])


def func_1c(x: np.ndarray) -> float:
    """ Computes and returns the function value for function 1c) at a given point x
        @param x Vector of size (2,)
    """
    x1, x2 = x[0], x[1]  # I like to match the code to the formula, hence the conversion
    squared_norm = x1 ** 2 + x2 ** 2  # since we are dealing with multiple values at once
    return x1 ** 2 + x1 * squared_norm + squared_norm


def grad_1c(x: np.ndarray) -> np.ndarray:
    """ Computes and returns the analytical gradient result for function 1c) at a given point x
        @param x Vector of size (2,)
    """
    x1, x2 = x[0], x[1]
    dx1 = 3 * x1 ** 2 + 4 * x1 + x2 ** 2
    dx2 = 2 * x1 * x2 + 2 * x2
    return np.array([dx1, dx2])


def func_2a(x: np.ndarray, c: np.ndarray, k: np.ndarray) -> float:
    """ Computes and returns the function value for function 2a) at a given point x
        @param x Vector of size (n,)
    """
    # np.multiply = Element wise multiply = Hadamard Product
    return 1 / 2 * np.sum(np.multiply(c, (x - k)) ** 2)


def grad_2a(x: np.ndarray, c: np.ndarray, k: np.ndarray) -> np.ndarray:
    """ Computes and returns the analytical gradient result for function 2a) at a given point x
        @param x Vector of size (n,)
    """
    return c ** 2 * (x - k)


def func_2b(x: np.ndarray, A: np.ndarray) -> float:
    """ Computes and returns the function value for function 2b) at a given point x
        @param x Vector of size (n,)
    """

    def h(arr: np.ndarray) -> np.ndarray:
        return 1 / 2 * arr ** 2 + 2 * arr

    return sum(h(A.dot(x)))


def grad_2b(x: np.ndarray, A: np.ndarray) -> np.ndarray:
    """ Computes and returns the analytical gradient result for function 2b) at a given point x
        @param x Vector of size (n,)
    """
    return A.T.dot(A.dot(x) + 2)


# Later we use approx_fprime, and it requires x to be the very first parameter.
def func_2c(alpha: float, x: np.ndarray, y: np.ndarray, b: np.ndarray, A: np.ndarray) -> float:
    """ Computes and returns the function value for function 2c) at a given point x
        @param x Vector of size (n,)
    """
    return 1 / 2 * np.sum((A.dot(x + alpha * y) - b) ** 2)


# Later we use approx_fprime, and it requires x to be the very first parameter.
def grad_2c(alpha: float, x: np.ndarray, y: np.ndarray, b: np.ndarray, A: np.ndarray) -> float:
    """ Computes and returns the analytical gradient result for function 2c) at a given point alpha
        @param alpha Vector of size (1,)
    """
    return (A.dot(x + alpha * y) - b).dot(A.dot(y))


def task3():
    """ Numerical Gradient Verification
        ax[0] to ax[2] Bar plot comparison, analytical vs numerical gradient for Task 1
        ax[3] to ax[5] Bar plot comparison, analytical vs numerical gradient for Task 2

    """
    print('\nTask 3')

    fig = plt.figure(figsize=(15, 8), constrained_layout=True)
    fig.suptitle('Task 3 - Barplots numerical vs analytical', fontsize=16)
    ax = [None, None, None, None, None, None]
    keys = ['a)', 'b)', 'c)']
    gs = fig.add_gridspec(7, 12)

    n = 2
    for i in range(3):
        ax[i] = fig.add_subplot(gs[1:4, 3 * i:(3 * i + 3)])
        ax[i].set_title('1 ' + keys[i])
        ax[i].set_xticks(np.arange(n))
        ax[i].set_xticklabels((r'$\frac{\partial}{\partial x_1}$', r'$\frac{\partial}{\partial x_2}$'), fontsize=16)

    n = 5
    n = n
    for k, i in enumerate(range(3, 6)):
        ax[i] = fig.add_subplot(gs[4:, 3 * k:(3 * k + 3)])
        ax[i].set_title('2 ' + keys[k])
        ax[i].set_xticks(np.arange(n))
        ax[i].set_xticklabels((r'$\frac{\partial}{\partial x_1}$', r'$\frac{\partial}{\partial x_2}$',
                               r'$\frac{\partial}{\partial x_3}$', r'$\frac{\partial}{\partial x_4}$',
                               r'$\frac{\partial}{\partial x_5}$'), fontsize=16)

        if i == n:
            ax[i].set_xticklabels((r'$\frac{\partial}{\partial \alpha}$', '', '', '', ''), fontsize=16)

    """ Start of your code
    """

    # Example for plot usage
    bw = 0.3  # bar width
    # bar (ceneters, values)
    np.random.seed(42)  # To generate always the same numbers
    rand = np.random.random
    """ ax[0] to ax[2] Bar plot comparison, analytical vs numerical gradient for Task 1 """
    table = {}
    for i, (name, grad, func) in enumerate([
        ("1a", grad_1a, func_1a),
        ("1b", grad_1b, func_1b),
        ("1c", grad_1c, func_1c)
    ]):
        x = rand(2)
        dx_analytical, dx_approx = grad(x).tolist(), approx_grad_task1(func, x).tolist()
        ax[i].bar([0 - bw / 2, 1 - bw / 2], dx_analytical, bw)
        ax[i].bar([0 + bw / 2, 1 + bw / 2], dx_approx, bw)
        table[name] = (dx_analytical, dx_approx)

    for i, (name, grad, func, args) in enumerate([
        ("2a", grad_2a, func_2a, [rand(n), rand(n), rand(n)]),
        ("2b", grad_2b, func_2b, [rand(n), rand((n, n))]),
    ]):
        dx_analytical, dx_approx = grad(*args).tolist(), approx_grad_task2(func, *args).tolist()

        bins = np.arange(start=0, stop=n)
        ax[i + 3].bar((bins - bw / 2).tolist(), dx_analytical, bw)
        ax[i + 3].bar((bins + bw / 2).tolist(), dx_approx, bw)
        table[name] = (dx_analytical, dx_approx)
    m = 3
    name, args = "2c", [rand(1), rand(n), rand(n), rand(m), rand((m, n))]
    dx_analytical, dx_approx = grad_2c(*args), approx_grad_task2(func_2c, *args).tolist()
    ax[5].bar([0 - bw / 2], dx_analytical, bw)
    ax[5].bar([0 + bw / 2], dx_approx, bw)
    table[name] = (dx_analytical, dx_approx)
    """ ax[3] to ax[5] Bar plot comparison, analytical vs numerical gradient for Task 2 """
    """ End of your code
    """

    print("""
    \\begin{center}
    \\begin{tabular}{|c|c|c|}
        \\hline
        Task & Analytical & Approximated \\\\
        \\hline""")
    for name in table.keys():
        analytical, approx = table[name]
        if name != "2c":
            print(f"        {name} & $\\nabla f(x) = \\begin{{bmatrix}} ")
            print("             ", end="")
            print(*analytical, sep="\\\\\n             ")
            print(f"        \\end{{bmatrix}}$ & $\\nabla f(x)=\\begin{{bmatrix}}")
            print("             ", end="")
            print(*approx, sep="\\\\\n             ")
            print(f"        \\end{{bmatrix}}$\\\\")
        else:
            print(f"        {name} & $\\nabla f(\\alpha)=({analytical})$ & $\\nabla f(a)={approx[0]}$\\\\")
        print("         \\hline")
    print("""    \\end{tabular}
    \\end{center}""")
    return fig


def func_5(x: np.ndarray, w0: np.ndarray, w1: np.ndarray) -> float:
    """ Computes and returns the function value for the neural network in Task 5 at a given point x
        @param x Vector of size (2,)
    """
    z = w0.dot(x)

    def sigma(t):
        return np.divide(1, 1 + np.exp(-t))

    a = sigma(z)
    return w1.dot(a)


def grad_5(x: np.ndarray, w0: np.ndarray, w1: np.ndarray) -> np.ndarray:
    """ Computes and returns the gradient for the neural network in Task 5 at a given point x
        @param x Vector of size (2,)
    """
    # Calculated on paper, for the real implementation of autodiff, look down to task 5.
    z = w0.dot(x)

    def dsigma_dt(t):
        return np.divide(np.exp(-t), (1 + np.exp(-t)) ** 2)

    dx_1 = w0[0, 0] * dsigma_dt(z[0]) * w1[0] + w0[1, 0] * dsigma_dt(z[1]) * w1[1]
    dx_2 = w0[0, 1] * dsigma_dt(z[0]) * w1[0] + w0[1, 1] * dsigma_dt(z[1]) * w1[1]
    return np.array([dx_1, dx_2])


def task5():
    """ Automatic Differentation 

    Check your results for the gradients obtained with automatic differentiation in forward/backward mode.
    Therefor use central differences to obtain a numeric approximation.

    """
    print('\nTask 5')

    """ Start of your code
    """

    x = np.array([1.0, 0.5])  # no transpose, since approx fails with multi dim input
    w0 = np.array([[1.0, 0.2],
                   [0.5, -1.]])
    w1 = np.array([1.0, 0.5])
    print("%-----------------------------------------------Forward-mode-----------------------------------------------")

    def fm_dot(w, dual_v):  # No type description since I don't want to import typing
        """
        Forward mode dot product.
        :param: w: nd_array is are the weights
        :param: dual_v: Tuple[nd_array, nd_array] dual variable consisting of actual
        value and value of the derivative
        returns value and derivative
        """
        v = w.dot(dual_v[0])
        dv = w.dot(dual_v[1])  # \dv{f}{x} = W y(x) \dv{x} = W dv{y}{x}
        return v, dv

    def fm_sigma(dual_v):
        """
        Forward mode dot product.
        :param: w: nd_array is are the weights
        :param: dual_v: Tuple[nd_array, nd_array] dual variable consisting of actual
        value and value of the derivative
        returns value and derivative
        """
        v = np.divide(1, 1 + np.exp(-dual_v[0]))
        dv = np.multiply(np.divide(np.exp(-dual_v[0]), (1 + np.exp(-dual_v[0])) ** 2), dual_v[1])
        return v, dv

    def fm_nn(dv: np.ndarray):
        """
        :param: dv, input for forward mode
        returns the result of the nn and the derivative 
        """
        z = fm_dot(w0, (x, dv))
        a = fm_sigma(z)
        return fm_dot(w1, a)

    y, fm_dx1 = fm_nn(np.array([1, 0]))  # call for input x1
    _, fm_dx2 = fm_nn(np.array([0, 1]))  # call for input x2
    print(f"%  f(x, W0, W1)={y}")
    print(f"%  Forward mode gradient: dx1={fm_dx1}, dx2={fm_dx2}")
    print("%----------------------------------------------------------------------------------------------------------")
    print("%-----------------------------------------------Backward-mode----------------------------------------------")
    cal_stack = []  # containing lambdas to calculate the respective derivative

    def bm_dot(w, x_):
        cal_stack.append(lambda dy: w.T.dot(dy))  # We are going in the reverse direction, hence the transpose.
        return w.dot(x_)

    def bm_sigma(x_):
        cal_stack.append(lambda dy: np.multiply(np.divide(np.exp(-x_), (1 + np.exp(-x_)) ** 2), dy))
        return np.divide(1, 1 + np.exp(-x_))

    def fm_nn():
        """
        :param: dv, input for forward mode
        returns the result of the nn and the derivative
        """
        z = bm_dot(w0, x)
        a = bm_sigma(z)
        return bm_dot(w1, a)

    print("%-------------------------------------------One-time-forward-pass------------------------------------------")
    print(f"%  f(x, W0, W1)={fm_nn()}")
    print("%-----------------------------Unwind-the-call-stack-and-compute-both-derivatives---------------------------")
    dy = 1
    for dv_func in reversed(cal_stack):
        dy = dv_func(dy)
    bm_dx1, bm_dx2 = dy
    print(f"%  Backward mode gradient: dx1={bm_dx1}, dx2={bm_dx2}")
    print("%----------------------------------------------------------------------------------------------------------")
    print("%------------------------------------------Calculated-derivative-------------------------------------------")
    dx1, dx2 = grad_5(x=x, w0=w0, w1=w1)
    print(f"%  Calculated gradient: {dx1=}, {dx2=}")
    print("%----------------------------------------------------------------------------------------------------------")
    print("%------------------------------------------Approximate-derivative------------------------------------------")
    print(f"%  f(x, W0, W1) ={func_5(x=x, w0=w0, w1=w1)}")
    dx1, dx2 = approx_fprime(x, func_5, 1e-9, w0, w1)
    print(f"%  Approximated gradient: {dx1=}, {dx2=}")
    print("%----------------------------------------------------------------------------------------------------------")
    """ End of your code
    """


if __name__ == '__main__':
    tasks = [task1, task3]

    pdf = PdfPages('figures.pdf')
    for task in tasks:
        retval = task()
        pdf.savefig(retval)
    pdf.close()

    task5()
