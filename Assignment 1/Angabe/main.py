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

    fig, ax = plt.subplots(1,3, figsize=(12,4))
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

    # Example for plot usage
    x1, x2 = np.meshgrid(np.linspace(-5, 5), np.linspace(-5, 5))
    ax[0].contour(x1, x2, x1**2 + 0.5*x2**2, 50)
    
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
    assert(len(x) == 2)
    pass

def approx_grad_task2(func: Callable, x: np.ndarray, *args) -> np.ndarray:
    
    """ Numerical Gradient Computation
        @param x Vector of size (n,)
        This function shall compute the gradient approximation for a given point 'x' and a function 'func'
        using scipy.optimize.approx_fprime(). (Task2 functions)
        @return The gradient approximation
    """
    pass

def func_1a(x: np.ndarray) -> float:
    """ Computes and returns the function value for function 1a) at a given point x
        @param x Vector of size (2,)
    """
    pass

def grad_1a(x: np.ndarray) -> np.ndarray:
    """ Computes and returns the analytical gradient result for function 1a) at a given point x
        @param x Vector of size (2,)
    """
    pass

def func_1b(x: np.ndarray) -> float:
    """ Computes and returns the function value for function 1b) at a given point x
        @param x Vector of size (2,)
    """
    pass

def grad_1b(x: np.ndarray) -> np.ndarray:
    """ Computes and returns the analytical gradient result for function 1b) at a given point x
        @param x Vector of size (2,)
    """
    pass

def func_1c(x: np.ndarray) -> float:
    """ Computes and returns the function value for function 1c) at a given point x
        @param x Vector of size (2,)
    """
    pass

def grad_1c(x: np.ndarray) -> np.ndarray:
    """ Computes and returns the analytical gradient result for function 1c) at a given point x
        @param x Vector of size (2,)
    """
    pass

def func_2a(x: np.ndarray, c: np.ndarray, k: np.ndarray) -> float:
    """ Computes and returns the function value for function 2a) at a given point x
        @param x Vector of size (n,)
    """
    pass

def grad_2a(x: np.ndarray, c: np.ndarray, k: np.ndarray) -> np.ndarray:
    """ Computes and returns the analytical gradient result for function 2a) at a given point x
        @param x Vector of size (n,)
    """
    pass

def func_2b(x: np.ndarray, A: np.ndarray) -> float:
    """ Computes and returns the function value for function 2b) at a given point x
        @param x Vector of size (n,)
    """
    pass

def grad_2b(x: np.ndarray, A: np.ndarray) -> np.ndarray:
    """ Computes and returns the analytical gradient result for function 2b) at a given point x
        @param x Vector of size (n,)
    """
    pass

def func_2c(alpha: float, x: np.ndarray, y: np.ndarray, b: np.ndarray, A: np.ndarray) -> float:
    """ Computes and returns the function value for function 2c) at a given point x
        @param x Vector of size (n,)
    """
    pass

def grad_2c(alpha: np.ndarray, x: np.ndarray, y: np.ndarray, b: np.ndarray, A: np.ndarray) -> np.ndarray:
    """ Computes and returns the analytical gradient result for function 2c) at a given point alpha
        @param alpha Vector of size (1,)
    """
    pass

def task3():

    """ Numerical Gradient Verification
        ax[0] to ax[2] Bar plot comparison, analytical vs numerical gradient for Task 1
        ax[3] to ax[5] Bar plot comparison, analytical vs numerical gradient for Task 2

    """
    print('\nTask 3')

    fig = plt.figure(figsize=(15,8), constrained_layout=True)
    fig.suptitle('Task 3 - Barplots numerical vs analytical', fontsize=16)
    ax = [None, None, None, None, None, None]
    keys = ['a)', 'b)', 'c)']
    gs = fig.add_gridspec(7, 12)

    n = 2
    for i in range(3):
        ax[i] = fig.add_subplot(gs[1:4, 3*i:(3*i+3)])
        ax[i].set_title('1 '+keys[i])
        ax[i].set_xticks(np.arange(n))
        ax[i].set_xticklabels((r'$\frac{\partial}{\partial x_1}$', r'$\frac{\partial}{\partial x_2}$'), fontsize=16)

    n = 5
    for k, i in enumerate(range(3,6)):
        ax[i] = fig.add_subplot(gs[4:, 3*k:(3*k+3)])
        ax[i].set_title('2 '+ keys[k])
        ax[i].set_xticks(np.arange(n))
        ax[i].set_xticklabels((r'$\frac{\partial}{\partial x_1}$', r'$\frac{\partial}{\partial x_2}$', r'$\frac{\partial}{\partial x_3}$', r'$\frac{\partial}{\partial x_4}$', r'$\frac{\partial}{\partial x_5}$'), fontsize=16)

        if i == 5:
            ax[i].set_xticklabels((r'$\frac{\partial}{\partial \alpha}$', '', '', '', ''), fontsize=16)

    """ Start of your code
    """

    # Example for plot usage
    bw = 0.3
    ax[0].bar([0-bw/2,1-bw/2], [1.5, 1.1], bw)
    ax[0].bar([0+bw/2,1+bw/2], [1.5, 1.1], bw)

    """ End of your code
    """
    return fig


def func_5(x: np.ndarray, w0: np.ndarray, w1: np.ndarray) -> float:
    """ Computes and returns the function value for the neural network in Task 5 at a given point x
        @param x Vector of size (2,)
    """
    pass 

def grad_5(x: np.ndarray, w0: np.ndarray, w1: np.ndarray) -> np.ndarray:
    """ Computes and returns the gradient for the neural network in Task 5 at a given point x
        @param x Vector of size (2,)
    """
    pass 

def task5():
    """ Automatic Differentation 

    Check your results for the gradients obtained with automatic differentiation in forward/backmord mode.
    Therefore use central differences to obtain a numeric approximation.

    """
    print('\nTask 5')

    """ Start of your code
    """

    x = ...
    w0 = ...
    w1 = ...


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
