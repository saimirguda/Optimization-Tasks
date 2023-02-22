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

np.random.seed(0)

def task():
    """ Neural Network Training
        Requirements for the plots:
            - fig1 (make sure that all curves include labels)
                - ax[0] logarithmic plot for training loss with constant step size and Armijo backtracking (include a label!)
                - ax[1] plot of training accuracy with constant step size and Armijo backtracking (include a label!)
            - fig2
                - ax[0] already plots the training data
                - ax[1] for the training with constant step size, plot the predicted class for a dense meshgrid over the input data range
                - ax[2] make the same plot as in ax[1] for the training using Armijo backtracking
            - fig3: (bonus task), this should be the same as fig2 but with weight decay
    """

    # load data
    with np.load('data.npz') as data_set: 
            # get the training data
            x_train = data_set['x_train']
            y_train = data_set['y_train']

            # get the test data
            x_test = data_set['x_test']
            y_test = data_set['y_test']

    print(f'Training set with {x_train.shape[0]} data samples.')
    print(f'Test set with {x_test.shape[0]} data samples.')

    # plot training loss/accuracy
    fig1, ax = plt.subplots(1,2)
    ax[0].set_title('Training loss')
    ax[1].set_title('Training accuracy')
    ax[0].legend()
    ax[1].legend()

    fig2, ax = plt.subplots(1,3,sharex=True,sharey=True,figsize=(10,3.5))
    ax[0].scatter(x_train[:,0],x_train[:,1],c=y_train), ax[0].set_title('Training Data'), ax[0].set_aspect('equal', 'box')
    ax[0].set_title('Training Data'), ax[0].set_aspect('equal', 'box')
    ax[1].set_title('Decision BD (constant)'), ax[1].set_aspect('equal', 'box')
    ax[2].set_title('Decision BD (Armijo)'), ax[2].set_aspect('equal', 'box')

    lam = 1e-3
    fig3, ax = plt.subplots(1,3,sharex=True,sharey=True,figsize=(10,3.5))
    plt.suptitle(r'Regularization $\lambda$=%.6f' %lam)
    ax[0].scatter(x_train[:,0],x_train[:,1],c=y_train), ax[0].set_title('Training Data'), ax[0].set_aspect('equal', 'box')
    ax[1].set_title('Decision BD (constant)'), ax[1].set_aspect('equal', 'box')
    ax[2].set_title('Decision BD (Armijo)'), ax[2].set_aspect('equal', 'box')

    """ Start of your code
        You are free to create any functions that are called from within task()
    """



    """ End of your code
    """

    return [fig1, fig2, fig3]

if __name__ == '__main__':
    pdf = PdfPages('figures.pdf')
    figures = task()
    for fig in figures:
            pdf.savefig(fig)
    pdf.close()
