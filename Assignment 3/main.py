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


def sigmoid(x):
    return 1. / (1 + np.exp(-x))


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


def loss(y, y_star):
    return -np.sum(np.multiply(y_star, np.log2(y))) / len(y)


def predict(x, W, b):
    return feed_forward(x, W, b)[0][-1]


def feed_forward(x, W, b, act=None):
    # Method from the lecture
    if act is None:
        act = [sigmoid, softmax]
    a = [x]
    z = [x]
    for i, (Wi, bi, acti) in enumerate(zip(W, b, act)):
        z.append(np.dot(Wi, a[i]) + bi)
        a.append(acti(z[-1]))
    return a, z


def d_sigmoid(x):
    # Method from the lecture
    act = sigmoid(x)
    return act * (1 - act)


def d_softmax(x):
    sm = softmax(x)
    jac = np.outer(sm, -sm)
    jac[np.diag_indices(len(sm))] = np.multiply(sm, 1 - sm)
    # jac:
    #  x_1 (1 - x_1) |    -x_1 x_2    |    -x_1 x_3
    #    -x_2 x_1    |  x_2 (1 - x_2) |    -x_2 x_3
    #    -x_3 x_1    |    -x_3 x_2    |  x_3 (1 - x_3)
    return jac


def d_loss(y, y_star):
    # this could lead to nan values if y is 0, but imo it is fine for this example
    return - y_star / (len(y) * np.log(2) * y)


def back_prop(y_star, W, b, d_act, a, z):
    # Method from the lecture + a little adaptation for the softmax
    # We know it's not fully applicable for every network, but we had a working backprob. in the first assignment.
    # And since this was shown in the lecture, it should be sufficient for this assignment.
    delta = [None] * len(a)
    # we know that the output layer has the softmax, hence the sum.
    delta[-1] = np.sum(d_act[-1](z[-1]) * (d_loss(a[-1], y_star)), axis=1)  # delta_L
    # We could
    for l in range(len(a) - 2, 0, -1):
        delta[l] = d_act[l - 1](z[l]) * W[l].T.dot(delta[l + 1])

    dW = [0.] * len(a)
    db = [0.] * len(a)
    for l in range(len(a) - 1):
        dW[l] = np.outer(delta[l + 1], a[l])
        db[l] = delta[l + 1]

    return dW, db, delta


def init_params(ni, nh, no):
    v = 1 / np.sqrt(ni), 1 / np.sqrt(nh)
    np.random.uniform()
    W = [np.random.uniform(-v[0], v[0], size=(nh, ni)), np.random.uniform(-v[0], v[0], size=(no, nh))]
    b = [np.random.uniform(-v[0], v[0], size=(nh,)), np.random.uniform(-v[0], v[0], size=(no,))]
    return W, b


def calculate_gradients(x, y_star, W, b):
    a, z = feed_forward(x, W, b)
    d_act = [d_sigmoid, d_softmax]
    dW, db, _ = back_prop(y_star, W, b, d_act, a, z)
    output = a[-1]
    return (dW[0], dW[1]), (db[0], db[1]), output


def armijo(loss_p, d_loss_p, _loss, alpha=10):
    t = alpha
    sigma = 1e-4
    beta = 0.5
    s_loss_p_loss_p = 0
    for d_param in d_loss_p:
        s_loss_p_loss_p += np.sum(d_param ** 2)

    while not (loss_p - _loss(t) >= - sigma * t * s_loss_p_loss_p):
        t *= beta
    return t


def steepest_descent(X, Y, W, b, t=1, use_armijo=False, num_iterations=5000):
    Loss, Accuracy = [], []
    data_length = len(X)
    for iteration in range(num_iterations):
        dW0, dW1, db0, db1 = None, None, None, None
        batch_loss, accuracy = 0, 0
        for x, y_star in zip(X, Y):
            if any([dW0 is None, dW1 is None, db0 is None, db1 is None]):
                (dW0, dW1), (db0, db1), out = calculate_gradients(x, y_star, W=W, b=b)
            else:
                (dW0_, dW1_), (db0_, db1_), out = calculate_gradients(x, y_star, W=W, b=b)
                dW0 += dW0_
                dW1 += dW1_
                db0 += db0_
                db1 += db1_
            batch_loss += loss(out, y_star)
            accuracy += 1 if np.argmax(y_star) == np.argmax(out) else 0
        dW0, dW1, db0, db1 = dW0 / data_length, dW1 / data_length, db0 / data_length, db1 / data_length
        Loss.append(batch_loss / data_length)
        Accuracy.append(accuracy / data_length)

        if use_armijo:
            def calc_loss(_t) -> float:
                total_loss = 0
                for _x, _y_star in zip(X, Y):
                    a, _ = feed_forward(_x, W=[W[0] - _t * dW0, W[1] - _t * dW1], b=[b[0] - _t * db0, b[1] - _t * db1])
                    total_loss += loss(a[-1], _y_star)
                return total_loss / data_length

            t = armijo(
                loss_p=batch_loss / data_length,
                d_loss_p=(dW0, dW1, db0, db1),
                _loss=calc_loss,
                alpha=10
            )

        W[0] -= t * dW0
        W[1] -= t * dW1
        b[0] -= t * db0
        b[1] -= t * db1
    return Loss, Accuracy


def test(X, Y, W, b):
    batch_loss, accuracy = 0, 0
    data_length = len(X)
    for x, y_star in zip(X, Y):
        out = predict(x, W, b)
        batch_loss += loss(out, y_star)
        accuracy += 1 if np.argmax(y_star) == np.argmax(out) else 0
    return batch_loss / data_length, accuracy / data_length


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
    fig1, ax1 = plt.subplots(1, 2)
    ax1[0].set_title('Training loss')
    ax1[1].set_title('Training accuracy')

    fig2, ax2 = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(10, 3.5))
    ax2[0].scatter(x_train[:, 0], x_train[:, 1], c=y_train), ax2[0].set_title('Training Data'), ax2[0].set_aspect(
        'equal',
        'box')
    ax2[0].set_title('Training Data'), ax2[0].set_aspect('equal', 'box')
    ax2[1].set_title('Decision BD (constant)'), ax2[1].set_aspect('equal', 'box')
    ax2[2].set_title('Decision BD (Armijo)'), ax2[2].set_aspect('equal', 'box')

    lam = 1e-3
    fig3, ax3 = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(10, 3.5))
    plt.suptitle(r'Regularization $\lambda$=%.6f' % lam)
    ax3[0].scatter(x_train[:, 0], x_train[:, 1], c=y_train)
    ax3[0].set_title('Training Data'), ax3[0].set_aspect('equal', 'box')
    ax3[1].set_title('Decision BD (constant)'), ax3[1].set_aspect('equal', 'box')
    ax3[2].set_title('Decision BD (Armijo)'), ax3[2].set_aspect('equal', 'box')

    """ Start of your code
        You are free to create any functions that are called from within task()
    """
    # Methods are defined above task() to keep the code more clean
    ni, nh, no = 2, 12, 3

    def reshape_to_wb(x):
        W0 = x[:nh * ni].reshape(nh, ni)
        W1 = x[nh * ni:nh * ni + no * nh].reshape(no, nh)
        b0 = x[nh * ni + no * nh:nh * ni + no * nh + nh]
        b1 = x[nh * ni + no * nh + nh:]
        assert len(b1) == no
        return W0, W1, b0, b1

    def f(x, input_, y_star):
        W0, W1, b0, b1 = reshape_to_wb(x)

        W = [W0, W1]
        b = [b0, b1]
        a, z = feed_forward(input_, W, b)
        L = loss(a[-1], y_star)
        return L
    # ------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------Check-Grad-------------------------------------------------------------
    W, b = init_params(ni=ni, nh=nh, no=no)

    x_init = np.concatenate((W[0].ravel(), W[1].ravel(), b[0].ravel(), b[1].ravel()))

    rand_index = np.random.randint(0, len(x_train))
    ipt, y_star = x_train[rand_index], y_train[rand_index]
    grads = approx_fprime(x_init, f, 1e-6, ipt, y_star)
    dW0, dW1, db0, db1 = reshape_to_wb(grads)
    dW, db, _ = calculate_gradients(ipt, y_star, W, b)

    print(
        f"Approx == Backprop",
        f"W0: {np.allclose(dW[0], dW0)}",
        f"W1: {np.allclose(dW[1], dW1)}",
        f"b0: {np.allclose(db[0], db0)}",
        f"b1: {np.allclose(db[1], db1)}",
        sep="\n\t"
    )
    n_iter = 5000
    # ------------------------------------------------------------------------------------------------------------------
    # --------------------------------------------Constant--------------------------------------------------------------
    loss_c, accuracy_c = steepest_descent(x_train, np.eye(3)[y_train], W, b, num_iterations=n_iter)
    ax1[0].set_yscale("log")
    ax1[0].plot(loss_c, label="constant")
    ax1[1].plot(accuracy_c, label="constant")
    x_1 = np.linspace(-1.5, 1.5, 61)
    x_2 = np.linspace(-1.5, 1.5, 61)
    xx, yy = np.meshgrid(x_1, x_2)
    class_prediction_c = np.argmax([predict((x, y), W, b) for x, y in zip(xx.flatten(), yy.flatten())], axis=1)
    ax2[1].contourf(x_1, x_2, class_prediction_c.reshape(xx.shape))
    test_loss_c, test_accuracy_c = test(X=x_test, Y=y_test, W=W, b=b)
    print(f"(t=1) Test loss: {test_loss_c}, Accuracy: {test_accuracy_c}")
    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------Armijo---------------------------------------------------------------
    print("Begin Training with Armijo(alpha=10, sigma = 1e-4, beta = 0.5)")
    W, b = init_params(ni=ni, nh=nh, no=no)  # To train the model from the same start
    loss_a, accuracy_a = steepest_descent(x_train, np.eye(3)[y_train], W, b, use_armijo=True, num_iterations=n_iter)
    ax1[0].plot(loss_a, label="Armijo")
    ax1[1].plot(accuracy_a, label="Armijo")
    class_prediction_a = np.argmax([predict((x, y), W, b) for x, y in zip(xx.flatten(), yy.flatten())], axis=1)
    ax2[2].contourf(x_1, x_2, class_prediction_a.reshape(xx.shape))
    test_loss_a, test_accuracy_a = test(X=x_test, Y=y_test, W=W, b=b)
    print(f"(t=Armijo) Test loss: {test_loss_a}, Accuracy: {test_accuracy_a}")

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    ax1[0].legend()
    ax1[1].legend()
    """ End of your code
    """

    return [fig1, fig2, fig3]


if __name__ == '__main__':
    pdf = PdfPages('figures.pdf')
    figures = task()
    for fig in figures:
        pdf.savefig(fig)
    pdf.close()
