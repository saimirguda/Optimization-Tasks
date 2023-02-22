import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def decompose_image_to_blocks(img, window_size):
    """ Rearrange img of (N,N) into non-overlapping blocks of (N_blocks,window_size**2).
        Make sure to determine N_blocks from the image size. 
    """
    # ------------------------------------------------------------------------------------------------------------------
    # ASSUMPTION: img.shape is divisible by window size and the image is square
    assert img.shape[0] % window_size == 0 and img.shape[1] == img.shape[1]
    # ------------------------------------------------------------------------------------------------------------------
    img_size = img.shape[0]
    n_blocks = (img_size // window_size) * (img_size // window_size)
    blocks = np.zeros((n_blocks, window_size, window_size))
    for i in range(n_blocks):
        col, row = int((i % (img_size / 8)) * 8), int((i // (img_size / 8)) * 8)
        blocks[i] = img[col:col + window_size, row: row + window_size]
    return blocks.reshape((n_blocks, window_size * window_size))


def rearrange_image_from_blocks(blocks, img_size):
    """ Function to rearrange non-overlapping blocks of (N_blocks,window_size**2) into img (N,N). """
    img = np.zeros(shape=(img_size, img_size))
    for i, block in enumerate(blocks):
        col, row = int((i % (img_size / 8)) * 8), int((i // (img_size / 8)) * 8)
        s = int(np.sqrt(block.shape))
        img[col:col +s, row: row + s] = block.reshape((s, s))
    return img


def DCT2_2D(d, nB):
    """ Function to get 2D DCT basis functions of size (d, d, nB, nB).
        d represents the dimensions of the DCT basis image 
        nB is the size of the non-overlapping blocks per dimension
        Reshape to (d**2, nB**2) to conveniently work with this. 
    """
    # since i, j = 1, ..., n and i * j = nB * nB
    n = nB

    def a_(ml: int) -> float:
        return 1 / np.sqrt(d) if ml == 1 else np.sqrt(2 / n)

    # There is a more pythonic way by generating matrix and using np.multiply (Hadamard product),
    # but this one is nice, since it forms directly from the assignment sheet.
    A = np.zeros(shape=(d, d, nB, nB))
    for i in np.arange(start=1, stop=d + 1, step=1):
        for j in np.arange(start=1, stop=d + 1, step=1):
            for l in np.arange(start=1, stop=nB + 1, step=1):
                for m in np.arange(start=1, stop=nB + 1, step=1):
                    A[i -1, j-1, l-1, m-1] = (
                            a_(l) * a_(m)
                            * np.cos(np.pi / n * (l - 1) * (i - 1 / 2))
                            * np.cos(np.pi / n * (m - 1) * (j - 1 / 2))
                    )

    return A.reshape(d * d, nB * nB)


def DCT2_1D(d, n):
    """ Function to get 1D DCT basis functions of size (d, n)
        n: signal dimension, d: basis functions
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Task 1.3 ---------------------------------------------------------------------------------------------------------
    def a_(j: int) -> float:
        return 1 / np.sqrt(n) if j == 1 else np.sqrt(2 / n)

    # Added + 1 to j an i, since range starts at 0 and the described algorithm starts with 1
    return np.array([[a_(j + 1) * np.cos(np.pi / n * j * (i + 1 / 2)) for j in range(d)] for i in range(n)])


def frank_wolfe(calc_pk, x_k):
    for k in np.arange(start=1, stop=200, step=1):
        p_k = calc_pk(x_k)
        tau_k = 2 / (k + 1)
        x_k = (1 - tau_k) * x_k + tau_k * p_k
    return x_k  # Signal has to be restored in the calling method


def task1(signal):
    """ Signal Denoising

        Requirements for the plots:
            -ax[0,0] - Results for low noise and K=15
            -ax[0,1] - Results for high noise and K=15
            -ax[1,0] - Results for low noise and K=100
            -ax[1,1] - Results for low noise and K=5

    """

    fig, ax = plt.subplots(2, 2, figsize=(16, 8))
    fig.suptitle('Task 1 - Signal denoising task', fontsize=16)

    ax[0, 0].set_title('a)')
    ax[0, 1].set_title('b)')
    ax[1, 0].set_title('c)')
    ax[1, 1].set_title('d)')

    var = (0.01 ** 2, 0.03 ** 2, 0.01 ** 2, 0.01 ** 2)
    K = (15, 15, 100, 5)

    """ Start of your code
    """
    np.random.seed(11703581)  # Set random seed to generate the same results
    plots_axs = [ax[0, 0], ax[0, 1], ax[1, 0], ax[1, 1]]
    for scale, d, ax_ in zip(var, K, plots_axs):
        noisy_signal = signal + np.random.normal(loc=0, scale=scale, size=signal.shape[0])
        A = DCT2_1D(d=d, n=noisy_signal.shape[0])

        def calc_pk(x_k):
            nabla_f = (A.dot(x_k) - noisy_signal).dot(A)
            p_k = np.zeros(shape=x_k.shape)
            p_k[np.argmin(nabla_f)] = 1
            return p_k

        x_k = np.random.randint(low=0, high=100, size=d)
        x_k = x_k / np.sum(x_k)  # -> to unit simplex
        x = frank_wolfe(calc_pk, x_k)
        ax_.plot(signal, label="Signal")
        ax_.plot(noisy_signal, label="Noisy")
        ax_.plot(A.dot(x), label="FW")
        ax_.legend()
    """ End of your code
    """
    return fig


def task2(img):
    """ Image Compression

        Requirements for fig1:
            - ax[0] the groundtruth grayscale image 
            - ax[1] the compressed image from the Frank Wolfe algorithm

        Requirements for fig2:
            - ax[0] the groundtruth grayscale image 
            - ax[1] the compressed image using LASSO and \lambda=0.01
            - ax[2] the compressed image using LASSO and \lambda=0.1
            - ax[3] the compressed image using LASSO and \lambda=1.0

    """

    fig1, ax1 = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    fig1.suptitle('Task 2 - Image compression', fontsize=16)
    ax1[0].set_title('GT')
    ax1[1].set_title('Cond. GD')
    ax1[0].imshow(img, 'gray')
    for ax_ in ax1:
        ax_.set_aspect('equal')
        ax_.get_xaxis().set_visible(False)
        ax_.get_yaxis().set_visible(False)

    blocks = decompose_image_to_blocks(img, window_size=8)
    A = DCT2_2D(d=8, nB=8)
    t = 0.01
    fw_img = np.zeros(shape=blocks.shape)
    for i, block in enumerate(blocks):
        def calc_pk(x):
            nabla_f = x - A.T.dot(block)
            # Since we should not tackle the first one.
            i = np.argmax(np.abs(nabla_f)[1:]) + 1
            e = np.zeros(shape=x.shape)
            e[i] = 1
            return -t * np.multiply(np.sign(nabla_f[i]), e)
        x_k = np.zeros(shape=block.shape)
        x_k = frank_wolfe(calc_pk, x_k=x_k)
        # FW would scale this in each iteration, instead of rewriting the algorithm to ignore the first value,
        # I reassigned the first DC basis and in the function to calculate pk the first DC basis will be ignored.
        x_k[0] = A.T.dot(block)[0]
        fw_img[i] = A.dot(x_k)
    ax1[1].imshow(rearrange_image_from_blocks(fw_img, img_size=256), 'gray')
    # lasso
    lamb_arr = np.array([0.01, 0.1, 1.])

    fig2, ax2 = plt.subplots(1, len(lamb_arr) + 1, sharex=True, sharey=True, figsize=(10, 4))
    plt.suptitle('Task 2 - LASSO', fontsize=16)
    ax2[0].set_title('GT')
    ax2[0].imshow(img, 'gray')
    for ax_ in ax2:
        ax_.set_aspect('equal')
        ax_.get_xaxis().set_visible(False)
        ax_.get_yaxis().set_visible(False)

    for l_idx, l in enumerate(lamb_arr):
        ax2[l_idx + 1].set_title(r'$\lambda$=%.2f' % l)

    """ Start of your code
    """
    for l_idx, l in enumerate(lamb_arr):
        lasso_img = np.zeros(blocks.shape)
        for i, block in enumerate(blocks):
            at_bs = A.T.dot(block)
            lam = l * np.ones(shape=at_bs.shape)
            x_k = np.multiply(np.clip(np.abs(at_bs)-lam, a_min=0, a_max=None), np.sign(at_bs))
            lasso_img[i] = A.dot(x_k)
        ax2[l_idx + 1].imshow(rearrange_image_from_blocks(lasso_img, img_size=256), 'gray')
    """ End of your code
    """
    return fig1, fig2


if __name__ == "__main__":
    # load 1D signal and 2D image
    with np.load('data.npz') as data:
        signal = data['sig']
        img = data['img']

    pdf = PdfPages('figures.pdf')
    fig_signal = task1(signal)
    figures_img = task2(img)

    pdf.savefig(fig_signal)
    for fig in figures_img:
        pdf.savefig(fig)
    pdf.close()
