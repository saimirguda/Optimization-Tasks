import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def decompose_image_to_blocks(img, window_size):
    """ Rearrange img of (N,N) into non-overlapping blocks of (N_blocks,window_size**2).
        Make sure to determine N_blocks from the image size. 
    """
    pass

def rearrange_image_from_blocks(blocks, img_size):
    """ Function to rearrange non-overlapping blocks of (N_blocks,window_size**2) into img (N,N). """
    pass

def DCT2_2D(d, nB):
    """ Function to get 2D DCT basis functions of size (d, d, nB, nB).
        d represents the dimensions of the DCT basis image 
        nB is the size of the non-overlapping blocks per dimension
        Reshape to (d**2, nB**2) to conveniently work with this. 
    """
    pass

def DCT2_1D(d, n):
    """ Function to get 1D DCT basis functions of size (d, n)
        n: signal dimension, d: basis functions 
    """
    pass

def task1(signal):
    """ Signal Denoising

        Requirements for the plots:
            -ax[0,0] - Results for low noise and K=15
            -ax[0,1] - Results for high noise and K=15
            -ax[1,0] - Results for low noise and K=100
            -ax[1,1] - Results for low noise and K=5

    """

    fig, ax = plt.subplots(2,2, figsize=(16,8))
    fig.suptitle('Task 1 - Signal denoising task', fontsize=16)

    ax[0,0].set_title('a)')
    ax[0,1].set_title('b)')
    ax[1,0].set_title('c)')
    ax[1,1].set_title('d)')

    
    var = (0.01**2, 0.03**2, 0.01**2, 0.01**2)
    K = (15, 15, 100, 5)

    """ Start of your code
    """
    



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

    fig1, ax1 = plt.subplots(1,2,figsize=(10,5),sharex=True,sharey=True)
    fig1.suptitle('Task 2 - Image compression', fontsize=16)
    ax1[0].set_title('GT')
    ax1[1].set_title('Cond. GD')
    ax1[0].imshow(img,'gray')
    for ax_ in ax1:
        ax_.set_aspect('equal')
        ax_.get_xaxis().set_visible(False)
        ax_.get_yaxis().set_visible(False)

    # lasso
    lamb_arr = np.array([0.01,0.1,1.])

    fig2, ax2 = plt.subplots(1,len(lamb_arr)+1,sharex=True,sharey=True,figsize=(10,4))
    plt.suptitle('Task 2 - LASSO', fontsize=16)
    ax2[0].set_title('GT')
    ax2[0].imshow(img,'gray')
    for ax_ in ax2:
        ax_.set_aspect('equal')
        ax_.get_xaxis().set_visible(False)
        ax_.get_yaxis().set_visible(False)

    for l_idx, l in enumerate(lamb_arr):
        ax2[l_idx+1].set_title(r'$\lambda$=%.2f' %l)

    """ Start of your code
    """
   


    
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
