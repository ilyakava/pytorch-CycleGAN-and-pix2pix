import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skimage.transform import radon, rescale, iradon
import numpy as np
import scipy
import scipy.misc
from scipy.fftpack import fftfreq
from numpy.fft import fft, ifft, fftshift
from warnings import warn
import pdb

from PIL import Image

from skimage.io import imread
import matplotlib.pyplot as plt

import torch
import torch.autograd as autograd
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from models.inv_radon_fourier_layer import InvRadonFourierLayer

dtype = torch.cuda.FloatTensor

if __name__ == '__main__':
    obj = imread('/scratch0/ilya/locDoc/data/siim-medical-images/337/ID_0004_AGE_0056_CONTRAST_1_CT.png', as_grey=True)
    obj2 = imread('/scratch0/ilya/locDoc/data/siim-medical-images/337/ID_0000_AGE_0060_CONTRAST_1_CT.png', as_grey=True)

    #obj = np.random.rand(256,256)
    nang = 50;
    ang = np.linspace(0., 180., nang, endpoint=False)
    proj = np.expand_dims(radon(obj, theta=ang, circle=False), axis=0)
    proj2 = np.expand_dims(radon(obj2, theta=ang, circle=False), axis=0)

    projs = np.stack([proj, proj2])

    projG = autograd.Variable(torch.from_numpy(projs).type(dtype))

    # test InvRadonLayer
    m = InvRadonFourierLayer(proj.shape[1], nang, obj.shape[0])

    fpbG = m(projG)
    fbp = (fpbG.data).cpu().numpy()


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5))
    ax1.imshow(obj, cmap=plt.cm.Greys_r)
    ax2.imshow(fbp[0,0,:,:], cmap=plt.cm.Greys_r)
    fig.tight_layout()
    plt.show()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5))
    ax1.imshow(obj2, cmap=plt.cm.Greys_r)
    ax2.imshow(fbp[1,0,:,:], cmap=plt.cm.Greys_r)
    fig.tight_layout()
    plt.show()

    pdb.set_trace()
