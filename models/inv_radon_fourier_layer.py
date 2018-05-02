import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.fftpack import fftfreq
from numpy.fft import fft, ifft, fftshift

import torch
import torch.autograd as autograd
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import pdb

from models.inv_radon_layer import fourier_fbp_filter

import pytorch_fft.fft.autograd as torchfft
fft2d = torchfft.Fft2d()
ifft2d = torchfft.Ifft2d()

dtype = torch.cuda.FloatTensor

class InvRadonFourierLayer(torch.nn.Module):
    def __init__(self, H_in, W_in, D_out, filter_type='ramp'):
        super(InvRadonFourierLayer, self).__init__()
        self.W_in = W_in
        self.D_out = D_out
        self.H_in = H_in
        self.hH_in = H_in // 2
        # assumes output is square
        # assumes W_in is number angles and angles were generated this way
        theta = np.linspace(0., 180., W_in, endpoint=False)
        th = (np.pi / 180.0) * theta
        # Construct the filter in Fourier domain
        fourier_filter = fourier_fbp_filter(H_in, filter_type)
        
        preG = np.reshape(fourier_filter, (1,1,len(fourier_filter),1))
        # this will be learned, initialized to ramp filter
        self.hG = Parameter(torch.from_numpy(preG).type(dtype))

        [X, Y] = np.mgrid[0:D_out, 0:D_out]
        xpr = X - int(D_out) // 2
        ypr = Y - int(D_out) // 2

        # prepare interpolation grids
        t4dim = np.zeros((self.W_in, 1, D_out, D_out, 2))
        for i in range(self.W_in):
            t = ypr * np.cos(th[i]) - xpr * np.sin(th[i])
            
            ty = t / (H_in // 2)
            txval = -1 + i*(2 / (W_in-1))
            tx = np.ones([D_out,D_out]) * txval
            t4dim[i,0,:,:,0] = tx
            t4dim[i,0,:,:,1] = ty
        # this is a constant
        self.tG = autograd.Variable(torch.from_numpy(t4dim).type(dtype))

    def forward(self, radon_imageG):
        radon_imageG_i = autograd.Variable(torch.zeros(radon_imageG.size()).type(dtype))
        radon_imageG_fft_r, radon_imageG_fft_i = fft2d(radon_imageG, radon_imageG_i)
        radon_imageG_fft_r *= self.hG
        radon_imageG_fft_i *= self.hG
        radon_filteredG, _ = ifft2d(radon_imageG_fft_r, radon_imageG_fft_i)

        # accumulator
        N_in = radon_imageG.size(0)
        preG = np.zeros((N_in,1, self.D_out, self.D_out))
        reconstructedG = autograd.Variable(torch.from_numpy(preG).type(dtype))
        # we need to extend the parameters of the grid sampling once per input size
        trepeatedG = self.tG.repeat(1,N_in,1,1,1)
        # accumulate
        for i in range(self.W_in):
            # bilinear mode is effectively linear since we are not using the 2nd dimension
            reconstructedG += F.grid_sample(radon_filteredG, trepeatedG[i,:,:,:,:], 'bilinear') # one backprojection

        return reconstructedG * np.pi / (2 * self.W_in)
