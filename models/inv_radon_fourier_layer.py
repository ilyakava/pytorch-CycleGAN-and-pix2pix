import numpy as np
from scipy.fftpack import fftfreq
from numpy.fft import fft, ifft, fftshift

import torch
import torch.autograd as autograd
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import pdb

import pytorch_fft.fft.autograd as torchfft
fft2d = fft.Fft2d()
ifft2d = fft.Ifft2d()

dtype = torch.cuda.FloatTensor

class InvRadonFourierLayer(torch.nn.Module):
    def __init__(self, H_in, W_in, D_out, filter_type='ramp'):
        super(InvRadonLayer, self).__init__()
        self.W_in = W_in
        self.D_out = D_out
        self.H_in = H_in
        self.hH_in = H_in // 2
        # assumes output is square
        # assumes W_in is number angles and angles were generated this way
        theta = np.linspace(0., 180., W_in, endpoint=False)
        th = (np.pi / 180.0) * theta
        # Construct the filter in Fourier domain
        if filter_type == 'ramp':
            f = fftfreq(H_in).reshape(-1, 1)   # digital frequency
            fourier_filter = 2 * np.abs(f)     # ramp filter
        else:
            fourier_filter = np.random.rand(H_in)

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
        pdb.set_trace()
        radon_filteredG = ifft2d(fft2d(radon_imageG) * self.hG)
        # filter
        radon_filteredG = F.conv2d(radon_image_paddedG, self.hG)
        # unpad
        radon_filteredG = radon_padded_filteredG[:,:,(self.hH_in+1):(self.hH_in+self.H_in+1),:]

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
