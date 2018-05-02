import numpy as np
from scipy.fftpack import fftfreq
from numpy.fft import fft, ifft, fftshift

import torch
import torch.autograd as autograd
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import pdb

dtype = torch.cuda.FloatTensor

def fourier_fbp_filter(H_in, W_in, filter_type='ramp'):
    f = fftfreq(H_in).reshape(-1, 1)   # digital frequency
    fourier_filter = 2 * np.abs(f)     # ramp filter
    omega = 2 * np.pi * f 
    if filter_type == 'ramp':
        pass
    elif filter_type == "shepp-logan":
        fourier_filter *= np.sinc(omega/(2*np.pi))
    elif filter_type == "cosine":
        fourier_filter *= np.cos(omega)
    elif filter_type == "hamming":
        fourier_filter *= (0.54 + 0.46 * np.cos(omega / 2))
    elif filter_type == "hann":
        fourier_filter *= (1 + np.cos(omega / 2)) / 2
    elif filter_type == 'none':
        fourier_filter[:] = 1
    elif filter_type == "shepp-logan-double": # seen in rice
        fourier_filter *= np.sinc(omega/(np.pi))
    elif filter_type == "rand":
        fourier_filter = np.random.rand(H_in,1) / W_in
    else:
        error('invalid filter type requests')
    return fourier_filter    

class InvRadonLayer(torch.nn.Module):
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
        fourier_filter = fourier_fbp_filter(H_in, W_in, filter_type)
        time_filter = fftshift(ifft(fourier_filter, axis=0).real)
        time_filterr = time_filter.tolist()
        time_filterr.reverse()
        time_filterr = np.array(time_filterr)

        preG = np.reshape(time_filterr, (1,1,len(time_filterr),1))
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
        # pad
        radon_image_paddedG = torch.cat([radon_imageG, radon_imageG, radon_imageG], dim=2)
        # filter
        radon_padded_filteredG = F.conv2d(radon_image_paddedG, self.hG)
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
