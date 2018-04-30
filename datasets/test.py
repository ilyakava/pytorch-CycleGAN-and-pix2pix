from skimage.transform import radon, rescale, iradon
import numpy as np
import scipy
import scipy.misc
from scipy.fftpack import fft, ifft, fftfreq
import pdb

from skimage.io import imread
import matplotlib.pyplot as plt

import torch
import torch.autograd as autograd
import torch.nn.functional as F


dtype = torch.cuda.FloatTensor

#torch.nn.functional.grid_sample(input, grid, mode='bilinear', padding_mode='zeros')
def torch_iradon(radon_image, theta=None, output_size=None,
           filter="ramp", interpolation="linear", circle=None):
    if radon_image.ndim != 2:
        raise ValueError('The input image must be 2-D')
    if theta is None:
        m, n = radon_image.shape
        theta = np.linspace(0, 180, n, endpoint=False)
    else:
        theta = np.asarray(theta)
    if len(theta) != radon_image.shape[1]:
        raise ValueError("The given ``theta`` does not match the number of "
                         "projections in ``radon_image``.")
    interpolation_types = ('linear', 'nearest', 'cubic')
    if interpolation not in interpolation_types:
        raise ValueError("Unknown interpolation: %s" % interpolation)
    if not output_size:
        # If output size not specified, estimate from input radon image
        if circle:
            output_size = radon_image.shape[0]
        else:
            output_size = int(np.floor(np.sqrt((radon_image.shape[0]) ** 2
                                               / 2.0)))
    if circle is None:
        warn('The default of `circle` in `skimage.transform.iradon` '
             'will change to `True` in version 0.15.')
        circle = False
    if circle:
        radon_image = _sinogram_circle_to_square(radon_image)

    th = (np.pi / 180.0) * theta
    # resize image to next power of two (but no less than 64) for
    # Fourier analysis; speeds up Fourier and lessens artifacts
    projection_size_padded = \
        max(64, int(2 ** np.ceil(np.log2(2 * radon_image.shape[0]))))
    pad_width = ((0, projection_size_padded - radon_image.shape[0]), (0, 0))
    img = np.pad(radon_image, pad_width, mode='constant', constant_values=0)

    # Construct the Fourier filter
    f = fftfreq(projection_size_padded).reshape(-1, 1)   # digital frequency
    omega = 2 * np.pi * f                                # angular frequency
    fourier_filter = 2 * np.abs(f)                       # ramp filter

    # Apply filter in Fourier domain
    projection = fft(img, axis=0) * fourier_filter
    radon_filtered = np.real(ifft(projection, axis=0))

    # Resize filtered image back to original size
    radon_filtered = radon_filtered[:radon_image.shape[0], :]
    reconstructed = np.zeros((output_size, output_size))
    # Determine the center of the projections (= center of sinogram)
    mid_index = radon_image.shape[0] // 2

    # #########################################

    [X, Y] = np.mgrid[0:output_size, 0:output_size]
    xpr = X - int(output_size) // 2
    ypr = Y - int(output_size) // 2

    preG = np.reshape(radon_filtered, (1,1)+radon_filtered.shape)
    radon_filteredG = autograd.Variable(torch.from_numpy(preG).type(dtype))

    preG = np.zeros((1,1, output_size, output_size))
    reconstructedG = autograd.Variable(torch.from_numpy(preG).type(dtype))


    for i in range(len(theta)):
        t = ypr * np.cos(th[i]) - xpr * np.sin(th[i])

        ty = t / (radon_filteredG.size(2) // 2)
        txval = -1 + i*(2 / (radon_filteredG.size(3)-1)) # (i - radon_filteredG.size(3) / 2) / (radon_filteredG.size(3) / 2)
        tx = np.ones(ty.shape) * txval
        t4dim = np.zeros((1,)+ ty.shape + (2,))
        t4dim[0,:,:,0] = tx
        t4dim[0,:,:,1] = ty
        tG = autograd.Variable(torch.from_numpy(t4dim).type(dtype))

        # bilinear mode is effectively linear since we are not using the 2nd dimension
        backprojectedG = F.grid_sample(radon_filteredG, tG, 'bilinear')
        
        reconstructedG += backprojectedG

    backprojected_test = np.reshape((reconstructedG.data).cpu().numpy(), ty.shape)
    pdb.set_trace()
    
    np.sum(np.abs(backprojected - backprojected_test))

if __name__ == '__main__':
    obj = imread('/scratch0/ilya/locDoc/data/siim-medical-images/337/ID_0004_AGE_0056_CONTRAST_1_CT.png', as_grey=True)
    #obj = np.random.rand(256,256)
    obj.shape
    ang = np.linspace(0., 180., 50, endpoint=False)
    proj = radon(obj, theta=ang, circle=False)
    # b 262
    rec = torch_iradon(proj, theta=ang, circle=False)