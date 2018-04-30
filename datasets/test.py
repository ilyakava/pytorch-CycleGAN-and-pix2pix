from skimage.transform import radon, rescale, iradon
import numpy as np
import scipy
import scipy.misc
from scipy.fftpack import fft, ifft, fftfreq, rfft, irfft
from warnings import warn
import pdb

from PIL import Image

from skimage.io import imread
import matplotlib.pyplot as plt

import torch
import torch.autograd as autograd
import torch.nn.functional as F


dtype = torch.cuda.FloatTensor

#torch.nn.functional.grid_sample(input, grid, mode='bilinear', padding_mode='zeros')
def torch_iradon(radon_image, theta, output_size=None,
           filter="ramp", circle=False):


    if len(theta) != radon_image.shape[1]:
        raise ValueError("The given ``theta`` does not match the number of "
                         "projections in ``radon_image``.")

    if not output_size:
        warn('estimating output size')
        # If output size not specified, estimate from input radon image
        output_size = int(np.floor(np.sqrt((radon_image.shape[0]) ** 2 / 2.0)))

    th = (np.pi / 180.0) * theta

    # resize image to next power of two (but no less than 64) for
    # Fourier analysis; speeds up Fourier and lessens artifacts
    projection_size_padded = max(64, int(2 ** np.ceil(np.log2(2 * radon_image.shape[0]))))
    pad_width = ((0, projection_size_padded - radon_image.shape[0]), (0, 0))
    img = np.pad(radon_image, pad_width, mode='constant', constant_values=0)

    # Construct the Fourier filter
    f = fftfreq(projection_size_padded).reshape(-1, 1)   # digital frequency
    omega = 2 * np.pi * f                                # angular frequency
    fourier_filter = 2 * np.abs(f)                       # ramp filter

    # Apply filter in Fourier domain
    projection = fft(img, axis=0) * fourier_filter
    radon_filtered = np.real(ifft(projection, axis=0))

    # just compare here

    # Resize filtered image back to original size
    radon_filtered = radon_filtered[:radon_image.shape[0], :]
    
    # pdb.set_trace()
    ####################

    preG = np.reshape(img, (1,1)+img.shape)
    radon_imageG = autograd.Variable(torch.from_numpy(preG).type(dtype))

    radon_image_paddedG = torch.cat([radon_imageG, radon_imageG], dim=2)


    mylist = (np.array(range(radon_image.shape[0])) - ((radon_image.shape[0] - 1) / 2)) / ((radon_image.shape[0] - 1) / 2)
    fourier_filter = 1 - np.abs(mylist)

    time_filter = (irfft(fourier_filter, axis=0).real)  # np.fft.fftshift
    # time_filter2 = time_filter.tolist()
    # time_filter2.reverse()
    # time_filter2 = np.array(time_filter2)
    #time_filter_padded = np.concatenate([time_filter, np.zeros((len(time_filter)-1))])
    preG = np.reshape(time_filter, (1,1,len(time_filter),1))
    hG = autograd.Variable(torch.from_numpy(preG).type(dtype))

    radon_padded_filteredG = F.conv2d(radon_image_paddedG, hG)
    radon_filteredG = radon_padded_filteredG[:,:,:radon_image.shape[0],:]

    radon_padded_filtered_test = (radon_padded_filteredG.data).cpu().numpy()
    radon_filtered_test = (radon_filteredG.data).cpu().numpy()

    # pdb.set_trace()

    # #########################################

    [X, Y] = np.mgrid[0:output_size, 0:output_size]
    xpr = X - int(output_size) // 2
    ypr = Y - int(output_size) // 2

    #preG = np.reshape(radon_filtered, (1,1)+radon_filtered.shape)
    #radon_filteredG = autograd.Variable(torch.from_numpy(preG).type(dtype))

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

    return reconstructedG * np.pi / (2 * len(th))

if __name__ == '__main__':
    obj = imread('/scratch0/ilya/locDoc/data/siim-medical-images/337/ID_0004_AGE_0056_CONTRAST_1_CT.png', as_grey=True)
    #obj = np.random.rand(256,256)
    ang = np.linspace(0., 180., 50, endpoint=False)
    proj = radon(obj, theta=ang, circle=False)
    # b 262
    # projG = 
    rec = torch_iradon(proj, theta=ang, circle=False)
    rec2 = iradon(proj, theta=ang, circle=False)
    reconstructed = (rec.data).cpu().numpy() # and get rid of first two dimensions
    rec3 = reconstructed[0,0,:,:]
    im = Image.fromarray(rec3).convert('RGB')
    im.save('/cfarhomes/ilyak/Desktop/time.png')
    plt.imshow(reconstructed[0,0,:,:], cmap='gray')
    plt.show()
    plt.imshow(rec2, cmap='gray')
    plt.show()
    pdb.set_trace()
    # rec4 = (rec3-np.min(rec3))/(np.max(rec3)-np.min(rec3)) * 255.0

