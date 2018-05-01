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


dtype = torch.cuda.FloatTensor

#torch.nn.functional.grid_sample(input, grid, mode='bilinear', padding_mode='zeros')

# egs in:
# http://pytorch.org/tutorials/beginner/pytorch_with_examples.html#pytorch-custom-nn-modules
# http://pytorch.org/docs/master/_modules/torch/nn/modules/linear.html

class InvRadonLayer(torch.nn.Module):
    def __init__(self, H_in, W_in, D_out):
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
        f = fftfreq(H_in).reshape(-1, 1)   # digital frequency
        fourier_filter = 2 * np.abs(f)     # ramp filter
        time_filter = fftshift(ifft(fourier_filter, axis=0).real)
        time_filterr = time_filter.tolist()
        time_filterr.reverse()
        time_filterr = np.array(time_filterr)

        preG = np.reshape(time_filterr, (1,1,len(time_filterr),1))
        # this will be learned, initialized to ramp filter
        self.hG = autograd.Variable(torch.from_numpy(preG).type(dtype))

        [X, Y] = np.mgrid[0:D_out, 0:D_out]
        xpr = X - int(D_out) // 2
        ypr = Y - int(D_out) // 2

        # prepare interpolation grids
        t4dim = np.zeros((self.W_in, 1, D_out, D_out, 2))
        for i in range(self.W_in):
            t = ypr * np.cos(th[i]) - xpr * np.sin(th[i])
            
            ty = t / (H_in // 2)
            txval = -1 + i*(2 / (W_in-1)) # (i - radon_filteredG.size(3) / 2) / (radon_filteredG.size(3) / 2)
            tx = np.ones([D_out,D_out]) * txval
            t4dim[i,0,:,:,0] = tx
            t4dim[i,0,:,:,1] = ty
        # this is a constant
        self.tG = autograd.Variable(torch.from_numpy(t4dim).type(dtype))

    def forward(self, radon_imageG):
        # this works only if 1st dimension of input is 1
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

def torch_iradon(radon_image, theta, output_size=None, filter="ramp", circle=False):
    # for comparrison

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
    H_in = radon_image.shape[0] #max(64, int(2 ** np.ceil(np.log2(2 * radon_image.shape[0]))))
    # pad_width = ((0, H_in - radon_image.shape[0]), (0, 0))
    # img = np.pad(radon_image, pad_width, mode='constant', constant_values=0)

    # Construct the Fourier filter
    f = fftfreq(H_in).reshape(-1, 1)   # digital frequency
    fourier_filter = 2 * np.abs(f)                       # ramp filter

    # Apply filter in Fourier domain
    # projection = fft(img, axis=0) * fourier_filter
    # radon_filtered = np.real(ifft(projection, axis=0))

    # just compare here

    # Resize filtered image back to original size
    # radon_filtered = radon_filtered[:radon_image.shape[0], :]

    ####################

    preG = np.reshape(radon_image, (1,1)+radon_image.shape)
    radon_imageG = autograd.Variable(torch.from_numpy(preG).type(dtype))

    radon_image_paddedG = torch.cat([radon_imageG, radon_imageG, radon_imageG], dim=2)

    time_filter = fftshift(ifft(fourier_filter, axis=0).real)
    time_filterr = time_filter.tolist()
    time_filterr.reverse()
    time_filterr = np.array(time_filterr)

    preG = np.reshape(time_filterr, (1,1,len(time_filterr),1))
    hG = autograd.Variable(torch.from_numpy(preG).type(dtype))

    radon_padded_filteredG = F.conv2d(radon_image_paddedG, hG)
    rilen = radon_image.shape[0]
    rihlen = radon_image.shape[0] // 2
    
    radon_filteredG = radon_padded_filteredG[:,:,(rihlen+1):(rihlen+rilen+1),:]

    # #########################################

    [X, Y] = np.mgrid[0:output_size, 0:output_size]
    xpr = X - int(output_size) // 2
    ypr = Y - int(output_size) // 2

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

def test_torch_iradon():
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
    # plt.imshow(reconstructed[0,0,:,:], cmap='gray')
    # plt.show()
    # plt.imshow(rec2, cmap='gray')
    # plt.show()

    rec2n = (rec2-np.min(rec2))/(np.max(rec2)-np.min(rec2)) * 255.0
    rec3n = (rec3-np.min(rec3))/(np.max(rec3)-np.min(rec3)) * 255.0
    showme = plt.imshow(rec2n - rec3n)
    plt.colorbar(showme)
    plt.show()
    
    pdb.set_trace()

    #     radon_padded_filtered_test = (radon_padded_filteredG.data).cpu().numpy()
    # radon_filtered_test = (radon_filteredG.data).cpu().numpy()

if __name__ == '__main__':
    obj = imread('/scratch0/ilya/locDoc/data/siim-medical-images/337/ID_0004_AGE_0056_CONTRAST_1_CT.png', as_grey=True)
    obj2 = imread('/scratch0/ilya/locDoc/data/siim-medical-images/337/ID_0000_AGE_0060_CONTRAST_1_CT.png', as_grey=True)

    #obj = np.random.rand(256,256)
    nang = 50;
    ang = np.linspace(0., 180., nang, endpoint=False)
    proj = np.expand_dims(radon(obj, theta=ang, circle=False), axis=0)
    proj2 = np.expand_dims(radon(obj2, theta=ang, circle=False), axis=0)

    projs = np.stack([proj, proj2])

    # recG = torch_iradon(proj, theta=ang, circle=False)
    # rec = (recG.data).cpu().numpy()
    # plt.imshow(rec[0,0,:,:])
    # plt.show()

    projG = autograd.Variable(torch.from_numpy(projs).type(dtype))

    # test InvRadonLayer
    m = InvRadonLayer(proj.shape[1], nang, obj.shape[0])

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
