from skimage.transform import radon, rescale, iradon
import numpy as np
import scipy
import scipy.misc
import pdb

from skimage.io import imread
import matplotlib.pyplot as plt

import torch
import torch.autograd as autograd
import torch.nn.functional as F

# lena = scipy.misc.lena()
obj = imread('/scratch0/ilya/locDoc/data/siim-medical-images/337/ID_0004_AGE_0056_CONTRAST_1_CT.png', as_grey=True)
obj = np.random.rand(256,256)
obj.shape
ang = np.linspace(0., 180., 50, endpoint=False)
proj = radon(obj, theta=ang, circle=False)
pdb.set_trace()
# b 262
rec = iradon(proj, theta=ang, circle=False)


dtype = torch.cuda.FloatTensor
autograd.Variable(torch.from_numpy(winO1.filters).type(dtype))

#torch.nn.functional.grid_sample(input, grid, mode='bilinear', padding_mode='zeros')
def torch_iradon():
    # t, x, radon_filtered[:, i]
    
    
    preG = np.reshape(radon_filtered, (1,1)+radon_filtered.shape)
    radon_filteredG = autograd.Variable(torch.from_numpy(preG).type(dtype))

    ty = t / (radon_filteredG.size(2) // 2)
    txval = (i - radon_filteredG.size(2) // 2) / (radon_filteredG.size(2) // 2)
    tx = np.ones(ty.shape) * txval
    t4dim = np.zeros((1,)+ ty.shape + (2,))
    t4dim[0,:,:,0] = tx
    t4dim[0,:,:,1] = ty
    tG = autograd.Variable(torch.from_numpy(t4dim).type(dtype))

    # bilinear mode is effectively linear since we are not using the 2nd dimension
    backprojectedG = F.grid_sample(radon_filteredG, tG, 'bilinear')
    backprojected_test = np.reshape((backprojectedG.data).cpu().numpy(), ty.shape)

    np.sum(np.abs(backprojected - backprojected_test))
