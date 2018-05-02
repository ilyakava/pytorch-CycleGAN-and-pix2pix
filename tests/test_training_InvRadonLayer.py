import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skimage.transform import radon, rescale, iradon
import numpy as np
import scipy
import scipy.misc
from scipy.fftpack import fftfreq
from numpy.fft import fft, ifft, fftshift, ifftshift
from warnings import warn
import pdb

from PIL import Image

from skimage.io import imread
import matplotlib.pyplot as plt

import torch
import torch.autograd as autograd
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from models.inv_radon_layer import InvRadonLayer

from tqdm import tqdm

import random
import visdom
vis = visdom.Visdom()

dtype = torch.cuda.FloatTensor
DATA_DIR = '/scratch0/ilya/locDoc/data/siim-medical-images/pix2pix_50_views/B/train/'
# projs_file = None or 

nang = 50;
ang = np.linspace(0., 180., nang, endpoint=False)
infiles = os.listdir(DATA_DIR)
eg_obj = imread(DATA_DIR + infiles[0])
eg_proj = radon(eg_obj, theta=ang, circle=False)
D_out = eg_obj.shape[0]
H_in, W_in = eg_proj.shape
N = len(infiles)

load_data = 1

if load_data == 0:
    projs = np.zeros([N, 1, H_in, W_in])
    ims = np.zeros([N, 1, D_out, D_out])

    for i in tqdm(range(N)):
        filename = infiles[i]
        obj = imread(DATA_DIR + filename) / 255.0
        proj = radon(obj, theta=ang, circle=False)
        projs[i,0,:,:] = proj
        ims[i,0,:,:] = obj
    #projs = (projs - np.min(projs)) / (np.max(projs) - np.min(projs))

    np.save('/scratch0/ilya/locDoc/data/siim-medical-images/85projs.npy', projs)
    np.save('/scratch0/ilya/locDoc/data/siim-medical-images/85ims.npy', ims)
else:
    projs = np.load('/scratch0/ilya/locDoc/data/siim-medical-images/85projs.npy')
    ims = np.load('/scratch0/ilya/locDoc/data/siim-medical-images/85ims.npy')

last_itr_visuals = []
losses = []

# Create random Tensors to hold inputs and outputs
x = autograd.Variable(torch.from_numpy(projs).type(dtype), requires_grad=True)
y = autograd.Variable(torch.from_numpy(ims).type(dtype), requires_grad=False)

# Construct our model by instantiating the class defined above
model = InvRadonLayer(H_in, W_in, D_out)

# Construct our loss function and an Optimizer. Training this strange model with
# vanilla stochastic gradient descent is tough, so we use momentum
criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=4e-14, momentum=0.9)
for t in range(500):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x)

    # Compute and print loss
    loss = criterion(y_pred, y)
    losses.append(loss.data[0])
    if t > -1:
        while len(last_itr_visuals) > 0:
            visual = last_itr_visuals.pop()
            vis.close(visual)
        
        last_itr_visuals.append(vis.line(np.array(losses), opts={'title': 'Losses for %i itrs' % t}))
        params = list(model.parameters())[0].data
        time_filter = params.cpu().numpy()[0,0,:,0]
        last_itr_visuals.append(vis.line(time_filter, opts={'title': 'Time filter'}))
        last_itr_visuals.append(vis.line(fftshift(fft(ifftshift(time_filter)).real), opts={'title': 'Fourier filter'}))
        k = 8#random.randint(0,N-1)
        y_predC = (y_pred.data).cpu().numpy()
        pred_img = y_predC[k,0,:,:]
        pred_img[pred_img < 0] = 0
        last_itr_visuals.append(vis.image(pred_img))
        last_itr_visuals.append(vis.image(ims[k,0,:,:], opts={'title': 'gt image k=%i' % k}))
        last_itr_visuals.append(vis.heatmap(np.flipud(y_predC[k,0,:,:]), opts={'title': 'pred heatmap'}))

    print(t, loss.data[0])

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
