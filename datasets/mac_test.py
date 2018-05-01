import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from skimage.transform import radon, rescale, iradon
import pdb
import scipy

from scipy.fftpack import fftfreq
from numpy.fft import fft, ifft
from PIL import Image
from skimage.io import imread

obj = imread('/Applications/MATLAB_R2017b.app/toolbox/images/imdata/cameraman.tif', as_grey=True)
ang = np.linspace(0., 180., 50, endpoint=False)
proj = radon(obj, theta=ang, circle=False)
radon_image = proj

# projection_size_padded = max(64, int(2 ** np.ceil(np.log2(2 * radon_image.shape[0]))))
# #projection_size_padded = 1024
# pad_width = ((0, projection_size_padded - radon_image.shape[0]), (0, 0))
# img = np.pad(radon_image, pad_width, mode='constant', constant_values=0)

# Construct the Fourier filter
projection_size = radon_image.shape[0]
f = fftfreq(projection_size).reshape(-1, 1)   # digital frequency
omega = 2 * np.pi * f                                # angular frequency
fourier_filter = 2 * np.abs(f)                       # ramp filter

# Apply filter in Fourier domain
# projection = fft(radon_image, axis=0) * fourier_filter
# radon_filtered = np.real(ifft(projection, axis=0))


time_filter = np.fft.fftshift(fft(fourier_filter, axis=0).real) / len(fourier_filter)
sig = radon_image[:,0]

fourier_res = ifft(fft(sig) * fourier_filter[:,0]).real
time_res = np.convolve(np.concatenate([sig,sig]), np.flipud(time_filter[:,0]),'same')

first_half_projection_size = projection_size // 2
time_res_best = np.fft.fftshift(time_res[first_half_projection_size:(first_half_projection_size+projection_size)])

plt.plot(fourier_res-time_res_best)
plt.show()
