from skimage.transform import radon, rescale, iradon
import numpy as np
import scipy.misc
import pdb

lena = scipy.misc.lena()
obj = lena[:256, :256]
obj.shape
ang = np.linspace(0., 180., 50, endpoint=False)
proj = radon(obj, theta=ang, circle=False)
pdb.set_trace()
rec = iradon(proj, theta=ang, circle=False)


