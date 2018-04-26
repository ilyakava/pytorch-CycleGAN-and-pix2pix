# I run this particular file in my venvconda environment

import numpy as np
from scipy.misc import toimage
from tqdm import tqdm
import odl
import pdb

OUT_DIR = '/scratch0/ilya/locDoc/data/ellipsoids/png/'

def random_ellipse():
	# from: https://github.com/adler-j/learned_gradient_tomography/blob/72c2dee62d4c619710b5a62cd04b33bf75635287/code/util.py
    return ((np.random.rand() - 0.3) * np.random.exponential(0.3),
            np.random.exponential() * 0.2, np.random.exponential() * 0.2,
            np.random.rand() - 0.5, np.random.rand() - 0.5,
            np.random.rand() * 2 * np.pi)


def random_phantom(spc):
	# from: https://github.com/adler-j/learned_gradient_tomography/blob/72c2dee62d4c619710b5a62cd04b33bf75635287/code/util.py
    n = np.random.poisson(100)
    ellipses = [random_ellipse() for _ in range(n)]
    return odl.phantom.ellipsoid_phantom(spc, ellipses)

size = 512
space = odl.uniform_discr([-64, -64], [64, 64], [size, size], dtype='float32')

for i in tqdm(range(700)):
	arr = random_phantom(space)
	toimage(arr).save(OUT_DIR + ('%03d.png' % i))
