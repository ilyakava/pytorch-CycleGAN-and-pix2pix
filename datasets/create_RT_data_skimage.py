from __future__ import print_function, division
import os
import numpy as np

from PIL import Image

from skimage.io import imread
from skimage import data_dir
from skimage.transform import radon, rescale, iradon

from tqdm import tqdm

import pdb

DATA_DIR = '/scratch0/public/siim-medical-images/337/'
# OUT_DIR = '/scratch0/public/rsna-bone-age/50_views/'
NPY_OUT_DIR = '/scratch0/public/siim-medical-images/50_views_npys_fixed/'
NANG = 50

ang = np.linspace(0., 180., NANG, endpoint=False)

for filename in tqdm(os.listdir(DATA_DIR)):
    obj = imread(DATA_DIR + filename) / 255.0
    proj = radon(obj, theta=ang, circle=False)
    np.save(NPY_OUT_DIR + filename, proj)

    #rec = iradon(proj, theta=ang, circle=False)
    #im = Image.fromarray(rec).convert('RGB')
    #im.save(OUT_DIR + filename)
    
