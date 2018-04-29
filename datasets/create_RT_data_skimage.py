from __future__ import print_function, division
import os
import numpy as np

from PIL import Image

from skimage.io import imread
from skimage import data_dir
from skimage.transform import radon, rescale, iradon

from tqdm import tqdm

import pdb

DATA_DIR = '/scratch0/public/rsna-bone-age/512/train/'
OUT_DIR = '/scratch0/public/rsna-bone-age/50_views/'
NANG = 50

ang = np.linspace(0., 180., NANG, endpoint=False)

for filename in tqdm(os.listdir(DATA_DIR)):
    obj = imread(DATA_DIR + filename, as_grey=True)
    proj = radon(obj, theta=ang, circle=False)

    rec = iradon(proj, theta=ang, circle=False)
    im = Image.fromarray(rec).convert('RGB')
    im.save(OUT_DIR + filename)
    