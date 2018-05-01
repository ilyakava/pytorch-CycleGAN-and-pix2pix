from __future__ import print_function, division
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import numpy as np
import sys

from PIL import Image

from skimage.io import imread
from skimage import data_dir
from skimage.transform import radon, rescale, iradon

from tqdm import tqdm

import pdb


def write_img(filename):
    obj = imread(DATA_DIR + filename, as_grey=True)
    proj = radon(obj, theta=ang, circle=False)

    rec = iradon(proj, theta=ang, circle=False)
    im = Image.fromarray(rec).convert('RGB')
    im.save(OUT_DIR + filename)

def main():
    global DATA_DIR
    global OUT_DIR
    global ang
    DATA_DIR = '/vulcan/scratch/snanduri/iradon/data/ellipsoids_large/png/'
    OUT_DIR = '/vulcan/scratch/snanduri/iradon/data/ellipsoids_large/50_views_6/'
    NANG = 50
    ang = np.linspace(0., 180., NANG, endpoint=False)
    img_list = os.listdir(DATA_DIR)
    len_img_list = len(img_list)
    num_parts = 6
    len_img_list_small = len_img_list//num_parts
    part_index = int(sys.argv[1]) #max = num_parts-1
    img_list_small = img_list[part_index*len_img_list_small+20000:(part_index+1)*len_img_list_small]
    with ThreadPoolExecutor() as executor:
        futures = executor.map(write_img, img_list_small)
    #    kwargs = {'total': len(img_list),'unit': 'write','unit_scale': True,'leave': True}
    #    for f in tqdm(as_completed(futures), **kwargs):
    #        pass

if __name__ == '__main__':
    main()
    
