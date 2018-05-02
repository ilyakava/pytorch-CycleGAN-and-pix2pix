# coppies the npy files of RT sinograms in the same pattern as A and B
# train/test/val sets were made by train_test_pix2pix_split.py

import os
import random as r

import pdb

A_SRC_DIR = '/scratch0/public/siim-medical-images/pix2pix_premerge/A'
C_DEST_DIR = '/scratch0/public/siim-medical-images/pix2pix_premerge/C'
C_SRC_DIR = '/scratch0/public/siim-medical-images/50_views_npys_fixed'

train = os.listdir('%s/train' % A_SRC_DIR)
val = os.listdir('%s/val' % A_SRC_DIR)
test = os.listdir('%s/test' % A_SRC_DIR)

os.system('mkdir -p %s/train' % C_DEST_DIR)
os.system('mkdir -p %s/test' % C_DEST_DIR)
os.system('mkdir -p %s/val' % C_DEST_DIR)

for name in train:
    os.system('cp %s/%s.npy %s/train/%s.npy' % (C_SRC_DIR, name, C_DEST_DIR, name))

for name in test:
    os.system('cp %s/%s.npy %s/test/%s.npy' % (C_SRC_DIR, name, C_DEST_DIR, name))

for name in val:
    os.system('cp %s/%s.npy %s/val/%s.npy' % (C_SRC_DIR, name, C_DEST_DIR, name))
