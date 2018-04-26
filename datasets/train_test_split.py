# setup before running the fold:
# python scripts/combine_A_and_B.py --fold_A /path/to/data/A --fold_B /path/to/data/B --fold_AB /path/to/data

import os
import random as r

import pdb

A_SRC_DIR = '/scratch0/ilya/locDoc/data/siim-medical-images/50views'
B_SRC_DIR = '/scratch0/ilya/locDoc/data/siim-medical-images/337'
DEST_DIR = '/scratch0/ilya/locDoc/data/siim-medical-images/pix2pix_50_views'

# VAL_SZ = 10
TEST_SZ = 10

train = os.listdir(A_SRC_DIR)
test = r.sample(train, TEST_SZ)
for x in test:
	train.remove(x)
# val = r.sample(train, VAL_SZ)
# for x in val:
# 	train.remove(x)

os.system('mkdir %s/trainA' % DEST_DIR)
os.system('mkdir %s/trainB' % DEST_DIR)
os.system('mkdir %s/testA' % DEST_DIR)
os.system('mkdir %s/testB' % DEST_DIR)

for name in train:
	os.system('cp %s/%s %s/trainA/%s' % (A_SRC_DIR, name, DEST_DIR, name))
	os.system('cp %s/%s %s/trainB/%s' % (B_SRC_DIR, name, DEST_DIR, name))

for name in test:
	os.system('cp %s/%s %s/testA/%s' % (A_SRC_DIR, name, DEST_DIR, name))
	os.system('cp %s/%s %s/testB/%s' % (B_SRC_DIR, name, DEST_DIR, name))

# for name in val:
# 	os.system('cp %s/%s %s/A/val/%s' % (A_SRC_DIR, name, DEST_DIR, name))
# 	os.system('cp %s/%s %s/B/val/%s' % (B_SRC_DIR, name, DEST_DIR, name))
 