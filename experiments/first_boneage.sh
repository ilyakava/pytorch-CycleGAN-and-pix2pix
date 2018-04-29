#!/bin/bash

python train.py --dataroot /scratch0/public/rsna-bone-age/pix2pixmerged_1k \
	 --name boneage50views --model pix2pix --which_model_netG unet_256 \
	 --which_direction AtoB --lambda_A 100 \
	 --dataset_mode aligned --no_lsgan --norm batch --pool_size 0
