#!/bin/bash

python train.py --dataroot /vulcan/scratch/snanduri/iradon/data/ellipsoids_large/pix2pixmerged \
	 --name experiment_1 --model pix2pix --which_model_netG unet_256 \
	 --which_direction AtoB --lambda_A 100 \
	 --dataset_mode aligned \
     --batchSize 600 --gpu_ids 0,1,2,3,4,5,6,7 \
     --checkpoints_dir /vulcan/scratch/snanduri/iradon/data/checkpoints/pix2pix/ellipsoids_large/ \

