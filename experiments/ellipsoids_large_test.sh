#!/bin/bash

python test.py --dataroot /vulcan/scratch/snanduri/iradon/data/ellipsoids_large/pix2pix_combined \
	 --name experiment_1 --model pix2pix --which_model_netG unet_256 \
	 --dataset_mode aligned --norm batch --phase test --gpu_ids 0,1,2,3,4,5,6,7 \
     --checkpoints_dir /vulcan/scratch/snanduri/iradon/data/checkpoints/pix2pix/ellipsoids_large/ \ 
     --results_dir /vulcan/scratch/snanduri/iradon/results/ellipsoids_large/
