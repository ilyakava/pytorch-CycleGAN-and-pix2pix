#!/bin/bash

python test.py --dataroot /vulcan/scratch/snanduri/iradon/data/siim-medical-images/pix2pixmerged \
	 --name experiment_0 --model pix2pix --which_model_netG unet_256 \
	 --dataset_mode aligned --norm batch --phase val \
     --checkpoints_dir /vulcan/scratch/snanduri/iradon/data/checkpoints/pix2pix/siim/
