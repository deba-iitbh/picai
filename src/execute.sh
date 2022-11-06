#!/bin/sh

python train.py \
	--max_threads 10 \
	--validate_n_epochs 3 \
	--validate_min_epoch 1 \
	--resume_training 0 \
	--num_epochs 2 \
	--weights_dir ./../output/unet \
	--overviews_dir ./../workdir/results/UNet/overviews/ \
	--folds 0 \
	--batch_size 15
